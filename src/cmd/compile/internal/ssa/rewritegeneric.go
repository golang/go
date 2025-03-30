// Code generated from _gen/generic.rules using 'go generate'; DO NOT EDIT.

package ssa

import "math"
import "math/bits"
import "cmd/internal/obj"
import "cmd/compile/internal/types"
import "cmd/compile/internal/ir"

func rewriteValuegeneric(v *Value) bool {
	switch v.Op {
	case OpAdd16:
		return rewriteValuegeneric_OpAdd16(v)
	case OpAdd32:
		return rewriteValuegeneric_OpAdd32(v)
	case OpAdd32F:
		return rewriteValuegeneric_OpAdd32F(v)
	case OpAdd64:
		return rewriteValuegeneric_OpAdd64(v)
	case OpAdd64F:
		return rewriteValuegeneric_OpAdd64F(v)
	case OpAdd64carry:
		return rewriteValuegeneric_OpAdd64carry(v)
	case OpAdd8:
		return rewriteValuegeneric_OpAdd8(v)
	case OpAddPtr:
		return rewriteValuegeneric_OpAddPtr(v)
	case OpAnd16:
		return rewriteValuegeneric_OpAnd16(v)
	case OpAnd32:
		return rewriteValuegeneric_OpAnd32(v)
	case OpAnd64:
		return rewriteValuegeneric_OpAnd64(v)
	case OpAnd8:
		return rewriteValuegeneric_OpAnd8(v)
	case OpAndB:
		return rewriteValuegeneric_OpAndB(v)
	case OpArraySelect:
		return rewriteValuegeneric_OpArraySelect(v)
	case OpBitLen16:
		return rewriteValuegeneric_OpBitLen16(v)
	case OpBitLen32:
		return rewriteValuegeneric_OpBitLen32(v)
	case OpBitLen64:
		return rewriteValuegeneric_OpBitLen64(v)
	case OpBitLen8:
		return rewriteValuegeneric_OpBitLen8(v)
	case OpCeil:
		return rewriteValuegeneric_OpCeil(v)
	case OpCom16:
		return rewriteValuegeneric_OpCom16(v)
	case OpCom32:
		return rewriteValuegeneric_OpCom32(v)
	case OpCom64:
		return rewriteValuegeneric_OpCom64(v)
	case OpCom8:
		return rewriteValuegeneric_OpCom8(v)
	case OpConstInterface:
		return rewriteValuegeneric_OpConstInterface(v)
	case OpConstSlice:
		return rewriteValuegeneric_OpConstSlice(v)
	case OpConstString:
		return rewriteValuegeneric_OpConstString(v)
	case OpConvert:
		return rewriteValuegeneric_OpConvert(v)
	case OpCtz16:
		return rewriteValuegeneric_OpCtz16(v)
	case OpCtz32:
		return rewriteValuegeneric_OpCtz32(v)
	case OpCtz64:
		return rewriteValuegeneric_OpCtz64(v)
	case OpCtz8:
		return rewriteValuegeneric_OpCtz8(v)
	case OpCvt32Fto32:
		return rewriteValuegeneric_OpCvt32Fto32(v)
	case OpCvt32Fto64:
		return rewriteValuegeneric_OpCvt32Fto64(v)
	case OpCvt32Fto64F:
		return rewriteValuegeneric_OpCvt32Fto64F(v)
	case OpCvt32to32F:
		return rewriteValuegeneric_OpCvt32to32F(v)
	case OpCvt32to64F:
		return rewriteValuegeneric_OpCvt32to64F(v)
	case OpCvt64Fto32:
		return rewriteValuegeneric_OpCvt64Fto32(v)
	case OpCvt64Fto32F:
		return rewriteValuegeneric_OpCvt64Fto32F(v)
	case OpCvt64Fto64:
		return rewriteValuegeneric_OpCvt64Fto64(v)
	case OpCvt64to32F:
		return rewriteValuegeneric_OpCvt64to32F(v)
	case OpCvt64to64F:
		return rewriteValuegeneric_OpCvt64to64F(v)
	case OpCvtBoolToUint8:
		return rewriteValuegeneric_OpCvtBoolToUint8(v)
	case OpDiv128u:
		return rewriteValuegeneric_OpDiv128u(v)
	case OpDiv16:
		return rewriteValuegeneric_OpDiv16(v)
	case OpDiv16u:
		return rewriteValuegeneric_OpDiv16u(v)
	case OpDiv32:
		return rewriteValuegeneric_OpDiv32(v)
	case OpDiv32F:
		return rewriteValuegeneric_OpDiv32F(v)
	case OpDiv32u:
		return rewriteValuegeneric_OpDiv32u(v)
	case OpDiv64:
		return rewriteValuegeneric_OpDiv64(v)
	case OpDiv64F:
		return rewriteValuegeneric_OpDiv64F(v)
	case OpDiv64u:
		return rewriteValuegeneric_OpDiv64u(v)
	case OpDiv8:
		return rewriteValuegeneric_OpDiv8(v)
	case OpDiv8u:
		return rewriteValuegeneric_OpDiv8u(v)
	case OpEq16:
		return rewriteValuegeneric_OpEq16(v)
	case OpEq32:
		return rewriteValuegeneric_OpEq32(v)
	case OpEq32F:
		return rewriteValuegeneric_OpEq32F(v)
	case OpEq64:
		return rewriteValuegeneric_OpEq64(v)
	case OpEq64F:
		return rewriteValuegeneric_OpEq64F(v)
	case OpEq8:
		return rewriteValuegeneric_OpEq8(v)
	case OpEqB:
		return rewriteValuegeneric_OpEqB(v)
	case OpEqInter:
		return rewriteValuegeneric_OpEqInter(v)
	case OpEqPtr:
		return rewriteValuegeneric_OpEqPtr(v)
	case OpEqSlice:
		return rewriteValuegeneric_OpEqSlice(v)
	case OpFloor:
		return rewriteValuegeneric_OpFloor(v)
	case OpIMake:
		return rewriteValuegeneric_OpIMake(v)
	case OpInterLECall:
		return rewriteValuegeneric_OpInterLECall(v)
	case OpIsInBounds:
		return rewriteValuegeneric_OpIsInBounds(v)
	case OpIsNonNil:
		return rewriteValuegeneric_OpIsNonNil(v)
	case OpIsSliceInBounds:
		return rewriteValuegeneric_OpIsSliceInBounds(v)
	case OpLeq16:
		return rewriteValuegeneric_OpLeq16(v)
	case OpLeq16U:
		return rewriteValuegeneric_OpLeq16U(v)
	case OpLeq32:
		return rewriteValuegeneric_OpLeq32(v)
	case OpLeq32F:
		return rewriteValuegeneric_OpLeq32F(v)
	case OpLeq32U:
		return rewriteValuegeneric_OpLeq32U(v)
	case OpLeq64:
		return rewriteValuegeneric_OpLeq64(v)
	case OpLeq64F:
		return rewriteValuegeneric_OpLeq64F(v)
	case OpLeq64U:
		return rewriteValuegeneric_OpLeq64U(v)
	case OpLeq8:
		return rewriteValuegeneric_OpLeq8(v)
	case OpLeq8U:
		return rewriteValuegeneric_OpLeq8U(v)
	case OpLess16:
		return rewriteValuegeneric_OpLess16(v)
	case OpLess16U:
		return rewriteValuegeneric_OpLess16U(v)
	case OpLess32:
		return rewriteValuegeneric_OpLess32(v)
	case OpLess32F:
		return rewriteValuegeneric_OpLess32F(v)
	case OpLess32U:
		return rewriteValuegeneric_OpLess32U(v)
	case OpLess64:
		return rewriteValuegeneric_OpLess64(v)
	case OpLess64F:
		return rewriteValuegeneric_OpLess64F(v)
	case OpLess64U:
		return rewriteValuegeneric_OpLess64U(v)
	case OpLess8:
		return rewriteValuegeneric_OpLess8(v)
	case OpLess8U:
		return rewriteValuegeneric_OpLess8U(v)
	case OpLoad:
		return rewriteValuegeneric_OpLoad(v)
	case OpLsh16x16:
		return rewriteValuegeneric_OpLsh16x16(v)
	case OpLsh16x32:
		return rewriteValuegeneric_OpLsh16x32(v)
	case OpLsh16x64:
		return rewriteValuegeneric_OpLsh16x64(v)
	case OpLsh16x8:
		return rewriteValuegeneric_OpLsh16x8(v)
	case OpLsh32x16:
		return rewriteValuegeneric_OpLsh32x16(v)
	case OpLsh32x32:
		return rewriteValuegeneric_OpLsh32x32(v)
	case OpLsh32x64:
		return rewriteValuegeneric_OpLsh32x64(v)
	case OpLsh32x8:
		return rewriteValuegeneric_OpLsh32x8(v)
	case OpLsh64x16:
		return rewriteValuegeneric_OpLsh64x16(v)
	case OpLsh64x32:
		return rewriteValuegeneric_OpLsh64x32(v)
	case OpLsh64x64:
		return rewriteValuegeneric_OpLsh64x64(v)
	case OpLsh64x8:
		return rewriteValuegeneric_OpLsh64x8(v)
	case OpLsh8x16:
		return rewriteValuegeneric_OpLsh8x16(v)
	case OpLsh8x32:
		return rewriteValuegeneric_OpLsh8x32(v)
	case OpLsh8x64:
		return rewriteValuegeneric_OpLsh8x64(v)
	case OpLsh8x8:
		return rewriteValuegeneric_OpLsh8x8(v)
	case OpMod16:
		return rewriteValuegeneric_OpMod16(v)
	case OpMod16u:
		return rewriteValuegeneric_OpMod16u(v)
	case OpMod32:
		return rewriteValuegeneric_OpMod32(v)
	case OpMod32u:
		return rewriteValuegeneric_OpMod32u(v)
	case OpMod64:
		return rewriteValuegeneric_OpMod64(v)
	case OpMod64u:
		return rewriteValuegeneric_OpMod64u(v)
	case OpMod8:
		return rewriteValuegeneric_OpMod8(v)
	case OpMod8u:
		return rewriteValuegeneric_OpMod8u(v)
	case OpMove:
		return rewriteValuegeneric_OpMove(v)
	case OpMul16:
		return rewriteValuegeneric_OpMul16(v)
	case OpMul32:
		return rewriteValuegeneric_OpMul32(v)
	case OpMul32F:
		return rewriteValuegeneric_OpMul32F(v)
	case OpMul32uhilo:
		return rewriteValuegeneric_OpMul32uhilo(v)
	case OpMul32uover:
		return rewriteValuegeneric_OpMul32uover(v)
	case OpMul64:
		return rewriteValuegeneric_OpMul64(v)
	case OpMul64F:
		return rewriteValuegeneric_OpMul64F(v)
	case OpMul64uhilo:
		return rewriteValuegeneric_OpMul64uhilo(v)
	case OpMul64uover:
		return rewriteValuegeneric_OpMul64uover(v)
	case OpMul8:
		return rewriteValuegeneric_OpMul8(v)
	case OpNeg16:
		return rewriteValuegeneric_OpNeg16(v)
	case OpNeg32:
		return rewriteValuegeneric_OpNeg32(v)
	case OpNeg32F:
		return rewriteValuegeneric_OpNeg32F(v)
	case OpNeg64:
		return rewriteValuegeneric_OpNeg64(v)
	case OpNeg64F:
		return rewriteValuegeneric_OpNeg64F(v)
	case OpNeg8:
		return rewriteValuegeneric_OpNeg8(v)
	case OpNeq16:
		return rewriteValuegeneric_OpNeq16(v)
	case OpNeq32:
		return rewriteValuegeneric_OpNeq32(v)
	case OpNeq32F:
		return rewriteValuegeneric_OpNeq32F(v)
	case OpNeq64:
		return rewriteValuegeneric_OpNeq64(v)
	case OpNeq64F:
		return rewriteValuegeneric_OpNeq64F(v)
	case OpNeq8:
		return rewriteValuegeneric_OpNeq8(v)
	case OpNeqB:
		return rewriteValuegeneric_OpNeqB(v)
	case OpNeqInter:
		return rewriteValuegeneric_OpNeqInter(v)
	case OpNeqPtr:
		return rewriteValuegeneric_OpNeqPtr(v)
	case OpNeqSlice:
		return rewriteValuegeneric_OpNeqSlice(v)
	case OpNilCheck:
		return rewriteValuegeneric_OpNilCheck(v)
	case OpNot:
		return rewriteValuegeneric_OpNot(v)
	case OpOffPtr:
		return rewriteValuegeneric_OpOffPtr(v)
	case OpOr16:
		return rewriteValuegeneric_OpOr16(v)
	case OpOr32:
		return rewriteValuegeneric_OpOr32(v)
	case OpOr64:
		return rewriteValuegeneric_OpOr64(v)
	case OpOr8:
		return rewriteValuegeneric_OpOr8(v)
	case OpOrB:
		return rewriteValuegeneric_OpOrB(v)
	case OpPhi:
		return rewriteValuegeneric_OpPhi(v)
	case OpPopCount16:
		return rewriteValuegeneric_OpPopCount16(v)
	case OpPopCount32:
		return rewriteValuegeneric_OpPopCount32(v)
	case OpPopCount64:
		return rewriteValuegeneric_OpPopCount64(v)
	case OpPopCount8:
		return rewriteValuegeneric_OpPopCount8(v)
	case OpPtrIndex:
		return rewriteValuegeneric_OpPtrIndex(v)
	case OpRotateLeft16:
		return rewriteValuegeneric_OpRotateLeft16(v)
	case OpRotateLeft32:
		return rewriteValuegeneric_OpRotateLeft32(v)
	case OpRotateLeft64:
		return rewriteValuegeneric_OpRotateLeft64(v)
	case OpRotateLeft8:
		return rewriteValuegeneric_OpRotateLeft8(v)
	case OpRound32F:
		return rewriteValuegeneric_OpRound32F(v)
	case OpRound64F:
		return rewriteValuegeneric_OpRound64F(v)
	case OpRoundToEven:
		return rewriteValuegeneric_OpRoundToEven(v)
	case OpRsh16Ux16:
		return rewriteValuegeneric_OpRsh16Ux16(v)
	case OpRsh16Ux32:
		return rewriteValuegeneric_OpRsh16Ux32(v)
	case OpRsh16Ux64:
		return rewriteValuegeneric_OpRsh16Ux64(v)
	case OpRsh16Ux8:
		return rewriteValuegeneric_OpRsh16Ux8(v)
	case OpRsh16x16:
		return rewriteValuegeneric_OpRsh16x16(v)
	case OpRsh16x32:
		return rewriteValuegeneric_OpRsh16x32(v)
	case OpRsh16x64:
		return rewriteValuegeneric_OpRsh16x64(v)
	case OpRsh16x8:
		return rewriteValuegeneric_OpRsh16x8(v)
	case OpRsh32Ux16:
		return rewriteValuegeneric_OpRsh32Ux16(v)
	case OpRsh32Ux32:
		return rewriteValuegeneric_OpRsh32Ux32(v)
	case OpRsh32Ux64:
		return rewriteValuegeneric_OpRsh32Ux64(v)
	case OpRsh32Ux8:
		return rewriteValuegeneric_OpRsh32Ux8(v)
	case OpRsh32x16:
		return rewriteValuegeneric_OpRsh32x16(v)
	case OpRsh32x32:
		return rewriteValuegeneric_OpRsh32x32(v)
	case OpRsh32x64:
		return rewriteValuegeneric_OpRsh32x64(v)
	case OpRsh32x8:
		return rewriteValuegeneric_OpRsh32x8(v)
	case OpRsh64Ux16:
		return rewriteValuegeneric_OpRsh64Ux16(v)
	case OpRsh64Ux32:
		return rewriteValuegeneric_OpRsh64Ux32(v)
	case OpRsh64Ux64:
		return rewriteValuegeneric_OpRsh64Ux64(v)
	case OpRsh64Ux8:
		return rewriteValuegeneric_OpRsh64Ux8(v)
	case OpRsh64x16:
		return rewriteValuegeneric_OpRsh64x16(v)
	case OpRsh64x32:
		return rewriteValuegeneric_OpRsh64x32(v)
	case OpRsh64x64:
		return rewriteValuegeneric_OpRsh64x64(v)
	case OpRsh64x8:
		return rewriteValuegeneric_OpRsh64x8(v)
	case OpRsh8Ux16:
		return rewriteValuegeneric_OpRsh8Ux16(v)
	case OpRsh8Ux32:
		return rewriteValuegeneric_OpRsh8Ux32(v)
	case OpRsh8Ux64:
		return rewriteValuegeneric_OpRsh8Ux64(v)
	case OpRsh8Ux8:
		return rewriteValuegeneric_OpRsh8Ux8(v)
	case OpRsh8x16:
		return rewriteValuegeneric_OpRsh8x16(v)
	case OpRsh8x32:
		return rewriteValuegeneric_OpRsh8x32(v)
	case OpRsh8x64:
		return rewriteValuegeneric_OpRsh8x64(v)
	case OpRsh8x8:
		return rewriteValuegeneric_OpRsh8x8(v)
	case OpSelect0:
		return rewriteValuegeneric_OpSelect0(v)
	case OpSelect1:
		return rewriteValuegeneric_OpSelect1(v)
	case OpSelectN:
		return rewriteValuegeneric_OpSelectN(v)
	case OpSignExt16to32:
		return rewriteValuegeneric_OpSignExt16to32(v)
	case OpSignExt16to64:
		return rewriteValuegeneric_OpSignExt16to64(v)
	case OpSignExt32to64:
		return rewriteValuegeneric_OpSignExt32to64(v)
	case OpSignExt8to16:
		return rewriteValuegeneric_OpSignExt8to16(v)
	case OpSignExt8to32:
		return rewriteValuegeneric_OpSignExt8to32(v)
	case OpSignExt8to64:
		return rewriteValuegeneric_OpSignExt8to64(v)
	case OpSliceCap:
		return rewriteValuegeneric_OpSliceCap(v)
	case OpSliceLen:
		return rewriteValuegeneric_OpSliceLen(v)
	case OpSlicePtr:
		return rewriteValuegeneric_OpSlicePtr(v)
	case OpSlicemask:
		return rewriteValuegeneric_OpSlicemask(v)
	case OpSqrt:
		return rewriteValuegeneric_OpSqrt(v)
	case OpStaticCall:
		return rewriteValuegeneric_OpStaticCall(v)
	case OpStaticLECall:
		return rewriteValuegeneric_OpStaticLECall(v)
	case OpStore:
		return rewriteValuegeneric_OpStore(v)
	case OpStringLen:
		return rewriteValuegeneric_OpStringLen(v)
	case OpStringPtr:
		return rewriteValuegeneric_OpStringPtr(v)
	case OpStructSelect:
		return rewriteValuegeneric_OpStructSelect(v)
	case OpSub16:
		return rewriteValuegeneric_OpSub16(v)
	case OpSub32:
		return rewriteValuegeneric_OpSub32(v)
	case OpSub32F:
		return rewriteValuegeneric_OpSub32F(v)
	case OpSub64:
		return rewriteValuegeneric_OpSub64(v)
	case OpSub64F:
		return rewriteValuegeneric_OpSub64F(v)
	case OpSub8:
		return rewriteValuegeneric_OpSub8(v)
	case OpTrunc:
		return rewriteValuegeneric_OpTrunc(v)
	case OpTrunc16to8:
		return rewriteValuegeneric_OpTrunc16to8(v)
	case OpTrunc32to16:
		return rewriteValuegeneric_OpTrunc32to16(v)
	case OpTrunc32to8:
		return rewriteValuegeneric_OpTrunc32to8(v)
	case OpTrunc64to16:
		return rewriteValuegeneric_OpTrunc64to16(v)
	case OpTrunc64to32:
		return rewriteValuegeneric_OpTrunc64to32(v)
	case OpTrunc64to8:
		return rewriteValuegeneric_OpTrunc64to8(v)
	case OpXor16:
		return rewriteValuegeneric_OpXor16(v)
	case OpXor32:
		return rewriteValuegeneric_OpXor32(v)
	case OpXor64:
		return rewriteValuegeneric_OpXor64(v)
	case OpXor8:
		return rewriteValuegeneric_OpXor8(v)
	case OpZero:
		return rewriteValuegeneric_OpZero(v)
	case OpZeroExt16to32:
		return rewriteValuegeneric_OpZeroExt16to32(v)
	case OpZeroExt16to64:
		return rewriteValuegeneric_OpZeroExt16to64(v)
	case OpZeroExt32to64:
		return rewriteValuegeneric_OpZeroExt32to64(v)
	case OpZeroExt8to16:
		return rewriteValuegeneric_OpZeroExt8to16(v)
	case OpZeroExt8to32:
		return rewriteValuegeneric_OpZeroExt8to32(v)
	case OpZeroExt8to64:
		return rewriteValuegeneric_OpZeroExt8to64(v)
	}
	return false
}
func rewriteValuegeneric_OpAdd16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (Add16 (Const16 [c]) (Const16 [d]))
	// result: (Const16 [c+d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1.AuxInt)
			v.reset(OpConst16)
			v.AuxInt = int16ToAuxInt(c + d)
			return true
		}
		break
	}
	// match: (Add16 <t> (Mul16 x y) (Mul16 x z))
	// result: (Mul16 x (Add16 <t> y z))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpMul16 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				if v_1.Op != OpMul16 {
					continue
				}
				_ = v_1.Args[1]
				v_1_0 := v_1.Args[0]
				v_1_1 := v_1.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, v_1_0, v_1_1 = _i2+1, v_1_1, v_1_0 {
					if x != v_1_0 {
						continue
					}
					z := v_1_1
					v.reset(OpMul16)
					v0 := b.NewValue0(v.Pos, OpAdd16, t)
					v0.AddArg2(y, z)
					v.AddArg2(x, v0)
					return true
				}
			}
		}
		break
	}
	// match: (Add16 (Const16 [0]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Add16 x (Neg16 y))
	// result: (Sub16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpNeg16 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpSub16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add16 (Com16 x) x)
	// result: (Const16 [-1])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom16 {
				continue
			}
			x := v_0.Args[0]
			if x != v_1 {
				continue
			}
			v.reset(OpConst16)
			v.AuxInt = int16ToAuxInt(-1)
			return true
		}
		break
	}
	// match: (Add16 (Sub16 x t) (Add16 t y))
	// result: (Add16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpSub16 {
				continue
			}
			t := v_0.Args[1]
			x := v_0.Args[0]
			if v_1.Op != OpAdd16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if t != v_1_0 {
					continue
				}
				y := v_1_1
				v.reset(OpAdd16)
				v.AddArg2(x, y)
				return true
			}
		}
		break
	}
	// match: (Add16 (Const16 [1]) (Com16 x))
	// result: (Neg16 x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 1 || v_1.Op != OpCom16 {
				continue
			}
			x := v_1.Args[0]
			v.reset(OpNeg16)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Add16 x (Sub16 y x))
	// result: y
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpSub16 {
				continue
			}
			_ = v_1.Args[1]
			y := v_1.Args[0]
			if x != v_1.Args[1] {
				continue
			}
			v.copyOf(y)
			return true
		}
		break
	}
	// match: (Add16 x (Add16 y (Sub16 z x)))
	// result: (Add16 y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAdd16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				y := v_1_0
				if v_1_1.Op != OpSub16 {
					continue
				}
				_ = v_1_1.Args[1]
				z := v_1_1.Args[0]
				if x != v_1_1.Args[1] {
					continue
				}
				v.reset(OpAdd16)
				v.AddArg2(y, z)
				return true
			}
		}
		break
	}
	// match: (Add16 (Add16 i:(Const16 <t>) z) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Add16 i (Add16 <t> z x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAdd16 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst16 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst16 && x.Op != OpConst16) {
					continue
				}
				v.reset(OpAdd16)
				v0 := b.NewValue0(v.Pos, OpAdd16, t)
				v0.AddArg2(z, x)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (Add16 (Sub16 i:(Const16 <t>) z) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Add16 i (Sub16 <t> x z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpSub16 {
				continue
			}
			z := v_0.Args[1]
			i := v_0.Args[0]
			if i.Op != OpConst16 {
				continue
			}
			t := i.Type
			x := v_1
			if !(z.Op != OpConst16 && x.Op != OpConst16) {
				continue
			}
			v.reset(OpAdd16)
			v0 := b.NewValue0(v.Pos, OpSub16, t)
			v0.AddArg2(x, z)
			v.AddArg2(i, v0)
			return true
		}
		break
	}
	// match: (Add16 (Const16 <t> [c]) (Add16 (Const16 <t> [d]) x))
	// result: (Add16 (Const16 <t> [c+d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpAdd16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst16 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt16(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpAdd16)
				v0 := b.NewValue0(v.Pos, OpConst16, t)
				v0.AuxInt = int16ToAuxInt(c + d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Add16 (Const16 <t> [c]) (Sub16 (Const16 <t> [d]) x))
	// result: (Sub16 (Const16 <t> [c+d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpSub16 {
				continue
			}
			x := v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpConst16 || v_1_0.Type != t {
				continue
			}
			d := auxIntToInt16(v_1_0.AuxInt)
			v.reset(OpSub16)
			v0 := b.NewValue0(v.Pos, OpConst16, t)
			v0.AuxInt = int16ToAuxInt(c + d)
			v.AddArg2(v0, x)
			return true
		}
		break
	}
	// match: (Add16 (Lsh16x64 x z:(Const64 <t> [c])) (Rsh16Ux64 x (Const64 [d])))
	// cond: c < 16 && d == 16-c && canRotate(config, 16)
	// result: (RotateLeft16 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLsh16x64 {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(z.AuxInt)
			if v_1.Op != OpRsh16Ux64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(c < 16 && d == 16-c && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Add16 left:(Lsh16x64 x y) right:(Rsh16Ux64 x (Sub64 (Const64 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh16x64 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh16Ux64 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub64 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst64 || auxIntToInt64(right_1_0.AuxInt) != 16 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add16 left:(Lsh16x32 x y) right:(Rsh16Ux32 x (Sub32 (Const32 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh16x32 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh16Ux32 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub32 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst32 || auxIntToInt32(right_1_0.AuxInt) != 16 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add16 left:(Lsh16x16 x y) right:(Rsh16Ux16 x (Sub16 (Const16 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh16x16 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh16Ux16 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub16 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst16 || auxIntToInt16(right_1_0.AuxInt) != 16 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add16 left:(Lsh16x8 x y) right:(Rsh16Ux8 x (Sub8 (Const8 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh16x8 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh16Ux8 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub8 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst8 || auxIntToInt8(right_1_0.AuxInt) != 16 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add16 right:(Rsh16Ux64 x y) left:(Lsh16x64 x z:(Sub64 (Const64 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh16Ux64 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh16x64 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub64 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst64 || auxIntToInt64(z_0.AuxInt) != 16 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Add16 right:(Rsh16Ux32 x y) left:(Lsh16x32 x z:(Sub32 (Const32 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh16Ux32 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh16x32 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub32 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst32 || auxIntToInt32(z_0.AuxInt) != 16 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Add16 right:(Rsh16Ux16 x y) left:(Lsh16x16 x z:(Sub16 (Const16 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh16Ux16 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh16x16 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub16 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst16 || auxIntToInt16(z_0.AuxInt) != 16 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Add16 right:(Rsh16Ux8 x y) left:(Lsh16x8 x z:(Sub8 (Const8 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh16Ux8 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh16x8 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub8 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst8 || auxIntToInt8(z_0.AuxInt) != 16 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpAdd32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (Add32 (Const32 [c]) (Const32 [d]))
	// result: (Const32 [c+d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			v.reset(OpConst32)
			v.AuxInt = int32ToAuxInt(c + d)
			return true
		}
		break
	}
	// match: (Add32 <t> (Mul32 x y) (Mul32 x z))
	// result: (Mul32 x (Add32 <t> y z))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpMul32 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				if v_1.Op != OpMul32 {
					continue
				}
				_ = v_1.Args[1]
				v_1_0 := v_1.Args[0]
				v_1_1 := v_1.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, v_1_0, v_1_1 = _i2+1, v_1_1, v_1_0 {
					if x != v_1_0 {
						continue
					}
					z := v_1_1
					v.reset(OpMul32)
					v0 := b.NewValue0(v.Pos, OpAdd32, t)
					v0.AddArg2(y, z)
					v.AddArg2(x, v0)
					return true
				}
			}
		}
		break
	}
	// match: (Add32 (Const32 [0]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Add32 x (Neg32 y))
	// result: (Sub32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpNeg32 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpSub32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add32 (Com32 x) x)
	// result: (Const32 [-1])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom32 {
				continue
			}
			x := v_0.Args[0]
			if x != v_1 {
				continue
			}
			v.reset(OpConst32)
			v.AuxInt = int32ToAuxInt(-1)
			return true
		}
		break
	}
	// match: (Add32 (Sub32 x t) (Add32 t y))
	// result: (Add32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpSub32 {
				continue
			}
			t := v_0.Args[1]
			x := v_0.Args[0]
			if v_1.Op != OpAdd32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if t != v_1_0 {
					continue
				}
				y := v_1_1
				v.reset(OpAdd32)
				v.AddArg2(x, y)
				return true
			}
		}
		break
	}
	// match: (Add32 (Const32 [1]) (Com32 x))
	// result: (Neg32 x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 1 || v_1.Op != OpCom32 {
				continue
			}
			x := v_1.Args[0]
			v.reset(OpNeg32)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Add32 x (Sub32 y x))
	// result: y
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpSub32 {
				continue
			}
			_ = v_1.Args[1]
			y := v_1.Args[0]
			if x != v_1.Args[1] {
				continue
			}
			v.copyOf(y)
			return true
		}
		break
	}
	// match: (Add32 x (Add32 y (Sub32 z x)))
	// result: (Add32 y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAdd32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				y := v_1_0
				if v_1_1.Op != OpSub32 {
					continue
				}
				_ = v_1_1.Args[1]
				z := v_1_1.Args[0]
				if x != v_1_1.Args[1] {
					continue
				}
				v.reset(OpAdd32)
				v.AddArg2(y, z)
				return true
			}
		}
		break
	}
	// match: (Add32 (Add32 i:(Const32 <t>) z) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Add32 i (Add32 <t> z x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAdd32 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst32 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst32 && x.Op != OpConst32) {
					continue
				}
				v.reset(OpAdd32)
				v0 := b.NewValue0(v.Pos, OpAdd32, t)
				v0.AddArg2(z, x)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (Add32 (Sub32 i:(Const32 <t>) z) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Add32 i (Sub32 <t> x z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpSub32 {
				continue
			}
			z := v_0.Args[1]
			i := v_0.Args[0]
			if i.Op != OpConst32 {
				continue
			}
			t := i.Type
			x := v_1
			if !(z.Op != OpConst32 && x.Op != OpConst32) {
				continue
			}
			v.reset(OpAdd32)
			v0 := b.NewValue0(v.Pos, OpSub32, t)
			v0.AddArg2(x, z)
			v.AddArg2(i, v0)
			return true
		}
		break
	}
	// match: (Add32 (Const32 <t> [c]) (Add32 (Const32 <t> [d]) x))
	// result: (Add32 (Const32 <t> [c+d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpAdd32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst32 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt32(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpAdd32)
				v0 := b.NewValue0(v.Pos, OpConst32, t)
				v0.AuxInt = int32ToAuxInt(c + d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Add32 (Const32 <t> [c]) (Sub32 (Const32 <t> [d]) x))
	// result: (Sub32 (Const32 <t> [c+d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpSub32 {
				continue
			}
			x := v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpConst32 || v_1_0.Type != t {
				continue
			}
			d := auxIntToInt32(v_1_0.AuxInt)
			v.reset(OpSub32)
			v0 := b.NewValue0(v.Pos, OpConst32, t)
			v0.AuxInt = int32ToAuxInt(c + d)
			v.AddArg2(v0, x)
			return true
		}
		break
	}
	// match: (Add32 (Lsh32x64 x z:(Const64 <t> [c])) (Rsh32Ux64 x (Const64 [d])))
	// cond: c < 32 && d == 32-c && canRotate(config, 32)
	// result: (RotateLeft32 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLsh32x64 {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(z.AuxInt)
			if v_1.Op != OpRsh32Ux64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(c < 32 && d == 32-c && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Add32 left:(Lsh32x64 x y) right:(Rsh32Ux64 x (Sub64 (Const64 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh32x64 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh32Ux64 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub64 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst64 || auxIntToInt64(right_1_0.AuxInt) != 32 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add32 left:(Lsh32x32 x y) right:(Rsh32Ux32 x (Sub32 (Const32 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh32x32 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh32Ux32 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub32 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst32 || auxIntToInt32(right_1_0.AuxInt) != 32 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add32 left:(Lsh32x16 x y) right:(Rsh32Ux16 x (Sub16 (Const16 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh32x16 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh32Ux16 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub16 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst16 || auxIntToInt16(right_1_0.AuxInt) != 32 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add32 left:(Lsh32x8 x y) right:(Rsh32Ux8 x (Sub8 (Const8 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh32x8 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh32Ux8 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub8 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst8 || auxIntToInt8(right_1_0.AuxInt) != 32 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add32 right:(Rsh32Ux64 x y) left:(Lsh32x64 x z:(Sub64 (Const64 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh32Ux64 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh32x64 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub64 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst64 || auxIntToInt64(z_0.AuxInt) != 32 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Add32 right:(Rsh32Ux32 x y) left:(Lsh32x32 x z:(Sub32 (Const32 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh32Ux32 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh32x32 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub32 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst32 || auxIntToInt32(z_0.AuxInt) != 32 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Add32 right:(Rsh32Ux16 x y) left:(Lsh32x16 x z:(Sub16 (Const16 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh32Ux16 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh32x16 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub16 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst16 || auxIntToInt16(z_0.AuxInt) != 32 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Add32 right:(Rsh32Ux8 x y) left:(Lsh32x8 x z:(Sub8 (Const8 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh32Ux8 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh32x8 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub8 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst8 || auxIntToInt8(z_0.AuxInt) != 32 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpAdd32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Add32F (Const32F [c]) (Const32F [d]))
	// cond: c+d == c+d
	// result: (Const32F [c+d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32F {
				continue
			}
			c := auxIntToFloat32(v_0.AuxInt)
			if v_1.Op != OpConst32F {
				continue
			}
			d := auxIntToFloat32(v_1.AuxInt)
			if !(c+d == c+d) {
				continue
			}
			v.reset(OpConst32F)
			v.AuxInt = float32ToAuxInt(c + d)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpAdd64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (Add64 (Const64 [c]) (Const64 [d]))
	// result: (Const64 [c+d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpConst64)
			v.AuxInt = int64ToAuxInt(c + d)
			return true
		}
		break
	}
	// match: (Add64 <t> (Mul64 x y) (Mul64 x z))
	// result: (Mul64 x (Add64 <t> y z))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpMul64 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				if v_1.Op != OpMul64 {
					continue
				}
				_ = v_1.Args[1]
				v_1_0 := v_1.Args[0]
				v_1_1 := v_1.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, v_1_0, v_1_1 = _i2+1, v_1_1, v_1_0 {
					if x != v_1_0 {
						continue
					}
					z := v_1_1
					v.reset(OpMul64)
					v0 := b.NewValue0(v.Pos, OpAdd64, t)
					v0.AddArg2(y, z)
					v.AddArg2(x, v0)
					return true
				}
			}
		}
		break
	}
	// match: (Add64 (Const64 [0]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Add64 x (Neg64 y))
	// result: (Sub64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpNeg64 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpSub64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add64 (Com64 x) x)
	// result: (Const64 [-1])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom64 {
				continue
			}
			x := v_0.Args[0]
			if x != v_1 {
				continue
			}
			v.reset(OpConst64)
			v.AuxInt = int64ToAuxInt(-1)
			return true
		}
		break
	}
	// match: (Add64 (Sub64 x t) (Add64 t y))
	// result: (Add64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpSub64 {
				continue
			}
			t := v_0.Args[1]
			x := v_0.Args[0]
			if v_1.Op != OpAdd64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if t != v_1_0 {
					continue
				}
				y := v_1_1
				v.reset(OpAdd64)
				v.AddArg2(x, y)
				return true
			}
		}
		break
	}
	// match: (Add64 (Const64 [1]) (Com64 x))
	// result: (Neg64 x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 1 || v_1.Op != OpCom64 {
				continue
			}
			x := v_1.Args[0]
			v.reset(OpNeg64)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Add64 x (Sub64 y x))
	// result: y
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpSub64 {
				continue
			}
			_ = v_1.Args[1]
			y := v_1.Args[0]
			if x != v_1.Args[1] {
				continue
			}
			v.copyOf(y)
			return true
		}
		break
	}
	// match: (Add64 x (Add64 y (Sub64 z x)))
	// result: (Add64 y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAdd64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				y := v_1_0
				if v_1_1.Op != OpSub64 {
					continue
				}
				_ = v_1_1.Args[1]
				z := v_1_1.Args[0]
				if x != v_1_1.Args[1] {
					continue
				}
				v.reset(OpAdd64)
				v.AddArg2(y, z)
				return true
			}
		}
		break
	}
	// match: (Add64 (Add64 i:(Const64 <t>) z) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Add64 i (Add64 <t> z x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAdd64 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst64 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst64 && x.Op != OpConst64) {
					continue
				}
				v.reset(OpAdd64)
				v0 := b.NewValue0(v.Pos, OpAdd64, t)
				v0.AddArg2(z, x)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (Add64 (Sub64 i:(Const64 <t>) z) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Add64 i (Sub64 <t> x z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpSub64 {
				continue
			}
			z := v_0.Args[1]
			i := v_0.Args[0]
			if i.Op != OpConst64 {
				continue
			}
			t := i.Type
			x := v_1
			if !(z.Op != OpConst64 && x.Op != OpConst64) {
				continue
			}
			v.reset(OpAdd64)
			v0 := b.NewValue0(v.Pos, OpSub64, t)
			v0.AddArg2(x, z)
			v.AddArg2(i, v0)
			return true
		}
		break
	}
	// match: (Add64 (Const64 <t> [c]) (Add64 (Const64 <t> [d]) x))
	// result: (Add64 (Const64 <t> [c+d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpAdd64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst64 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt64(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpAdd64)
				v0 := b.NewValue0(v.Pos, OpConst64, t)
				v0.AuxInt = int64ToAuxInt(c + d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Add64 (Const64 <t> [c]) (Sub64 (Const64 <t> [d]) x))
	// result: (Sub64 (Const64 <t> [c+d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpSub64 {
				continue
			}
			x := v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpConst64 || v_1_0.Type != t {
				continue
			}
			d := auxIntToInt64(v_1_0.AuxInt)
			v.reset(OpSub64)
			v0 := b.NewValue0(v.Pos, OpConst64, t)
			v0.AuxInt = int64ToAuxInt(c + d)
			v.AddArg2(v0, x)
			return true
		}
		break
	}
	// match: (Add64 (Lsh64x64 x z:(Const64 <t> [c])) (Rsh64Ux64 x (Const64 [d])))
	// cond: c < 64 && d == 64-c && canRotate(config, 64)
	// result: (RotateLeft64 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLsh64x64 {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(z.AuxInt)
			if v_1.Op != OpRsh64Ux64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(c < 64 && d == 64-c && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Add64 left:(Lsh64x64 x y) right:(Rsh64Ux64 x (Sub64 (Const64 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh64x64 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh64Ux64 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub64 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst64 || auxIntToInt64(right_1_0.AuxInt) != 64 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add64 left:(Lsh64x32 x y) right:(Rsh64Ux32 x (Sub32 (Const32 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh64x32 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh64Ux32 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub32 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst32 || auxIntToInt32(right_1_0.AuxInt) != 64 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add64 left:(Lsh64x16 x y) right:(Rsh64Ux16 x (Sub16 (Const16 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh64x16 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh64Ux16 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub16 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst16 || auxIntToInt16(right_1_0.AuxInt) != 64 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add64 left:(Lsh64x8 x y) right:(Rsh64Ux8 x (Sub8 (Const8 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh64x8 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh64Ux8 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub8 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst8 || auxIntToInt8(right_1_0.AuxInt) != 64 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add64 right:(Rsh64Ux64 x y) left:(Lsh64x64 x z:(Sub64 (Const64 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh64Ux64 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh64x64 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub64 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst64 || auxIntToInt64(z_0.AuxInt) != 64 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Add64 right:(Rsh64Ux32 x y) left:(Lsh64x32 x z:(Sub32 (Const32 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh64Ux32 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh64x32 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub32 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst32 || auxIntToInt32(z_0.AuxInt) != 64 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Add64 right:(Rsh64Ux16 x y) left:(Lsh64x16 x z:(Sub16 (Const16 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh64Ux16 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh64x16 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub16 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst16 || auxIntToInt16(z_0.AuxInt) != 64 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Add64 right:(Rsh64Ux8 x y) left:(Lsh64x8 x z:(Sub8 (Const8 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh64Ux8 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh64x8 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub8 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst8 || auxIntToInt8(z_0.AuxInt) != 64 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpAdd64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Add64F (Const64F [c]) (Const64F [d]))
	// cond: c+d == c+d
	// result: (Const64F [c+d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64F {
				continue
			}
			c := auxIntToFloat64(v_0.AuxInt)
			if v_1.Op != OpConst64F {
				continue
			}
			d := auxIntToFloat64(v_1.AuxInt)
			if !(c+d == c+d) {
				continue
			}
			v.reset(OpConst64F)
			v.AuxInt = float64ToAuxInt(c + d)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpAdd64carry(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Add64carry (Const64 <t> [x]) (Const64 [y]) (Const64 [c]))
	// cond: c >= 0 && c <= 1
	// result: (MakeTuple (Const64 <t> [bitsAdd64(x, y, c).sum]) (Const64 <t> [bitsAdd64(x, y, c).carry]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			t := v_0.Type
			x := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			y := auxIntToInt64(v_1.AuxInt)
			if v_2.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_2.AuxInt)
			if !(c >= 0 && c <= 1) {
				continue
			}
			v.reset(OpMakeTuple)
			v0 := b.NewValue0(v.Pos, OpConst64, t)
			v0.AuxInt = int64ToAuxInt(bitsAdd64(x, y, c).sum)
			v1 := b.NewValue0(v.Pos, OpConst64, t)
			v1.AuxInt = int64ToAuxInt(bitsAdd64(x, y, c).carry)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpAdd8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (Add8 (Const8 [c]) (Const8 [d]))
	// result: (Const8 [c+d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1.AuxInt)
			v.reset(OpConst8)
			v.AuxInt = int8ToAuxInt(c + d)
			return true
		}
		break
	}
	// match: (Add8 <t> (Mul8 x y) (Mul8 x z))
	// result: (Mul8 x (Add8 <t> y z))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpMul8 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				if v_1.Op != OpMul8 {
					continue
				}
				_ = v_1.Args[1]
				v_1_0 := v_1.Args[0]
				v_1_1 := v_1.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, v_1_0, v_1_1 = _i2+1, v_1_1, v_1_0 {
					if x != v_1_0 {
						continue
					}
					z := v_1_1
					v.reset(OpMul8)
					v0 := b.NewValue0(v.Pos, OpAdd8, t)
					v0.AddArg2(y, z)
					v.AddArg2(x, v0)
					return true
				}
			}
		}
		break
	}
	// match: (Add8 (Const8 [0]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Add8 x (Neg8 y))
	// result: (Sub8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpNeg8 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpSub8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add8 (Com8 x) x)
	// result: (Const8 [-1])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom8 {
				continue
			}
			x := v_0.Args[0]
			if x != v_1 {
				continue
			}
			v.reset(OpConst8)
			v.AuxInt = int8ToAuxInt(-1)
			return true
		}
		break
	}
	// match: (Add8 (Sub8 x t) (Add8 t y))
	// result: (Add8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpSub8 {
				continue
			}
			t := v_0.Args[1]
			x := v_0.Args[0]
			if v_1.Op != OpAdd8 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if t != v_1_0 {
					continue
				}
				y := v_1_1
				v.reset(OpAdd8)
				v.AddArg2(x, y)
				return true
			}
		}
		break
	}
	// match: (Add8 (Const8 [1]) (Com8 x))
	// result: (Neg8 x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 1 || v_1.Op != OpCom8 {
				continue
			}
			x := v_1.Args[0]
			v.reset(OpNeg8)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Add8 x (Sub8 y x))
	// result: y
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpSub8 {
				continue
			}
			_ = v_1.Args[1]
			y := v_1.Args[0]
			if x != v_1.Args[1] {
				continue
			}
			v.copyOf(y)
			return true
		}
		break
	}
	// match: (Add8 x (Add8 y (Sub8 z x)))
	// result: (Add8 y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAdd8 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				y := v_1_0
				if v_1_1.Op != OpSub8 {
					continue
				}
				_ = v_1_1.Args[1]
				z := v_1_1.Args[0]
				if x != v_1_1.Args[1] {
					continue
				}
				v.reset(OpAdd8)
				v.AddArg2(y, z)
				return true
			}
		}
		break
	}
	// match: (Add8 (Add8 i:(Const8 <t>) z) x)
	// cond: (z.Op != OpConst8 && x.Op != OpConst8)
	// result: (Add8 i (Add8 <t> z x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAdd8 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst8 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst8 && x.Op != OpConst8) {
					continue
				}
				v.reset(OpAdd8)
				v0 := b.NewValue0(v.Pos, OpAdd8, t)
				v0.AddArg2(z, x)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (Add8 (Sub8 i:(Const8 <t>) z) x)
	// cond: (z.Op != OpConst8 && x.Op != OpConst8)
	// result: (Add8 i (Sub8 <t> x z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpSub8 {
				continue
			}
			z := v_0.Args[1]
			i := v_0.Args[0]
			if i.Op != OpConst8 {
				continue
			}
			t := i.Type
			x := v_1
			if !(z.Op != OpConst8 && x.Op != OpConst8) {
				continue
			}
			v.reset(OpAdd8)
			v0 := b.NewValue0(v.Pos, OpSub8, t)
			v0.AddArg2(x, z)
			v.AddArg2(i, v0)
			return true
		}
		break
	}
	// match: (Add8 (Const8 <t> [c]) (Add8 (Const8 <t> [d]) x))
	// result: (Add8 (Const8 <t> [c+d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpAdd8 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst8 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt8(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpAdd8)
				v0 := b.NewValue0(v.Pos, OpConst8, t)
				v0.AuxInt = int8ToAuxInt(c + d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Add8 (Const8 <t> [c]) (Sub8 (Const8 <t> [d]) x))
	// result: (Sub8 (Const8 <t> [c+d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpSub8 {
				continue
			}
			x := v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpConst8 || v_1_0.Type != t {
				continue
			}
			d := auxIntToInt8(v_1_0.AuxInt)
			v.reset(OpSub8)
			v0 := b.NewValue0(v.Pos, OpConst8, t)
			v0.AuxInt = int8ToAuxInt(c + d)
			v.AddArg2(v0, x)
			return true
		}
		break
	}
	// match: (Add8 (Lsh8x64 x z:(Const64 <t> [c])) (Rsh8Ux64 x (Const64 [d])))
	// cond: c < 8 && d == 8-c && canRotate(config, 8)
	// result: (RotateLeft8 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLsh8x64 {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(z.AuxInt)
			if v_1.Op != OpRsh8Ux64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(c < 8 && d == 8-c && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Add8 left:(Lsh8x64 x y) right:(Rsh8Ux64 x (Sub64 (Const64 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh8x64 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh8Ux64 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub64 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst64 || auxIntToInt64(right_1_0.AuxInt) != 8 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add8 left:(Lsh8x32 x y) right:(Rsh8Ux32 x (Sub32 (Const32 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh8x32 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh8Ux32 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub32 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst32 || auxIntToInt32(right_1_0.AuxInt) != 8 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add8 left:(Lsh8x16 x y) right:(Rsh8Ux16 x (Sub16 (Const16 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh8x16 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh8Ux16 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub16 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst16 || auxIntToInt16(right_1_0.AuxInt) != 8 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add8 left:(Lsh8x8 x y) right:(Rsh8Ux8 x (Sub8 (Const8 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh8x8 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh8Ux8 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub8 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst8 || auxIntToInt8(right_1_0.AuxInt) != 8 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Add8 right:(Rsh8Ux64 x y) left:(Lsh8x64 x z:(Sub64 (Const64 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh8Ux64 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh8x64 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub64 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst64 || auxIntToInt64(z_0.AuxInt) != 8 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Add8 right:(Rsh8Ux32 x y) left:(Lsh8x32 x z:(Sub32 (Const32 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh8Ux32 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh8x32 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub32 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst32 || auxIntToInt32(z_0.AuxInt) != 8 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Add8 right:(Rsh8Ux16 x y) left:(Lsh8x16 x z:(Sub16 (Const16 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh8Ux16 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh8x16 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub16 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst16 || auxIntToInt16(z_0.AuxInt) != 8 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Add8 right:(Rsh8Ux8 x y) left:(Lsh8x8 x z:(Sub8 (Const8 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh8Ux8 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh8x8 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub8 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst8 || auxIntToInt8(z_0.AuxInt) != 8 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpAddPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AddPtr <t> x (Const64 [c]))
	// result: (OffPtr <t> x [c])
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpOffPtr)
		v.Type = t
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (AddPtr <t> x (Const32 [c]))
	// result: (OffPtr <t> x [int64(c)])
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpOffPtr)
		v.Type = t
		v.AuxInt = int64ToAuxInt(int64(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAnd16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (And16 (Const16 [c]) (Const16 [d]))
	// result: (Const16 [c&d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1.AuxInt)
			v.reset(OpConst16)
			v.AuxInt = int16ToAuxInt(c & d)
			return true
		}
		break
	}
	// match: (And16 <t> (Com16 x) (Com16 y))
	// result: (Com16 (Or16 <t> x y))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom16 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != OpCom16 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpCom16)
			v0 := b.NewValue0(v.Pos, OpOr16, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (And16 (Const16 [m]) (Rsh16Ux64 _ (Const64 [c])))
	// cond: c >= int64(16-ntz16(m))
	// result: (Const16 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			m := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpRsh16Ux64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_1.AuxInt)
			if !(c >= int64(16-ntz16(m))) {
				continue
			}
			v.reset(OpConst16)
			v.AuxInt = int16ToAuxInt(0)
			return true
		}
		break
	}
	// match: (And16 (Const16 [m]) (Lsh16x64 _ (Const64 [c])))
	// cond: c >= int64(16-nlz16(m))
	// result: (Const16 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			m := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpLsh16x64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_1.AuxInt)
			if !(c >= int64(16-nlz16(m))) {
				continue
			}
			v.reset(OpConst16)
			v.AuxInt = int16ToAuxInt(0)
			return true
		}
		break
	}
	// match: (And16 x x)
	// result: x
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (And16 (Const16 [-1]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != -1 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (And16 (Const16 [0]) _)
	// result: (Const16 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
				continue
			}
			v.reset(OpConst16)
			v.AuxInt = int16ToAuxInt(0)
			return true
		}
		break
	}
	// match: (And16 (Com16 x) x)
	// result: (Const16 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom16 {
				continue
			}
			x := v_0.Args[0]
			if x != v_1 {
				continue
			}
			v.reset(OpConst16)
			v.AuxInt = int16ToAuxInt(0)
			return true
		}
		break
	}
	// match: (And16 x (And16 x y))
	// result: (And16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAnd16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				y := v_1_1
				v.reset(OpAnd16)
				v.AddArg2(x, y)
				return true
			}
		}
		break
	}
	// match: (And16 (And16 i:(Const16 <t>) z) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (And16 i (And16 <t> z x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAnd16 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst16 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst16 && x.Op != OpConst16) {
					continue
				}
				v.reset(OpAnd16)
				v0 := b.NewValue0(v.Pos, OpAnd16, t)
				v0.AddArg2(z, x)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (And16 (Const16 <t> [c]) (And16 (Const16 <t> [d]) x))
	// result: (And16 (Const16 <t> [c&d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpAnd16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst16 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt16(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpAnd16)
				v0 := b.NewValue0(v.Pos, OpConst16, t)
				v0.AuxInt = int16ToAuxInt(c & d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpAnd32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (And32 (Const32 [c]) (Const32 [d]))
	// result: (Const32 [c&d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			v.reset(OpConst32)
			v.AuxInt = int32ToAuxInt(c & d)
			return true
		}
		break
	}
	// match: (And32 <t> (Com32 x) (Com32 y))
	// result: (Com32 (Or32 <t> x y))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom32 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != OpCom32 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpCom32)
			v0 := b.NewValue0(v.Pos, OpOr32, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (And32 (Const32 [m]) (Rsh32Ux64 _ (Const64 [c])))
	// cond: c >= int64(32-ntz32(m))
	// result: (Const32 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			m := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpRsh32Ux64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_1.AuxInt)
			if !(c >= int64(32-ntz32(m))) {
				continue
			}
			v.reset(OpConst32)
			v.AuxInt = int32ToAuxInt(0)
			return true
		}
		break
	}
	// match: (And32 (Const32 [m]) (Lsh32x64 _ (Const64 [c])))
	// cond: c >= int64(32-nlz32(m))
	// result: (Const32 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			m := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpLsh32x64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_1.AuxInt)
			if !(c >= int64(32-nlz32(m))) {
				continue
			}
			v.reset(OpConst32)
			v.AuxInt = int32ToAuxInt(0)
			return true
		}
		break
	}
	// match: (And32 x x)
	// result: x
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (And32 (Const32 [-1]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != -1 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (And32 (Const32 [0]) _)
	// result: (Const32 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
				continue
			}
			v.reset(OpConst32)
			v.AuxInt = int32ToAuxInt(0)
			return true
		}
		break
	}
	// match: (And32 (Com32 x) x)
	// result: (Const32 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom32 {
				continue
			}
			x := v_0.Args[0]
			if x != v_1 {
				continue
			}
			v.reset(OpConst32)
			v.AuxInt = int32ToAuxInt(0)
			return true
		}
		break
	}
	// match: (And32 x (And32 x y))
	// result: (And32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAnd32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				y := v_1_1
				v.reset(OpAnd32)
				v.AddArg2(x, y)
				return true
			}
		}
		break
	}
	// match: (And32 (And32 i:(Const32 <t>) z) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (And32 i (And32 <t> z x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAnd32 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst32 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst32 && x.Op != OpConst32) {
					continue
				}
				v.reset(OpAnd32)
				v0 := b.NewValue0(v.Pos, OpAnd32, t)
				v0.AddArg2(z, x)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (And32 (Const32 <t> [c]) (And32 (Const32 <t> [d]) x))
	// result: (And32 (Const32 <t> [c&d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpAnd32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst32 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt32(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpAnd32)
				v0 := b.NewValue0(v.Pos, OpConst32, t)
				v0.AuxInt = int32ToAuxInt(c & d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpAnd64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (And64 (Const64 [c]) (Const64 [d]))
	// result: (Const64 [c&d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpConst64)
			v.AuxInt = int64ToAuxInt(c & d)
			return true
		}
		break
	}
	// match: (And64 <t> (Com64 x) (Com64 y))
	// result: (Com64 (Or64 <t> x y))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom64 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != OpCom64 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpCom64)
			v0 := b.NewValue0(v.Pos, OpOr64, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (And64 (Const64 [m]) (Rsh64Ux64 _ (Const64 [c])))
	// cond: c >= int64(64-ntz64(m))
	// result: (Const64 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			m := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpRsh64Ux64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_1.AuxInt)
			if !(c >= int64(64-ntz64(m))) {
				continue
			}
			v.reset(OpConst64)
			v.AuxInt = int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (And64 (Const64 [m]) (Lsh64x64 _ (Const64 [c])))
	// cond: c >= int64(64-nlz64(m))
	// result: (Const64 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			m := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpLsh64x64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_1.AuxInt)
			if !(c >= int64(64-nlz64(m))) {
				continue
			}
			v.reset(OpConst64)
			v.AuxInt = int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (And64 x x)
	// result: x
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (And64 (Const64 [-1]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != -1 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (And64 (Const64 [0]) _)
	// result: (Const64 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
				continue
			}
			v.reset(OpConst64)
			v.AuxInt = int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (And64 (Com64 x) x)
	// result: (Const64 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom64 {
				continue
			}
			x := v_0.Args[0]
			if x != v_1 {
				continue
			}
			v.reset(OpConst64)
			v.AuxInt = int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (And64 x (And64 x y))
	// result: (And64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAnd64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				y := v_1_1
				v.reset(OpAnd64)
				v.AddArg2(x, y)
				return true
			}
		}
		break
	}
	// match: (And64 (And64 i:(Const64 <t>) z) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (And64 i (And64 <t> z x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAnd64 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst64 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst64 && x.Op != OpConst64) {
					continue
				}
				v.reset(OpAnd64)
				v0 := b.NewValue0(v.Pos, OpAnd64, t)
				v0.AddArg2(z, x)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (And64 (Const64 <t> [c]) (And64 (Const64 <t> [d]) x))
	// result: (And64 (Const64 <t> [c&d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpAnd64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst64 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt64(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpAnd64)
				v0 := b.NewValue0(v.Pos, OpConst64, t)
				v0.AuxInt = int64ToAuxInt(c & d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpAnd8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (And8 (Const8 [c]) (Const8 [d]))
	// result: (Const8 [c&d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1.AuxInt)
			v.reset(OpConst8)
			v.AuxInt = int8ToAuxInt(c & d)
			return true
		}
		break
	}
	// match: (And8 <t> (Com8 x) (Com8 y))
	// result: (Com8 (Or8 <t> x y))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom8 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != OpCom8 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpCom8)
			v0 := b.NewValue0(v.Pos, OpOr8, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (And8 (Const8 [m]) (Rsh8Ux64 _ (Const64 [c])))
	// cond: c >= int64(8-ntz8(m))
	// result: (Const8 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			m := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpRsh8Ux64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_1.AuxInt)
			if !(c >= int64(8-ntz8(m))) {
				continue
			}
			v.reset(OpConst8)
			v.AuxInt = int8ToAuxInt(0)
			return true
		}
		break
	}
	// match: (And8 (Const8 [m]) (Lsh8x64 _ (Const64 [c])))
	// cond: c >= int64(8-nlz8(m))
	// result: (Const8 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			m := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpLsh8x64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_1.AuxInt)
			if !(c >= int64(8-nlz8(m))) {
				continue
			}
			v.reset(OpConst8)
			v.AuxInt = int8ToAuxInt(0)
			return true
		}
		break
	}
	// match: (And8 x x)
	// result: x
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (And8 (Const8 [-1]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != -1 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (And8 (Const8 [0]) _)
	// result: (Const8 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
				continue
			}
			v.reset(OpConst8)
			v.AuxInt = int8ToAuxInt(0)
			return true
		}
		break
	}
	// match: (And8 (Com8 x) x)
	// result: (Const8 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom8 {
				continue
			}
			x := v_0.Args[0]
			if x != v_1 {
				continue
			}
			v.reset(OpConst8)
			v.AuxInt = int8ToAuxInt(0)
			return true
		}
		break
	}
	// match: (And8 x (And8 x y))
	// result: (And8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAnd8 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				y := v_1_1
				v.reset(OpAnd8)
				v.AddArg2(x, y)
				return true
			}
		}
		break
	}
	// match: (And8 (And8 i:(Const8 <t>) z) x)
	// cond: (z.Op != OpConst8 && x.Op != OpConst8)
	// result: (And8 i (And8 <t> z x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAnd8 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst8 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst8 && x.Op != OpConst8) {
					continue
				}
				v.reset(OpAnd8)
				v0 := b.NewValue0(v.Pos, OpAnd8, t)
				v0.AddArg2(z, x)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (And8 (Const8 <t> [c]) (And8 (Const8 <t> [d]) x))
	// result: (And8 (Const8 <t> [c&d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpAnd8 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst8 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt8(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpAnd8)
				v0 := b.NewValue0(v.Pos, OpConst8, t)
				v0.AuxInt = int8ToAuxInt(c & d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpAndB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (AndB (Leq64 (Const64 [c]) x) (Less64 x (Const64 [d])))
	// cond: d >= c
	// result: (Less64U (Sub64 <x.Type> x (Const64 <x.Type> [c])) (Const64 <x.Type> [d-c]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq64 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpLess64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(d >= c) {
				continue
			}
			v.reset(OpLess64U)
			v0 := b.NewValue0(v.Pos, OpSub64, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v1.AuxInt = int64ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v2.AuxInt = int64ToAuxInt(d - c)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Leq64 (Const64 [c]) x) (Leq64 x (Const64 [d])))
	// cond: d >= c
	// result: (Leq64U (Sub64 <x.Type> x (Const64 <x.Type> [c])) (Const64 <x.Type> [d-c]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq64 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpLeq64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(d >= c) {
				continue
			}
			v.reset(OpLeq64U)
			v0 := b.NewValue0(v.Pos, OpSub64, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v1.AuxInt = int64ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v2.AuxInt = int64ToAuxInt(d - c)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Leq32 (Const32 [c]) x) (Less32 x (Const32 [d])))
	// cond: d >= c
	// result: (Less32U (Sub32 <x.Type> x (Const32 <x.Type> [c])) (Const32 <x.Type> [d-c]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq32 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpLess32 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1_1.AuxInt)
			if !(d >= c) {
				continue
			}
			v.reset(OpLess32U)
			v0 := b.NewValue0(v.Pos, OpSub32, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v1.AuxInt = int32ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v2.AuxInt = int32ToAuxInt(d - c)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Leq32 (Const32 [c]) x) (Leq32 x (Const32 [d])))
	// cond: d >= c
	// result: (Leq32U (Sub32 <x.Type> x (Const32 <x.Type> [c])) (Const32 <x.Type> [d-c]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq32 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpLeq32 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1_1.AuxInt)
			if !(d >= c) {
				continue
			}
			v.reset(OpLeq32U)
			v0 := b.NewValue0(v.Pos, OpSub32, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v1.AuxInt = int32ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v2.AuxInt = int32ToAuxInt(d - c)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Leq16 (Const16 [c]) x) (Less16 x (Const16 [d])))
	// cond: d >= c
	// result: (Less16U (Sub16 <x.Type> x (Const16 <x.Type> [c])) (Const16 <x.Type> [d-c]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq16 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0.AuxInt)
			if v_1.Op != OpLess16 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1_1.AuxInt)
			if !(d >= c) {
				continue
			}
			v.reset(OpLess16U)
			v0 := b.NewValue0(v.Pos, OpSub16, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v1.AuxInt = int16ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v2.AuxInt = int16ToAuxInt(d - c)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Leq16 (Const16 [c]) x) (Leq16 x (Const16 [d])))
	// cond: d >= c
	// result: (Leq16U (Sub16 <x.Type> x (Const16 <x.Type> [c])) (Const16 <x.Type> [d-c]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq16 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0.AuxInt)
			if v_1.Op != OpLeq16 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1_1.AuxInt)
			if !(d >= c) {
				continue
			}
			v.reset(OpLeq16U)
			v0 := b.NewValue0(v.Pos, OpSub16, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v1.AuxInt = int16ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v2.AuxInt = int16ToAuxInt(d - c)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Leq8 (Const8 [c]) x) (Less8 x (Const8 [d])))
	// cond: d >= c
	// result: (Less8U (Sub8 <x.Type> x (Const8 <x.Type> [c])) (Const8 <x.Type> [d-c]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq8 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0.AuxInt)
			if v_1.Op != OpLess8 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1_1.AuxInt)
			if !(d >= c) {
				continue
			}
			v.reset(OpLess8U)
			v0 := b.NewValue0(v.Pos, OpSub8, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v1.AuxInt = int8ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v2.AuxInt = int8ToAuxInt(d - c)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Leq8 (Const8 [c]) x) (Leq8 x (Const8 [d])))
	// cond: d >= c
	// result: (Leq8U (Sub8 <x.Type> x (Const8 <x.Type> [c])) (Const8 <x.Type> [d-c]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq8 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0.AuxInt)
			if v_1.Op != OpLeq8 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1_1.AuxInt)
			if !(d >= c) {
				continue
			}
			v.reset(OpLeq8U)
			v0 := b.NewValue0(v.Pos, OpSub8, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v1.AuxInt = int8ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v2.AuxInt = int8ToAuxInt(d - c)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Less64 (Const64 [c]) x) (Less64 x (Const64 [d])))
	// cond: d >= c+1 && c+1 > c
	// result: (Less64U (Sub64 <x.Type> x (Const64 <x.Type> [c+1])) (Const64 <x.Type> [d-c-1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess64 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpLess64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(d >= c+1 && c+1 > c) {
				continue
			}
			v.reset(OpLess64U)
			v0 := b.NewValue0(v.Pos, OpSub64, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v1.AuxInt = int64ToAuxInt(c + 1)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v2.AuxInt = int64ToAuxInt(d - c - 1)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Less64 (Const64 [c]) x) (Leq64 x (Const64 [d])))
	// cond: d >= c+1 && c+1 > c
	// result: (Leq64U (Sub64 <x.Type> x (Const64 <x.Type> [c+1])) (Const64 <x.Type> [d-c-1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess64 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpLeq64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(d >= c+1 && c+1 > c) {
				continue
			}
			v.reset(OpLeq64U)
			v0 := b.NewValue0(v.Pos, OpSub64, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v1.AuxInt = int64ToAuxInt(c + 1)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v2.AuxInt = int64ToAuxInt(d - c - 1)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Less32 (Const32 [c]) x) (Less32 x (Const32 [d])))
	// cond: d >= c+1 && c+1 > c
	// result: (Less32U (Sub32 <x.Type> x (Const32 <x.Type> [c+1])) (Const32 <x.Type> [d-c-1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess32 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpLess32 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1_1.AuxInt)
			if !(d >= c+1 && c+1 > c) {
				continue
			}
			v.reset(OpLess32U)
			v0 := b.NewValue0(v.Pos, OpSub32, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v1.AuxInt = int32ToAuxInt(c + 1)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v2.AuxInt = int32ToAuxInt(d - c - 1)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Less32 (Const32 [c]) x) (Leq32 x (Const32 [d])))
	// cond: d >= c+1 && c+1 > c
	// result: (Leq32U (Sub32 <x.Type> x (Const32 <x.Type> [c+1])) (Const32 <x.Type> [d-c-1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess32 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpLeq32 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1_1.AuxInt)
			if !(d >= c+1 && c+1 > c) {
				continue
			}
			v.reset(OpLeq32U)
			v0 := b.NewValue0(v.Pos, OpSub32, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v1.AuxInt = int32ToAuxInt(c + 1)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v2.AuxInt = int32ToAuxInt(d - c - 1)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Less16 (Const16 [c]) x) (Less16 x (Const16 [d])))
	// cond: d >= c+1 && c+1 > c
	// result: (Less16U (Sub16 <x.Type> x (Const16 <x.Type> [c+1])) (Const16 <x.Type> [d-c-1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess16 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0.AuxInt)
			if v_1.Op != OpLess16 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1_1.AuxInt)
			if !(d >= c+1 && c+1 > c) {
				continue
			}
			v.reset(OpLess16U)
			v0 := b.NewValue0(v.Pos, OpSub16, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v1.AuxInt = int16ToAuxInt(c + 1)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v2.AuxInt = int16ToAuxInt(d - c - 1)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Less16 (Const16 [c]) x) (Leq16 x (Const16 [d])))
	// cond: d >= c+1 && c+1 > c
	// result: (Leq16U (Sub16 <x.Type> x (Const16 <x.Type> [c+1])) (Const16 <x.Type> [d-c-1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess16 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0.AuxInt)
			if v_1.Op != OpLeq16 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1_1.AuxInt)
			if !(d >= c+1 && c+1 > c) {
				continue
			}
			v.reset(OpLeq16U)
			v0 := b.NewValue0(v.Pos, OpSub16, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v1.AuxInt = int16ToAuxInt(c + 1)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v2.AuxInt = int16ToAuxInt(d - c - 1)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Less8 (Const8 [c]) x) (Less8 x (Const8 [d])))
	// cond: d >= c+1 && c+1 > c
	// result: (Less8U (Sub8 <x.Type> x (Const8 <x.Type> [c+1])) (Const8 <x.Type> [d-c-1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess8 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0.AuxInt)
			if v_1.Op != OpLess8 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1_1.AuxInt)
			if !(d >= c+1 && c+1 > c) {
				continue
			}
			v.reset(OpLess8U)
			v0 := b.NewValue0(v.Pos, OpSub8, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v1.AuxInt = int8ToAuxInt(c + 1)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v2.AuxInt = int8ToAuxInt(d - c - 1)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Less8 (Const8 [c]) x) (Leq8 x (Const8 [d])))
	// cond: d >= c+1 && c+1 > c
	// result: (Leq8U (Sub8 <x.Type> x (Const8 <x.Type> [c+1])) (Const8 <x.Type> [d-c-1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess8 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0.AuxInt)
			if v_1.Op != OpLeq8 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1_1.AuxInt)
			if !(d >= c+1 && c+1 > c) {
				continue
			}
			v.reset(OpLeq8U)
			v0 := b.NewValue0(v.Pos, OpSub8, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v1.AuxInt = int8ToAuxInt(c + 1)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v2.AuxInt = int8ToAuxInt(d - c - 1)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Leq64U (Const64 [c]) x) (Less64U x (Const64 [d])))
	// cond: uint64(d) >= uint64(c)
	// result: (Less64U (Sub64 <x.Type> x (Const64 <x.Type> [c])) (Const64 <x.Type> [d-c]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq64U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpLess64U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(uint64(d) >= uint64(c)) {
				continue
			}
			v.reset(OpLess64U)
			v0 := b.NewValue0(v.Pos, OpSub64, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v1.AuxInt = int64ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v2.AuxInt = int64ToAuxInt(d - c)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Leq64U (Const64 [c]) x) (Leq64U x (Const64 [d])))
	// cond: uint64(d) >= uint64(c)
	// result: (Leq64U (Sub64 <x.Type> x (Const64 <x.Type> [c])) (Const64 <x.Type> [d-c]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq64U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpLeq64U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(uint64(d) >= uint64(c)) {
				continue
			}
			v.reset(OpLeq64U)
			v0 := b.NewValue0(v.Pos, OpSub64, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v1.AuxInt = int64ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v2.AuxInt = int64ToAuxInt(d - c)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Leq32U (Const32 [c]) x) (Less32U x (Const32 [d])))
	// cond: uint32(d) >= uint32(c)
	// result: (Less32U (Sub32 <x.Type> x (Const32 <x.Type> [c])) (Const32 <x.Type> [d-c]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq32U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpLess32U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1_1.AuxInt)
			if !(uint32(d) >= uint32(c)) {
				continue
			}
			v.reset(OpLess32U)
			v0 := b.NewValue0(v.Pos, OpSub32, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v1.AuxInt = int32ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v2.AuxInt = int32ToAuxInt(d - c)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Leq32U (Const32 [c]) x) (Leq32U x (Const32 [d])))
	// cond: uint32(d) >= uint32(c)
	// result: (Leq32U (Sub32 <x.Type> x (Const32 <x.Type> [c])) (Const32 <x.Type> [d-c]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq32U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpLeq32U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1_1.AuxInt)
			if !(uint32(d) >= uint32(c)) {
				continue
			}
			v.reset(OpLeq32U)
			v0 := b.NewValue0(v.Pos, OpSub32, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v1.AuxInt = int32ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v2.AuxInt = int32ToAuxInt(d - c)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Leq16U (Const16 [c]) x) (Less16U x (Const16 [d])))
	// cond: uint16(d) >= uint16(c)
	// result: (Less16U (Sub16 <x.Type> x (Const16 <x.Type> [c])) (Const16 <x.Type> [d-c]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq16U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0.AuxInt)
			if v_1.Op != OpLess16U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1_1.AuxInt)
			if !(uint16(d) >= uint16(c)) {
				continue
			}
			v.reset(OpLess16U)
			v0 := b.NewValue0(v.Pos, OpSub16, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v1.AuxInt = int16ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v2.AuxInt = int16ToAuxInt(d - c)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Leq16U (Const16 [c]) x) (Leq16U x (Const16 [d])))
	// cond: uint16(d) >= uint16(c)
	// result: (Leq16U (Sub16 <x.Type> x (Const16 <x.Type> [c])) (Const16 <x.Type> [d-c]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq16U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0.AuxInt)
			if v_1.Op != OpLeq16U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1_1.AuxInt)
			if !(uint16(d) >= uint16(c)) {
				continue
			}
			v.reset(OpLeq16U)
			v0 := b.NewValue0(v.Pos, OpSub16, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v1.AuxInt = int16ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v2.AuxInt = int16ToAuxInt(d - c)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Leq8U (Const8 [c]) x) (Less8U x (Const8 [d])))
	// cond: uint8(d) >= uint8(c)
	// result: (Less8U (Sub8 <x.Type> x (Const8 <x.Type> [c])) (Const8 <x.Type> [d-c]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq8U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0.AuxInt)
			if v_1.Op != OpLess8U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1_1.AuxInt)
			if !(uint8(d) >= uint8(c)) {
				continue
			}
			v.reset(OpLess8U)
			v0 := b.NewValue0(v.Pos, OpSub8, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v1.AuxInt = int8ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v2.AuxInt = int8ToAuxInt(d - c)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Leq8U (Const8 [c]) x) (Leq8U x (Const8 [d])))
	// cond: uint8(d) >= uint8(c)
	// result: (Leq8U (Sub8 <x.Type> x (Const8 <x.Type> [c])) (Const8 <x.Type> [d-c]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq8U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0.AuxInt)
			if v_1.Op != OpLeq8U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1_1.AuxInt)
			if !(uint8(d) >= uint8(c)) {
				continue
			}
			v.reset(OpLeq8U)
			v0 := b.NewValue0(v.Pos, OpSub8, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v1.AuxInt = int8ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v2.AuxInt = int8ToAuxInt(d - c)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Less64U (Const64 [c]) x) (Less64U x (Const64 [d])))
	// cond: uint64(d) >= uint64(c+1) && uint64(c+1) > uint64(c)
	// result: (Less64U (Sub64 <x.Type> x (Const64 <x.Type> [c+1])) (Const64 <x.Type> [d-c-1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess64U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpLess64U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(uint64(d) >= uint64(c+1) && uint64(c+1) > uint64(c)) {
				continue
			}
			v.reset(OpLess64U)
			v0 := b.NewValue0(v.Pos, OpSub64, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v1.AuxInt = int64ToAuxInt(c + 1)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v2.AuxInt = int64ToAuxInt(d - c - 1)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Less64U (Const64 [c]) x) (Leq64U x (Const64 [d])))
	// cond: uint64(d) >= uint64(c+1) && uint64(c+1) > uint64(c)
	// result: (Leq64U (Sub64 <x.Type> x (Const64 <x.Type> [c+1])) (Const64 <x.Type> [d-c-1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess64U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpLeq64U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(uint64(d) >= uint64(c+1) && uint64(c+1) > uint64(c)) {
				continue
			}
			v.reset(OpLeq64U)
			v0 := b.NewValue0(v.Pos, OpSub64, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v1.AuxInt = int64ToAuxInt(c + 1)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v2.AuxInt = int64ToAuxInt(d - c - 1)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Less32U (Const32 [c]) x) (Less32U x (Const32 [d])))
	// cond: uint32(d) >= uint32(c+1) && uint32(c+1) > uint32(c)
	// result: (Less32U (Sub32 <x.Type> x (Const32 <x.Type> [c+1])) (Const32 <x.Type> [d-c-1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess32U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpLess32U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1_1.AuxInt)
			if !(uint32(d) >= uint32(c+1) && uint32(c+1) > uint32(c)) {
				continue
			}
			v.reset(OpLess32U)
			v0 := b.NewValue0(v.Pos, OpSub32, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v1.AuxInt = int32ToAuxInt(c + 1)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v2.AuxInt = int32ToAuxInt(d - c - 1)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Less32U (Const32 [c]) x) (Leq32U x (Const32 [d])))
	// cond: uint32(d) >= uint32(c+1) && uint32(c+1) > uint32(c)
	// result: (Leq32U (Sub32 <x.Type> x (Const32 <x.Type> [c+1])) (Const32 <x.Type> [d-c-1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess32U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpLeq32U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1_1.AuxInt)
			if !(uint32(d) >= uint32(c+1) && uint32(c+1) > uint32(c)) {
				continue
			}
			v.reset(OpLeq32U)
			v0 := b.NewValue0(v.Pos, OpSub32, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v1.AuxInt = int32ToAuxInt(c + 1)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v2.AuxInt = int32ToAuxInt(d - c - 1)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Less16U (Const16 [c]) x) (Less16U x (Const16 [d])))
	// cond: uint16(d) >= uint16(c+1) && uint16(c+1) > uint16(c)
	// result: (Less16U (Sub16 <x.Type> x (Const16 <x.Type> [c+1])) (Const16 <x.Type> [d-c-1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess16U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0.AuxInt)
			if v_1.Op != OpLess16U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1_1.AuxInt)
			if !(uint16(d) >= uint16(c+1) && uint16(c+1) > uint16(c)) {
				continue
			}
			v.reset(OpLess16U)
			v0 := b.NewValue0(v.Pos, OpSub16, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v1.AuxInt = int16ToAuxInt(c + 1)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v2.AuxInt = int16ToAuxInt(d - c - 1)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Less16U (Const16 [c]) x) (Leq16U x (Const16 [d])))
	// cond: uint16(d) >= uint16(c+1) && uint16(c+1) > uint16(c)
	// result: (Leq16U (Sub16 <x.Type> x (Const16 <x.Type> [c+1])) (Const16 <x.Type> [d-c-1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess16U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0.AuxInt)
			if v_1.Op != OpLeq16U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1_1.AuxInt)
			if !(uint16(d) >= uint16(c+1) && uint16(c+1) > uint16(c)) {
				continue
			}
			v.reset(OpLeq16U)
			v0 := b.NewValue0(v.Pos, OpSub16, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v1.AuxInt = int16ToAuxInt(c + 1)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v2.AuxInt = int16ToAuxInt(d - c - 1)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Less8U (Const8 [c]) x) (Less8U x (Const8 [d])))
	// cond: uint8(d) >= uint8(c+1) && uint8(c+1) > uint8(c)
	// result: (Less8U (Sub8 <x.Type> x (Const8 <x.Type> [c+1])) (Const8 <x.Type> [d-c-1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess8U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0.AuxInt)
			if v_1.Op != OpLess8U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1_1.AuxInt)
			if !(uint8(d) >= uint8(c+1) && uint8(c+1) > uint8(c)) {
				continue
			}
			v.reset(OpLess8U)
			v0 := b.NewValue0(v.Pos, OpSub8, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v1.AuxInt = int8ToAuxInt(c + 1)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v2.AuxInt = int8ToAuxInt(d - c - 1)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (AndB (Less8U (Const8 [c]) x) (Leq8U x (Const8 [d])))
	// cond: uint8(d) >= uint8(c+1) && uint8(c+1) > uint8(c)
	// result: (Leq8U (Sub8 <x.Type> x (Const8 <x.Type> [c+1])) (Const8 <x.Type> [d-c-1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess8U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0.AuxInt)
			if v_1.Op != OpLeq8U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1_1.AuxInt)
			if !(uint8(d) >= uint8(c+1) && uint8(c+1) > uint8(c)) {
				continue
			}
			v.reset(OpLeq8U)
			v0 := b.NewValue0(v.Pos, OpSub8, x.Type)
			v1 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v1.AuxInt = int8ToAuxInt(c + 1)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v2.AuxInt = int8ToAuxInt(d - c - 1)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpArraySelect(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ArraySelect (ArrayMake1 x))
	// result: x
	for {
		if v_0.Op != OpArrayMake1 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (ArraySelect [0] (IData x))
	// result: (IData x)
	for {
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpIData {
			break
		}
		x := v_0.Args[0]
		v.reset(OpIData)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpBitLen16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (BitLen16 (Const16 [c]))
	// cond: config.PtrSize == 8
	// result: (Const64 [int64(bits.Len16(uint16(c)))])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(bits.Len16(uint16(c))))
		return true
	}
	// match: (BitLen16 (Const16 [c]))
	// cond: config.PtrSize == 4
	// result: (Const32 [int32(bits.Len16(uint16(c)))])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(bits.Len16(uint16(c))))
		return true
	}
	return false
}
func rewriteValuegeneric_OpBitLen32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (BitLen32 (Const32 [c]))
	// cond: config.PtrSize == 8
	// result: (Const64 [int64(bits.Len32(uint32(c)))])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(bits.Len32(uint32(c))))
		return true
	}
	// match: (BitLen32 (Const32 [c]))
	// cond: config.PtrSize == 4
	// result: (Const32 [int32(bits.Len32(uint32(c)))])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(bits.Len32(uint32(c))))
		return true
	}
	return false
}
func rewriteValuegeneric_OpBitLen64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (BitLen64 (Const64 [c]))
	// cond: config.PtrSize == 8
	// result: (Const64 [int64(bits.Len64(uint64(c)))])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(bits.Len64(uint64(c))))
		return true
	}
	// match: (BitLen64 (Const64 [c]))
	// cond: config.PtrSize == 4
	// result: (Const32 [int32(bits.Len64(uint64(c)))])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(bits.Len64(uint64(c))))
		return true
	}
	return false
}
func rewriteValuegeneric_OpBitLen8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (BitLen8 (Const8 [c]))
	// cond: config.PtrSize == 8
	// result: (Const64 [int64(bits.Len8(uint8(c)))])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(bits.Len8(uint8(c))))
		return true
	}
	// match: (BitLen8 (Const8 [c]))
	// cond: config.PtrSize == 4
	// result: (Const32 [int32(bits.Len8(uint8(c)))])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(bits.Len8(uint8(c))))
		return true
	}
	return false
}
func rewriteValuegeneric_OpCeil(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Ceil (Const64F [c]))
	// result: (Const64F [math.Ceil(c)])
	for {
		if v_0.Op != OpConst64F {
			break
		}
		c := auxIntToFloat64(v_0.AuxInt)
		v.reset(OpConst64F)
		v.AuxInt = float64ToAuxInt(math.Ceil(c))
		return true
	}
	return false
}
func rewriteValuegeneric_OpCom16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Com16 (Com16 x))
	// result: x
	for {
		if v_0.Op != OpCom16 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Com16 (Const16 [c]))
	// result: (Const16 [^c])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(^c)
		return true
	}
	// match: (Com16 (Add16 (Const16 [-1]) x))
	// result: (Neg16 x)
	for {
		if v_0.Op != OpAdd16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConst16 || auxIntToInt16(v_0_0.AuxInt) != -1 {
				continue
			}
			x := v_0_1
			v.reset(OpNeg16)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpCom32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Com32 (Com32 x))
	// result: x
	for {
		if v_0.Op != OpCom32 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Com32 (Const32 [c]))
	// result: (Const32 [^c])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(^c)
		return true
	}
	// match: (Com32 (Add32 (Const32 [-1]) x))
	// result: (Neg32 x)
	for {
		if v_0.Op != OpAdd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConst32 || auxIntToInt32(v_0_0.AuxInt) != -1 {
				continue
			}
			x := v_0_1
			v.reset(OpNeg32)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpCom64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Com64 (Com64 x))
	// result: x
	for {
		if v_0.Op != OpCom64 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Com64 (Const64 [c]))
	// result: (Const64 [^c])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(^c)
		return true
	}
	// match: (Com64 (Add64 (Const64 [-1]) x))
	// result: (Neg64 x)
	for {
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConst64 || auxIntToInt64(v_0_0.AuxInt) != -1 {
				continue
			}
			x := v_0_1
			v.reset(OpNeg64)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpCom8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Com8 (Com8 x))
	// result: x
	for {
		if v_0.Op != OpCom8 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Com8 (Const8 [c]))
	// result: (Const8 [^c])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(^c)
		return true
	}
	// match: (Com8 (Add8 (Const8 [-1]) x))
	// result: (Neg8 x)
	for {
		if v_0.Op != OpAdd8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConst8 || auxIntToInt8(v_0_0.AuxInt) != -1 {
				continue
			}
			x := v_0_1
			v.reset(OpNeg8)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpConstInterface(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ConstInterface)
	// result: (IMake (ConstNil <typ.Uintptr>) (ConstNil <typ.BytePtr>))
	for {
		v.reset(OpIMake)
		v0 := b.NewValue0(v.Pos, OpConstNil, typ.Uintptr)
		v1 := b.NewValue0(v.Pos, OpConstNil, typ.BytePtr)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuegeneric_OpConstSlice(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (ConstSlice)
	// cond: config.PtrSize == 4
	// result: (SliceMake (ConstNil <v.Type.Elem().PtrTo()>) (Const32 <typ.Int> [0]) (Const32 <typ.Int> [0]))
	for {
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpSliceMake)
		v0 := b.NewValue0(v.Pos, OpConstNil, v.Type.Elem().PtrTo())
		v1 := b.NewValue0(v.Pos, OpConst32, typ.Int)
		v1.AuxInt = int32ToAuxInt(0)
		v.AddArg3(v0, v1, v1)
		return true
	}
	// match: (ConstSlice)
	// cond: config.PtrSize == 8
	// result: (SliceMake (ConstNil <v.Type.Elem().PtrTo()>) (Const64 <typ.Int> [0]) (Const64 <typ.Int> [0]))
	for {
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpSliceMake)
		v0 := b.NewValue0(v.Pos, OpConstNil, v.Type.Elem().PtrTo())
		v1 := b.NewValue0(v.Pos, OpConst64, typ.Int)
		v1.AuxInt = int64ToAuxInt(0)
		v.AddArg3(v0, v1, v1)
		return true
	}
	return false
}
func rewriteValuegeneric_OpConstString(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	fe := b.Func.fe
	typ := &b.Func.Config.Types
	// match: (ConstString {str})
	// cond: config.PtrSize == 4 && str == ""
	// result: (StringMake (ConstNil) (Const32 <typ.Int> [0]))
	for {
		str := auxToString(v.Aux)
		if !(config.PtrSize == 4 && str == "") {
			break
		}
		v.reset(OpStringMake)
		v0 := b.NewValue0(v.Pos, OpConstNil, typ.BytePtr)
		v1 := b.NewValue0(v.Pos, OpConst32, typ.Int)
		v1.AuxInt = int32ToAuxInt(0)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (ConstString {str})
	// cond: config.PtrSize == 8 && str == ""
	// result: (StringMake (ConstNil) (Const64 <typ.Int> [0]))
	for {
		str := auxToString(v.Aux)
		if !(config.PtrSize == 8 && str == "") {
			break
		}
		v.reset(OpStringMake)
		v0 := b.NewValue0(v.Pos, OpConstNil, typ.BytePtr)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.Int)
		v1.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (ConstString {str})
	// cond: config.PtrSize == 4 && str != ""
	// result: (StringMake (Addr <typ.BytePtr> {fe.StringData(str)} (SB)) (Const32 <typ.Int> [int32(len(str))]))
	for {
		str := auxToString(v.Aux)
		if !(config.PtrSize == 4 && str != "") {
			break
		}
		v.reset(OpStringMake)
		v0 := b.NewValue0(v.Pos, OpAddr, typ.BytePtr)
		v0.Aux = symToAux(fe.StringData(str))
		v1 := b.NewValue0(v.Pos, OpSB, typ.Uintptr)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.Int)
		v2.AuxInt = int32ToAuxInt(int32(len(str)))
		v.AddArg2(v0, v2)
		return true
	}
	// match: (ConstString {str})
	// cond: config.PtrSize == 8 && str != ""
	// result: (StringMake (Addr <typ.BytePtr> {fe.StringData(str)} (SB)) (Const64 <typ.Int> [int64(len(str))]))
	for {
		str := auxToString(v.Aux)
		if !(config.PtrSize == 8 && str != "") {
			break
		}
		v.reset(OpStringMake)
		v0 := b.NewValue0(v.Pos, OpAddr, typ.BytePtr)
		v0.Aux = symToAux(fe.StringData(str))
		v1 := b.NewValue0(v.Pos, OpSB, typ.Uintptr)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.Int)
		v2.AuxInt = int64ToAuxInt(int64(len(str)))
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValuegeneric_OpConvert(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Convert (Add64 (Convert ptr mem) off) mem)
	// result: (AddPtr ptr off)
	for {
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConvert {
				continue
			}
			mem := v_0_0.Args[1]
			ptr := v_0_0.Args[0]
			off := v_0_1
			if mem != v_1 {
				continue
			}
			v.reset(OpAddPtr)
			v.AddArg2(ptr, off)
			return true
		}
		break
	}
	// match: (Convert (Add32 (Convert ptr mem) off) mem)
	// result: (AddPtr ptr off)
	for {
		if v_0.Op != OpAdd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConvert {
				continue
			}
			mem := v_0_0.Args[1]
			ptr := v_0_0.Args[0]
			off := v_0_1
			if mem != v_1 {
				continue
			}
			v.reset(OpAddPtr)
			v.AddArg2(ptr, off)
			return true
		}
		break
	}
	// match: (Convert (Convert ptr mem) mem)
	// result: ptr
	for {
		if v_0.Op != OpConvert {
			break
		}
		mem := v_0.Args[1]
		ptr := v_0.Args[0]
		if mem != v_1 {
			break
		}
		v.copyOf(ptr)
		return true
	}
	// match: (Convert a:(Add64 (Add64 (Convert ptr mem) off1) off2) mem)
	// result: (AddPtr ptr (Add64 <a.Type> off1 off2))
	for {
		a := v_0
		if a.Op != OpAdd64 {
			break
		}
		_ = a.Args[1]
		a_0 := a.Args[0]
		a_1 := a.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, a_0, a_1 = _i0+1, a_1, a_0 {
			if a_0.Op != OpAdd64 {
				continue
			}
			_ = a_0.Args[1]
			a_0_0 := a_0.Args[0]
			a_0_1 := a_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, a_0_0, a_0_1 = _i1+1, a_0_1, a_0_0 {
				if a_0_0.Op != OpConvert {
					continue
				}
				mem := a_0_0.Args[1]
				ptr := a_0_0.Args[0]
				off1 := a_0_1
				off2 := a_1
				if mem != v_1 {
					continue
				}
				v.reset(OpAddPtr)
				v0 := b.NewValue0(v.Pos, OpAdd64, a.Type)
				v0.AddArg2(off1, off2)
				v.AddArg2(ptr, v0)
				return true
			}
		}
		break
	}
	// match: (Convert a:(Add32 (Add32 (Convert ptr mem) off1) off2) mem)
	// result: (AddPtr ptr (Add32 <a.Type> off1 off2))
	for {
		a := v_0
		if a.Op != OpAdd32 {
			break
		}
		_ = a.Args[1]
		a_0 := a.Args[0]
		a_1 := a.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, a_0, a_1 = _i0+1, a_1, a_0 {
			if a_0.Op != OpAdd32 {
				continue
			}
			_ = a_0.Args[1]
			a_0_0 := a_0.Args[0]
			a_0_1 := a_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, a_0_0, a_0_1 = _i1+1, a_0_1, a_0_0 {
				if a_0_0.Op != OpConvert {
					continue
				}
				mem := a_0_0.Args[1]
				ptr := a_0_0.Args[0]
				off1 := a_0_1
				off2 := a_1
				if mem != v_1 {
					continue
				}
				v.reset(OpAddPtr)
				v0 := b.NewValue0(v.Pos, OpAdd32, a.Type)
				v0.AddArg2(off1, off2)
				v.AddArg2(ptr, v0)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpCtz16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (Ctz16 (Const16 [c]))
	// cond: config.PtrSize == 4
	// result: (Const32 [int32(ntz16(c))])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(ntz16(c)))
		return true
	}
	// match: (Ctz16 (Const16 [c]))
	// cond: config.PtrSize == 8
	// result: (Const64 [int64(ntz16(c))])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(ntz16(c)))
		return true
	}
	return false
}
func rewriteValuegeneric_OpCtz32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (Ctz32 (Const32 [c]))
	// cond: config.PtrSize == 4
	// result: (Const32 [int32(ntz32(c))])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(ntz32(c)))
		return true
	}
	// match: (Ctz32 (Const32 [c]))
	// cond: config.PtrSize == 8
	// result: (Const64 [int64(ntz32(c))])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(ntz32(c)))
		return true
	}
	return false
}
func rewriteValuegeneric_OpCtz64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (Ctz64 (Const64 [c]))
	// cond: config.PtrSize == 4
	// result: (Const32 [int32(ntz64(c))])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(ntz64(c)))
		return true
	}
	// match: (Ctz64 (Const64 [c]))
	// cond: config.PtrSize == 8
	// result: (Const64 [int64(ntz64(c))])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(ntz64(c)))
		return true
	}
	return false
}
func rewriteValuegeneric_OpCtz8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (Ctz8 (Const8 [c]))
	// cond: config.PtrSize == 4
	// result: (Const32 [int32(ntz8(c))])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(ntz8(c)))
		return true
	}
	// match: (Ctz8 (Const8 [c]))
	// cond: config.PtrSize == 8
	// result: (Const64 [int64(ntz8(c))])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(ntz8(c)))
		return true
	}
	return false
}
func rewriteValuegeneric_OpCvt32Fto32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt32Fto32 (Const32F [c]))
	// result: (Const32 [int32(c)])
	for {
		if v_0.Op != OpConst32F {
			break
		}
		c := auxIntToFloat32(v_0.AuxInt)
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(c))
		return true
	}
	return false
}
func rewriteValuegeneric_OpCvt32Fto64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt32Fto64 (Const32F [c]))
	// result: (Const64 [int64(c)])
	for {
		if v_0.Op != OpConst32F {
			break
		}
		c := auxIntToFloat32(v_0.AuxInt)
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(c))
		return true
	}
	return false
}
func rewriteValuegeneric_OpCvt32Fto64F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt32Fto64F (Const32F [c]))
	// result: (Const64F [float64(c)])
	for {
		if v_0.Op != OpConst32F {
			break
		}
		c := auxIntToFloat32(v_0.AuxInt)
		v.reset(OpConst64F)
		v.AuxInt = float64ToAuxInt(float64(c))
		return true
	}
	return false
}
func rewriteValuegeneric_OpCvt32to32F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt32to32F (Const32 [c]))
	// result: (Const32F [float32(c)])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		v.reset(OpConst32F)
		v.AuxInt = float32ToAuxInt(float32(c))
		return true
	}
	return false
}
func rewriteValuegeneric_OpCvt32to64F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt32to64F (Const32 [c]))
	// result: (Const64F [float64(c)])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		v.reset(OpConst64F)
		v.AuxInt = float64ToAuxInt(float64(c))
		return true
	}
	return false
}
func rewriteValuegeneric_OpCvt64Fto32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64Fto32 (Const64F [c]))
	// result: (Const32 [int32(c)])
	for {
		if v_0.Op != OpConst64F {
			break
		}
		c := auxIntToFloat64(v_0.AuxInt)
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(c))
		return true
	}
	return false
}
func rewriteValuegeneric_OpCvt64Fto32F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64Fto32F (Const64F [c]))
	// result: (Const32F [float32(c)])
	for {
		if v_0.Op != OpConst64F {
			break
		}
		c := auxIntToFloat64(v_0.AuxInt)
		v.reset(OpConst32F)
		v.AuxInt = float32ToAuxInt(float32(c))
		return true
	}
	// match: (Cvt64Fto32F sqrt0:(Sqrt (Cvt32Fto64F x)))
	// cond: sqrt0.Uses==1
	// result: (Sqrt32 x)
	for {
		sqrt0 := v_0
		if sqrt0.Op != OpSqrt {
			break
		}
		sqrt0_0 := sqrt0.Args[0]
		if sqrt0_0.Op != OpCvt32Fto64F {
			break
		}
		x := sqrt0_0.Args[0]
		if !(sqrt0.Uses == 1) {
			break
		}
		v.reset(OpSqrt32)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpCvt64Fto64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64Fto64 (Const64F [c]))
	// result: (Const64 [int64(c)])
	for {
		if v_0.Op != OpConst64F {
			break
		}
		c := auxIntToFloat64(v_0.AuxInt)
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(c))
		return true
	}
	return false
}
func rewriteValuegeneric_OpCvt64to32F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64to32F (Const64 [c]))
	// result: (Const32F [float32(c)])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpConst32F)
		v.AuxInt = float32ToAuxInt(float32(c))
		return true
	}
	return false
}
func rewriteValuegeneric_OpCvt64to64F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64to64F (Const64 [c]))
	// result: (Const64F [float64(c)])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpConst64F)
		v.AuxInt = float64ToAuxInt(float64(c))
		return true
	}
	return false
}
func rewriteValuegeneric_OpCvtBoolToUint8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CvtBoolToUint8 (ConstBool [false]))
	// result: (Const8 [0])
	for {
		if v_0.Op != OpConstBool || auxIntToBool(v_0.AuxInt) != false {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	// match: (CvtBoolToUint8 (ConstBool [true]))
	// result: (Const8 [1])
	for {
		if v_0.Op != OpConstBool || auxIntToBool(v_0.AuxInt) != true {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv128u(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Div128u <t> (Const64 [0]) lo y)
	// result: (MakeTuple (Div64u <t.FieldType(0)> lo y) (Mod64u <t.FieldType(1)> lo y))
	for {
		t := v.Type
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		lo := v_1
		y := v_2
		v.reset(OpMakeTuple)
		v0 := b.NewValue0(v.Pos, OpDiv64u, t.FieldType(0))
		v0.AddArg2(lo, y)
		v1 := b.NewValue0(v.Pos, OpMod64u, t.FieldType(1))
		v1.AddArg2(lo, y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 (Const16 [c]) (Const16 [d]))
	// cond: d != 0
	// result: (Const16 [c/d])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if v_1.Op != OpConst16 {
			break
		}
		d := auxIntToInt16(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(c / d)
		return true
	}
	// match: (Div16 n (Const16 [c]))
	// cond: isNonNegative(n) && isPowerOfTwo(c)
	// result: (Rsh16Ux64 n (Const64 <typ.UInt64> [log16(c)]))
	for {
		n := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(isNonNegative(n) && isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh16Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(log16(c))
		v.AddArg2(n, v0)
		return true
	}
	// match: (Div16 <t> n (Const16 [c]))
	// cond: c < 0 && c != -1<<15
	// result: (Neg16 (Div16 <t> n (Const16 <t> [-c])))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(c < 0 && c != -1<<15) {
			break
		}
		v.reset(OpNeg16)
		v0 := b.NewValue0(v.Pos, OpDiv16, t)
		v1 := b.NewValue0(v.Pos, OpConst16, t)
		v1.AuxInt = int16ToAuxInt(-c)
		v0.AddArg2(n, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Div16 <t> x (Const16 [-1<<15]))
	// result: (Rsh16Ux64 (And16 <t> x (Neg16 <t> x)) (Const64 <typ.UInt64> [15]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 || auxIntToInt16(v_1.AuxInt) != -1<<15 {
			break
		}
		v.reset(OpRsh16Ux64)
		v0 := b.NewValue0(v.Pos, OpAnd16, t)
		v1 := b.NewValue0(v.Pos, OpNeg16, t)
		v1.AddArg(x)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(15)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Div16 <t> n (Const16 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Rsh16x64 (Add16 <t> n (Rsh16Ux64 <t> (Rsh16x64 <t> n (Const64 <typ.UInt64> [15])) (Const64 <typ.UInt64> [int64(16-log16(c))]))) (Const64 <typ.UInt64> [int64(log16(c))]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh16x64)
		v0 := b.NewValue0(v.Pos, OpAdd16, t)
		v1 := b.NewValue0(v.Pos, OpRsh16Ux64, t)
		v2 := b.NewValue0(v.Pos, OpRsh16x64, t)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(15)
		v2.AddArg2(n, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(int64(16 - log16(c)))
		v1.AddArg2(v2, v4)
		v0.AddArg2(n, v1)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(int64(log16(c)))
		v.AddArg2(v0, v5)
		return true
	}
	// match: (Div16 <t> x (Const16 [c]))
	// cond: smagicOK16(c)
	// result: (Sub16 <t> (Rsh32x64 <t> (Mul32 <typ.UInt32> (Const32 <typ.UInt32> [int32(smagic16(c).m)]) (SignExt16to32 x)) (Const64 <typ.UInt64> [16+smagic16(c).s])) (Rsh32x64 <t> (SignExt16to32 x) (Const64 <typ.UInt64> [31])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(smagicOK16(c)) {
			break
		}
		v.reset(OpSub16)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v2.AuxInt = int32ToAuxInt(int32(smagic16(c).m))
		v3 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v3.AddArg(x)
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(16 + smagic16(c).s)
		v0.AddArg2(v1, v4)
		v5 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = int64ToAuxInt(31)
		v5.AddArg2(v3, v6)
		v.AddArg2(v0, v5)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Div16u (Const16 [c]) (Const16 [d]))
	// cond: d != 0
	// result: (Const16 [int16(uint16(c)/uint16(d))])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if v_1.Op != OpConst16 {
			break
		}
		d := auxIntToInt16(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(int16(uint16(c) / uint16(d)))
		return true
	}
	// match: (Div16u n (Const16 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Rsh16Ux64 n (Const64 <typ.UInt64> [log16(c)]))
	for {
		n := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh16Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(log16(c))
		v.AddArg2(n, v0)
		return true
	}
	// match: (Div16u x (Const16 [c]))
	// cond: umagicOK16(c) && config.RegSize == 8
	// result: (Trunc64to16 (Rsh64Ux64 <typ.UInt64> (Mul64 <typ.UInt64> (Const64 <typ.UInt64> [int64(1<<16+umagic16(c).m)]) (ZeroExt16to64 x)) (Const64 <typ.UInt64> [16+umagic16(c).s])))
	for {
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(umagicOK16(c) && config.RegSize == 8) {
			break
		}
		v.reset(OpTrunc64to16)
		v0 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(int64(1<<16 + umagic16(c).m))
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(x)
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(16 + umagic16(c).s)
		v0.AddArg2(v1, v4)
		v.AddArg(v0)
		return true
	}
	// match: (Div16u x (Const16 [c]))
	// cond: umagicOK16(c) && config.RegSize == 4 && umagic16(c).m&1 == 0
	// result: (Trunc32to16 (Rsh32Ux64 <typ.UInt32> (Mul32 <typ.UInt32> (Const32 <typ.UInt32> [int32(1<<15+umagic16(c).m/2)]) (ZeroExt16to32 x)) (Const64 <typ.UInt64> [16+umagic16(c).s-1])))
	for {
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(umagicOK16(c) && config.RegSize == 4 && umagic16(c).m&1 == 0) {
			break
		}
		v.reset(OpTrunc32to16)
		v0 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v2.AuxInt = int32ToAuxInt(int32(1<<15 + umagic16(c).m/2))
		v3 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v3.AddArg(x)
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(16 + umagic16(c).s - 1)
		v0.AddArg2(v1, v4)
		v.AddArg(v0)
		return true
	}
	// match: (Div16u x (Const16 [c]))
	// cond: umagicOK16(c) && config.RegSize == 4 && c&1 == 0
	// result: (Trunc32to16 (Rsh32Ux64 <typ.UInt32> (Mul32 <typ.UInt32> (Const32 <typ.UInt32> [int32(1<<15+(umagic16(c).m+1)/2)]) (Rsh32Ux64 <typ.UInt32> (ZeroExt16to32 x) (Const64 <typ.UInt64> [1]))) (Const64 <typ.UInt64> [16+umagic16(c).s-2])))
	for {
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(umagicOK16(c) && config.RegSize == 4 && c&1 == 0) {
			break
		}
		v.reset(OpTrunc32to16)
		v0 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v2.AuxInt = int32ToAuxInt(int32(1<<15 + (umagic16(c).m+1)/2))
		v3 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v4 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v4.AddArg(x)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(1)
		v3.AddArg2(v4, v5)
		v1.AddArg2(v2, v3)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = int64ToAuxInt(16 + umagic16(c).s - 2)
		v0.AddArg2(v1, v6)
		v.AddArg(v0)
		return true
	}
	// match: (Div16u x (Const16 [c]))
	// cond: umagicOK16(c) && config.RegSize == 4 && config.useAvg
	// result: (Trunc32to16 (Rsh32Ux64 <typ.UInt32> (Avg32u (Lsh32x64 <typ.UInt32> (ZeroExt16to32 x) (Const64 <typ.UInt64> [16])) (Mul32 <typ.UInt32> (Const32 <typ.UInt32> [int32(umagic16(c).m)]) (ZeroExt16to32 x))) (Const64 <typ.UInt64> [16+umagic16(c).s-1])))
	for {
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(umagicOK16(c) && config.RegSize == 4 && config.useAvg) {
			break
		}
		v.reset(OpTrunc32to16)
		v0 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpAvg32u, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpLsh32x64, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(16)
		v2.AddArg2(v3, v4)
		v5 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v6 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v6.AuxInt = int32ToAuxInt(int32(umagic16(c).m))
		v5.AddArg2(v6, v3)
		v1.AddArg2(v2, v5)
		v7 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v7.AuxInt = int64ToAuxInt(16 + umagic16(c).s - 1)
		v0.AddArg2(v1, v7)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Div32 (Const32 [c]) (Const32 [d]))
	// cond: d != 0
	// result: (Const32 [c/d])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if v_1.Op != OpConst32 {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(c / d)
		return true
	}
	// match: (Div32 n (Const32 [c]))
	// cond: isNonNegative(n) && isPowerOfTwo(c)
	// result: (Rsh32Ux64 n (Const64 <typ.UInt64> [log32(c)]))
	for {
		n := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(isNonNegative(n) && isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh32Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(log32(c))
		v.AddArg2(n, v0)
		return true
	}
	// match: (Div32 <t> n (Const32 [c]))
	// cond: c < 0 && c != -1<<31
	// result: (Neg32 (Div32 <t> n (Const32 <t> [-c])))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(c < 0 && c != -1<<31) {
			break
		}
		v.reset(OpNeg32)
		v0 := b.NewValue0(v.Pos, OpDiv32, t)
		v1 := b.NewValue0(v.Pos, OpConst32, t)
		v1.AuxInt = int32ToAuxInt(-c)
		v0.AddArg2(n, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Div32 <t> x (Const32 [-1<<31]))
	// result: (Rsh32Ux64 (And32 <t> x (Neg32 <t> x)) (Const64 <typ.UInt64> [31]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 || auxIntToInt32(v_1.AuxInt) != -1<<31 {
			break
		}
		v.reset(OpRsh32Ux64)
		v0 := b.NewValue0(v.Pos, OpAnd32, t)
		v1 := b.NewValue0(v.Pos, OpNeg32, t)
		v1.AddArg(x)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(31)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Div32 <t> n (Const32 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Rsh32x64 (Add32 <t> n (Rsh32Ux64 <t> (Rsh32x64 <t> n (Const64 <typ.UInt64> [31])) (Const64 <typ.UInt64> [int64(32-log32(c))]))) (Const64 <typ.UInt64> [int64(log32(c))]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh32x64)
		v0 := b.NewValue0(v.Pos, OpAdd32, t)
		v1 := b.NewValue0(v.Pos, OpRsh32Ux64, t)
		v2 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(31)
		v2.AddArg2(n, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(int64(32 - log32(c)))
		v1.AddArg2(v2, v4)
		v0.AddArg2(n, v1)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(int64(log32(c)))
		v.AddArg2(v0, v5)
		return true
	}
	// match: (Div32 <t> x (Const32 [c]))
	// cond: smagicOK32(c) && config.RegSize == 8
	// result: (Sub32 <t> (Rsh64x64 <t> (Mul64 <typ.UInt64> (Const64 <typ.UInt64> [int64(smagic32(c).m)]) (SignExt32to64 x)) (Const64 <typ.UInt64> [32+smagic32(c).s])) (Rsh64x64 <t> (SignExt32to64 x) (Const64 <typ.UInt64> [63])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(smagicOK32(c) && config.RegSize == 8) {
			break
		}
		v.reset(OpSub32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v1 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(int64(smagic32(c).m))
		v3 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v3.AddArg(x)
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(32 + smagic32(c).s)
		v0.AddArg2(v1, v4)
		v5 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = int64ToAuxInt(63)
		v5.AddArg2(v3, v6)
		v.AddArg2(v0, v5)
		return true
	}
	// match: (Div32 <t> x (Const32 [c]))
	// cond: smagicOK32(c) && config.RegSize == 4 && smagic32(c).m&1 == 0 && config.useHmul
	// result: (Sub32 <t> (Rsh32x64 <t> (Hmul32 <t> (Const32 <typ.UInt32> [int32(smagic32(c).m/2)]) x) (Const64 <typ.UInt64> [smagic32(c).s-1])) (Rsh32x64 <t> x (Const64 <typ.UInt64> [31])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(smagicOK32(c) && config.RegSize == 4 && smagic32(c).m&1 == 0 && config.useHmul) {
			break
		}
		v.reset(OpSub32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v1 := b.NewValue0(v.Pos, OpHmul32, t)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v2.AuxInt = int32ToAuxInt(int32(smagic32(c).m / 2))
		v1.AddArg2(v2, x)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(smagic32(c).s - 1)
		v0.AddArg2(v1, v3)
		v4 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(31)
		v4.AddArg2(x, v5)
		v.AddArg2(v0, v4)
		return true
	}
	// match: (Div32 <t> x (Const32 [c]))
	// cond: smagicOK32(c) && config.RegSize == 4 && smagic32(c).m&1 != 0 && config.useHmul
	// result: (Sub32 <t> (Rsh32x64 <t> (Add32 <t> (Hmul32 <t> (Const32 <typ.UInt32> [int32(smagic32(c).m)]) x) x) (Const64 <typ.UInt64> [smagic32(c).s])) (Rsh32x64 <t> x (Const64 <typ.UInt64> [31])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(smagicOK32(c) && config.RegSize == 4 && smagic32(c).m&1 != 0 && config.useHmul) {
			break
		}
		v.reset(OpSub32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v1 := b.NewValue0(v.Pos, OpAdd32, t)
		v2 := b.NewValue0(v.Pos, OpHmul32, t)
		v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v3.AuxInt = int32ToAuxInt(int32(smagic32(c).m))
		v2.AddArg2(v3, x)
		v1.AddArg2(v2, x)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(smagic32(c).s)
		v0.AddArg2(v1, v4)
		v5 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = int64ToAuxInt(31)
		v5.AddArg2(x, v6)
		v.AddArg2(v0, v5)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Div32F (Const32F [c]) (Const32F [d]))
	// cond: c/d == c/d
	// result: (Const32F [c/d])
	for {
		if v_0.Op != OpConst32F {
			break
		}
		c := auxIntToFloat32(v_0.AuxInt)
		if v_1.Op != OpConst32F {
			break
		}
		d := auxIntToFloat32(v_1.AuxInt)
		if !(c/d == c/d) {
			break
		}
		v.reset(OpConst32F)
		v.AuxInt = float32ToAuxInt(c / d)
		return true
	}
	// match: (Div32F x (Const32F <t> [c]))
	// cond: reciprocalExact32(c)
	// result: (Mul32F x (Const32F <t> [1/c]))
	for {
		x := v_0
		if v_1.Op != OpConst32F {
			break
		}
		t := v_1.Type
		c := auxIntToFloat32(v_1.AuxInt)
		if !(reciprocalExact32(c)) {
			break
		}
		v.reset(OpMul32F)
		v0 := b.NewValue0(v.Pos, OpConst32F, t)
		v0.AuxInt = float32ToAuxInt(1 / c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Div32u (Const32 [c]) (Const32 [d]))
	// cond: d != 0
	// result: (Const32 [int32(uint32(c)/uint32(d))])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if v_1.Op != OpConst32 {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(uint32(c) / uint32(d)))
		return true
	}
	// match: (Div32u n (Const32 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Rsh32Ux64 n (Const64 <typ.UInt64> [log32(c)]))
	for {
		n := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh32Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(log32(c))
		v.AddArg2(n, v0)
		return true
	}
	// match: (Div32u x (Const32 [c]))
	// cond: umagicOK32(c) && config.RegSize == 4 && umagic32(c).m&1 == 0 && config.useHmul
	// result: (Rsh32Ux64 <typ.UInt32> (Hmul32u <typ.UInt32> (Const32 <typ.UInt32> [int32(1<<31+umagic32(c).m/2)]) x) (Const64 <typ.UInt64> [umagic32(c).s-1]))
	for {
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && config.RegSize == 4 && umagic32(c).m&1 == 0 && config.useHmul) {
			break
		}
		v.reset(OpRsh32Ux64)
		v.Type = typ.UInt32
		v0 := b.NewValue0(v.Pos, OpHmul32u, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v1.AuxInt = int32ToAuxInt(int32(1<<31 + umagic32(c).m/2))
		v0.AddArg2(v1, x)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(umagic32(c).s - 1)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Div32u x (Const32 [c]))
	// cond: umagicOK32(c) && config.RegSize == 4 && c&1 == 0 && config.useHmul
	// result: (Rsh32Ux64 <typ.UInt32> (Hmul32u <typ.UInt32> (Const32 <typ.UInt32> [int32(1<<31+(umagic32(c).m+1)/2)]) (Rsh32Ux64 <typ.UInt32> x (Const64 <typ.UInt64> [1]))) (Const64 <typ.UInt64> [umagic32(c).s-2]))
	for {
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && config.RegSize == 4 && c&1 == 0 && config.useHmul) {
			break
		}
		v.reset(OpRsh32Ux64)
		v.Type = typ.UInt32
		v0 := b.NewValue0(v.Pos, OpHmul32u, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v1.AuxInt = int32ToAuxInt(int32(1<<31 + (umagic32(c).m+1)/2))
		v2 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(1)
		v2.AddArg2(x, v3)
		v0.AddArg2(v1, v2)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(umagic32(c).s - 2)
		v.AddArg2(v0, v4)
		return true
	}
	// match: (Div32u x (Const32 [c]))
	// cond: umagicOK32(c) && config.RegSize == 4 && config.useAvg && config.useHmul
	// result: (Rsh32Ux64 <typ.UInt32> (Avg32u x (Hmul32u <typ.UInt32> (Const32 <typ.UInt32> [int32(umagic32(c).m)]) x)) (Const64 <typ.UInt64> [umagic32(c).s-1]))
	for {
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && config.RegSize == 4 && config.useAvg && config.useHmul) {
			break
		}
		v.reset(OpRsh32Ux64)
		v.Type = typ.UInt32
		v0 := b.NewValue0(v.Pos, OpAvg32u, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpHmul32u, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v2.AuxInt = int32ToAuxInt(int32(umagic32(c).m))
		v1.AddArg2(v2, x)
		v0.AddArg2(x, v1)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(umagic32(c).s - 1)
		v.AddArg2(v0, v3)
		return true
	}
	// match: (Div32u x (Const32 [c]))
	// cond: umagicOK32(c) && config.RegSize == 8 && umagic32(c).m&1 == 0
	// result: (Trunc64to32 (Rsh64Ux64 <typ.UInt64> (Mul64 <typ.UInt64> (Const64 <typ.UInt64> [int64(1<<31+umagic32(c).m/2)]) (ZeroExt32to64 x)) (Const64 <typ.UInt64> [32+umagic32(c).s-1])))
	for {
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && config.RegSize == 8 && umagic32(c).m&1 == 0) {
			break
		}
		v.reset(OpTrunc64to32)
		v0 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(int64(1<<31 + umagic32(c).m/2))
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(x)
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(32 + umagic32(c).s - 1)
		v0.AddArg2(v1, v4)
		v.AddArg(v0)
		return true
	}
	// match: (Div32u x (Const32 [c]))
	// cond: umagicOK32(c) && config.RegSize == 8 && c&1 == 0
	// result: (Trunc64to32 (Rsh64Ux64 <typ.UInt64> (Mul64 <typ.UInt64> (Const64 <typ.UInt64> [int64(1<<31+(umagic32(c).m+1)/2)]) (Rsh64Ux64 <typ.UInt64> (ZeroExt32to64 x) (Const64 <typ.UInt64> [1]))) (Const64 <typ.UInt64> [32+umagic32(c).s-2])))
	for {
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && config.RegSize == 8 && c&1 == 0) {
			break
		}
		v.reset(OpTrunc64to32)
		v0 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(int64(1<<31 + (umagic32(c).m+1)/2))
		v3 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(x)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(1)
		v3.AddArg2(v4, v5)
		v1.AddArg2(v2, v3)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = int64ToAuxInt(32 + umagic32(c).s - 2)
		v0.AddArg2(v1, v6)
		v.AddArg(v0)
		return true
	}
	// match: (Div32u x (Const32 [c]))
	// cond: umagicOK32(c) && config.RegSize == 8 && config.useAvg
	// result: (Trunc64to32 (Rsh64Ux64 <typ.UInt64> (Avg64u (Lsh64x64 <typ.UInt64> (ZeroExt32to64 x) (Const64 <typ.UInt64> [32])) (Mul64 <typ.UInt64> (Const64 <typ.UInt32> [int64(umagic32(c).m)]) (ZeroExt32to64 x))) (Const64 <typ.UInt64> [32+umagic32(c).s-1])))
	for {
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && config.RegSize == 8 && config.useAvg) {
			break
		}
		v.reset(OpTrunc64to32)
		v0 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpAvg64u, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpLsh64x64, typ.UInt64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(32)
		v2.AddArg2(v3, v4)
		v5 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt32)
		v6.AuxInt = int64ToAuxInt(int64(umagic32(c).m))
		v5.AddArg2(v6, v3)
		v1.AddArg2(v2, v5)
		v7 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v7.AuxInt = int64ToAuxInt(32 + umagic32(c).s - 1)
		v0.AddArg2(v1, v7)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Div64 (Const64 [c]) (Const64 [d]))
	// cond: d != 0
	// result: (Const64 [c/d])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(c / d)
		return true
	}
	// match: (Div64 n (Const64 [c]))
	// cond: isNonNegative(n) && isPowerOfTwo(c)
	// result: (Rsh64Ux64 n (Const64 <typ.UInt64> [log64(c)]))
	for {
		n := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isNonNegative(n) && isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh64Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg2(n, v0)
		return true
	}
	// match: (Div64 n (Const64 [-1<<63]))
	// cond: isNonNegative(n)
	// result: (Const64 [0])
	for {
		n := v_0
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != -1<<63 || !(isNonNegative(n)) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (Div64 <t> n (Const64 [c]))
	// cond: c < 0 && c != -1<<63
	// result: (Neg64 (Div64 <t> n (Const64 <t> [-c])))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(c < 0 && c != -1<<63) {
			break
		}
		v.reset(OpNeg64)
		v0 := b.NewValue0(v.Pos, OpDiv64, t)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(-c)
		v0.AddArg2(n, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Div64 <t> x (Const64 [-1<<63]))
	// result: (Rsh64Ux64 (And64 <t> x (Neg64 <t> x)) (Const64 <typ.UInt64> [63]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != -1<<63 {
			break
		}
		v.reset(OpRsh64Ux64)
		v0 := b.NewValue0(v.Pos, OpAnd64, t)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v1.AddArg(x)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(63)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Div64 <t> n (Const64 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Rsh64x64 (Add64 <t> n (Rsh64Ux64 <t> (Rsh64x64 <t> n (Const64 <typ.UInt64> [63])) (Const64 <typ.UInt64> [int64(64-log64(c))]))) (Const64 <typ.UInt64> [int64(log64(c))]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh64x64)
		v0 := b.NewValue0(v.Pos, OpAdd64, t)
		v1 := b.NewValue0(v.Pos, OpRsh64Ux64, t)
		v2 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(63)
		v2.AddArg2(n, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(int64(64 - log64(c)))
		v1.AddArg2(v2, v4)
		v0.AddArg2(n, v1)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(int64(log64(c)))
		v.AddArg2(v0, v5)
		return true
	}
	// match: (Div64 <t> x (Const64 [c]))
	// cond: smagicOK64(c) && smagic64(c).m&1 == 0 && config.useHmul
	// result: (Sub64 <t> (Rsh64x64 <t> (Hmul64 <t> (Const64 <typ.UInt64> [int64(smagic64(c).m/2)]) x) (Const64 <typ.UInt64> [smagic64(c).s-1])) (Rsh64x64 <t> x (Const64 <typ.UInt64> [63])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(smagicOK64(c) && smagic64(c).m&1 == 0 && config.useHmul) {
			break
		}
		v.reset(OpSub64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v1 := b.NewValue0(v.Pos, OpHmul64, t)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(int64(smagic64(c).m / 2))
		v1.AddArg2(v2, x)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(smagic64(c).s - 1)
		v0.AddArg2(v1, v3)
		v4 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(63)
		v4.AddArg2(x, v5)
		v.AddArg2(v0, v4)
		return true
	}
	// match: (Div64 <t> x (Const64 [c]))
	// cond: smagicOK64(c) && smagic64(c).m&1 != 0 && config.useHmul
	// result: (Sub64 <t> (Rsh64x64 <t> (Add64 <t> (Hmul64 <t> (Const64 <typ.UInt64> [int64(smagic64(c).m)]) x) x) (Const64 <typ.UInt64> [smagic64(c).s])) (Rsh64x64 <t> x (Const64 <typ.UInt64> [63])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(smagicOK64(c) && smagic64(c).m&1 != 0 && config.useHmul) {
			break
		}
		v.reset(OpSub64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v1 := b.NewValue0(v.Pos, OpAdd64, t)
		v2 := b.NewValue0(v.Pos, OpHmul64, t)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(int64(smagic64(c).m))
		v2.AddArg2(v3, x)
		v1.AddArg2(v2, x)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(smagic64(c).s)
		v0.AddArg2(v1, v4)
		v5 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = int64ToAuxInt(63)
		v5.AddArg2(x, v6)
		v.AddArg2(v0, v5)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Div64F (Const64F [c]) (Const64F [d]))
	// cond: c/d == c/d
	// result: (Const64F [c/d])
	for {
		if v_0.Op != OpConst64F {
			break
		}
		c := auxIntToFloat64(v_0.AuxInt)
		if v_1.Op != OpConst64F {
			break
		}
		d := auxIntToFloat64(v_1.AuxInt)
		if !(c/d == c/d) {
			break
		}
		v.reset(OpConst64F)
		v.AuxInt = float64ToAuxInt(c / d)
		return true
	}
	// match: (Div64F x (Const64F <t> [c]))
	// cond: reciprocalExact64(c)
	// result: (Mul64F x (Const64F <t> [1/c]))
	for {
		x := v_0
		if v_1.Op != OpConst64F {
			break
		}
		t := v_1.Type
		c := auxIntToFloat64(v_1.AuxInt)
		if !(reciprocalExact64(c)) {
			break
		}
		v.reset(OpMul64F)
		v0 := b.NewValue0(v.Pos, OpConst64F, t)
		v0.AuxInt = float64ToAuxInt(1 / c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Div64u (Const64 [c]) (Const64 [d]))
	// cond: d != 0
	// result: (Const64 [int64(uint64(c)/uint64(d))])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) / uint64(d)))
		return true
	}
	// match: (Div64u n (Const64 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Rsh64Ux64 n (Const64 <typ.UInt64> [log64(c)]))
	for {
		n := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh64Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg2(n, v0)
		return true
	}
	// match: (Div64u n (Const64 [-1<<63]))
	// result: (Rsh64Ux64 n (Const64 <typ.UInt64> [63]))
	for {
		n := v_0
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != -1<<63 {
			break
		}
		v.reset(OpRsh64Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(63)
		v.AddArg2(n, v0)
		return true
	}
	// match: (Div64u x (Const64 [c]))
	// cond: c > 0 && c <= 0xFFFF && umagicOK32(int32(c)) && config.RegSize == 4 && config.useHmul
	// result: (Add64 (Add64 <typ.UInt64> (Add64 <typ.UInt64> (Lsh64x64 <typ.UInt64> (ZeroExt32to64 (Div32u <typ.UInt32> (Trunc64to32 <typ.UInt32> (Rsh64Ux64 <typ.UInt64> x (Const64 <typ.UInt64> [32]))) (Const32 <typ.UInt32> [int32(c)]))) (Const64 <typ.UInt64> [32])) (ZeroExt32to64 (Div32u <typ.UInt32> (Trunc64to32 <typ.UInt32> x) (Const32 <typ.UInt32> [int32(c)])))) (Mul64 <typ.UInt64> (ZeroExt32to64 <typ.UInt64> (Mod32u <typ.UInt32> (Trunc64to32 <typ.UInt32> (Rsh64Ux64 <typ.UInt64> x (Const64 <typ.UInt64> [32]))) (Const32 <typ.UInt32> [int32(c)]))) (Const64 <typ.UInt64> [int64((1<<32)/c)]))) (ZeroExt32to64 (Div32u <typ.UInt32> (Add32 <typ.UInt32> (Mod32u <typ.UInt32> (Trunc64to32 <typ.UInt32> x) (Const32 <typ.UInt32> [int32(c)])) (Mul32 <typ.UInt32> (Mod32u <typ.UInt32> (Trunc64to32 <typ.UInt32> (Rsh64Ux64 <typ.UInt64> x (Const64 <typ.UInt64> [32]))) (Const32 <typ.UInt32> [int32(c)])) (Const32 <typ.UInt32> [int32((1<<32)%c)]))) (Const32 <typ.UInt32> [int32(c)]))))
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(c > 0 && c <= 0xFFFF && umagicOK32(int32(c)) && config.RegSize == 4 && config.useHmul) {
			break
		}
		v.reset(OpAdd64)
		v0 := b.NewValue0(v.Pos, OpAdd64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpAdd64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpLsh64x64, typ.UInt64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4 := b.NewValue0(v.Pos, OpDiv32u, typ.UInt32)
		v5 := b.NewValue0(v.Pos, OpTrunc64to32, typ.UInt32)
		v6 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v7 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v7.AuxInt = int64ToAuxInt(32)
		v6.AddArg2(x, v7)
		v5.AddArg(v6)
		v8 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v8.AuxInt = int32ToAuxInt(int32(c))
		v4.AddArg2(v5, v8)
		v3.AddArg(v4)
		v2.AddArg2(v3, v7)
		v9 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v10 := b.NewValue0(v.Pos, OpDiv32u, typ.UInt32)
		v11 := b.NewValue0(v.Pos, OpTrunc64to32, typ.UInt32)
		v11.AddArg(x)
		v10.AddArg2(v11, v8)
		v9.AddArg(v10)
		v1.AddArg2(v2, v9)
		v12 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
		v13 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v14 := b.NewValue0(v.Pos, OpMod32u, typ.UInt32)
		v14.AddArg2(v5, v8)
		v13.AddArg(v14)
		v15 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v15.AuxInt = int64ToAuxInt(int64((1 << 32) / c))
		v12.AddArg2(v13, v15)
		v0.AddArg2(v1, v12)
		v16 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v17 := b.NewValue0(v.Pos, OpDiv32u, typ.UInt32)
		v18 := b.NewValue0(v.Pos, OpAdd32, typ.UInt32)
		v19 := b.NewValue0(v.Pos, OpMod32u, typ.UInt32)
		v19.AddArg2(v11, v8)
		v20 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v21 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v21.AuxInt = int32ToAuxInt(int32((1 << 32) % c))
		v20.AddArg2(v14, v21)
		v18.AddArg2(v19, v20)
		v17.AddArg2(v18, v8)
		v16.AddArg(v17)
		v.AddArg2(v0, v16)
		return true
	}
	// match: (Div64u x (Const64 [c]))
	// cond: umagicOK64(c) && config.RegSize == 8 && umagic64(c).m&1 == 0 && config.useHmul
	// result: (Rsh64Ux64 <typ.UInt64> (Hmul64u <typ.UInt64> (Const64 <typ.UInt64> [int64(1<<63+umagic64(c).m/2)]) x) (Const64 <typ.UInt64> [umagic64(c).s-1]))
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(umagicOK64(c) && config.RegSize == 8 && umagic64(c).m&1 == 0 && config.useHmul) {
			break
		}
		v.reset(OpRsh64Ux64)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpHmul64u, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(int64(1<<63 + umagic64(c).m/2))
		v0.AddArg2(v1, x)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(umagic64(c).s - 1)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Div64u x (Const64 [c]))
	// cond: umagicOK64(c) && config.RegSize == 8 && c&1 == 0 && config.useHmul
	// result: (Rsh64Ux64 <typ.UInt64> (Hmul64u <typ.UInt64> (Const64 <typ.UInt64> [int64(1<<63+(umagic64(c).m+1)/2)]) (Rsh64Ux64 <typ.UInt64> x (Const64 <typ.UInt64> [1]))) (Const64 <typ.UInt64> [umagic64(c).s-2]))
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(umagicOK64(c) && config.RegSize == 8 && c&1 == 0 && config.useHmul) {
			break
		}
		v.reset(OpRsh64Ux64)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpHmul64u, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(int64(1<<63 + (umagic64(c).m+1)/2))
		v2 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(1)
		v2.AddArg2(x, v3)
		v0.AddArg2(v1, v2)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(umagic64(c).s - 2)
		v.AddArg2(v0, v4)
		return true
	}
	// match: (Div64u x (Const64 [c]))
	// cond: umagicOK64(c) && config.RegSize == 8 && config.useAvg && config.useHmul
	// result: (Rsh64Ux64 <typ.UInt64> (Avg64u x (Hmul64u <typ.UInt64> (Const64 <typ.UInt64> [int64(umagic64(c).m)]) x)) (Const64 <typ.UInt64> [umagic64(c).s-1]))
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(umagicOK64(c) && config.RegSize == 8 && config.useAvg && config.useHmul) {
			break
		}
		v.reset(OpRsh64Ux64)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpAvg64u, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpHmul64u, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(int64(umagic64(c).m))
		v1.AddArg2(v2, x)
		v0.AddArg2(x, v1)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(umagic64(c).s - 1)
		v.AddArg2(v0, v3)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 (Const8 [c]) (Const8 [d]))
	// cond: d != 0
	// result: (Const8 [c/d])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if v_1.Op != OpConst8 {
			break
		}
		d := auxIntToInt8(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(c / d)
		return true
	}
	// match: (Div8 n (Const8 [c]))
	// cond: isNonNegative(n) && isPowerOfTwo(c)
	// result: (Rsh8Ux64 n (Const64 <typ.UInt64> [log8(c)]))
	for {
		n := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		if !(isNonNegative(n) && isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh8Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(log8(c))
		v.AddArg2(n, v0)
		return true
	}
	// match: (Div8 <t> n (Const8 [c]))
	// cond: c < 0 && c != -1<<7
	// result: (Neg8 (Div8 <t> n (Const8 <t> [-c])))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		if !(c < 0 && c != -1<<7) {
			break
		}
		v.reset(OpNeg8)
		v0 := b.NewValue0(v.Pos, OpDiv8, t)
		v1 := b.NewValue0(v.Pos, OpConst8, t)
		v1.AuxInt = int8ToAuxInt(-c)
		v0.AddArg2(n, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Div8 <t> x (Const8 [-1<<7 ]))
	// result: (Rsh8Ux64 (And8 <t> x (Neg8 <t> x)) (Const64 <typ.UInt64> [7 ]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 || auxIntToInt8(v_1.AuxInt) != -1<<7 {
			break
		}
		v.reset(OpRsh8Ux64)
		v0 := b.NewValue0(v.Pos, OpAnd8, t)
		v1 := b.NewValue0(v.Pos, OpNeg8, t)
		v1.AddArg(x)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(7)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Div8 <t> n (Const8 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Rsh8x64 (Add8 <t> n (Rsh8Ux64 <t> (Rsh8x64 <t> n (Const64 <typ.UInt64> [ 7])) (Const64 <typ.UInt64> [int64( 8-log8(c))]))) (Const64 <typ.UInt64> [int64(log8(c))]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh8x64)
		v0 := b.NewValue0(v.Pos, OpAdd8, t)
		v1 := b.NewValue0(v.Pos, OpRsh8Ux64, t)
		v2 := b.NewValue0(v.Pos, OpRsh8x64, t)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(7)
		v2.AddArg2(n, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(int64(8 - log8(c)))
		v1.AddArg2(v2, v4)
		v0.AddArg2(n, v1)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(int64(log8(c)))
		v.AddArg2(v0, v5)
		return true
	}
	// match: (Div8 <t> x (Const8 [c]))
	// cond: smagicOK8(c)
	// result: (Sub8 <t> (Rsh32x64 <t> (Mul32 <typ.UInt32> (Const32 <typ.UInt32> [int32(smagic8(c).m)]) (SignExt8to32 x)) (Const64 <typ.UInt64> [8+smagic8(c).s])) (Rsh32x64 <t> (SignExt8to32 x) (Const64 <typ.UInt64> [31])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		if !(smagicOK8(c)) {
			break
		}
		v.reset(OpSub8)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v2.AuxInt = int32ToAuxInt(int32(smagic8(c).m))
		v3 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v3.AddArg(x)
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(8 + smagic8(c).s)
		v0.AddArg2(v1, v4)
		v5 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = int64ToAuxInt(31)
		v5.AddArg2(v3, v6)
		v.AddArg2(v0, v5)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u (Const8 [c]) (Const8 [d]))
	// cond: d != 0
	// result: (Const8 [int8(uint8(c)/uint8(d))])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if v_1.Op != OpConst8 {
			break
		}
		d := auxIntToInt8(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(int8(uint8(c) / uint8(d)))
		return true
	}
	// match: (Div8u n (Const8 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Rsh8Ux64 n (Const64 <typ.UInt64> [log8(c)]))
	for {
		n := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh8Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(log8(c))
		v.AddArg2(n, v0)
		return true
	}
	// match: (Div8u x (Const8 [c]))
	// cond: umagicOK8(c)
	// result: (Trunc32to8 (Rsh32Ux64 <typ.UInt32> (Mul32 <typ.UInt32> (Const32 <typ.UInt32> [int32(1<<8+umagic8(c).m)]) (ZeroExt8to32 x)) (Const64 <typ.UInt64> [8+umagic8(c).s])))
	for {
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		if !(umagicOK8(c)) {
			break
		}
		v.reset(OpTrunc32to8)
		v0 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v2.AuxInt = int32ToAuxInt(int32(1<<8 + umagic8(c).m))
		v3 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v3.AddArg(x)
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(8 + umagic8(c).s)
		v0.AddArg2(v1, v4)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpEq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Eq16 x x)
	// result: (ConstBool [true])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Eq16 (Const16 <t> [c]) (Add16 (Const16 <t> [d]) x))
	// result: (Eq16 (Const16 <t> [c-d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpAdd16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst16 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt16(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpEq16)
				v0 := b.NewValue0(v.Pos, OpConst16, t)
				v0.AuxInt = int16ToAuxInt(c - d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Eq16 (Const16 [c]) (Const16 [d]))
	// result: (ConstBool [c == d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c == d)
			return true
		}
		break
	}
	// match: (Eq16 (Mod16u x (Const16 [c])) (Const16 [0]))
	// cond: x.Op != OpConst16 && udivisibleOK16(c) && !hasSmallRotate(config)
	// result: (Eq32 (Mod32u <typ.UInt32> (ZeroExt16to32 <typ.UInt32> x) (Const32 <typ.UInt32> [int32(uint16(c))])) (Const32 <typ.UInt32> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpMod16u {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_1.AuxInt)
			if v_1.Op != OpConst16 || auxIntToInt16(v_1.AuxInt) != 0 || !(x.Op != OpConst16 && udivisibleOK16(c) && !hasSmallRotate(config)) {
				continue
			}
			v.reset(OpEq32)
			v0 := b.NewValue0(v.Pos, OpMod32u, typ.UInt32)
			v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
			v1.AddArg(x)
			v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
			v2.AuxInt = int32ToAuxInt(int32(uint16(c)))
			v0.AddArg2(v1, v2)
			v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
			v3.AuxInt = int32ToAuxInt(0)
			v.AddArg2(v0, v3)
			return true
		}
		break
	}
	// match: (Eq16 (Mod16 x (Const16 [c])) (Const16 [0]))
	// cond: x.Op != OpConst16 && sdivisibleOK16(c) && !hasSmallRotate(config)
	// result: (Eq32 (Mod32 <typ.Int32> (SignExt16to32 <typ.Int32> x) (Const32 <typ.Int32> [int32(c)])) (Const32 <typ.Int32> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpMod16 {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_1.AuxInt)
			if v_1.Op != OpConst16 || auxIntToInt16(v_1.AuxInt) != 0 || !(x.Op != OpConst16 && sdivisibleOK16(c) && !hasSmallRotate(config)) {
				continue
			}
			v.reset(OpEq32)
			v0 := b.NewValue0(v.Pos, OpMod32, typ.Int32)
			v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
			v1.AddArg(x)
			v2 := b.NewValue0(v.Pos, OpConst32, typ.Int32)
			v2.AuxInt = int32ToAuxInt(int32(c))
			v0.AddArg2(v1, v2)
			v3 := b.NewValue0(v.Pos, OpConst32, typ.Int32)
			v3.AuxInt = int32ToAuxInt(0)
			v.AddArg2(v0, v3)
			return true
		}
		break
	}
	// match: (Eq16 x (Mul16 (Const16 [c]) (Trunc64to16 (Rsh64Ux64 mul:(Mul64 (Const64 [m]) (ZeroExt16to64 x)) (Const64 [s]))) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(1<<16+umagic16(c).m) && s == 16+umagic16(c).s && x.Op != OpConst16 && udivisibleOK16(c)
	// result: (Leq16U (RotateLeft16 <typ.UInt16> (Mul16 <typ.UInt16> (Const16 <typ.UInt16> [int16(udivisible16(c).m)]) x) (Const16 <typ.UInt16> [int16(16-udivisible16(c).k)]) ) (Const16 <typ.UInt16> [int16(udivisible16(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst16 {
					continue
				}
				c := auxIntToInt16(v_1_0.AuxInt)
				if v_1_1.Op != OpTrunc64to16 {
					continue
				}
				v_1_1_0 := v_1_1.Args[0]
				if v_1_1_0.Op != OpRsh64Ux64 {
					continue
				}
				_ = v_1_1_0.Args[1]
				mul := v_1_1_0.Args[0]
				if mul.Op != OpMul64 {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst64 {
						continue
					}
					m := auxIntToInt64(mul_0.AuxInt)
					if mul_1.Op != OpZeroExt16to64 || x != mul_1.Args[0] {
						continue
					}
					v_1_1_0_1 := v_1_1_0.Args[1]
					if v_1_1_0_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_0_1.AuxInt)
					if !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(1<<16+umagic16(c).m) && s == 16+umagic16(c).s && x.Op != OpConst16 && udivisibleOK16(c)) {
						continue
					}
					v.reset(OpLeq16U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft16, typ.UInt16)
					v1 := b.NewValue0(v.Pos, OpMul16, typ.UInt16)
					v2 := b.NewValue0(v.Pos, OpConst16, typ.UInt16)
					v2.AuxInt = int16ToAuxInt(int16(udivisible16(c).m))
					v1.AddArg2(v2, x)
					v3 := b.NewValue0(v.Pos, OpConst16, typ.UInt16)
					v3.AuxInt = int16ToAuxInt(int16(16 - udivisible16(c).k))
					v0.AddArg2(v1, v3)
					v4 := b.NewValue0(v.Pos, OpConst16, typ.UInt16)
					v4.AuxInt = int16ToAuxInt(int16(udivisible16(c).max))
					v.AddArg2(v0, v4)
					return true
				}
			}
		}
		break
	}
	// match: (Eq16 x (Mul16 (Const16 [c]) (Trunc32to16 (Rsh32Ux64 mul:(Mul32 (Const32 [m]) (ZeroExt16to32 x)) (Const64 [s]))) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(1<<15+umagic16(c).m/2) && s == 16+umagic16(c).s-1 && x.Op != OpConst16 && udivisibleOK16(c)
	// result: (Leq16U (RotateLeft16 <typ.UInt16> (Mul16 <typ.UInt16> (Const16 <typ.UInt16> [int16(udivisible16(c).m)]) x) (Const16 <typ.UInt16> [int16(16-udivisible16(c).k)]) ) (Const16 <typ.UInt16> [int16(udivisible16(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst16 {
					continue
				}
				c := auxIntToInt16(v_1_0.AuxInt)
				if v_1_1.Op != OpTrunc32to16 {
					continue
				}
				v_1_1_0 := v_1_1.Args[0]
				if v_1_1_0.Op != OpRsh32Ux64 {
					continue
				}
				_ = v_1_1_0.Args[1]
				mul := v_1_1_0.Args[0]
				if mul.Op != OpMul32 {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst32 {
						continue
					}
					m := auxIntToInt32(mul_0.AuxInt)
					if mul_1.Op != OpZeroExt16to32 || x != mul_1.Args[0] {
						continue
					}
					v_1_1_0_1 := v_1_1_0.Args[1]
					if v_1_1_0_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_0_1.AuxInt)
					if !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(1<<15+umagic16(c).m/2) && s == 16+umagic16(c).s-1 && x.Op != OpConst16 && udivisibleOK16(c)) {
						continue
					}
					v.reset(OpLeq16U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft16, typ.UInt16)
					v1 := b.NewValue0(v.Pos, OpMul16, typ.UInt16)
					v2 := b.NewValue0(v.Pos, OpConst16, typ.UInt16)
					v2.AuxInt = int16ToAuxInt(int16(udivisible16(c).m))
					v1.AddArg2(v2, x)
					v3 := b.NewValue0(v.Pos, OpConst16, typ.UInt16)
					v3.AuxInt = int16ToAuxInt(int16(16 - udivisible16(c).k))
					v0.AddArg2(v1, v3)
					v4 := b.NewValue0(v.Pos, OpConst16, typ.UInt16)
					v4.AuxInt = int16ToAuxInt(int16(udivisible16(c).max))
					v.AddArg2(v0, v4)
					return true
				}
			}
		}
		break
	}
	// match: (Eq16 x (Mul16 (Const16 [c]) (Trunc32to16 (Rsh32Ux64 mul:(Mul32 (Const32 [m]) (Rsh32Ux64 (ZeroExt16to32 x) (Const64 [1]))) (Const64 [s]))) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(1<<15+(umagic16(c).m+1)/2) && s == 16+umagic16(c).s-2 && x.Op != OpConst16 && udivisibleOK16(c)
	// result: (Leq16U (RotateLeft16 <typ.UInt16> (Mul16 <typ.UInt16> (Const16 <typ.UInt16> [int16(udivisible16(c).m)]) x) (Const16 <typ.UInt16> [int16(16-udivisible16(c).k)]) ) (Const16 <typ.UInt16> [int16(udivisible16(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst16 {
					continue
				}
				c := auxIntToInt16(v_1_0.AuxInt)
				if v_1_1.Op != OpTrunc32to16 {
					continue
				}
				v_1_1_0 := v_1_1.Args[0]
				if v_1_1_0.Op != OpRsh32Ux64 {
					continue
				}
				_ = v_1_1_0.Args[1]
				mul := v_1_1_0.Args[0]
				if mul.Op != OpMul32 {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst32 {
						continue
					}
					m := auxIntToInt32(mul_0.AuxInt)
					if mul_1.Op != OpRsh32Ux64 {
						continue
					}
					_ = mul_1.Args[1]
					mul_1_0 := mul_1.Args[0]
					if mul_1_0.Op != OpZeroExt16to32 || x != mul_1_0.Args[0] {
						continue
					}
					mul_1_1 := mul_1.Args[1]
					if mul_1_1.Op != OpConst64 || auxIntToInt64(mul_1_1.AuxInt) != 1 {
						continue
					}
					v_1_1_0_1 := v_1_1_0.Args[1]
					if v_1_1_0_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_0_1.AuxInt)
					if !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(1<<15+(umagic16(c).m+1)/2) && s == 16+umagic16(c).s-2 && x.Op != OpConst16 && udivisibleOK16(c)) {
						continue
					}
					v.reset(OpLeq16U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft16, typ.UInt16)
					v1 := b.NewValue0(v.Pos, OpMul16, typ.UInt16)
					v2 := b.NewValue0(v.Pos, OpConst16, typ.UInt16)
					v2.AuxInt = int16ToAuxInt(int16(udivisible16(c).m))
					v1.AddArg2(v2, x)
					v3 := b.NewValue0(v.Pos, OpConst16, typ.UInt16)
					v3.AuxInt = int16ToAuxInt(int16(16 - udivisible16(c).k))
					v0.AddArg2(v1, v3)
					v4 := b.NewValue0(v.Pos, OpConst16, typ.UInt16)
					v4.AuxInt = int16ToAuxInt(int16(udivisible16(c).max))
					v.AddArg2(v0, v4)
					return true
				}
			}
		}
		break
	}
	// match: (Eq16 x (Mul16 (Const16 [c]) (Trunc32to16 (Rsh32Ux64 (Avg32u (Lsh32x64 (ZeroExt16to32 x) (Const64 [16])) mul:(Mul32 (Const32 [m]) (ZeroExt16to32 x))) (Const64 [s]))) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(umagic16(c).m) && s == 16+umagic16(c).s-1 && x.Op != OpConst16 && udivisibleOK16(c)
	// result: (Leq16U (RotateLeft16 <typ.UInt16> (Mul16 <typ.UInt16> (Const16 <typ.UInt16> [int16(udivisible16(c).m)]) x) (Const16 <typ.UInt16> [int16(16-udivisible16(c).k)]) ) (Const16 <typ.UInt16> [int16(udivisible16(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst16 {
					continue
				}
				c := auxIntToInt16(v_1_0.AuxInt)
				if v_1_1.Op != OpTrunc32to16 {
					continue
				}
				v_1_1_0 := v_1_1.Args[0]
				if v_1_1_0.Op != OpRsh32Ux64 {
					continue
				}
				_ = v_1_1_0.Args[1]
				v_1_1_0_0 := v_1_1_0.Args[0]
				if v_1_1_0_0.Op != OpAvg32u {
					continue
				}
				_ = v_1_1_0_0.Args[1]
				v_1_1_0_0_0 := v_1_1_0_0.Args[0]
				if v_1_1_0_0_0.Op != OpLsh32x64 {
					continue
				}
				_ = v_1_1_0_0_0.Args[1]
				v_1_1_0_0_0_0 := v_1_1_0_0_0.Args[0]
				if v_1_1_0_0_0_0.Op != OpZeroExt16to32 || x != v_1_1_0_0_0_0.Args[0] {
					continue
				}
				v_1_1_0_0_0_1 := v_1_1_0_0_0.Args[1]
				if v_1_1_0_0_0_1.Op != OpConst64 || auxIntToInt64(v_1_1_0_0_0_1.AuxInt) != 16 {
					continue
				}
				mul := v_1_1_0_0.Args[1]
				if mul.Op != OpMul32 {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst32 {
						continue
					}
					m := auxIntToInt32(mul_0.AuxInt)
					if mul_1.Op != OpZeroExt16to32 || x != mul_1.Args[0] {
						continue
					}
					v_1_1_0_1 := v_1_1_0.Args[1]
					if v_1_1_0_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_0_1.AuxInt)
					if !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(umagic16(c).m) && s == 16+umagic16(c).s-1 && x.Op != OpConst16 && udivisibleOK16(c)) {
						continue
					}
					v.reset(OpLeq16U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft16, typ.UInt16)
					v1 := b.NewValue0(v.Pos, OpMul16, typ.UInt16)
					v2 := b.NewValue0(v.Pos, OpConst16, typ.UInt16)
					v2.AuxInt = int16ToAuxInt(int16(udivisible16(c).m))
					v1.AddArg2(v2, x)
					v3 := b.NewValue0(v.Pos, OpConst16, typ.UInt16)
					v3.AuxInt = int16ToAuxInt(int16(16 - udivisible16(c).k))
					v0.AddArg2(v1, v3)
					v4 := b.NewValue0(v.Pos, OpConst16, typ.UInt16)
					v4.AuxInt = int16ToAuxInt(int16(udivisible16(c).max))
					v.AddArg2(v0, v4)
					return true
				}
			}
		}
		break
	}
	// match: (Eq16 x (Mul16 (Const16 [c]) (Sub16 (Rsh32x64 mul:(Mul32 (Const32 [m]) (SignExt16to32 x)) (Const64 [s])) (Rsh32x64 (SignExt16to32 x) (Const64 [31]))) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(smagic16(c).m) && s == 16+smagic16(c).s && x.Op != OpConst16 && sdivisibleOK16(c)
	// result: (Leq16U (RotateLeft16 <typ.UInt16> (Add16 <typ.UInt16> (Mul16 <typ.UInt16> (Const16 <typ.UInt16> [int16(sdivisible16(c).m)]) x) (Const16 <typ.UInt16> [int16(sdivisible16(c).a)]) ) (Const16 <typ.UInt16> [int16(16-sdivisible16(c).k)]) ) (Const16 <typ.UInt16> [int16(sdivisible16(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst16 {
					continue
				}
				c := auxIntToInt16(v_1_0.AuxInt)
				if v_1_1.Op != OpSub16 {
					continue
				}
				_ = v_1_1.Args[1]
				v_1_1_0 := v_1_1.Args[0]
				if v_1_1_0.Op != OpRsh32x64 {
					continue
				}
				_ = v_1_1_0.Args[1]
				mul := v_1_1_0.Args[0]
				if mul.Op != OpMul32 {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst32 {
						continue
					}
					m := auxIntToInt32(mul_0.AuxInt)
					if mul_1.Op != OpSignExt16to32 || x != mul_1.Args[0] {
						continue
					}
					v_1_1_0_1 := v_1_1_0.Args[1]
					if v_1_1_0_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_0_1.AuxInt)
					v_1_1_1 := v_1_1.Args[1]
					if v_1_1_1.Op != OpRsh32x64 {
						continue
					}
					_ = v_1_1_1.Args[1]
					v_1_1_1_0 := v_1_1_1.Args[0]
					if v_1_1_1_0.Op != OpSignExt16to32 || x != v_1_1_1_0.Args[0] {
						continue
					}
					v_1_1_1_1 := v_1_1_1.Args[1]
					if v_1_1_1_1.Op != OpConst64 || auxIntToInt64(v_1_1_1_1.AuxInt) != 31 || !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(smagic16(c).m) && s == 16+smagic16(c).s && x.Op != OpConst16 && sdivisibleOK16(c)) {
						continue
					}
					v.reset(OpLeq16U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft16, typ.UInt16)
					v1 := b.NewValue0(v.Pos, OpAdd16, typ.UInt16)
					v2 := b.NewValue0(v.Pos, OpMul16, typ.UInt16)
					v3 := b.NewValue0(v.Pos, OpConst16, typ.UInt16)
					v3.AuxInt = int16ToAuxInt(int16(sdivisible16(c).m))
					v2.AddArg2(v3, x)
					v4 := b.NewValue0(v.Pos, OpConst16, typ.UInt16)
					v4.AuxInt = int16ToAuxInt(int16(sdivisible16(c).a))
					v1.AddArg2(v2, v4)
					v5 := b.NewValue0(v.Pos, OpConst16, typ.UInt16)
					v5.AuxInt = int16ToAuxInt(int16(16 - sdivisible16(c).k))
					v0.AddArg2(v1, v5)
					v6 := b.NewValue0(v.Pos, OpConst16, typ.UInt16)
					v6.AuxInt = int16ToAuxInt(int16(sdivisible16(c).max))
					v.AddArg2(v0, v6)
					return true
				}
			}
		}
		break
	}
	// match: (Eq16 n (Lsh16x64 (Rsh16x64 (Add16 <t> n (Rsh16Ux64 <t> (Rsh16x64 <t> n (Const64 <typ.UInt64> [15])) (Const64 <typ.UInt64> [kbar]))) (Const64 <typ.UInt64> [k])) (Const64 <typ.UInt64> [k])) )
	// cond: k > 0 && k < 15 && kbar == 16 - k
	// result: (Eq16 (And16 <t> n (Const16 <t> [1<<uint(k)-1])) (Const16 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			n := v_0
			if v_1.Op != OpLsh16x64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRsh16x64 {
				continue
			}
			_ = v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			if v_1_0_0.Op != OpAdd16 {
				continue
			}
			t := v_1_0_0.Type
			_ = v_1_0_0.Args[1]
			v_1_0_0_0 := v_1_0_0.Args[0]
			v_1_0_0_1 := v_1_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0_0_0, v_1_0_0_1 = _i1+1, v_1_0_0_1, v_1_0_0_0 {
				if n != v_1_0_0_0 || v_1_0_0_1.Op != OpRsh16Ux64 || v_1_0_0_1.Type != t {
					continue
				}
				_ = v_1_0_0_1.Args[1]
				v_1_0_0_1_0 := v_1_0_0_1.Args[0]
				if v_1_0_0_1_0.Op != OpRsh16x64 || v_1_0_0_1_0.Type != t {
					continue
				}
				_ = v_1_0_0_1_0.Args[1]
				if n != v_1_0_0_1_0.Args[0] {
					continue
				}
				v_1_0_0_1_0_1 := v_1_0_0_1_0.Args[1]
				if v_1_0_0_1_0_1.Op != OpConst64 || v_1_0_0_1_0_1.Type != typ.UInt64 || auxIntToInt64(v_1_0_0_1_0_1.AuxInt) != 15 {
					continue
				}
				v_1_0_0_1_1 := v_1_0_0_1.Args[1]
				if v_1_0_0_1_1.Op != OpConst64 || v_1_0_0_1_1.Type != typ.UInt64 {
					continue
				}
				kbar := auxIntToInt64(v_1_0_0_1_1.AuxInt)
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst64 || v_1_0_1.Type != typ.UInt64 {
					continue
				}
				k := auxIntToInt64(v_1_0_1.AuxInt)
				v_1_1 := v_1.Args[1]
				if v_1_1.Op != OpConst64 || v_1_1.Type != typ.UInt64 || auxIntToInt64(v_1_1.AuxInt) != k || !(k > 0 && k < 15 && kbar == 16-k) {
					continue
				}
				v.reset(OpEq16)
				v0 := b.NewValue0(v.Pos, OpAnd16, t)
				v1 := b.NewValue0(v.Pos, OpConst16, t)
				v1.AuxInt = int16ToAuxInt(1<<uint(k) - 1)
				v0.AddArg2(n, v1)
				v2 := b.NewValue0(v.Pos, OpConst16, t)
				v2.AuxInt = int16ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq16 s:(Sub16 x y) (Const16 [0]))
	// cond: s.Uses == 1
	// result: (Eq16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s := v_0
			if s.Op != OpSub16 {
				continue
			}
			y := s.Args[1]
			x := s.Args[0]
			if v_1.Op != OpConst16 || auxIntToInt16(v_1.AuxInt) != 0 || !(s.Uses == 1) {
				continue
			}
			v.reset(OpEq16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Eq16 (And16 <t> x (Const16 <t> [y])) (Const16 <t> [y]))
	// cond: oneBit16(y)
	// result: (Neq16 (And16 <t> x (Const16 <t> [y])) (Const16 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAnd16 {
				continue
			}
			t := v_0.Type
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				if v_0_1.Op != OpConst16 || v_0_1.Type != t {
					continue
				}
				y := auxIntToInt16(v_0_1.AuxInt)
				if v_1.Op != OpConst16 || v_1.Type != t || auxIntToInt16(v_1.AuxInt) != y || !(oneBit16(y)) {
					continue
				}
				v.reset(OpNeq16)
				v0 := b.NewValue0(v.Pos, OpAnd16, t)
				v1 := b.NewValue0(v.Pos, OpConst16, t)
				v1.AuxInt = int16ToAuxInt(y)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst16, t)
				v2.AuxInt = int16ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpEq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq32 x x)
	// result: (ConstBool [true])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Eq32 (Const32 <t> [c]) (Add32 (Const32 <t> [d]) x))
	// result: (Eq32 (Const32 <t> [c-d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpAdd32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst32 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt32(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpEq32)
				v0 := b.NewValue0(v.Pos, OpConst32, t)
				v0.AuxInt = int32ToAuxInt(c - d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Eq32 (Const32 [c]) (Const32 [d]))
	// result: (ConstBool [c == d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c == d)
			return true
		}
		break
	}
	// match: (Eq32 x (Mul32 (Const32 [c]) (Rsh32Ux64 mul:(Hmul32u (Const32 [m]) x) (Const64 [s])) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(1<<31+umagic32(c).m/2) && s == umagic32(c).s-1 && x.Op != OpConst32 && udivisibleOK32(c)
	// result: (Leq32U (RotateLeft32 <typ.UInt32> (Mul32 <typ.UInt32> (Const32 <typ.UInt32> [int32(udivisible32(c).m)]) x) (Const32 <typ.UInt32> [int32(32-udivisible32(c).k)]) ) (Const32 <typ.UInt32> [int32(udivisible32(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst32 {
					continue
				}
				c := auxIntToInt32(v_1_0.AuxInt)
				if v_1_1.Op != OpRsh32Ux64 {
					continue
				}
				_ = v_1_1.Args[1]
				mul := v_1_1.Args[0]
				if mul.Op != OpHmul32u {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst32 {
						continue
					}
					m := auxIntToInt32(mul_0.AuxInt)
					if x != mul_1 {
						continue
					}
					v_1_1_1 := v_1_1.Args[1]
					if v_1_1_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_1.AuxInt)
					if !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(1<<31+umagic32(c).m/2) && s == umagic32(c).s-1 && x.Op != OpConst32 && udivisibleOK32(c)) {
						continue
					}
					v.reset(OpLeq32U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft32, typ.UInt32)
					v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
					v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v2.AuxInt = int32ToAuxInt(int32(udivisible32(c).m))
					v1.AddArg2(v2, x)
					v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v3.AuxInt = int32ToAuxInt(int32(32 - udivisible32(c).k))
					v0.AddArg2(v1, v3)
					v4 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v4.AuxInt = int32ToAuxInt(int32(udivisible32(c).max))
					v.AddArg2(v0, v4)
					return true
				}
			}
		}
		break
	}
	// match: (Eq32 x (Mul32 (Const32 [c]) (Rsh32Ux64 mul:(Hmul32u (Const32 <typ.UInt32> [m]) (Rsh32Ux64 x (Const64 [1]))) (Const64 [s])) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(1<<31+(umagic32(c).m+1)/2) && s == umagic32(c).s-2 && x.Op != OpConst32 && udivisibleOK32(c)
	// result: (Leq32U (RotateLeft32 <typ.UInt32> (Mul32 <typ.UInt32> (Const32 <typ.UInt32> [int32(udivisible32(c).m)]) x) (Const32 <typ.UInt32> [int32(32-udivisible32(c).k)]) ) (Const32 <typ.UInt32> [int32(udivisible32(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst32 {
					continue
				}
				c := auxIntToInt32(v_1_0.AuxInt)
				if v_1_1.Op != OpRsh32Ux64 {
					continue
				}
				_ = v_1_1.Args[1]
				mul := v_1_1.Args[0]
				if mul.Op != OpHmul32u {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst32 || mul_0.Type != typ.UInt32 {
						continue
					}
					m := auxIntToInt32(mul_0.AuxInt)
					if mul_1.Op != OpRsh32Ux64 {
						continue
					}
					_ = mul_1.Args[1]
					if x != mul_1.Args[0] {
						continue
					}
					mul_1_1 := mul_1.Args[1]
					if mul_1_1.Op != OpConst64 || auxIntToInt64(mul_1_1.AuxInt) != 1 {
						continue
					}
					v_1_1_1 := v_1_1.Args[1]
					if v_1_1_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_1.AuxInt)
					if !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(1<<31+(umagic32(c).m+1)/2) && s == umagic32(c).s-2 && x.Op != OpConst32 && udivisibleOK32(c)) {
						continue
					}
					v.reset(OpLeq32U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft32, typ.UInt32)
					v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
					v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v2.AuxInt = int32ToAuxInt(int32(udivisible32(c).m))
					v1.AddArg2(v2, x)
					v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v3.AuxInt = int32ToAuxInt(int32(32 - udivisible32(c).k))
					v0.AddArg2(v1, v3)
					v4 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v4.AuxInt = int32ToAuxInt(int32(udivisible32(c).max))
					v.AddArg2(v0, v4)
					return true
				}
			}
		}
		break
	}
	// match: (Eq32 x (Mul32 (Const32 [c]) (Rsh32Ux64 (Avg32u x mul:(Hmul32u (Const32 [m]) x)) (Const64 [s])) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(umagic32(c).m) && s == umagic32(c).s-1 && x.Op != OpConst32 && udivisibleOK32(c)
	// result: (Leq32U (RotateLeft32 <typ.UInt32> (Mul32 <typ.UInt32> (Const32 <typ.UInt32> [int32(udivisible32(c).m)]) x) (Const32 <typ.UInt32> [int32(32-udivisible32(c).k)]) ) (Const32 <typ.UInt32> [int32(udivisible32(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst32 {
					continue
				}
				c := auxIntToInt32(v_1_0.AuxInt)
				if v_1_1.Op != OpRsh32Ux64 {
					continue
				}
				_ = v_1_1.Args[1]
				v_1_1_0 := v_1_1.Args[0]
				if v_1_1_0.Op != OpAvg32u {
					continue
				}
				_ = v_1_1_0.Args[1]
				if x != v_1_1_0.Args[0] {
					continue
				}
				mul := v_1_1_0.Args[1]
				if mul.Op != OpHmul32u {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst32 {
						continue
					}
					m := auxIntToInt32(mul_0.AuxInt)
					if x != mul_1 {
						continue
					}
					v_1_1_1 := v_1_1.Args[1]
					if v_1_1_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_1.AuxInt)
					if !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(umagic32(c).m) && s == umagic32(c).s-1 && x.Op != OpConst32 && udivisibleOK32(c)) {
						continue
					}
					v.reset(OpLeq32U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft32, typ.UInt32)
					v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
					v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v2.AuxInt = int32ToAuxInt(int32(udivisible32(c).m))
					v1.AddArg2(v2, x)
					v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v3.AuxInt = int32ToAuxInt(int32(32 - udivisible32(c).k))
					v0.AddArg2(v1, v3)
					v4 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v4.AuxInt = int32ToAuxInt(int32(udivisible32(c).max))
					v.AddArg2(v0, v4)
					return true
				}
			}
		}
		break
	}
	// match: (Eq32 x (Mul32 (Const32 [c]) (Trunc64to32 (Rsh64Ux64 mul:(Mul64 (Const64 [m]) (ZeroExt32to64 x)) (Const64 [s]))) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(1<<31+umagic32(c).m/2) && s == 32+umagic32(c).s-1 && x.Op != OpConst32 && udivisibleOK32(c)
	// result: (Leq32U (RotateLeft32 <typ.UInt32> (Mul32 <typ.UInt32> (Const32 <typ.UInt32> [int32(udivisible32(c).m)]) x) (Const32 <typ.UInt32> [int32(32-udivisible32(c).k)]) ) (Const32 <typ.UInt32> [int32(udivisible32(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst32 {
					continue
				}
				c := auxIntToInt32(v_1_0.AuxInt)
				if v_1_1.Op != OpTrunc64to32 {
					continue
				}
				v_1_1_0 := v_1_1.Args[0]
				if v_1_1_0.Op != OpRsh64Ux64 {
					continue
				}
				_ = v_1_1_0.Args[1]
				mul := v_1_1_0.Args[0]
				if mul.Op != OpMul64 {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst64 {
						continue
					}
					m := auxIntToInt64(mul_0.AuxInt)
					if mul_1.Op != OpZeroExt32to64 || x != mul_1.Args[0] {
						continue
					}
					v_1_1_0_1 := v_1_1_0.Args[1]
					if v_1_1_0_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_0_1.AuxInt)
					if !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(1<<31+umagic32(c).m/2) && s == 32+umagic32(c).s-1 && x.Op != OpConst32 && udivisibleOK32(c)) {
						continue
					}
					v.reset(OpLeq32U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft32, typ.UInt32)
					v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
					v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v2.AuxInt = int32ToAuxInt(int32(udivisible32(c).m))
					v1.AddArg2(v2, x)
					v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v3.AuxInt = int32ToAuxInt(int32(32 - udivisible32(c).k))
					v0.AddArg2(v1, v3)
					v4 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v4.AuxInt = int32ToAuxInt(int32(udivisible32(c).max))
					v.AddArg2(v0, v4)
					return true
				}
			}
		}
		break
	}
	// match: (Eq32 x (Mul32 (Const32 [c]) (Trunc64to32 (Rsh64Ux64 mul:(Mul64 (Const64 [m]) (Rsh64Ux64 (ZeroExt32to64 x) (Const64 [1]))) (Const64 [s]))) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(1<<31+(umagic32(c).m+1)/2) && s == 32+umagic32(c).s-2 && x.Op != OpConst32 && udivisibleOK32(c)
	// result: (Leq32U (RotateLeft32 <typ.UInt32> (Mul32 <typ.UInt32> (Const32 <typ.UInt32> [int32(udivisible32(c).m)]) x) (Const32 <typ.UInt32> [int32(32-udivisible32(c).k)]) ) (Const32 <typ.UInt32> [int32(udivisible32(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst32 {
					continue
				}
				c := auxIntToInt32(v_1_0.AuxInt)
				if v_1_1.Op != OpTrunc64to32 {
					continue
				}
				v_1_1_0 := v_1_1.Args[0]
				if v_1_1_0.Op != OpRsh64Ux64 {
					continue
				}
				_ = v_1_1_0.Args[1]
				mul := v_1_1_0.Args[0]
				if mul.Op != OpMul64 {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst64 {
						continue
					}
					m := auxIntToInt64(mul_0.AuxInt)
					if mul_1.Op != OpRsh64Ux64 {
						continue
					}
					_ = mul_1.Args[1]
					mul_1_0 := mul_1.Args[0]
					if mul_1_0.Op != OpZeroExt32to64 || x != mul_1_0.Args[0] {
						continue
					}
					mul_1_1 := mul_1.Args[1]
					if mul_1_1.Op != OpConst64 || auxIntToInt64(mul_1_1.AuxInt) != 1 {
						continue
					}
					v_1_1_0_1 := v_1_1_0.Args[1]
					if v_1_1_0_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_0_1.AuxInt)
					if !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(1<<31+(umagic32(c).m+1)/2) && s == 32+umagic32(c).s-2 && x.Op != OpConst32 && udivisibleOK32(c)) {
						continue
					}
					v.reset(OpLeq32U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft32, typ.UInt32)
					v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
					v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v2.AuxInt = int32ToAuxInt(int32(udivisible32(c).m))
					v1.AddArg2(v2, x)
					v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v3.AuxInt = int32ToAuxInt(int32(32 - udivisible32(c).k))
					v0.AddArg2(v1, v3)
					v4 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v4.AuxInt = int32ToAuxInt(int32(udivisible32(c).max))
					v.AddArg2(v0, v4)
					return true
				}
			}
		}
		break
	}
	// match: (Eq32 x (Mul32 (Const32 [c]) (Trunc64to32 (Rsh64Ux64 (Avg64u (Lsh64x64 (ZeroExt32to64 x) (Const64 [32])) mul:(Mul64 (Const64 [m]) (ZeroExt32to64 x))) (Const64 [s]))) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(umagic32(c).m) && s == 32+umagic32(c).s-1 && x.Op != OpConst32 && udivisibleOK32(c)
	// result: (Leq32U (RotateLeft32 <typ.UInt32> (Mul32 <typ.UInt32> (Const32 <typ.UInt32> [int32(udivisible32(c).m)]) x) (Const32 <typ.UInt32> [int32(32-udivisible32(c).k)]) ) (Const32 <typ.UInt32> [int32(udivisible32(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst32 {
					continue
				}
				c := auxIntToInt32(v_1_0.AuxInt)
				if v_1_1.Op != OpTrunc64to32 {
					continue
				}
				v_1_1_0 := v_1_1.Args[0]
				if v_1_1_0.Op != OpRsh64Ux64 {
					continue
				}
				_ = v_1_1_0.Args[1]
				v_1_1_0_0 := v_1_1_0.Args[0]
				if v_1_1_0_0.Op != OpAvg64u {
					continue
				}
				_ = v_1_1_0_0.Args[1]
				v_1_1_0_0_0 := v_1_1_0_0.Args[0]
				if v_1_1_0_0_0.Op != OpLsh64x64 {
					continue
				}
				_ = v_1_1_0_0_0.Args[1]
				v_1_1_0_0_0_0 := v_1_1_0_0_0.Args[0]
				if v_1_1_0_0_0_0.Op != OpZeroExt32to64 || x != v_1_1_0_0_0_0.Args[0] {
					continue
				}
				v_1_1_0_0_0_1 := v_1_1_0_0_0.Args[1]
				if v_1_1_0_0_0_1.Op != OpConst64 || auxIntToInt64(v_1_1_0_0_0_1.AuxInt) != 32 {
					continue
				}
				mul := v_1_1_0_0.Args[1]
				if mul.Op != OpMul64 {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst64 {
						continue
					}
					m := auxIntToInt64(mul_0.AuxInt)
					if mul_1.Op != OpZeroExt32to64 || x != mul_1.Args[0] {
						continue
					}
					v_1_1_0_1 := v_1_1_0.Args[1]
					if v_1_1_0_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_0_1.AuxInt)
					if !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(umagic32(c).m) && s == 32+umagic32(c).s-1 && x.Op != OpConst32 && udivisibleOK32(c)) {
						continue
					}
					v.reset(OpLeq32U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft32, typ.UInt32)
					v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
					v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v2.AuxInt = int32ToAuxInt(int32(udivisible32(c).m))
					v1.AddArg2(v2, x)
					v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v3.AuxInt = int32ToAuxInt(int32(32 - udivisible32(c).k))
					v0.AddArg2(v1, v3)
					v4 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v4.AuxInt = int32ToAuxInt(int32(udivisible32(c).max))
					v.AddArg2(v0, v4)
					return true
				}
			}
		}
		break
	}
	// match: (Eq32 x (Mul32 (Const32 [c]) (Sub32 (Rsh64x64 mul:(Mul64 (Const64 [m]) (SignExt32to64 x)) (Const64 [s])) (Rsh64x64 (SignExt32to64 x) (Const64 [63]))) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(smagic32(c).m) && s == 32+smagic32(c).s && x.Op != OpConst32 && sdivisibleOK32(c)
	// result: (Leq32U (RotateLeft32 <typ.UInt32> (Add32 <typ.UInt32> (Mul32 <typ.UInt32> (Const32 <typ.UInt32> [int32(sdivisible32(c).m)]) x) (Const32 <typ.UInt32> [int32(sdivisible32(c).a)]) ) (Const32 <typ.UInt32> [int32(32-sdivisible32(c).k)]) ) (Const32 <typ.UInt32> [int32(sdivisible32(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst32 {
					continue
				}
				c := auxIntToInt32(v_1_0.AuxInt)
				if v_1_1.Op != OpSub32 {
					continue
				}
				_ = v_1_1.Args[1]
				v_1_1_0 := v_1_1.Args[0]
				if v_1_1_0.Op != OpRsh64x64 {
					continue
				}
				_ = v_1_1_0.Args[1]
				mul := v_1_1_0.Args[0]
				if mul.Op != OpMul64 {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst64 {
						continue
					}
					m := auxIntToInt64(mul_0.AuxInt)
					if mul_1.Op != OpSignExt32to64 || x != mul_1.Args[0] {
						continue
					}
					v_1_1_0_1 := v_1_1_0.Args[1]
					if v_1_1_0_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_0_1.AuxInt)
					v_1_1_1 := v_1_1.Args[1]
					if v_1_1_1.Op != OpRsh64x64 {
						continue
					}
					_ = v_1_1_1.Args[1]
					v_1_1_1_0 := v_1_1_1.Args[0]
					if v_1_1_1_0.Op != OpSignExt32to64 || x != v_1_1_1_0.Args[0] {
						continue
					}
					v_1_1_1_1 := v_1_1_1.Args[1]
					if v_1_1_1_1.Op != OpConst64 || auxIntToInt64(v_1_1_1_1.AuxInt) != 63 || !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(smagic32(c).m) && s == 32+smagic32(c).s && x.Op != OpConst32 && sdivisibleOK32(c)) {
						continue
					}
					v.reset(OpLeq32U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft32, typ.UInt32)
					v1 := b.NewValue0(v.Pos, OpAdd32, typ.UInt32)
					v2 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
					v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v3.AuxInt = int32ToAuxInt(int32(sdivisible32(c).m))
					v2.AddArg2(v3, x)
					v4 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v4.AuxInt = int32ToAuxInt(int32(sdivisible32(c).a))
					v1.AddArg2(v2, v4)
					v5 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v5.AuxInt = int32ToAuxInt(int32(32 - sdivisible32(c).k))
					v0.AddArg2(v1, v5)
					v6 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v6.AuxInt = int32ToAuxInt(int32(sdivisible32(c).max))
					v.AddArg2(v0, v6)
					return true
				}
			}
		}
		break
	}
	// match: (Eq32 x (Mul32 (Const32 [c]) (Sub32 (Rsh32x64 mul:(Hmul32 (Const32 [m]) x) (Const64 [s])) (Rsh32x64 x (Const64 [31]))) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(smagic32(c).m/2) && s == smagic32(c).s-1 && x.Op != OpConst32 && sdivisibleOK32(c)
	// result: (Leq32U (RotateLeft32 <typ.UInt32> (Add32 <typ.UInt32> (Mul32 <typ.UInt32> (Const32 <typ.UInt32> [int32(sdivisible32(c).m)]) x) (Const32 <typ.UInt32> [int32(sdivisible32(c).a)]) ) (Const32 <typ.UInt32> [int32(32-sdivisible32(c).k)]) ) (Const32 <typ.UInt32> [int32(sdivisible32(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst32 {
					continue
				}
				c := auxIntToInt32(v_1_0.AuxInt)
				if v_1_1.Op != OpSub32 {
					continue
				}
				_ = v_1_1.Args[1]
				v_1_1_0 := v_1_1.Args[0]
				if v_1_1_0.Op != OpRsh32x64 {
					continue
				}
				_ = v_1_1_0.Args[1]
				mul := v_1_1_0.Args[0]
				if mul.Op != OpHmul32 {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst32 {
						continue
					}
					m := auxIntToInt32(mul_0.AuxInt)
					if x != mul_1 {
						continue
					}
					v_1_1_0_1 := v_1_1_0.Args[1]
					if v_1_1_0_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_0_1.AuxInt)
					v_1_1_1 := v_1_1.Args[1]
					if v_1_1_1.Op != OpRsh32x64 {
						continue
					}
					_ = v_1_1_1.Args[1]
					if x != v_1_1_1.Args[0] {
						continue
					}
					v_1_1_1_1 := v_1_1_1.Args[1]
					if v_1_1_1_1.Op != OpConst64 || auxIntToInt64(v_1_1_1_1.AuxInt) != 31 || !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(smagic32(c).m/2) && s == smagic32(c).s-1 && x.Op != OpConst32 && sdivisibleOK32(c)) {
						continue
					}
					v.reset(OpLeq32U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft32, typ.UInt32)
					v1 := b.NewValue0(v.Pos, OpAdd32, typ.UInt32)
					v2 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
					v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v3.AuxInt = int32ToAuxInt(int32(sdivisible32(c).m))
					v2.AddArg2(v3, x)
					v4 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v4.AuxInt = int32ToAuxInt(int32(sdivisible32(c).a))
					v1.AddArg2(v2, v4)
					v5 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v5.AuxInt = int32ToAuxInt(int32(32 - sdivisible32(c).k))
					v0.AddArg2(v1, v5)
					v6 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
					v6.AuxInt = int32ToAuxInt(int32(sdivisible32(c).max))
					v.AddArg2(v0, v6)
					return true
				}
			}
		}
		break
	}
	// match: (Eq32 x (Mul32 (Const32 [c]) (Sub32 (Rsh32x64 (Add32 mul:(Hmul32 (Const32 [m]) x) x) (Const64 [s])) (Rsh32x64 x (Const64 [31]))) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(smagic32(c).m) && s == smagic32(c).s && x.Op != OpConst32 && sdivisibleOK32(c)
	// result: (Leq32U (RotateLeft32 <typ.UInt32> (Add32 <typ.UInt32> (Mul32 <typ.UInt32> (Const32 <typ.UInt32> [int32(sdivisible32(c).m)]) x) (Const32 <typ.UInt32> [int32(sdivisible32(c).a)]) ) (Const32 <typ.UInt32> [int32(32-sdivisible32(c).k)]) ) (Const32 <typ.UInt32> [int32(sdivisible32(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst32 {
					continue
				}
				c := auxIntToInt32(v_1_0.AuxInt)
				if v_1_1.Op != OpSub32 {
					continue
				}
				_ = v_1_1.Args[1]
				v_1_1_0 := v_1_1.Args[0]
				if v_1_1_0.Op != OpRsh32x64 {
					continue
				}
				_ = v_1_1_0.Args[1]
				v_1_1_0_0 := v_1_1_0.Args[0]
				if v_1_1_0_0.Op != OpAdd32 {
					continue
				}
				_ = v_1_1_0_0.Args[1]
				v_1_1_0_0_0 := v_1_1_0_0.Args[0]
				v_1_1_0_0_1 := v_1_1_0_0.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, v_1_1_0_0_0, v_1_1_0_0_1 = _i2+1, v_1_1_0_0_1, v_1_1_0_0_0 {
					mul := v_1_1_0_0_0
					if mul.Op != OpHmul32 {
						continue
					}
					_ = mul.Args[1]
					mul_0 := mul.Args[0]
					mul_1 := mul.Args[1]
					for _i3 := 0; _i3 <= 1; _i3, mul_0, mul_1 = _i3+1, mul_1, mul_0 {
						if mul_0.Op != OpConst32 {
							continue
						}
						m := auxIntToInt32(mul_0.AuxInt)
						if x != mul_1 || x != v_1_1_0_0_1 {
							continue
						}
						v_1_1_0_1 := v_1_1_0.Args[1]
						if v_1_1_0_1.Op != OpConst64 {
							continue
						}
						s := auxIntToInt64(v_1_1_0_1.AuxInt)
						v_1_1_1 := v_1_1.Args[1]
						if v_1_1_1.Op != OpRsh32x64 {
							continue
						}
						_ = v_1_1_1.Args[1]
						if x != v_1_1_1.Args[0] {
							continue
						}
						v_1_1_1_1 := v_1_1_1.Args[1]
						if v_1_1_1_1.Op != OpConst64 || auxIntToInt64(v_1_1_1_1.AuxInt) != 31 || !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(smagic32(c).m) && s == smagic32(c).s && x.Op != OpConst32 && sdivisibleOK32(c)) {
							continue
						}
						v.reset(OpLeq32U)
						v0 := b.NewValue0(v.Pos, OpRotateLeft32, typ.UInt32)
						v1 := b.NewValue0(v.Pos, OpAdd32, typ.UInt32)
						v2 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
						v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
						v3.AuxInt = int32ToAuxInt(int32(sdivisible32(c).m))
						v2.AddArg2(v3, x)
						v4 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
						v4.AuxInt = int32ToAuxInt(int32(sdivisible32(c).a))
						v1.AddArg2(v2, v4)
						v5 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
						v5.AuxInt = int32ToAuxInt(int32(32 - sdivisible32(c).k))
						v0.AddArg2(v1, v5)
						v6 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
						v6.AuxInt = int32ToAuxInt(int32(sdivisible32(c).max))
						v.AddArg2(v0, v6)
						return true
					}
				}
			}
		}
		break
	}
	// match: (Eq32 n (Lsh32x64 (Rsh32x64 (Add32 <t> n (Rsh32Ux64 <t> (Rsh32x64 <t> n (Const64 <typ.UInt64> [31])) (Const64 <typ.UInt64> [kbar]))) (Const64 <typ.UInt64> [k])) (Const64 <typ.UInt64> [k])) )
	// cond: k > 0 && k < 31 && kbar == 32 - k
	// result: (Eq32 (And32 <t> n (Const32 <t> [1<<uint(k)-1])) (Const32 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			n := v_0
			if v_1.Op != OpLsh32x64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRsh32x64 {
				continue
			}
			_ = v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			if v_1_0_0.Op != OpAdd32 {
				continue
			}
			t := v_1_0_0.Type
			_ = v_1_0_0.Args[1]
			v_1_0_0_0 := v_1_0_0.Args[0]
			v_1_0_0_1 := v_1_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0_0_0, v_1_0_0_1 = _i1+1, v_1_0_0_1, v_1_0_0_0 {
				if n != v_1_0_0_0 || v_1_0_0_1.Op != OpRsh32Ux64 || v_1_0_0_1.Type != t {
					continue
				}
				_ = v_1_0_0_1.Args[1]
				v_1_0_0_1_0 := v_1_0_0_1.Args[0]
				if v_1_0_0_1_0.Op != OpRsh32x64 || v_1_0_0_1_0.Type != t {
					continue
				}
				_ = v_1_0_0_1_0.Args[1]
				if n != v_1_0_0_1_0.Args[0] {
					continue
				}
				v_1_0_0_1_0_1 := v_1_0_0_1_0.Args[1]
				if v_1_0_0_1_0_1.Op != OpConst64 || v_1_0_0_1_0_1.Type != typ.UInt64 || auxIntToInt64(v_1_0_0_1_0_1.AuxInt) != 31 {
					continue
				}
				v_1_0_0_1_1 := v_1_0_0_1.Args[1]
				if v_1_0_0_1_1.Op != OpConst64 || v_1_0_0_1_1.Type != typ.UInt64 {
					continue
				}
				kbar := auxIntToInt64(v_1_0_0_1_1.AuxInt)
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst64 || v_1_0_1.Type != typ.UInt64 {
					continue
				}
				k := auxIntToInt64(v_1_0_1.AuxInt)
				v_1_1 := v_1.Args[1]
				if v_1_1.Op != OpConst64 || v_1_1.Type != typ.UInt64 || auxIntToInt64(v_1_1.AuxInt) != k || !(k > 0 && k < 31 && kbar == 32-k) {
					continue
				}
				v.reset(OpEq32)
				v0 := b.NewValue0(v.Pos, OpAnd32, t)
				v1 := b.NewValue0(v.Pos, OpConst32, t)
				v1.AuxInt = int32ToAuxInt(1<<uint(k) - 1)
				v0.AddArg2(n, v1)
				v2 := b.NewValue0(v.Pos, OpConst32, t)
				v2.AuxInt = int32ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq32 s:(Sub32 x y) (Const32 [0]))
	// cond: s.Uses == 1
	// result: (Eq32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s := v_0
			if s.Op != OpSub32 {
				continue
			}
			y := s.Args[1]
			x := s.Args[0]
			if v_1.Op != OpConst32 || auxIntToInt32(v_1.AuxInt) != 0 || !(s.Uses == 1) {
				continue
			}
			v.reset(OpEq32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Eq32 (And32 <t> x (Const32 <t> [y])) (Const32 <t> [y]))
	// cond: oneBit32(y)
	// result: (Neq32 (And32 <t> x (Const32 <t> [y])) (Const32 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAnd32 {
				continue
			}
			t := v_0.Type
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				if v_0_1.Op != OpConst32 || v_0_1.Type != t {
					continue
				}
				y := auxIntToInt32(v_0_1.AuxInt)
				if v_1.Op != OpConst32 || v_1.Type != t || auxIntToInt32(v_1.AuxInt) != y || !(oneBit32(y)) {
					continue
				}
				v.reset(OpNeq32)
				v0 := b.NewValue0(v.Pos, OpAnd32, t)
				v1 := b.NewValue0(v.Pos, OpConst32, t)
				v1.AuxInt = int32ToAuxInt(y)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst32, t)
				v2.AuxInt = int32ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpEq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Eq32F (Const32F [c]) (Const32F [d]))
	// result: (ConstBool [c == d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32F {
				continue
			}
			c := auxIntToFloat32(v_0.AuxInt)
			if v_1.Op != OpConst32F {
				continue
			}
			d := auxIntToFloat32(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c == d)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpEq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq64 x x)
	// result: (ConstBool [true])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Eq64 (Const64 <t> [c]) (Add64 (Const64 <t> [d]) x))
	// result: (Eq64 (Const64 <t> [c-d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpAdd64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst64 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt64(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpEq64)
				v0 := b.NewValue0(v.Pos, OpConst64, t)
				v0.AuxInt = int64ToAuxInt(c - d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Eq64 (Const64 [c]) (Const64 [d]))
	// result: (ConstBool [c == d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c == d)
			return true
		}
		break
	}
	// match: (Eq64 x (Mul64 (Const64 [c]) (Rsh64Ux64 mul:(Hmul64u (Const64 [m]) x) (Const64 [s])) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(1<<63+umagic64(c).m/2) && s == umagic64(c).s-1 && x.Op != OpConst64 && udivisibleOK64(c)
	// result: (Leq64U (RotateLeft64 <typ.UInt64> (Mul64 <typ.UInt64> (Const64 <typ.UInt64> [int64(udivisible64(c).m)]) x) (Const64 <typ.UInt64> [64-udivisible64(c).k]) ) (Const64 <typ.UInt64> [int64(udivisible64(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst64 {
					continue
				}
				c := auxIntToInt64(v_1_0.AuxInt)
				if v_1_1.Op != OpRsh64Ux64 {
					continue
				}
				_ = v_1_1.Args[1]
				mul := v_1_1.Args[0]
				if mul.Op != OpHmul64u {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst64 {
						continue
					}
					m := auxIntToInt64(mul_0.AuxInt)
					if x != mul_1 {
						continue
					}
					v_1_1_1 := v_1_1.Args[1]
					if v_1_1_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_1.AuxInt)
					if !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(1<<63+umagic64(c).m/2) && s == umagic64(c).s-1 && x.Op != OpConst64 && udivisibleOK64(c)) {
						continue
					}
					v.reset(OpLeq64U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft64, typ.UInt64)
					v1 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
					v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
					v2.AuxInt = int64ToAuxInt(int64(udivisible64(c).m))
					v1.AddArg2(v2, x)
					v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
					v3.AuxInt = int64ToAuxInt(64 - udivisible64(c).k)
					v0.AddArg2(v1, v3)
					v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
					v4.AuxInt = int64ToAuxInt(int64(udivisible64(c).max))
					v.AddArg2(v0, v4)
					return true
				}
			}
		}
		break
	}
	// match: (Eq64 x (Mul64 (Const64 [c]) (Rsh64Ux64 mul:(Hmul64u (Const64 [m]) (Rsh64Ux64 x (Const64 [1]))) (Const64 [s])) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(1<<63+(umagic64(c).m+1)/2) && s == umagic64(c).s-2 && x.Op != OpConst64 && udivisibleOK64(c)
	// result: (Leq64U (RotateLeft64 <typ.UInt64> (Mul64 <typ.UInt64> (Const64 <typ.UInt64> [int64(udivisible64(c).m)]) x) (Const64 <typ.UInt64> [64-udivisible64(c).k]) ) (Const64 <typ.UInt64> [int64(udivisible64(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst64 {
					continue
				}
				c := auxIntToInt64(v_1_0.AuxInt)
				if v_1_1.Op != OpRsh64Ux64 {
					continue
				}
				_ = v_1_1.Args[1]
				mul := v_1_1.Args[0]
				if mul.Op != OpHmul64u {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst64 {
						continue
					}
					m := auxIntToInt64(mul_0.AuxInt)
					if mul_1.Op != OpRsh64Ux64 {
						continue
					}
					_ = mul_1.Args[1]
					if x != mul_1.Args[0] {
						continue
					}
					mul_1_1 := mul_1.Args[1]
					if mul_1_1.Op != OpConst64 || auxIntToInt64(mul_1_1.AuxInt) != 1 {
						continue
					}
					v_1_1_1 := v_1_1.Args[1]
					if v_1_1_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_1.AuxInt)
					if !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(1<<63+(umagic64(c).m+1)/2) && s == umagic64(c).s-2 && x.Op != OpConst64 && udivisibleOK64(c)) {
						continue
					}
					v.reset(OpLeq64U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft64, typ.UInt64)
					v1 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
					v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
					v2.AuxInt = int64ToAuxInt(int64(udivisible64(c).m))
					v1.AddArg2(v2, x)
					v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
					v3.AuxInt = int64ToAuxInt(64 - udivisible64(c).k)
					v0.AddArg2(v1, v3)
					v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
					v4.AuxInt = int64ToAuxInt(int64(udivisible64(c).max))
					v.AddArg2(v0, v4)
					return true
				}
			}
		}
		break
	}
	// match: (Eq64 x (Mul64 (Const64 [c]) (Rsh64Ux64 (Avg64u x mul:(Hmul64u (Const64 [m]) x)) (Const64 [s])) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(umagic64(c).m) && s == umagic64(c).s-1 && x.Op != OpConst64 && udivisibleOK64(c)
	// result: (Leq64U (RotateLeft64 <typ.UInt64> (Mul64 <typ.UInt64> (Const64 <typ.UInt64> [int64(udivisible64(c).m)]) x) (Const64 <typ.UInt64> [64-udivisible64(c).k]) ) (Const64 <typ.UInt64> [int64(udivisible64(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst64 {
					continue
				}
				c := auxIntToInt64(v_1_0.AuxInt)
				if v_1_1.Op != OpRsh64Ux64 {
					continue
				}
				_ = v_1_1.Args[1]
				v_1_1_0 := v_1_1.Args[0]
				if v_1_1_0.Op != OpAvg64u {
					continue
				}
				_ = v_1_1_0.Args[1]
				if x != v_1_1_0.Args[0] {
					continue
				}
				mul := v_1_1_0.Args[1]
				if mul.Op != OpHmul64u {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst64 {
						continue
					}
					m := auxIntToInt64(mul_0.AuxInt)
					if x != mul_1 {
						continue
					}
					v_1_1_1 := v_1_1.Args[1]
					if v_1_1_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_1.AuxInt)
					if !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(umagic64(c).m) && s == umagic64(c).s-1 && x.Op != OpConst64 && udivisibleOK64(c)) {
						continue
					}
					v.reset(OpLeq64U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft64, typ.UInt64)
					v1 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
					v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
					v2.AuxInt = int64ToAuxInt(int64(udivisible64(c).m))
					v1.AddArg2(v2, x)
					v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
					v3.AuxInt = int64ToAuxInt(64 - udivisible64(c).k)
					v0.AddArg2(v1, v3)
					v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
					v4.AuxInt = int64ToAuxInt(int64(udivisible64(c).max))
					v.AddArg2(v0, v4)
					return true
				}
			}
		}
		break
	}
	// match: (Eq64 x (Mul64 (Const64 [c]) (Sub64 (Rsh64x64 mul:(Hmul64 (Const64 [m]) x) (Const64 [s])) (Rsh64x64 x (Const64 [63]))) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(smagic64(c).m/2) && s == smagic64(c).s-1 && x.Op != OpConst64 && sdivisibleOK64(c)
	// result: (Leq64U (RotateLeft64 <typ.UInt64> (Add64 <typ.UInt64> (Mul64 <typ.UInt64> (Const64 <typ.UInt64> [int64(sdivisible64(c).m)]) x) (Const64 <typ.UInt64> [int64(sdivisible64(c).a)]) ) (Const64 <typ.UInt64> [64-sdivisible64(c).k]) ) (Const64 <typ.UInt64> [int64(sdivisible64(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst64 {
					continue
				}
				c := auxIntToInt64(v_1_0.AuxInt)
				if v_1_1.Op != OpSub64 {
					continue
				}
				_ = v_1_1.Args[1]
				v_1_1_0 := v_1_1.Args[0]
				if v_1_1_0.Op != OpRsh64x64 {
					continue
				}
				_ = v_1_1_0.Args[1]
				mul := v_1_1_0.Args[0]
				if mul.Op != OpHmul64 {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst64 {
						continue
					}
					m := auxIntToInt64(mul_0.AuxInt)
					if x != mul_1 {
						continue
					}
					v_1_1_0_1 := v_1_1_0.Args[1]
					if v_1_1_0_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_0_1.AuxInt)
					v_1_1_1 := v_1_1.Args[1]
					if v_1_1_1.Op != OpRsh64x64 {
						continue
					}
					_ = v_1_1_1.Args[1]
					if x != v_1_1_1.Args[0] {
						continue
					}
					v_1_1_1_1 := v_1_1_1.Args[1]
					if v_1_1_1_1.Op != OpConst64 || auxIntToInt64(v_1_1_1_1.AuxInt) != 63 || !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(smagic64(c).m/2) && s == smagic64(c).s-1 && x.Op != OpConst64 && sdivisibleOK64(c)) {
						continue
					}
					v.reset(OpLeq64U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft64, typ.UInt64)
					v1 := b.NewValue0(v.Pos, OpAdd64, typ.UInt64)
					v2 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
					v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
					v3.AuxInt = int64ToAuxInt(int64(sdivisible64(c).m))
					v2.AddArg2(v3, x)
					v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
					v4.AuxInt = int64ToAuxInt(int64(sdivisible64(c).a))
					v1.AddArg2(v2, v4)
					v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
					v5.AuxInt = int64ToAuxInt(64 - sdivisible64(c).k)
					v0.AddArg2(v1, v5)
					v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
					v6.AuxInt = int64ToAuxInt(int64(sdivisible64(c).max))
					v.AddArg2(v0, v6)
					return true
				}
			}
		}
		break
	}
	// match: (Eq64 x (Mul64 (Const64 [c]) (Sub64 (Rsh64x64 (Add64 mul:(Hmul64 (Const64 [m]) x) x) (Const64 [s])) (Rsh64x64 x (Const64 [63]))) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(smagic64(c).m) && s == smagic64(c).s && x.Op != OpConst64 && sdivisibleOK64(c)
	// result: (Leq64U (RotateLeft64 <typ.UInt64> (Add64 <typ.UInt64> (Mul64 <typ.UInt64> (Const64 <typ.UInt64> [int64(sdivisible64(c).m)]) x) (Const64 <typ.UInt64> [int64(sdivisible64(c).a)]) ) (Const64 <typ.UInt64> [64-sdivisible64(c).k]) ) (Const64 <typ.UInt64> [int64(sdivisible64(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst64 {
					continue
				}
				c := auxIntToInt64(v_1_0.AuxInt)
				if v_1_1.Op != OpSub64 {
					continue
				}
				_ = v_1_1.Args[1]
				v_1_1_0 := v_1_1.Args[0]
				if v_1_1_0.Op != OpRsh64x64 {
					continue
				}
				_ = v_1_1_0.Args[1]
				v_1_1_0_0 := v_1_1_0.Args[0]
				if v_1_1_0_0.Op != OpAdd64 {
					continue
				}
				_ = v_1_1_0_0.Args[1]
				v_1_1_0_0_0 := v_1_1_0_0.Args[0]
				v_1_1_0_0_1 := v_1_1_0_0.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, v_1_1_0_0_0, v_1_1_0_0_1 = _i2+1, v_1_1_0_0_1, v_1_1_0_0_0 {
					mul := v_1_1_0_0_0
					if mul.Op != OpHmul64 {
						continue
					}
					_ = mul.Args[1]
					mul_0 := mul.Args[0]
					mul_1 := mul.Args[1]
					for _i3 := 0; _i3 <= 1; _i3, mul_0, mul_1 = _i3+1, mul_1, mul_0 {
						if mul_0.Op != OpConst64 {
							continue
						}
						m := auxIntToInt64(mul_0.AuxInt)
						if x != mul_1 || x != v_1_1_0_0_1 {
							continue
						}
						v_1_1_0_1 := v_1_1_0.Args[1]
						if v_1_1_0_1.Op != OpConst64 {
							continue
						}
						s := auxIntToInt64(v_1_1_0_1.AuxInt)
						v_1_1_1 := v_1_1.Args[1]
						if v_1_1_1.Op != OpRsh64x64 {
							continue
						}
						_ = v_1_1_1.Args[1]
						if x != v_1_1_1.Args[0] {
							continue
						}
						v_1_1_1_1 := v_1_1_1.Args[1]
						if v_1_1_1_1.Op != OpConst64 || auxIntToInt64(v_1_1_1_1.AuxInt) != 63 || !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int64(smagic64(c).m) && s == smagic64(c).s && x.Op != OpConst64 && sdivisibleOK64(c)) {
							continue
						}
						v.reset(OpLeq64U)
						v0 := b.NewValue0(v.Pos, OpRotateLeft64, typ.UInt64)
						v1 := b.NewValue0(v.Pos, OpAdd64, typ.UInt64)
						v2 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
						v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
						v3.AuxInt = int64ToAuxInt(int64(sdivisible64(c).m))
						v2.AddArg2(v3, x)
						v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
						v4.AuxInt = int64ToAuxInt(int64(sdivisible64(c).a))
						v1.AddArg2(v2, v4)
						v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
						v5.AuxInt = int64ToAuxInt(64 - sdivisible64(c).k)
						v0.AddArg2(v1, v5)
						v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
						v6.AuxInt = int64ToAuxInt(int64(sdivisible64(c).max))
						v.AddArg2(v0, v6)
						return true
					}
				}
			}
		}
		break
	}
	// match: (Eq64 n (Lsh64x64 (Rsh64x64 (Add64 <t> n (Rsh64Ux64 <t> (Rsh64x64 <t> n (Const64 <typ.UInt64> [63])) (Const64 <typ.UInt64> [kbar]))) (Const64 <typ.UInt64> [k])) (Const64 <typ.UInt64> [k])) )
	// cond: k > 0 && k < 63 && kbar == 64 - k
	// result: (Eq64 (And64 <t> n (Const64 <t> [1<<uint(k)-1])) (Const64 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			n := v_0
			if v_1.Op != OpLsh64x64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRsh64x64 {
				continue
			}
			_ = v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			if v_1_0_0.Op != OpAdd64 {
				continue
			}
			t := v_1_0_0.Type
			_ = v_1_0_0.Args[1]
			v_1_0_0_0 := v_1_0_0.Args[0]
			v_1_0_0_1 := v_1_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0_0_0, v_1_0_0_1 = _i1+1, v_1_0_0_1, v_1_0_0_0 {
				if n != v_1_0_0_0 || v_1_0_0_1.Op != OpRsh64Ux64 || v_1_0_0_1.Type != t {
					continue
				}
				_ = v_1_0_0_1.Args[1]
				v_1_0_0_1_0 := v_1_0_0_1.Args[0]
				if v_1_0_0_1_0.Op != OpRsh64x64 || v_1_0_0_1_0.Type != t {
					continue
				}
				_ = v_1_0_0_1_0.Args[1]
				if n != v_1_0_0_1_0.Args[0] {
					continue
				}
				v_1_0_0_1_0_1 := v_1_0_0_1_0.Args[1]
				if v_1_0_0_1_0_1.Op != OpConst64 || v_1_0_0_1_0_1.Type != typ.UInt64 || auxIntToInt64(v_1_0_0_1_0_1.AuxInt) != 63 {
					continue
				}
				v_1_0_0_1_1 := v_1_0_0_1.Args[1]
				if v_1_0_0_1_1.Op != OpConst64 || v_1_0_0_1_1.Type != typ.UInt64 {
					continue
				}
				kbar := auxIntToInt64(v_1_0_0_1_1.AuxInt)
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst64 || v_1_0_1.Type != typ.UInt64 {
					continue
				}
				k := auxIntToInt64(v_1_0_1.AuxInt)
				v_1_1 := v_1.Args[1]
				if v_1_1.Op != OpConst64 || v_1_1.Type != typ.UInt64 || auxIntToInt64(v_1_1.AuxInt) != k || !(k > 0 && k < 63 && kbar == 64-k) {
					continue
				}
				v.reset(OpEq64)
				v0 := b.NewValue0(v.Pos, OpAnd64, t)
				v1 := b.NewValue0(v.Pos, OpConst64, t)
				v1.AuxInt = int64ToAuxInt(1<<uint(k) - 1)
				v0.AddArg2(n, v1)
				v2 := b.NewValue0(v.Pos, OpConst64, t)
				v2.AuxInt = int64ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq64 s:(Sub64 x y) (Const64 [0]))
	// cond: s.Uses == 1
	// result: (Eq64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s := v_0
			if s.Op != OpSub64 {
				continue
			}
			y := s.Args[1]
			x := s.Args[0]
			if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 0 || !(s.Uses == 1) {
				continue
			}
			v.reset(OpEq64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Eq64 (And64 <t> x (Const64 <t> [y])) (Const64 <t> [y]))
	// cond: oneBit64(y)
	// result: (Neq64 (And64 <t> x (Const64 <t> [y])) (Const64 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAnd64 {
				continue
			}
			t := v_0.Type
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				if v_0_1.Op != OpConst64 || v_0_1.Type != t {
					continue
				}
				y := auxIntToInt64(v_0_1.AuxInt)
				if v_1.Op != OpConst64 || v_1.Type != t || auxIntToInt64(v_1.AuxInt) != y || !(oneBit64(y)) {
					continue
				}
				v.reset(OpNeq64)
				v0 := b.NewValue0(v.Pos, OpAnd64, t)
				v1 := b.NewValue0(v.Pos, OpConst64, t)
				v1.AuxInt = int64ToAuxInt(y)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst64, t)
				v2.AuxInt = int64ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpEq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Eq64F (Const64F [c]) (Const64F [d]))
	// result: (ConstBool [c == d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64F {
				continue
			}
			c := auxIntToFloat64(v_0.AuxInt)
			if v_1.Op != OpConst64F {
				continue
			}
			d := auxIntToFloat64(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c == d)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpEq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Eq8 x x)
	// result: (ConstBool [true])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Eq8 (Const8 <t> [c]) (Add8 (Const8 <t> [d]) x))
	// result: (Eq8 (Const8 <t> [c-d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpAdd8 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst8 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt8(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpEq8)
				v0 := b.NewValue0(v.Pos, OpConst8, t)
				v0.AuxInt = int8ToAuxInt(c - d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Eq8 (Const8 [c]) (Const8 [d]))
	// result: (ConstBool [c == d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c == d)
			return true
		}
		break
	}
	// match: (Eq8 (Mod8u x (Const8 [c])) (Const8 [0]))
	// cond: x.Op != OpConst8 && udivisibleOK8(c) && !hasSmallRotate(config)
	// result: (Eq32 (Mod32u <typ.UInt32> (ZeroExt8to32 <typ.UInt32> x) (Const32 <typ.UInt32> [int32(uint8(c))])) (Const32 <typ.UInt32> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpMod8u {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_1.AuxInt)
			if v_1.Op != OpConst8 || auxIntToInt8(v_1.AuxInt) != 0 || !(x.Op != OpConst8 && udivisibleOK8(c) && !hasSmallRotate(config)) {
				continue
			}
			v.reset(OpEq32)
			v0 := b.NewValue0(v.Pos, OpMod32u, typ.UInt32)
			v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
			v1.AddArg(x)
			v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
			v2.AuxInt = int32ToAuxInt(int32(uint8(c)))
			v0.AddArg2(v1, v2)
			v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
			v3.AuxInt = int32ToAuxInt(0)
			v.AddArg2(v0, v3)
			return true
		}
		break
	}
	// match: (Eq8 (Mod8 x (Const8 [c])) (Const8 [0]))
	// cond: x.Op != OpConst8 && sdivisibleOK8(c) && !hasSmallRotate(config)
	// result: (Eq32 (Mod32 <typ.Int32> (SignExt8to32 <typ.Int32> x) (Const32 <typ.Int32> [int32(c)])) (Const32 <typ.Int32> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpMod8 {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_1.AuxInt)
			if v_1.Op != OpConst8 || auxIntToInt8(v_1.AuxInt) != 0 || !(x.Op != OpConst8 && sdivisibleOK8(c) && !hasSmallRotate(config)) {
				continue
			}
			v.reset(OpEq32)
			v0 := b.NewValue0(v.Pos, OpMod32, typ.Int32)
			v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
			v1.AddArg(x)
			v2 := b.NewValue0(v.Pos, OpConst32, typ.Int32)
			v2.AuxInt = int32ToAuxInt(int32(c))
			v0.AddArg2(v1, v2)
			v3 := b.NewValue0(v.Pos, OpConst32, typ.Int32)
			v3.AuxInt = int32ToAuxInt(0)
			v.AddArg2(v0, v3)
			return true
		}
		break
	}
	// match: (Eq8 x (Mul8 (Const8 [c]) (Trunc32to8 (Rsh32Ux64 mul:(Mul32 (Const32 [m]) (ZeroExt8to32 x)) (Const64 [s]))) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(1<<8+umagic8(c).m) && s == 8+umagic8(c).s && x.Op != OpConst8 && udivisibleOK8(c)
	// result: (Leq8U (RotateLeft8 <typ.UInt8> (Mul8 <typ.UInt8> (Const8 <typ.UInt8> [int8(udivisible8(c).m)]) x) (Const8 <typ.UInt8> [int8(8-udivisible8(c).k)]) ) (Const8 <typ.UInt8> [int8(udivisible8(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul8 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst8 {
					continue
				}
				c := auxIntToInt8(v_1_0.AuxInt)
				if v_1_1.Op != OpTrunc32to8 {
					continue
				}
				v_1_1_0 := v_1_1.Args[0]
				if v_1_1_0.Op != OpRsh32Ux64 {
					continue
				}
				_ = v_1_1_0.Args[1]
				mul := v_1_1_0.Args[0]
				if mul.Op != OpMul32 {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst32 {
						continue
					}
					m := auxIntToInt32(mul_0.AuxInt)
					if mul_1.Op != OpZeroExt8to32 || x != mul_1.Args[0] {
						continue
					}
					v_1_1_0_1 := v_1_1_0.Args[1]
					if v_1_1_0_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_0_1.AuxInt)
					if !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(1<<8+umagic8(c).m) && s == 8+umagic8(c).s && x.Op != OpConst8 && udivisibleOK8(c)) {
						continue
					}
					v.reset(OpLeq8U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft8, typ.UInt8)
					v1 := b.NewValue0(v.Pos, OpMul8, typ.UInt8)
					v2 := b.NewValue0(v.Pos, OpConst8, typ.UInt8)
					v2.AuxInt = int8ToAuxInt(int8(udivisible8(c).m))
					v1.AddArg2(v2, x)
					v3 := b.NewValue0(v.Pos, OpConst8, typ.UInt8)
					v3.AuxInt = int8ToAuxInt(int8(8 - udivisible8(c).k))
					v0.AddArg2(v1, v3)
					v4 := b.NewValue0(v.Pos, OpConst8, typ.UInt8)
					v4.AuxInt = int8ToAuxInt(int8(udivisible8(c).max))
					v.AddArg2(v0, v4)
					return true
				}
			}
		}
		break
	}
	// match: (Eq8 x (Mul8 (Const8 [c]) (Sub8 (Rsh32x64 mul:(Mul32 (Const32 [m]) (SignExt8to32 x)) (Const64 [s])) (Rsh32x64 (SignExt8to32 x) (Const64 [31]))) ) )
	// cond: v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(smagic8(c).m) && s == 8+smagic8(c).s && x.Op != OpConst8 && sdivisibleOK8(c)
	// result: (Leq8U (RotateLeft8 <typ.UInt8> (Add8 <typ.UInt8> (Mul8 <typ.UInt8> (Const8 <typ.UInt8> [int8(sdivisible8(c).m)]) x) (Const8 <typ.UInt8> [int8(sdivisible8(c).a)]) ) (Const8 <typ.UInt8> [int8(8-sdivisible8(c).k)]) ) (Const8 <typ.UInt8> [int8(sdivisible8(c).max)]) )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul8 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst8 {
					continue
				}
				c := auxIntToInt8(v_1_0.AuxInt)
				if v_1_1.Op != OpSub8 {
					continue
				}
				_ = v_1_1.Args[1]
				v_1_1_0 := v_1_1.Args[0]
				if v_1_1_0.Op != OpRsh32x64 {
					continue
				}
				_ = v_1_1_0.Args[1]
				mul := v_1_1_0.Args[0]
				if mul.Op != OpMul32 {
					continue
				}
				_ = mul.Args[1]
				mul_0 := mul.Args[0]
				mul_1 := mul.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, mul_0, mul_1 = _i2+1, mul_1, mul_0 {
					if mul_0.Op != OpConst32 {
						continue
					}
					m := auxIntToInt32(mul_0.AuxInt)
					if mul_1.Op != OpSignExt8to32 || x != mul_1.Args[0] {
						continue
					}
					v_1_1_0_1 := v_1_1_0.Args[1]
					if v_1_1_0_1.Op != OpConst64 {
						continue
					}
					s := auxIntToInt64(v_1_1_0_1.AuxInt)
					v_1_1_1 := v_1_1.Args[1]
					if v_1_1_1.Op != OpRsh32x64 {
						continue
					}
					_ = v_1_1_1.Args[1]
					v_1_1_1_0 := v_1_1_1.Args[0]
					if v_1_1_1_0.Op != OpSignExt8to32 || x != v_1_1_1_0.Args[0] {
						continue
					}
					v_1_1_1_1 := v_1_1_1.Args[1]
					if v_1_1_1_1.Op != OpConst64 || auxIntToInt64(v_1_1_1_1.AuxInt) != 31 || !(v.Block.Func.pass.name != "opt" && mul.Uses == 1 && m == int32(smagic8(c).m) && s == 8+smagic8(c).s && x.Op != OpConst8 && sdivisibleOK8(c)) {
						continue
					}
					v.reset(OpLeq8U)
					v0 := b.NewValue0(v.Pos, OpRotateLeft8, typ.UInt8)
					v1 := b.NewValue0(v.Pos, OpAdd8, typ.UInt8)
					v2 := b.NewValue0(v.Pos, OpMul8, typ.UInt8)
					v3 := b.NewValue0(v.Pos, OpConst8, typ.UInt8)
					v3.AuxInt = int8ToAuxInt(int8(sdivisible8(c).m))
					v2.AddArg2(v3, x)
					v4 := b.NewValue0(v.Pos, OpConst8, typ.UInt8)
					v4.AuxInt = int8ToAuxInt(int8(sdivisible8(c).a))
					v1.AddArg2(v2, v4)
					v5 := b.NewValue0(v.Pos, OpConst8, typ.UInt8)
					v5.AuxInt = int8ToAuxInt(int8(8 - sdivisible8(c).k))
					v0.AddArg2(v1, v5)
					v6 := b.NewValue0(v.Pos, OpConst8, typ.UInt8)
					v6.AuxInt = int8ToAuxInt(int8(sdivisible8(c).max))
					v.AddArg2(v0, v6)
					return true
				}
			}
		}
		break
	}
	// match: (Eq8 n (Lsh8x64 (Rsh8x64 (Add8 <t> n (Rsh8Ux64 <t> (Rsh8x64 <t> n (Const64 <typ.UInt64> [ 7])) (Const64 <typ.UInt64> [kbar]))) (Const64 <typ.UInt64> [k])) (Const64 <typ.UInt64> [k])) )
	// cond: k > 0 && k < 7 && kbar == 8 - k
	// result: (Eq8 (And8 <t> n (Const8 <t> [1<<uint(k)-1])) (Const8 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			n := v_0
			if v_1.Op != OpLsh8x64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRsh8x64 {
				continue
			}
			_ = v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			if v_1_0_0.Op != OpAdd8 {
				continue
			}
			t := v_1_0_0.Type
			_ = v_1_0_0.Args[1]
			v_1_0_0_0 := v_1_0_0.Args[0]
			v_1_0_0_1 := v_1_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0_0_0, v_1_0_0_1 = _i1+1, v_1_0_0_1, v_1_0_0_0 {
				if n != v_1_0_0_0 || v_1_0_0_1.Op != OpRsh8Ux64 || v_1_0_0_1.Type != t {
					continue
				}
				_ = v_1_0_0_1.Args[1]
				v_1_0_0_1_0 := v_1_0_0_1.Args[0]
				if v_1_0_0_1_0.Op != OpRsh8x64 || v_1_0_0_1_0.Type != t {
					continue
				}
				_ = v_1_0_0_1_0.Args[1]
				if n != v_1_0_0_1_0.Args[0] {
					continue
				}
				v_1_0_0_1_0_1 := v_1_0_0_1_0.Args[1]
				if v_1_0_0_1_0_1.Op != OpConst64 || v_1_0_0_1_0_1.Type != typ.UInt64 || auxIntToInt64(v_1_0_0_1_0_1.AuxInt) != 7 {
					continue
				}
				v_1_0_0_1_1 := v_1_0_0_1.Args[1]
				if v_1_0_0_1_1.Op != OpConst64 || v_1_0_0_1_1.Type != typ.UInt64 {
					continue
				}
				kbar := auxIntToInt64(v_1_0_0_1_1.AuxInt)
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst64 || v_1_0_1.Type != typ.UInt64 {
					continue
				}
				k := auxIntToInt64(v_1_0_1.AuxInt)
				v_1_1 := v_1.Args[1]
				if v_1_1.Op != OpConst64 || v_1_1.Type != typ.UInt64 || auxIntToInt64(v_1_1.AuxInt) != k || !(k > 0 && k < 7 && kbar == 8-k) {
					continue
				}
				v.reset(OpEq8)
				v0 := b.NewValue0(v.Pos, OpAnd8, t)
				v1 := b.NewValue0(v.Pos, OpConst8, t)
				v1.AuxInt = int8ToAuxInt(1<<uint(k) - 1)
				v0.AddArg2(n, v1)
				v2 := b.NewValue0(v.Pos, OpConst8, t)
				v2.AuxInt = int8ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq8 s:(Sub8 x y) (Const8 [0]))
	// cond: s.Uses == 1
	// result: (Eq8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s := v_0
			if s.Op != OpSub8 {
				continue
			}
			y := s.Args[1]
			x := s.Args[0]
			if v_1.Op != OpConst8 || auxIntToInt8(v_1.AuxInt) != 0 || !(s.Uses == 1) {
				continue
			}
			v.reset(OpEq8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Eq8 (And8 <t> x (Const8 <t> [y])) (Const8 <t> [y]))
	// cond: oneBit8(y)
	// result: (Neq8 (And8 <t> x (Const8 <t> [y])) (Const8 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAnd8 {
				continue
			}
			t := v_0.Type
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				if v_0_1.Op != OpConst8 || v_0_1.Type != t {
					continue
				}
				y := auxIntToInt8(v_0_1.AuxInt)
				if v_1.Op != OpConst8 || v_1.Type != t || auxIntToInt8(v_1.AuxInt) != y || !(oneBit8(y)) {
					continue
				}
				v.reset(OpNeq8)
				v0 := b.NewValue0(v.Pos, OpAnd8, t)
				v1 := b.NewValue0(v.Pos, OpConst8, t)
				v1.AuxInt = int8ToAuxInt(y)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst8, t)
				v2.AuxInt = int8ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpEqB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (EqB (ConstBool [c]) (ConstBool [d]))
	// result: (ConstBool [c == d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConstBool {
				continue
			}
			c := auxIntToBool(v_0.AuxInt)
			if v_1.Op != OpConstBool {
				continue
			}
			d := auxIntToBool(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c == d)
			return true
		}
		break
	}
	// match: (EqB (ConstBool [false]) x)
	// result: (Not x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConstBool || auxIntToBool(v_0.AuxInt) != false {
				continue
			}
			x := v_1
			v.reset(OpNot)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (EqB (ConstBool [true]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConstBool || auxIntToBool(v_0.AuxInt) != true {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpEqInter(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqInter x y)
	// result: (EqPtr (ITab x) (ITab y))
	for {
		x := v_0
		y := v_1
		v.reset(OpEqPtr)
		v0 := b.NewValue0(v.Pos, OpITab, typ.Uintptr)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpITab, typ.Uintptr)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuegeneric_OpEqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqPtr x x)
	// result: (ConstBool [true])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (EqPtr (Addr {x} _) (Addr {y} _))
	// result: (ConstBool [x == y])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAddr {
				continue
			}
			x := auxToSym(v_0.Aux)
			if v_1.Op != OpAddr {
				continue
			}
			y := auxToSym(v_1.Aux)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(x == y)
			return true
		}
		break
	}
	// match: (EqPtr (Addr {x} _) (OffPtr [o] (Addr {y} _)))
	// result: (ConstBool [x == y && o == 0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAddr {
				continue
			}
			x := auxToSym(v_0.Aux)
			if v_1.Op != OpOffPtr {
				continue
			}
			o := auxIntToInt64(v_1.AuxInt)
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpAddr {
				continue
			}
			y := auxToSym(v_1_0.Aux)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(x == y && o == 0)
			return true
		}
		break
	}
	// match: (EqPtr (OffPtr [o1] (Addr {x} _)) (OffPtr [o2] (Addr {y} _)))
	// result: (ConstBool [x == y && o1 == o2])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpOffPtr {
				continue
			}
			o1 := auxIntToInt64(v_0.AuxInt)
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAddr {
				continue
			}
			x := auxToSym(v_0_0.Aux)
			if v_1.Op != OpOffPtr {
				continue
			}
			o2 := auxIntToInt64(v_1.AuxInt)
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpAddr {
				continue
			}
			y := auxToSym(v_1_0.Aux)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(x == y && o1 == o2)
			return true
		}
		break
	}
	// match: (EqPtr (LocalAddr {x} _ _) (LocalAddr {y} _ _))
	// result: (ConstBool [x == y])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLocalAddr {
				continue
			}
			x := auxToSym(v_0.Aux)
			if v_1.Op != OpLocalAddr {
				continue
			}
			y := auxToSym(v_1.Aux)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(x == y)
			return true
		}
		break
	}
	// match: (EqPtr (LocalAddr {x} _ _) (OffPtr [o] (LocalAddr {y} _ _)))
	// result: (ConstBool [x == y && o == 0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLocalAddr {
				continue
			}
			x := auxToSym(v_0.Aux)
			if v_1.Op != OpOffPtr {
				continue
			}
			o := auxIntToInt64(v_1.AuxInt)
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpLocalAddr {
				continue
			}
			y := auxToSym(v_1_0.Aux)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(x == y && o == 0)
			return true
		}
		break
	}
	// match: (EqPtr (OffPtr [o1] (LocalAddr {x} _ _)) (OffPtr [o2] (LocalAddr {y} _ _)))
	// result: (ConstBool [x == y && o1 == o2])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpOffPtr {
				continue
			}
			o1 := auxIntToInt64(v_0.AuxInt)
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpLocalAddr {
				continue
			}
			x := auxToSym(v_0_0.Aux)
			if v_1.Op != OpOffPtr {
				continue
			}
			o2 := auxIntToInt64(v_1.AuxInt)
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpLocalAddr {
				continue
			}
			y := auxToSym(v_1_0.Aux)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(x == y && o1 == o2)
			return true
		}
		break
	}
	// match: (EqPtr (OffPtr [o1] p1) p2)
	// cond: isSamePtr(p1, p2)
	// result: (ConstBool [o1 == 0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpOffPtr {
				continue
			}
			o1 := auxIntToInt64(v_0.AuxInt)
			p1 := v_0.Args[0]
			p2 := v_1
			if !(isSamePtr(p1, p2)) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(o1 == 0)
			return true
		}
		break
	}
	// match: (EqPtr (OffPtr [o1] p1) (OffPtr [o2] p2))
	// cond: isSamePtr(p1, p2)
	// result: (ConstBool [o1 == o2])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpOffPtr {
				continue
			}
			o1 := auxIntToInt64(v_0.AuxInt)
			p1 := v_0.Args[0]
			if v_1.Op != OpOffPtr {
				continue
			}
			o2 := auxIntToInt64(v_1.AuxInt)
			p2 := v_1.Args[0]
			if !(isSamePtr(p1, p2)) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(o1 == o2)
			return true
		}
		break
	}
	// match: (EqPtr (Const32 [c]) (Const32 [d]))
	// result: (ConstBool [c == d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c == d)
			return true
		}
		break
	}
	// match: (EqPtr (Const64 [c]) (Const64 [d]))
	// result: (ConstBool [c == d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c == d)
			return true
		}
		break
	}
	// match: (EqPtr (Convert (Addr {x} _) _) (Addr {y} _))
	// result: (ConstBool [x==y])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConvert {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAddr {
				continue
			}
			x := auxToSym(v_0_0.Aux)
			if v_1.Op != OpAddr {
				continue
			}
			y := auxToSym(v_1.Aux)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(x == y)
			return true
		}
		break
	}
	// match: (EqPtr (LocalAddr _ _) (Addr _))
	// result: (ConstBool [false])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLocalAddr || v_1.Op != OpAddr {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(false)
			return true
		}
		break
	}
	// match: (EqPtr (OffPtr (LocalAddr _ _)) (Addr _))
	// result: (ConstBool [false])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpOffPtr {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpLocalAddr || v_1.Op != OpAddr {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(false)
			return true
		}
		break
	}
	// match: (EqPtr (LocalAddr _ _) (OffPtr (Addr _)))
	// result: (ConstBool [false])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLocalAddr || v_1.Op != OpOffPtr {
				continue
			}
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpAddr {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(false)
			return true
		}
		break
	}
	// match: (EqPtr (OffPtr (LocalAddr _ _)) (OffPtr (Addr _)))
	// result: (ConstBool [false])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpOffPtr {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpLocalAddr || v_1.Op != OpOffPtr {
				continue
			}
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpAddr {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(false)
			return true
		}
		break
	}
	// match: (EqPtr (AddPtr p1 o1) p2)
	// cond: isSamePtr(p1, p2)
	// result: (Not (IsNonNil o1))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAddPtr {
				continue
			}
			o1 := v_0.Args[1]
			p1 := v_0.Args[0]
			p2 := v_1
			if !(isSamePtr(p1, p2)) {
				continue
			}
			v.reset(OpNot)
			v0 := b.NewValue0(v.Pos, OpIsNonNil, typ.Bool)
			v0.AddArg(o1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (EqPtr (Const32 [0]) p)
	// result: (Not (IsNonNil p))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
				continue
			}
			p := v_1
			v.reset(OpNot)
			v0 := b.NewValue0(v.Pos, OpIsNonNil, typ.Bool)
			v0.AddArg(p)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (EqPtr (Const64 [0]) p)
	// result: (Not (IsNonNil p))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
				continue
			}
			p := v_1
			v.reset(OpNot)
			v0 := b.NewValue0(v.Pos, OpIsNonNil, typ.Bool)
			v0.AddArg(p)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (EqPtr (ConstNil) p)
	// result: (Not (IsNonNil p))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConstNil {
				continue
			}
			p := v_1
			v.reset(OpNot)
			v0 := b.NewValue0(v.Pos, OpIsNonNil, typ.Bool)
			v0.AddArg(p)
			v.AddArg(v0)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpEqSlice(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqSlice x y)
	// result: (EqPtr (SlicePtr x) (SlicePtr y))
	for {
		x := v_0
		y := v_1
		v.reset(OpEqPtr)
		v0 := b.NewValue0(v.Pos, OpSlicePtr, typ.BytePtr)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSlicePtr, typ.BytePtr)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuegeneric_OpFloor(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Floor (Const64F [c]))
	// result: (Const64F [math.Floor(c)])
	for {
		if v_0.Op != OpConst64F {
			break
		}
		c := auxIntToFloat64(v_0.AuxInt)
		v.reset(OpConst64F)
		v.AuxInt = float64ToAuxInt(math.Floor(c))
		return true
	}
	return false
}
func rewriteValuegeneric_OpIMake(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IMake _typ (StructMake val))
	// result: (IMake _typ val)
	for {
		_typ := v_0
		if v_1.Op != OpStructMake || len(v_1.Args) != 1 {
			break
		}
		val := v_1.Args[0]
		v.reset(OpIMake)
		v.AddArg2(_typ, val)
		return true
	}
	// match: (IMake _typ (ArrayMake1 val))
	// result: (IMake _typ val)
	for {
		_typ := v_0
		if v_1.Op != OpArrayMake1 {
			break
		}
		val := v_1.Args[0]
		v.reset(OpIMake)
		v.AddArg2(_typ, val)
		return true
	}
	return false
}
func rewriteValuegeneric_OpInterLECall(v *Value) bool {
	// match: (InterLECall [argsize] {auxCall} (Addr {fn} (SB)) ___)
	// result: devirtLECall(v, fn.(*obj.LSym))
	for {
		if len(v.Args) < 1 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAddr {
			break
		}
		fn := auxToSym(v_0.Aux)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpSB {
			break
		}
		v.copyOf(devirtLECall(v, fn.(*obj.LSym)))
		return true
	}
	return false
}
func rewriteValuegeneric_OpIsInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IsInBounds (ZeroExt8to32 _) (Const32 [c]))
	// cond: (1 << 8) <= c
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpZeroExt8to32 || v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !((1 << 8) <= c) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsInBounds (ZeroExt8to64 _) (Const64 [c]))
	// cond: (1 << 8) <= c
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpZeroExt8to64 || v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !((1 << 8) <= c) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsInBounds (ZeroExt16to32 _) (Const32 [c]))
	// cond: (1 << 16) <= c
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpZeroExt16to32 || v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !((1 << 16) <= c) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsInBounds (ZeroExt16to64 _) (Const64 [c]))
	// cond: (1 << 16) <= c
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpZeroExt16to64 || v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !((1 << 16) <= c) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsInBounds x x)
	// result: (ConstBool [false])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (IsInBounds (And8 (Const8 [c]) _) (Const8 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpAnd8 {
			break
		}
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0.AuxInt)
			if v_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1.AuxInt)
			if !(0 <= c && c < d) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (IsInBounds (ZeroExt8to16 (And8 (Const8 [c]) _)) (Const16 [d]))
	// cond: 0 <= c && int16(c) < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpZeroExt8to16 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd8 {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0_0, v_0_0_1 = _i0+1, v_0_0_1, v_0_0_0 {
			if v_0_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0_0.AuxInt)
			if v_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1.AuxInt)
			if !(0 <= c && int16(c) < d) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (IsInBounds (ZeroExt8to32 (And8 (Const8 [c]) _)) (Const32 [d]))
	// cond: 0 <= c && int32(c) < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpZeroExt8to32 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd8 {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0_0, v_0_0_1 = _i0+1, v_0_0_1, v_0_0_0 {
			if v_0_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0_0.AuxInt)
			if v_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			if !(0 <= c && int32(c) < d) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (IsInBounds (ZeroExt8to64 (And8 (Const8 [c]) _)) (Const64 [d]))
	// cond: 0 <= c && int64(c) < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpZeroExt8to64 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd8 {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0_0, v_0_0_1 = _i0+1, v_0_0_1, v_0_0_0 {
			if v_0_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0_0.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			if !(0 <= c && int64(c) < d) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (IsInBounds (And16 (Const16 [c]) _) (Const16 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpAnd16 {
			break
		}
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0.AuxInt)
			if v_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1.AuxInt)
			if !(0 <= c && c < d) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (IsInBounds (ZeroExt16to32 (And16 (Const16 [c]) _)) (Const32 [d]))
	// cond: 0 <= c && int32(c) < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpZeroExt16to32 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd16 {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0_0, v_0_0_1 = _i0+1, v_0_0_1, v_0_0_0 {
			if v_0_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0_0.AuxInt)
			if v_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			if !(0 <= c && int32(c) < d) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (IsInBounds (ZeroExt16to64 (And16 (Const16 [c]) _)) (Const64 [d]))
	// cond: 0 <= c && int64(c) < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpZeroExt16to64 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd16 {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0_0, v_0_0_1 = _i0+1, v_0_0_1, v_0_0_0 {
			if v_0_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0_0.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			if !(0 <= c && int64(c) < d) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (IsInBounds (And32 (Const32 [c]) _) (Const32 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpAnd32 {
			break
		}
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			if !(0 <= c && c < d) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (IsInBounds (ZeroExt32to64 (And32 (Const32 [c]) _)) (Const64 [d]))
	// cond: 0 <= c && int64(c) < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpZeroExt32to64 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd32 {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0_0, v_0_0_1 = _i0+1, v_0_0_1, v_0_0_0 {
			if v_0_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0_0.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			if !(0 <= c && int64(c) < d) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (IsInBounds (And64 (Const64 [c]) _) (Const64 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpAnd64 {
			break
		}
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			if !(0 <= c && c < d) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (IsInBounds (Const32 [c]) (Const32 [d]))
	// result: (ConstBool [0 <= c && c < d])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if v_1.Op != OpConst32 {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(0 <= c && c < d)
		return true
	}
	// match: (IsInBounds (Const64 [c]) (Const64 [d]))
	// result: (ConstBool [0 <= c && c < d])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(0 <= c && c < d)
		return true
	}
	// match: (IsInBounds (Mod32u _ y) y)
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpMod32u {
			break
		}
		y := v_0.Args[1]
		if y != v_1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsInBounds (Mod64u _ y) y)
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpMod64u {
			break
		}
		y := v_0.Args[1]
		if y != v_1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsInBounds (ZeroExt8to64 (Rsh8Ux64 _ (Const64 [c]))) (Const64 [d]))
	// cond: 0 < c && c < 8 && 1<<uint( 8-c)-1 < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpZeroExt8to64 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRsh8Ux64 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(0 < c && c < 8 && 1<<uint(8-c)-1 < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsInBounds (ZeroExt8to32 (Rsh8Ux64 _ (Const64 [c]))) (Const32 [d]))
	// cond: 0 < c && c < 8 && 1<<uint( 8-c)-1 < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpZeroExt8to32 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRsh8Ux64 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_0_1.AuxInt)
		if v_1.Op != OpConst32 {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		if !(0 < c && c < 8 && 1<<uint(8-c)-1 < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsInBounds (ZeroExt8to16 (Rsh8Ux64 _ (Const64 [c]))) (Const16 [d]))
	// cond: 0 < c && c < 8 && 1<<uint( 8-c)-1 < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpZeroExt8to16 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRsh8Ux64 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_0_1.AuxInt)
		if v_1.Op != OpConst16 {
			break
		}
		d := auxIntToInt16(v_1.AuxInt)
		if !(0 < c && c < 8 && 1<<uint(8-c)-1 < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsInBounds (Rsh8Ux64 _ (Const64 [c])) (Const64 [d]))
	// cond: 0 < c && c < 8 && 1<<uint( 8-c)-1 < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpRsh8Ux64 {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(0 < c && c < 8 && 1<<uint(8-c)-1 < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsInBounds (ZeroExt16to64 (Rsh16Ux64 _ (Const64 [c]))) (Const64 [d]))
	// cond: 0 < c && c < 16 && 1<<uint(16-c)-1 < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpZeroExt16to64 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRsh16Ux64 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(0 < c && c < 16 && 1<<uint(16-c)-1 < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsInBounds (ZeroExt16to32 (Rsh16Ux64 _ (Const64 [c]))) (Const64 [d]))
	// cond: 0 < c && c < 16 && 1<<uint(16-c)-1 < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpZeroExt16to32 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRsh16Ux64 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(0 < c && c < 16 && 1<<uint(16-c)-1 < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsInBounds (Rsh16Ux64 _ (Const64 [c])) (Const64 [d]))
	// cond: 0 < c && c < 16 && 1<<uint(16-c)-1 < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpRsh16Ux64 {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(0 < c && c < 16 && 1<<uint(16-c)-1 < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsInBounds (ZeroExt32to64 (Rsh32Ux64 _ (Const64 [c]))) (Const64 [d]))
	// cond: 0 < c && c < 32 && 1<<uint(32-c)-1 < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpZeroExt32to64 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRsh32Ux64 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(0 < c && c < 32 && 1<<uint(32-c)-1 < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsInBounds (Rsh32Ux64 _ (Const64 [c])) (Const64 [d]))
	// cond: 0 < c && c < 32 && 1<<uint(32-c)-1 < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpRsh32Ux64 {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(0 < c && c < 32 && 1<<uint(32-c)-1 < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsInBounds (Rsh64Ux64 _ (Const64 [c])) (Const64 [d]))
	// cond: 0 < c && c < 64 && 1<<uint(64-c)-1 < d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpRsh64Ux64 {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(0 < c && c < 64 && 1<<uint(64-c)-1 < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	return false
}
func rewriteValuegeneric_OpIsNonNil(v *Value) bool {
	v_0 := v.Args[0]
	// match: (IsNonNil (ConstNil))
	// result: (ConstBool [false])
	for {
		if v_0.Op != OpConstNil {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (IsNonNil (Const32 [c]))
	// result: (ConstBool [c != 0])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(c != 0)
		return true
	}
	// match: (IsNonNil (Const64 [c]))
	// result: (ConstBool [c != 0])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(c != 0)
		return true
	}
	// match: (IsNonNil (Addr _) )
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpAddr {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsNonNil (Convert (Addr _) _))
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConvert {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAddr {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsNonNil (LocalAddr _ _))
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpLocalAddr {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	return false
}
func rewriteValuegeneric_OpIsSliceInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IsSliceInBounds x x)
	// result: (ConstBool [true])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsSliceInBounds (And32 (Const32 [c]) _) (Const32 [d]))
	// cond: 0 <= c && c <= d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpAnd32 {
			break
		}
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			if !(0 <= c && c <= d) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (IsSliceInBounds (And64 (Const64 [c]) _) (Const64 [d]))
	// cond: 0 <= c && c <= d
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpAnd64 {
			break
		}
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			if !(0 <= c && c <= d) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (IsSliceInBounds (Const32 [0]) _)
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsSliceInBounds (Const64 [0]) _)
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (IsSliceInBounds (Const32 [c]) (Const32 [d]))
	// result: (ConstBool [0 <= c && c <= d])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if v_1.Op != OpConst32 {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(0 <= c && c <= d)
		return true
	}
	// match: (IsSliceInBounds (Const64 [c]) (Const64 [d]))
	// result: (ConstBool [0 <= c && c <= d])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(0 <= c && c <= d)
		return true
	}
	// match: (IsSliceInBounds (SliceLen x) (SliceCap x))
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpSliceLen {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpSliceCap || x != v_1.Args[0] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq16 (Const16 [c]) (Const16 [d]))
	// result: (ConstBool [c <= d])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if v_1.Op != OpConst16 {
			break
		}
		d := auxIntToInt16(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(c <= d)
		return true
	}
	// match: (Leq16 (Const16 [0]) (And16 _ (Const16 [c])))
	// cond: c >= 0
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 || v_1.Op != OpAnd16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if v_1_1.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_1_1.AuxInt)
			if !(c >= 0) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (Leq16 (Const16 [0]) (Rsh16Ux64 _ (Const64 [c])))
	// cond: c > 0
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 || v_1.Op != OpRsh16Ux64 {
			break
		}
		_ = v_1.Args[1]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1_1.AuxInt)
		if !(c > 0) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq16 x (Const16 <t> [-1]))
	// result: (Less16 x (Const16 <t> [0]))
	for {
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		t := v_1.Type
		if auxIntToInt16(v_1.AuxInt) != -1 {
			break
		}
		v.reset(OpLess16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Leq16 (Const16 <t> [1]) x)
	// result: (Less16 (Const16 <t> [0]) x)
	for {
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		if auxIntToInt16(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		v.reset(OpLess16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Leq16 (Const16 [math.MinInt16]) _)
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != math.MinInt16 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq16 _ (Const16 [math.MaxInt16]))
	// result: (ConstBool [true])
	for {
		if v_1.Op != OpConst16 || auxIntToInt16(v_1.AuxInt) != math.MaxInt16 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq16 x c:(Const16 [math.MinInt16]))
	// result: (Eq16 x c)
	for {
		x := v_0
		c := v_1
		if c.Op != OpConst16 || auxIntToInt16(c.AuxInt) != math.MinInt16 {
			break
		}
		v.reset(OpEq16)
		v.AddArg2(x, c)
		return true
	}
	// match: (Leq16 c:(Const16 [math.MaxInt16]) x)
	// result: (Eq16 x c)
	for {
		c := v_0
		if c.Op != OpConst16 || auxIntToInt16(c.AuxInt) != math.MaxInt16 {
			break
		}
		x := v_1
		v.reset(OpEq16)
		v.AddArg2(x, c)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq16U (Const16 [c]) (Const16 [d]))
	// result: (ConstBool [uint16(c) <= uint16(d)])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if v_1.Op != OpConst16 {
			break
		}
		d := auxIntToInt16(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(uint16(c) <= uint16(d))
		return true
	}
	// match: (Leq16U (Const16 <t> [1]) x)
	// result: (Neq16 (Const16 <t> [0]) x)
	for {
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		if auxIntToInt16(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		v.reset(OpNeq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Leq16U (Const16 [0]) _)
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq16U _ (Const16 [-1]))
	// result: (ConstBool [true])
	for {
		if v_1.Op != OpConst16 || auxIntToInt16(v_1.AuxInt) != -1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq16U x c:(Const16 [0]))
	// result: (Eq16 x c)
	for {
		x := v_0
		c := v_1
		if c.Op != OpConst16 || auxIntToInt16(c.AuxInt) != 0 {
			break
		}
		v.reset(OpEq16)
		v.AddArg2(x, c)
		return true
	}
	// match: (Leq16U c:(Const16 [-1]) x)
	// result: (Eq16 x c)
	for {
		c := v_0
		if c.Op != OpConst16 || auxIntToInt16(c.AuxInt) != -1 {
			break
		}
		x := v_1
		v.reset(OpEq16)
		v.AddArg2(x, c)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32 (Const32 [c]) (Const32 [d]))
	// result: (ConstBool [c <= d])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if v_1.Op != OpConst32 {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(c <= d)
		return true
	}
	// match: (Leq32 (Const32 [0]) (And32 _ (Const32 [c])))
	// cond: c >= 0
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 || v_1.Op != OpAnd32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if v_1_1.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_1_1.AuxInt)
			if !(c >= 0) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (Leq32 (Const32 [0]) (Rsh32Ux64 _ (Const64 [c])))
	// cond: c > 0
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 || v_1.Op != OpRsh32Ux64 {
			break
		}
		_ = v_1.Args[1]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1_1.AuxInt)
		if !(c > 0) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq32 x (Const32 <t> [-1]))
	// result: (Less32 x (Const32 <t> [0]))
	for {
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		t := v_1.Type
		if auxIntToInt32(v_1.AuxInt) != -1 {
			break
		}
		v.reset(OpLess32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Leq32 (Const32 <t> [1]) x)
	// result: (Less32 (Const32 <t> [0]) x)
	for {
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		if auxIntToInt32(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		v.reset(OpLess32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Leq32 (Const32 [math.MinInt32]) _)
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != math.MinInt32 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq32 _ (Const32 [math.MaxInt32]))
	// result: (ConstBool [true])
	for {
		if v_1.Op != OpConst32 || auxIntToInt32(v_1.AuxInt) != math.MaxInt32 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq32 x c:(Const32 [math.MinInt32]))
	// result: (Eq32 x c)
	for {
		x := v_0
		c := v_1
		if c.Op != OpConst32 || auxIntToInt32(c.AuxInt) != math.MinInt32 {
			break
		}
		v.reset(OpEq32)
		v.AddArg2(x, c)
		return true
	}
	// match: (Leq32 c:(Const32 [math.MaxInt32]) x)
	// result: (Eq32 x c)
	for {
		c := v_0
		if c.Op != OpConst32 || auxIntToInt32(c.AuxInt) != math.MaxInt32 {
			break
		}
		x := v_1
		v.reset(OpEq32)
		v.AddArg2(x, c)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Leq32F (Const32F [c]) (Const32F [d]))
	// result: (ConstBool [c <= d])
	for {
		if v_0.Op != OpConst32F {
			break
		}
		c := auxIntToFloat32(v_0.AuxInt)
		if v_1.Op != OpConst32F {
			break
		}
		d := auxIntToFloat32(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(c <= d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32U (Const32 [c]) (Const32 [d]))
	// result: (ConstBool [uint32(c) <= uint32(d)])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if v_1.Op != OpConst32 {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(uint32(c) <= uint32(d))
		return true
	}
	// match: (Leq32U (Const32 <t> [1]) x)
	// result: (Neq32 (Const32 <t> [0]) x)
	for {
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		if auxIntToInt32(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		v.reset(OpNeq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Leq32U (Const32 [0]) _)
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq32U _ (Const32 [-1]))
	// result: (ConstBool [true])
	for {
		if v_1.Op != OpConst32 || auxIntToInt32(v_1.AuxInt) != -1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq32U x c:(Const32 [0]))
	// result: (Eq32 x c)
	for {
		x := v_0
		c := v_1
		if c.Op != OpConst32 || auxIntToInt32(c.AuxInt) != 0 {
			break
		}
		v.reset(OpEq32)
		v.AddArg2(x, c)
		return true
	}
	// match: (Leq32U c:(Const32 [-1]) x)
	// result: (Eq32 x c)
	for {
		c := v_0
		if c.Op != OpConst32 || auxIntToInt32(c.AuxInt) != -1 {
			break
		}
		x := v_1
		v.reset(OpEq32)
		v.AddArg2(x, c)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64 (Const64 [c]) (Const64 [d]))
	// result: (ConstBool [c <= d])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(c <= d)
		return true
	}
	// match: (Leq64 (Const64 [0]) (And64 _ (Const64 [c])))
	// cond: c >= 0
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 || v_1.Op != OpAnd64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if v_1_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_1.AuxInt)
			if !(c >= 0) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (Leq64 (Const64 [0]) (Rsh64Ux64 _ (Const64 [c])))
	// cond: c > 0
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 || v_1.Op != OpRsh64Ux64 {
			break
		}
		_ = v_1.Args[1]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1_1.AuxInt)
		if !(c > 0) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq64 x (Const64 <t> [-1]))
	// result: (Less64 x (Const64 <t> [0]))
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		if auxIntToInt64(v_1.AuxInt) != -1 {
			break
		}
		v.reset(OpLess64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Leq64 (Const64 <t> [1]) x)
	// result: (Less64 (Const64 <t> [0]) x)
	for {
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		if auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		v.reset(OpLess64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Leq64 (Const64 [math.MinInt64]) _)
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != math.MinInt64 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq64 _ (Const64 [math.MaxInt64]))
	// result: (ConstBool [true])
	for {
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != math.MaxInt64 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq64 x c:(Const64 [math.MinInt64]))
	// result: (Eq64 x c)
	for {
		x := v_0
		c := v_1
		if c.Op != OpConst64 || auxIntToInt64(c.AuxInt) != math.MinInt64 {
			break
		}
		v.reset(OpEq64)
		v.AddArg2(x, c)
		return true
	}
	// match: (Leq64 c:(Const64 [math.MaxInt64]) x)
	// result: (Eq64 x c)
	for {
		c := v_0
		if c.Op != OpConst64 || auxIntToInt64(c.AuxInt) != math.MaxInt64 {
			break
		}
		x := v_1
		v.reset(OpEq64)
		v.AddArg2(x, c)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Leq64F (Const64F [c]) (Const64F [d]))
	// result: (ConstBool [c <= d])
	for {
		if v_0.Op != OpConst64F {
			break
		}
		c := auxIntToFloat64(v_0.AuxInt)
		if v_1.Op != OpConst64F {
			break
		}
		d := auxIntToFloat64(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(c <= d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64U (Const64 [c]) (Const64 [d]))
	// result: (ConstBool [uint64(c) <= uint64(d)])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(uint64(c) <= uint64(d))
		return true
	}
	// match: (Leq64U (Const64 <t> [1]) x)
	// result: (Neq64 (Const64 <t> [0]) x)
	for {
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		if auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		v.reset(OpNeq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Leq64U (Const64 [0]) _)
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq64U _ (Const64 [-1]))
	// result: (ConstBool [true])
	for {
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != -1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq64U x c:(Const64 [0]))
	// result: (Eq64 x c)
	for {
		x := v_0
		c := v_1
		if c.Op != OpConst64 || auxIntToInt64(c.AuxInt) != 0 {
			break
		}
		v.reset(OpEq64)
		v.AddArg2(x, c)
		return true
	}
	// match: (Leq64U c:(Const64 [-1]) x)
	// result: (Eq64 x c)
	for {
		c := v_0
		if c.Op != OpConst64 || auxIntToInt64(c.AuxInt) != -1 {
			break
		}
		x := v_1
		v.reset(OpEq64)
		v.AddArg2(x, c)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq8 (Const8 [c]) (Const8 [d]))
	// result: (ConstBool [c <= d])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if v_1.Op != OpConst8 {
			break
		}
		d := auxIntToInt8(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(c <= d)
		return true
	}
	// match: (Leq8 (Const8 [0]) (And8 _ (Const8 [c])))
	// cond: c >= 0
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 || v_1.Op != OpAnd8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if v_1_1.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_1_1.AuxInt)
			if !(c >= 0) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (Leq8 (Const8 [0]) (Rsh8Ux64 _ (Const64 [c])))
	// cond: c > 0
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 || v_1.Op != OpRsh8Ux64 {
			break
		}
		_ = v_1.Args[1]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1_1.AuxInt)
		if !(c > 0) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq8 x (Const8 <t> [-1]))
	// result: (Less8 x (Const8 <t> [0]))
	for {
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		t := v_1.Type
		if auxIntToInt8(v_1.AuxInt) != -1 {
			break
		}
		v.reset(OpLess8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Leq8 (Const8 <t> [1]) x)
	// result: (Less8 (Const8 <t> [0]) x)
	for {
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		if auxIntToInt8(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		v.reset(OpLess8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Leq8 (Const8 [math.MinInt8 ]) _)
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != math.MinInt8 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq8 _ (Const8 [math.MaxInt8 ]))
	// result: (ConstBool [true])
	for {
		if v_1.Op != OpConst8 || auxIntToInt8(v_1.AuxInt) != math.MaxInt8 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq8 x c:(Const8 [math.MinInt8 ]))
	// result: (Eq8 x c)
	for {
		x := v_0
		c := v_1
		if c.Op != OpConst8 || auxIntToInt8(c.AuxInt) != math.MinInt8 {
			break
		}
		v.reset(OpEq8)
		v.AddArg2(x, c)
		return true
	}
	// match: (Leq8 c:(Const8 [math.MaxInt8 ]) x)
	// result: (Eq8 x c)
	for {
		c := v_0
		if c.Op != OpConst8 || auxIntToInt8(c.AuxInt) != math.MaxInt8 {
			break
		}
		x := v_1
		v.reset(OpEq8)
		v.AddArg2(x, c)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq8U (Const8 [c]) (Const8 [d]))
	// result: (ConstBool [ uint8(c) <= uint8(d)])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if v_1.Op != OpConst8 {
			break
		}
		d := auxIntToInt8(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(uint8(c) <= uint8(d))
		return true
	}
	// match: (Leq8U (Const8 <t> [1]) x)
	// result: (Neq8 (Const8 <t> [0]) x)
	for {
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		if auxIntToInt8(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		v.reset(OpNeq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Leq8U (Const8 [0]) _)
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq8U _ (Const8 [-1]))
	// result: (ConstBool [true])
	for {
		if v_1.Op != OpConst8 || auxIntToInt8(v_1.AuxInt) != -1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (Leq8U x c:(Const8 [0]))
	// result: (Eq8 x c)
	for {
		x := v_0
		c := v_1
		if c.Op != OpConst8 || auxIntToInt8(c.AuxInt) != 0 {
			break
		}
		v.reset(OpEq8)
		v.AddArg2(x, c)
		return true
	}
	// match: (Leq8U c:(Const8 [-1]) x)
	// result: (Eq8 x c)
	for {
		c := v_0
		if c.Op != OpConst8 || auxIntToInt8(c.AuxInt) != -1 {
			break
		}
		x := v_1
		v.reset(OpEq8)
		v.AddArg2(x, c)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less16 (Const16 [c]) (Const16 [d]))
	// result: (ConstBool [c < d])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if v_1.Op != OpConst16 {
			break
		}
		d := auxIntToInt16(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(c < d)
		return true
	}
	// match: (Less16 (Const16 <t> [0]) x)
	// cond: isNonNegative(x)
	// result: (Neq16 (Const16 <t> [0]) x)
	for {
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		if auxIntToInt16(v_0.AuxInt) != 0 {
			break
		}
		x := v_1
		if !(isNonNegative(x)) {
			break
		}
		v.reset(OpNeq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Less16 x (Const16 <t> [1]))
	// cond: isNonNegative(x)
	// result: (Eq16 (Const16 <t> [0]) x)
	for {
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		t := v_1.Type
		if auxIntToInt16(v_1.AuxInt) != 1 || !(isNonNegative(x)) {
			break
		}
		v.reset(OpEq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Less16 x (Const16 <t> [1]))
	// result: (Leq16 x (Const16 <t> [0]))
	for {
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		t := v_1.Type
		if auxIntToInt16(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpLeq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less16 (Const16 <t> [-1]) x)
	// result: (Leq16 (Const16 <t> [0]) x)
	for {
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		if auxIntToInt16(v_0.AuxInt) != -1 {
			break
		}
		x := v_1
		v.reset(OpLeq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Less16 _ (Const16 [math.MinInt16]))
	// result: (ConstBool [false])
	for {
		if v_1.Op != OpConst16 || auxIntToInt16(v_1.AuxInt) != math.MinInt16 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Less16 (Const16 [math.MaxInt16]) _)
	// result: (ConstBool [false])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != math.MaxInt16 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Less16 x (Const16 <t> [math.MinInt16+1]))
	// result: (Eq16 x (Const16 <t> [math.MinInt16]))
	for {
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		t := v_1.Type
		if auxIntToInt16(v_1.AuxInt) != math.MinInt16+1 {
			break
		}
		v.reset(OpEq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(math.MinInt16)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less16 (Const16 <t> [math.MaxInt16-1]) x)
	// result: (Eq16 x (Const16 <t> [math.MaxInt16]))
	for {
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		if auxIntToInt16(v_0.AuxInt) != math.MaxInt16-1 {
			break
		}
		x := v_1
		v.reset(OpEq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(math.MaxInt16)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less16U (Const16 [c]) (Const16 [d]))
	// result: (ConstBool [uint16(c) < uint16(d)])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if v_1.Op != OpConst16 {
			break
		}
		d := auxIntToInt16(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(uint16(c) < uint16(d))
		return true
	}
	// match: (Less16U x (Const16 <t> [1]))
	// result: (Eq16 (Const16 <t> [0]) x)
	for {
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		t := v_1.Type
		if auxIntToInt16(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpEq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Less16U _ (Const16 [0]))
	// result: (ConstBool [false])
	for {
		if v_1.Op != OpConst16 || auxIntToInt16(v_1.AuxInt) != 0 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Less16U (Const16 [-1]) _)
	// result: (ConstBool [false])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != -1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Less16U x (Const16 <t> [1]))
	// result: (Eq16 x (Const16 <t> [0]))
	for {
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		t := v_1.Type
		if auxIntToInt16(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpEq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less16U (Const16 <t> [-2]) x)
	// result: (Eq16 x (Const16 <t> [-1]))
	for {
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		if auxIntToInt16(v_0.AuxInt) != -2 {
			break
		}
		x := v_1
		v.reset(OpEq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(-1)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32 (Const32 [c]) (Const32 [d]))
	// result: (ConstBool [c < d])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if v_1.Op != OpConst32 {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(c < d)
		return true
	}
	// match: (Less32 (Const32 <t> [0]) x)
	// cond: isNonNegative(x)
	// result: (Neq32 (Const32 <t> [0]) x)
	for {
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		if auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		x := v_1
		if !(isNonNegative(x)) {
			break
		}
		v.reset(OpNeq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Less32 x (Const32 <t> [1]))
	// cond: isNonNegative(x)
	// result: (Eq32 (Const32 <t> [0]) x)
	for {
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		t := v_1.Type
		if auxIntToInt32(v_1.AuxInt) != 1 || !(isNonNegative(x)) {
			break
		}
		v.reset(OpEq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Less32 x (Const32 <t> [1]))
	// result: (Leq32 x (Const32 <t> [0]))
	for {
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		t := v_1.Type
		if auxIntToInt32(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpLeq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less32 (Const32 <t> [-1]) x)
	// result: (Leq32 (Const32 <t> [0]) x)
	for {
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		if auxIntToInt32(v_0.AuxInt) != -1 {
			break
		}
		x := v_1
		v.reset(OpLeq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Less32 _ (Const32 [math.MinInt32]))
	// result: (ConstBool [false])
	for {
		if v_1.Op != OpConst32 || auxIntToInt32(v_1.AuxInt) != math.MinInt32 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Less32 (Const32 [math.MaxInt32]) _)
	// result: (ConstBool [false])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != math.MaxInt32 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Less32 x (Const32 <t> [math.MinInt32+1]))
	// result: (Eq32 x (Const32 <t> [math.MinInt32]))
	for {
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		t := v_1.Type
		if auxIntToInt32(v_1.AuxInt) != math.MinInt32+1 {
			break
		}
		v.reset(OpEq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(math.MinInt32)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less32 (Const32 <t> [math.MaxInt32-1]) x)
	// result: (Eq32 x (Const32 <t> [math.MaxInt32]))
	for {
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		if auxIntToInt32(v_0.AuxInt) != math.MaxInt32-1 {
			break
		}
		x := v_1
		v.reset(OpEq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(math.MaxInt32)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less32F (Const32F [c]) (Const32F [d]))
	// result: (ConstBool [c < d])
	for {
		if v_0.Op != OpConst32F {
			break
		}
		c := auxIntToFloat32(v_0.AuxInt)
		if v_1.Op != OpConst32F {
			break
		}
		d := auxIntToFloat32(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(c < d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32U (Const32 [c]) (Const32 [d]))
	// result: (ConstBool [uint32(c) < uint32(d)])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if v_1.Op != OpConst32 {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(uint32(c) < uint32(d))
		return true
	}
	// match: (Less32U x (Const32 <t> [1]))
	// result: (Eq32 (Const32 <t> [0]) x)
	for {
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		t := v_1.Type
		if auxIntToInt32(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpEq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Less32U _ (Const32 [0]))
	// result: (ConstBool [false])
	for {
		if v_1.Op != OpConst32 || auxIntToInt32(v_1.AuxInt) != 0 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Less32U (Const32 [-1]) _)
	// result: (ConstBool [false])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != -1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Less32U x (Const32 <t> [1]))
	// result: (Eq32 x (Const32 <t> [0]))
	for {
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		t := v_1.Type
		if auxIntToInt32(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpEq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less32U (Const32 <t> [-2]) x)
	// result: (Eq32 x (Const32 <t> [-1]))
	for {
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		if auxIntToInt32(v_0.AuxInt) != -2 {
			break
		}
		x := v_1
		v.reset(OpEq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(-1)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64 (Const64 [c]) (Const64 [d]))
	// result: (ConstBool [c < d])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(c < d)
		return true
	}
	// match: (Less64 (Const64 <t> [0]) x)
	// cond: isNonNegative(x)
	// result: (Neq64 (Const64 <t> [0]) x)
	for {
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		if auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_1
		if !(isNonNegative(x)) {
			break
		}
		v.reset(OpNeq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Less64 x (Const64 <t> [1]))
	// cond: isNonNegative(x)
	// result: (Eq64 (Const64 <t> [0]) x)
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		if auxIntToInt64(v_1.AuxInt) != 1 || !(isNonNegative(x)) {
			break
		}
		v.reset(OpEq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Less64 x (Const64 <t> [1]))
	// result: (Leq64 x (Const64 <t> [0]))
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		if auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpLeq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less64 (Const64 <t> [-1]) x)
	// result: (Leq64 (Const64 <t> [0]) x)
	for {
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		if auxIntToInt64(v_0.AuxInt) != -1 {
			break
		}
		x := v_1
		v.reset(OpLeq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Less64 _ (Const64 [math.MinInt64]))
	// result: (ConstBool [false])
	for {
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != math.MinInt64 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Less64 (Const64 [math.MaxInt64]) _)
	// result: (ConstBool [false])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != math.MaxInt64 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Less64 x (Const64 <t> [math.MinInt64+1]))
	// result: (Eq64 x (Const64 <t> [math.MinInt64]))
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		if auxIntToInt64(v_1.AuxInt) != math.MinInt64+1 {
			break
		}
		v.reset(OpEq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(math.MinInt64)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less64 (Const64 <t> [math.MaxInt64-1]) x)
	// result: (Eq64 x (Const64 <t> [math.MaxInt64]))
	for {
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		if auxIntToInt64(v_0.AuxInt) != math.MaxInt64-1 {
			break
		}
		x := v_1
		v.reset(OpEq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(math.MaxInt64)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less64F (Const64F [c]) (Const64F [d]))
	// result: (ConstBool [c < d])
	for {
		if v_0.Op != OpConst64F {
			break
		}
		c := auxIntToFloat64(v_0.AuxInt)
		if v_1.Op != OpConst64F {
			break
		}
		d := auxIntToFloat64(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(c < d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64U (Const64 [c]) (Const64 [d]))
	// result: (ConstBool [uint64(c) < uint64(d)])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(uint64(c) < uint64(d))
		return true
	}
	// match: (Less64U x (Const64 <t> [1]))
	// result: (Eq64 (Const64 <t> [0]) x)
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		if auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpEq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Less64U _ (Const64 [0]))
	// result: (ConstBool [false])
	for {
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Less64U (Const64 [-1]) _)
	// result: (ConstBool [false])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != -1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Less64U x (Const64 <t> [1]))
	// result: (Eq64 x (Const64 <t> [0]))
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		if auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpEq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less64U (Const64 <t> [-2]) x)
	// result: (Eq64 x (Const64 <t> [-1]))
	for {
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		if auxIntToInt64(v_0.AuxInt) != -2 {
			break
		}
		x := v_1
		v.reset(OpEq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(-1)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less8 (Const8 [c]) (Const8 [d]))
	// result: (ConstBool [c < d])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if v_1.Op != OpConst8 {
			break
		}
		d := auxIntToInt8(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(c < d)
		return true
	}
	// match: (Less8 (Const8 <t> [0]) x)
	// cond: isNonNegative(x)
	// result: (Neq8 (Const8 <t> [0]) x)
	for {
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		if auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		x := v_1
		if !(isNonNegative(x)) {
			break
		}
		v.reset(OpNeq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Less8 x (Const8 <t> [1]))
	// cond: isNonNegative(x)
	// result: (Eq8 (Const8 <t> [0]) x)
	for {
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		t := v_1.Type
		if auxIntToInt8(v_1.AuxInt) != 1 || !(isNonNegative(x)) {
			break
		}
		v.reset(OpEq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Less8 x (Const8 <t> [1]))
	// result: (Leq8 x (Const8 <t> [0]))
	for {
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		t := v_1.Type
		if auxIntToInt8(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpLeq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less8 (Const8 <t> [-1]) x)
	// result: (Leq8 (Const8 <t> [0]) x)
	for {
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		if auxIntToInt8(v_0.AuxInt) != -1 {
			break
		}
		x := v_1
		v.reset(OpLeq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Less8 _ (Const8 [math.MinInt8 ]))
	// result: (ConstBool [false])
	for {
		if v_1.Op != OpConst8 || auxIntToInt8(v_1.AuxInt) != math.MinInt8 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Less8 (Const8 [math.MaxInt8 ]) _)
	// result: (ConstBool [false])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != math.MaxInt8 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Less8 x (Const8 <t> [math.MinInt8 +1]))
	// result: (Eq8 x (Const8 <t> [math.MinInt8 ]))
	for {
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		t := v_1.Type
		if auxIntToInt8(v_1.AuxInt) != math.MinInt8+1 {
			break
		}
		v.reset(OpEq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(math.MinInt8)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less8 (Const8 <t> [math.MaxInt8 -1]) x)
	// result: (Eq8 x (Const8 <t> [math.MaxInt8 ]))
	for {
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		if auxIntToInt8(v_0.AuxInt) != math.MaxInt8-1 {
			break
		}
		x := v_1
		v.reset(OpEq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(math.MaxInt8)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less8U (Const8 [c]) (Const8 [d]))
	// result: (ConstBool [ uint8(c) < uint8(d)])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if v_1.Op != OpConst8 {
			break
		}
		d := auxIntToInt8(v_1.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(uint8(c) < uint8(d))
		return true
	}
	// match: (Less8U x (Const8 <t> [1]))
	// result: (Eq8 (Const8 <t> [0]) x)
	for {
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		t := v_1.Type
		if auxIntToInt8(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpEq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Less8U _ (Const8 [0]))
	// result: (ConstBool [false])
	for {
		if v_1.Op != OpConst8 || auxIntToInt8(v_1.AuxInt) != 0 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Less8U (Const8 [-1]) _)
	// result: (ConstBool [false])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != -1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Less8U x (Const8 <t> [1]))
	// result: (Eq8 x (Const8 <t> [0]))
	for {
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		t := v_1.Type
		if auxIntToInt8(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpEq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less8U (Const8 <t> [-2]) x)
	// result: (Eq8 x (Const8 <t> [-1]))
	for {
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		if auxIntToInt8(v_0.AuxInt) != -2 {
			break
		}
		x := v_1
		v.reset(OpEq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(-1)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLoad(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Load <t1> p1 (Store {t2} p2 x _))
	// cond: isSamePtr(p1, p2) && copyCompatibleType(t1, x.Type) && t1.Size() == t2.Size()
	// result: x
	for {
		t1 := v.Type
		p1 := v_0
		if v_1.Op != OpStore {
			break
		}
		t2 := auxToType(v_1.Aux)
		x := v_1.Args[1]
		p2 := v_1.Args[0]
		if !(isSamePtr(p1, p2) && copyCompatibleType(t1, x.Type) && t1.Size() == t2.Size()) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Load <t1> p1 (Store {t2} p2 _ (Store {t3} p3 x _)))
	// cond: isSamePtr(p1, p3) && copyCompatibleType(t1, x.Type) && t1.Size() == t3.Size() && disjoint(p3, t3.Size(), p2, t2.Size())
	// result: x
	for {
		t1 := v.Type
		p1 := v_0
		if v_1.Op != OpStore {
			break
		}
		t2 := auxToType(v_1.Aux)
		_ = v_1.Args[2]
		p2 := v_1.Args[0]
		v_1_2 := v_1.Args[2]
		if v_1_2.Op != OpStore {
			break
		}
		t3 := auxToType(v_1_2.Aux)
		x := v_1_2.Args[1]
		p3 := v_1_2.Args[0]
		if !(isSamePtr(p1, p3) && copyCompatibleType(t1, x.Type) && t1.Size() == t3.Size() && disjoint(p3, t3.Size(), p2, t2.Size())) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Load <t1> p1 (Store {t2} p2 _ (Store {t3} p3 _ (Store {t4} p4 x _))))
	// cond: isSamePtr(p1, p4) && copyCompatibleType(t1, x.Type) && t1.Size() == t4.Size() && disjoint(p4, t4.Size(), p2, t2.Size()) && disjoint(p4, t4.Size(), p3, t3.Size())
	// result: x
	for {
		t1 := v.Type
		p1 := v_0
		if v_1.Op != OpStore {
			break
		}
		t2 := auxToType(v_1.Aux)
		_ = v_1.Args[2]
		p2 := v_1.Args[0]
		v_1_2 := v_1.Args[2]
		if v_1_2.Op != OpStore {
			break
		}
		t3 := auxToType(v_1_2.Aux)
		_ = v_1_2.Args[2]
		p3 := v_1_2.Args[0]
		v_1_2_2 := v_1_2.Args[2]
		if v_1_2_2.Op != OpStore {
			break
		}
		t4 := auxToType(v_1_2_2.Aux)
		x := v_1_2_2.Args[1]
		p4 := v_1_2_2.Args[0]
		if !(isSamePtr(p1, p4) && copyCompatibleType(t1, x.Type) && t1.Size() == t4.Size() && disjoint(p4, t4.Size(), p2, t2.Size()) && disjoint(p4, t4.Size(), p3, t3.Size())) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Load <t1> p1 (Store {t2} p2 _ (Store {t3} p3 _ (Store {t4} p4 _ (Store {t5} p5 x _)))))
	// cond: isSamePtr(p1, p5) && copyCompatibleType(t1, x.Type) && t1.Size() == t5.Size() && disjoint(p5, t5.Size(), p2, t2.Size()) && disjoint(p5, t5.Size(), p3, t3.Size()) && disjoint(p5, t5.Size(), p4, t4.Size())
	// result: x
	for {
		t1 := v.Type
		p1 := v_0
		if v_1.Op != OpStore {
			break
		}
		t2 := auxToType(v_1.Aux)
		_ = v_1.Args[2]
		p2 := v_1.Args[0]
		v_1_2 := v_1.Args[2]
		if v_1_2.Op != OpStore {
			break
		}
		t3 := auxToType(v_1_2.Aux)
		_ = v_1_2.Args[2]
		p3 := v_1_2.Args[0]
		v_1_2_2 := v_1_2.Args[2]
		if v_1_2_2.Op != OpStore {
			break
		}
		t4 := auxToType(v_1_2_2.Aux)
		_ = v_1_2_2.Args[2]
		p4 := v_1_2_2.Args[0]
		v_1_2_2_2 := v_1_2_2.Args[2]
		if v_1_2_2_2.Op != OpStore {
			break
		}
		t5 := auxToType(v_1_2_2_2.Aux)
		x := v_1_2_2_2.Args[1]
		p5 := v_1_2_2_2.Args[0]
		if !(isSamePtr(p1, p5) && copyCompatibleType(t1, x.Type) && t1.Size() == t5.Size() && disjoint(p5, t5.Size(), p2, t2.Size()) && disjoint(p5, t5.Size(), p3, t3.Size()) && disjoint(p5, t5.Size(), p4, t4.Size())) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Load <t1> p1 (Store {t2} p2 (Const64 [x]) _))
	// cond: isSamePtr(p1,p2) && t2.Size() == 8 && is64BitFloat(t1) && !math.IsNaN(math.Float64frombits(uint64(x)))
	// result: (Const64F [math.Float64frombits(uint64(x))])
	for {
		t1 := v.Type
		p1 := v_0
		if v_1.Op != OpStore {
			break
		}
		t2 := auxToType(v_1.Aux)
		_ = v_1.Args[1]
		p2 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst64 {
			break
		}
		x := auxIntToInt64(v_1_1.AuxInt)
		if !(isSamePtr(p1, p2) && t2.Size() == 8 && is64BitFloat(t1) && !math.IsNaN(math.Float64frombits(uint64(x)))) {
			break
		}
		v.reset(OpConst64F)
		v.AuxInt = float64ToAuxInt(math.Float64frombits(uint64(x)))
		return true
	}
	// match: (Load <t1> p1 (Store {t2} p2 (Const32 [x]) _))
	// cond: isSamePtr(p1,p2) && t2.Size() == 4 && is32BitFloat(t1) && !math.IsNaN(float64(math.Float32frombits(uint32(x))))
	// result: (Const32F [math.Float32frombits(uint32(x))])
	for {
		t1 := v.Type
		p1 := v_0
		if v_1.Op != OpStore {
			break
		}
		t2 := auxToType(v_1.Aux)
		_ = v_1.Args[1]
		p2 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst32 {
			break
		}
		x := auxIntToInt32(v_1_1.AuxInt)
		if !(isSamePtr(p1, p2) && t2.Size() == 4 && is32BitFloat(t1) && !math.IsNaN(float64(math.Float32frombits(uint32(x))))) {
			break
		}
		v.reset(OpConst32F)
		v.AuxInt = float32ToAuxInt(math.Float32frombits(uint32(x)))
		return true
	}
	// match: (Load <t1> p1 (Store {t2} p2 (Const64F [x]) _))
	// cond: isSamePtr(p1,p2) && t2.Size() == 8 && is64BitInt(t1)
	// result: (Const64 [int64(math.Float64bits(x))])
	for {
		t1 := v.Type
		p1 := v_0
		if v_1.Op != OpStore {
			break
		}
		t2 := auxToType(v_1.Aux)
		_ = v_1.Args[1]
		p2 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst64F {
			break
		}
		x := auxIntToFloat64(v_1_1.AuxInt)
		if !(isSamePtr(p1, p2) && t2.Size() == 8 && is64BitInt(t1)) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(math.Float64bits(x)))
		return true
	}
	// match: (Load <t1> p1 (Store {t2} p2 (Const32F [x]) _))
	// cond: isSamePtr(p1,p2) && t2.Size() == 4 && is32BitInt(t1)
	// result: (Const32 [int32(math.Float32bits(x))])
	for {
		t1 := v.Type
		p1 := v_0
		if v_1.Op != OpStore {
			break
		}
		t2 := auxToType(v_1.Aux)
		_ = v_1.Args[1]
		p2 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst32F {
			break
		}
		x := auxIntToFloat32(v_1_1.AuxInt)
		if !(isSamePtr(p1, p2) && t2.Size() == 4 && is32BitInt(t1)) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(math.Float32bits(x)))
		return true
	}
	// match: (Load <t1> op:(OffPtr [o1] p1) (Store {t2} p2 _ mem:(Zero [n] p3 _)))
	// cond: o1 >= 0 && o1+t1.Size() <= n && isSamePtr(p1, p3) && CanSSA(t1) && disjoint(op, t1.Size(), p2, t2.Size())
	// result: @mem.Block (Load <t1> (OffPtr <op.Type> [o1] p3) mem)
	for {
		t1 := v.Type
		op := v_0
		if op.Op != OpOffPtr {
			break
		}
		o1 := auxIntToInt64(op.AuxInt)
		p1 := op.Args[0]
		if v_1.Op != OpStore {
			break
		}
		t2 := auxToType(v_1.Aux)
		_ = v_1.Args[2]
		p2 := v_1.Args[0]
		mem := v_1.Args[2]
		if mem.Op != OpZero {
			break
		}
		n := auxIntToInt64(mem.AuxInt)
		p3 := mem.Args[0]
		if !(o1 >= 0 && o1+t1.Size() <= n && isSamePtr(p1, p3) && CanSSA(t1) && disjoint(op, t1.Size(), p2, t2.Size())) {
			break
		}
		b = mem.Block
		v0 := b.NewValue0(v.Pos, OpLoad, t1)
		v.copyOf(v0)
		v1 := b.NewValue0(v.Pos, OpOffPtr, op.Type)
		v1.AuxInt = int64ToAuxInt(o1)
		v1.AddArg(p3)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (Load <t1> op:(OffPtr [o1] p1) (Store {t2} p2 _ (Store {t3} p3 _ mem:(Zero [n] p4 _))))
	// cond: o1 >= 0 && o1+t1.Size() <= n && isSamePtr(p1, p4) && CanSSA(t1) && disjoint(op, t1.Size(), p2, t2.Size()) && disjoint(op, t1.Size(), p3, t3.Size())
	// result: @mem.Block (Load <t1> (OffPtr <op.Type> [o1] p4) mem)
	for {
		t1 := v.Type
		op := v_0
		if op.Op != OpOffPtr {
			break
		}
		o1 := auxIntToInt64(op.AuxInt)
		p1 := op.Args[0]
		if v_1.Op != OpStore {
			break
		}
		t2 := auxToType(v_1.Aux)
		_ = v_1.Args[2]
		p2 := v_1.Args[0]
		v_1_2 := v_1.Args[2]
		if v_1_2.Op != OpStore {
			break
		}
		t3 := auxToType(v_1_2.Aux)
		_ = v_1_2.Args[2]
		p3 := v_1_2.Args[0]
		mem := v_1_2.Args[2]
		if mem.Op != OpZero {
			break
		}
		n := auxIntToInt64(mem.AuxInt)
		p4 := mem.Args[0]
		if !(o1 >= 0 && o1+t1.Size() <= n && isSamePtr(p1, p4) && CanSSA(t1) && disjoint(op, t1.Size(), p2, t2.Size()) && disjoint(op, t1.Size(), p3, t3.Size())) {
			break
		}
		b = mem.Block
		v0 := b.NewValue0(v.Pos, OpLoad, t1)
		v.copyOf(v0)
		v1 := b.NewValue0(v.Pos, OpOffPtr, op.Type)
		v1.AuxInt = int64ToAuxInt(o1)
		v1.AddArg(p4)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (Load <t1> op:(OffPtr [o1] p1) (Store {t2} p2 _ (Store {t3} p3 _ (Store {t4} p4 _ mem:(Zero [n] p5 _)))))
	// cond: o1 >= 0 && o1+t1.Size() <= n && isSamePtr(p1, p5) && CanSSA(t1) && disjoint(op, t1.Size(), p2, t2.Size()) && disjoint(op, t1.Size(), p3, t3.Size()) && disjoint(op, t1.Size(), p4, t4.Size())
	// result: @mem.Block (Load <t1> (OffPtr <op.Type> [o1] p5) mem)
	for {
		t1 := v.Type
		op := v_0
		if op.Op != OpOffPtr {
			break
		}
		o1 := auxIntToInt64(op.AuxInt)
		p1 := op.Args[0]
		if v_1.Op != OpStore {
			break
		}
		t2 := auxToType(v_1.Aux)
		_ = v_1.Args[2]
		p2 := v_1.Args[0]
		v_1_2 := v_1.Args[2]
		if v_1_2.Op != OpStore {
			break
		}
		t3 := auxToType(v_1_2.Aux)
		_ = v_1_2.Args[2]
		p3 := v_1_2.Args[0]
		v_1_2_2 := v_1_2.Args[2]
		if v_1_2_2.Op != OpStore {
			break
		}
		t4 := auxToType(v_1_2_2.Aux)
		_ = v_1_2_2.Args[2]
		p4 := v_1_2_2.Args[0]
		mem := v_1_2_2.Args[2]
		if mem.Op != OpZero {
			break
		}
		n := auxIntToInt64(mem.AuxInt)
		p5 := mem.Args[0]
		if !(o1 >= 0 && o1+t1.Size() <= n && isSamePtr(p1, p5) && CanSSA(t1) && disjoint(op, t1.Size(), p2, t2.Size()) && disjoint(op, t1.Size(), p3, t3.Size()) && disjoint(op, t1.Size(), p4, t4.Size())) {
			break
		}
		b = mem.Block
		v0 := b.NewValue0(v.Pos, OpLoad, t1)
		v.copyOf(v0)
		v1 := b.NewValue0(v.Pos, OpOffPtr, op.Type)
		v1.AuxInt = int64ToAuxInt(o1)
		v1.AddArg(p5)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (Load <t1> op:(OffPtr [o1] p1) (Store {t2} p2 _ (Store {t3} p3 _ (Store {t4} p4 _ (Store {t5} p5 _ mem:(Zero [n] p6 _))))))
	// cond: o1 >= 0 && o1+t1.Size() <= n && isSamePtr(p1, p6) && CanSSA(t1) && disjoint(op, t1.Size(), p2, t2.Size()) && disjoint(op, t1.Size(), p3, t3.Size()) && disjoint(op, t1.Size(), p4, t4.Size()) && disjoint(op, t1.Size(), p5, t5.Size())
	// result: @mem.Block (Load <t1> (OffPtr <op.Type> [o1] p6) mem)
	for {
		t1 := v.Type
		op := v_0
		if op.Op != OpOffPtr {
			break
		}
		o1 := auxIntToInt64(op.AuxInt)
		p1 := op.Args[0]
		if v_1.Op != OpStore {
			break
		}
		t2 := auxToType(v_1.Aux)
		_ = v_1.Args[2]
		p2 := v_1.Args[0]
		v_1_2 := v_1.Args[2]
		if v_1_2.Op != OpStore {
			break
		}
		t3 := auxToType(v_1_2.Aux)
		_ = v_1_2.Args[2]
		p3 := v_1_2.Args[0]
		v_1_2_2 := v_1_2.Args[2]
		if v_1_2_2.Op != OpStore {
			break
		}
		t4 := auxToType(v_1_2_2.Aux)
		_ = v_1_2_2.Args[2]
		p4 := v_1_2_2.Args[0]
		v_1_2_2_2 := v_1_2_2.Args[2]
		if v_1_2_2_2.Op != OpStore {
			break
		}
		t5 := auxToType(v_1_2_2_2.Aux)
		_ = v_1_2_2_2.Args[2]
		p5 := v_1_2_2_2.Args[0]
		mem := v_1_2_2_2.Args[2]
		if mem.Op != OpZero {
			break
		}
		n := auxIntToInt64(mem.AuxInt)
		p6 := mem.Args[0]
		if !(o1 >= 0 && o1+t1.Size() <= n && isSamePtr(p1, p6) && CanSSA(t1) && disjoint(op, t1.Size(), p2, t2.Size()) && disjoint(op, t1.Size(), p3, t3.Size()) && disjoint(op, t1.Size(), p4, t4.Size()) && disjoint(op, t1.Size(), p5, t5.Size())) {
			break
		}
		b = mem.Block
		v0 := b.NewValue0(v.Pos, OpLoad, t1)
		v.copyOf(v0)
		v1 := b.NewValue0(v.Pos, OpOffPtr, op.Type)
		v1.AuxInt = int64ToAuxInt(o1)
		v1.AddArg(p6)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (Load <t1> (OffPtr [o] p1) (Zero [n] p2 _))
	// cond: t1.IsBoolean() && isSamePtr(p1, p2) && n >= o + 1
	// result: (ConstBool [false])
	for {
		t1 := v.Type
		if v_0.Op != OpOffPtr {
			break
		}
		o := auxIntToInt64(v_0.AuxInt)
		p1 := v_0.Args[0]
		if v_1.Op != OpZero {
			break
		}
		n := auxIntToInt64(v_1.AuxInt)
		p2 := v_1.Args[0]
		if !(t1.IsBoolean() && isSamePtr(p1, p2) && n >= o+1) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Load <t1> (OffPtr [o] p1) (Zero [n] p2 _))
	// cond: is8BitInt(t1) && isSamePtr(p1, p2) && n >= o + 1
	// result: (Const8 [0])
	for {
		t1 := v.Type
		if v_0.Op != OpOffPtr {
			break
		}
		o := auxIntToInt64(v_0.AuxInt)
		p1 := v_0.Args[0]
		if v_1.Op != OpZero {
			break
		}
		n := auxIntToInt64(v_1.AuxInt)
		p2 := v_1.Args[0]
		if !(is8BitInt(t1) && isSamePtr(p1, p2) && n >= o+1) {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	// match: (Load <t1> (OffPtr [o] p1) (Zero [n] p2 _))
	// cond: is16BitInt(t1) && isSamePtr(p1, p2) && n >= o + 2
	// result: (Const16 [0])
	for {
		t1 := v.Type
		if v_0.Op != OpOffPtr {
			break
		}
		o := auxIntToInt64(v_0.AuxInt)
		p1 := v_0.Args[0]
		if v_1.Op != OpZero {
			break
		}
		n := auxIntToInt64(v_1.AuxInt)
		p2 := v_1.Args[0]
		if !(is16BitInt(t1) && isSamePtr(p1, p2) && n >= o+2) {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	// match: (Load <t1> (OffPtr [o] p1) (Zero [n] p2 _))
	// cond: is32BitInt(t1) && isSamePtr(p1, p2) && n >= o + 4
	// result: (Const32 [0])
	for {
		t1 := v.Type
		if v_0.Op != OpOffPtr {
			break
		}
		o := auxIntToInt64(v_0.AuxInt)
		p1 := v_0.Args[0]
		if v_1.Op != OpZero {
			break
		}
		n := auxIntToInt64(v_1.AuxInt)
		p2 := v_1.Args[0]
		if !(is32BitInt(t1) && isSamePtr(p1, p2) && n >= o+4) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (Load <t1> (OffPtr [o] p1) (Zero [n] p2 _))
	// cond: is64BitInt(t1) && isSamePtr(p1, p2) && n >= o + 8
	// result: (Const64 [0])
	for {
		t1 := v.Type
		if v_0.Op != OpOffPtr {
			break
		}
		o := auxIntToInt64(v_0.AuxInt)
		p1 := v_0.Args[0]
		if v_1.Op != OpZero {
			break
		}
		n := auxIntToInt64(v_1.AuxInt)
		p2 := v_1.Args[0]
		if !(is64BitInt(t1) && isSamePtr(p1, p2) && n >= o+8) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (Load <t1> (OffPtr [o] p1) (Zero [n] p2 _))
	// cond: is32BitFloat(t1) && isSamePtr(p1, p2) && n >= o + 4
	// result: (Const32F [0])
	for {
		t1 := v.Type
		if v_0.Op != OpOffPtr {
			break
		}
		o := auxIntToInt64(v_0.AuxInt)
		p1 := v_0.Args[0]
		if v_1.Op != OpZero {
			break
		}
		n := auxIntToInt64(v_1.AuxInt)
		p2 := v_1.Args[0]
		if !(is32BitFloat(t1) && isSamePtr(p1, p2) && n >= o+4) {
			break
		}
		v.reset(OpConst32F)
		v.AuxInt = float32ToAuxInt(0)
		return true
	}
	// match: (Load <t1> (OffPtr [o] p1) (Zero [n] p2 _))
	// cond: is64BitFloat(t1) && isSamePtr(p1, p2) && n >= o + 8
	// result: (Const64F [0])
	for {
		t1 := v.Type
		if v_0.Op != OpOffPtr {
			break
		}
		o := auxIntToInt64(v_0.AuxInt)
		p1 := v_0.Args[0]
		if v_1.Op != OpZero {
			break
		}
		n := auxIntToInt64(v_1.AuxInt)
		p2 := v_1.Args[0]
		if !(is64BitFloat(t1) && isSamePtr(p1, p2) && n >= o+8) {
			break
		}
		v.reset(OpConst64F)
		v.AuxInt = float64ToAuxInt(0)
		return true
	}
	// match: (Load <t> _ _)
	// cond: t.IsStruct() && CanSSA(t) && !t.IsSIMD()
	// result: rewriteStructLoad(v)
	for {
		t := v.Type
		if !(t.IsStruct() && CanSSA(t) && !t.IsSIMD()) {
			break
		}
		v.copyOf(rewriteStructLoad(v))
		return true
	}
	// match: (Load <t> _ _)
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
	// match: (Load <t> ptr mem)
	// cond: t.IsArray() && t.NumElem() == 1 && CanSSA(t)
	// result: (ArrayMake1 (Load <t.Elem()> ptr mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.IsArray() && t.NumElem() == 1 && CanSSA(t)) {
			break
		}
		v.reset(OpArrayMake1)
		v0 := b.NewValue0(v.Pos, OpLoad, t.Elem())
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	// match: (Load <typ.BytePtr> (OffPtr [off] (Addr {s} sb) ) _)
	// cond: isFixedSym(s, off)
	// result: (Addr {fixedSym(b.Func, s, off)} sb)
	for {
		if v.Type != typ.BytePtr || v_0.Op != OpOffPtr {
			break
		}
		off := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAddr {
			break
		}
		s := auxToSym(v_0_0.Aux)
		sb := v_0_0.Args[0]
		if !(isFixedSym(s, off)) {
			break
		}
		v.reset(OpAddr)
		v.Aux = symToAux(fixedSym(b.Func, s, off))
		v.AddArg(sb)
		return true
	}
	// match: (Load <typ.BytePtr> (OffPtr [off] (Convert (Addr {s} sb) _) ) _)
	// cond: isFixedSym(s, off)
	// result: (Addr {fixedSym(b.Func, s, off)} sb)
	for {
		if v.Type != typ.BytePtr || v_0.Op != OpOffPtr {
			break
		}
		off := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConvert {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAddr {
			break
		}
		s := auxToSym(v_0_0_0.Aux)
		sb := v_0_0_0.Args[0]
		if !(isFixedSym(s, off)) {
			break
		}
		v.reset(OpAddr)
		v.Aux = symToAux(fixedSym(b.Func, s, off))
		v.AddArg(sb)
		return true
	}
	// match: (Load <typ.BytePtr> (OffPtr [off] (ITab (IMake (Addr {s} sb) _))) _)
	// cond: isFixedSym(s, off)
	// result: (Addr {fixedSym(b.Func, s, off)} sb)
	for {
		if v.Type != typ.BytePtr || v_0.Op != OpOffPtr {
			break
		}
		off := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpITab {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpIMake {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpAddr {
			break
		}
		s := auxToSym(v_0_0_0_0.Aux)
		sb := v_0_0_0_0.Args[0]
		if !(isFixedSym(s, off)) {
			break
		}
		v.reset(OpAddr)
		v.Aux = symToAux(fixedSym(b.Func, s, off))
		v.AddArg(sb)
		return true
	}
	// match: (Load <typ.BytePtr> (OffPtr [off] (ITab (IMake (Convert (Addr {s} sb) _) _))) _)
	// cond: isFixedSym(s, off)
	// result: (Addr {fixedSym(b.Func, s, off)} sb)
	for {
		if v.Type != typ.BytePtr || v_0.Op != OpOffPtr {
			break
		}
		off := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpITab {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpIMake {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpConvert {
			break
		}
		v_0_0_0_0_0 := v_0_0_0_0.Args[0]
		if v_0_0_0_0_0.Op != OpAddr {
			break
		}
		s := auxToSym(v_0_0_0_0_0.Aux)
		sb := v_0_0_0_0_0.Args[0]
		if !(isFixedSym(s, off)) {
			break
		}
		v.reset(OpAddr)
		v.Aux = symToAux(fixedSym(b.Func, s, off))
		v.AddArg(sb)
		return true
	}
	// match: (Load <typ.Uintptr> (OffPtr [off] (Addr {s} sb) ) _)
	// cond: isFixedSym(s, off)
	// result: (Addr {fixedSym(b.Func, s, off)} sb)
	for {
		if v.Type != typ.Uintptr || v_0.Op != OpOffPtr {
			break
		}
		off := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAddr {
			break
		}
		s := auxToSym(v_0_0.Aux)
		sb := v_0_0.Args[0]
		if !(isFixedSym(s, off)) {
			break
		}
		v.reset(OpAddr)
		v.Aux = symToAux(fixedSym(b.Func, s, off))
		v.AddArg(sb)
		return true
	}
	// match: (Load <typ.Uintptr> (OffPtr [off] (Convert (Addr {s} sb) _) ) _)
	// cond: isFixedSym(s, off)
	// result: (Addr {fixedSym(b.Func, s, off)} sb)
	for {
		if v.Type != typ.Uintptr || v_0.Op != OpOffPtr {
			break
		}
		off := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConvert {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAddr {
			break
		}
		s := auxToSym(v_0_0_0.Aux)
		sb := v_0_0_0.Args[0]
		if !(isFixedSym(s, off)) {
			break
		}
		v.reset(OpAddr)
		v.Aux = symToAux(fixedSym(b.Func, s, off))
		v.AddArg(sb)
		return true
	}
	// match: (Load <typ.Uintptr> (OffPtr [off] (ITab (IMake (Addr {s} sb) _))) _)
	// cond: isFixedSym(s, off)
	// result: (Addr {fixedSym(b.Func, s, off)} sb)
	for {
		if v.Type != typ.Uintptr || v_0.Op != OpOffPtr {
			break
		}
		off := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpITab {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpIMake {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpAddr {
			break
		}
		s := auxToSym(v_0_0_0_0.Aux)
		sb := v_0_0_0_0.Args[0]
		if !(isFixedSym(s, off)) {
			break
		}
		v.reset(OpAddr)
		v.Aux = symToAux(fixedSym(b.Func, s, off))
		v.AddArg(sb)
		return true
	}
	// match: (Load <typ.Uintptr> (OffPtr [off] (ITab (IMake (Convert (Addr {s} sb) _) _))) _)
	// cond: isFixedSym(s, off)
	// result: (Addr {fixedSym(b.Func, s, off)} sb)
	for {
		if v.Type != typ.Uintptr || v_0.Op != OpOffPtr {
			break
		}
		off := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpITab {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpIMake {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpConvert {
			break
		}
		v_0_0_0_0_0 := v_0_0_0_0.Args[0]
		if v_0_0_0_0_0.Op != OpAddr {
			break
		}
		s := auxToSym(v_0_0_0_0_0.Aux)
		sb := v_0_0_0_0_0.Args[0]
		if !(isFixedSym(s, off)) {
			break
		}
		v.reset(OpAddr)
		v.Aux = symToAux(fixedSym(b.Func, s, off))
		v.AddArg(sb)
		return true
	}
	// match: (Load <t> (OffPtr [off] (Addr {sym} _) ) _)
	// cond: t.IsInteger() && t.Size() == 4 && isFixed32(config, sym, off)
	// result: (Const32 [fixed32(config, sym, off)])
	for {
		t := v.Type
		if v_0.Op != OpOffPtr {
			break
		}
		off := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAddr {
			break
		}
		sym := auxToSym(v_0_0.Aux)
		if !(t.IsInteger() && t.Size() == 4 && isFixed32(config, sym, off)) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(fixed32(config, sym, off))
		return true
	}
	// match: (Load <t> (OffPtr [off] (Convert (Addr {sym} _) _) ) _)
	// cond: t.IsInteger() && t.Size() == 4 && isFixed32(config, sym, off)
	// result: (Const32 [fixed32(config, sym, off)])
	for {
		t := v.Type
		if v_0.Op != OpOffPtr {
			break
		}
		off := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConvert {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAddr {
			break
		}
		sym := auxToSym(v_0_0_0.Aux)
		if !(t.IsInteger() && t.Size() == 4 && isFixed32(config, sym, off)) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(fixed32(config, sym, off))
		return true
	}
	// match: (Load <t> (OffPtr [off] (ITab (IMake (Addr {sym} _) _))) _)
	// cond: t.IsInteger() && t.Size() == 4 && isFixed32(config, sym, off)
	// result: (Const32 [fixed32(config, sym, off)])
	for {
		t := v.Type
		if v_0.Op != OpOffPtr {
			break
		}
		off := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpITab {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpIMake {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpAddr {
			break
		}
		sym := auxToSym(v_0_0_0_0.Aux)
		if !(t.IsInteger() && t.Size() == 4 && isFixed32(config, sym, off)) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(fixed32(config, sym, off))
		return true
	}
	// match: (Load <t> (OffPtr [off] (ITab (IMake (Convert (Addr {sym} _) _) _))) _)
	// cond: t.IsInteger() && t.Size() == 4 && isFixed32(config, sym, off)
	// result: (Const32 [fixed32(config, sym, off)])
	for {
		t := v.Type
		if v_0.Op != OpOffPtr {
			break
		}
		off := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpITab {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpIMake {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpConvert {
			break
		}
		v_0_0_0_0_0 := v_0_0_0_0.Args[0]
		if v_0_0_0_0_0.Op != OpAddr {
			break
		}
		sym := auxToSym(v_0_0_0_0_0.Aux)
		if !(t.IsInteger() && t.Size() == 4 && isFixed32(config, sym, off)) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(fixed32(config, sym, off))
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh16x16 <t> x (Const16 [c]))
	// result: (Lsh16x64 x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		v.reset(OpLsh16x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint16(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh16x16 (Const16 [0]) _)
	// result: (Const16 [0])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh16x32 <t> x (Const32 [c]))
	// result: (Lsh16x64 x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpLsh16x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint32(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh16x32 (Const16 [0]) _)
	// result: (Const16 [0])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x64 (Const16 [c]) (Const64 [d]))
	// result: (Const16 [c << uint64(d)])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(c << uint64(d))
		return true
	}
	// match: (Lsh16x64 x (Const64 [0]))
	// result: x
	for {
		x := v_0
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Lsh16x64 (Const16 [0]) _)
	// result: (Const16 [0])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	// match: (Lsh16x64 _ (Const64 [c]))
	// cond: uint64(c) >= 16
	// result: (Const16 [0])
	for {
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 16) {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	// match: (Lsh16x64 <t> (Lsh16x64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Lsh16x64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		if v_0.Op != OpLsh16x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpLsh16x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(c + d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh16x64 i:(Rsh16x64 x (Const64 [c])) (Const64 [c]))
	// cond: c >= 0 && c < 16 && i.Uses == 1
	// result: (And16 x (Const16 <v.Type> [int16(-1) << c]))
	for {
		i := v_0
		if i.Op != OpRsh16x64 {
			break
		}
		_ = i.Args[1]
		x := i.Args[0]
		i_1 := i.Args[1]
		if i_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(i_1.AuxInt)
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != c || !(c >= 0 && c < 16 && i.Uses == 1) {
			break
		}
		v.reset(OpAnd16)
		v0 := b.NewValue0(v.Pos, OpConst16, v.Type)
		v0.AuxInt = int16ToAuxInt(int16(-1) << c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh16x64 i:(Rsh16Ux64 x (Const64 [c])) (Const64 [c]))
	// cond: c >= 0 && c < 16 && i.Uses == 1
	// result: (And16 x (Const16 <v.Type> [int16(-1) << c]))
	for {
		i := v_0
		if i.Op != OpRsh16Ux64 {
			break
		}
		_ = i.Args[1]
		x := i.Args[0]
		i_1 := i.Args[1]
		if i_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(i_1.AuxInt)
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != c || !(c >= 0 && c < 16 && i.Uses == 1) {
			break
		}
		v.reset(OpAnd16)
		v0 := b.NewValue0(v.Pos, OpConst16, v.Type)
		v0.AuxInt = int16ToAuxInt(int16(-1) << c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh16x64 (Rsh16Ux64 (Lsh16x64 x (Const64 [c1])) (Const64 [c2])) (Const64 [c3]))
	// cond: uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)
	// result: (Lsh16x64 x (Const64 <typ.UInt64> [c1-c2+c3]))
	for {
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
		c1 := auxIntToInt64(v_0_0_1.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c2 := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		c3 := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)) {
			break
		}
		v.reset(OpLsh16x64)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c1 - c2 + c3)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh16x64 (And16 (Rsh16x64 <t> x (Const64 <t2> [c])) (Const16 [d])) (Const64 [e]))
	// cond: c >= e
	// result: (And16 (Rsh16x64 <t> x (Const64 <t2> [c-e])) (Const16 <t> [d<<e]))
	for {
		if v_0.Op != OpAnd16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRsh16x64 {
				continue
			}
			t := v_0_0.Type
			_ = v_0_0.Args[1]
			x := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			if v_0_0_1.Op != OpConst64 {
				continue
			}
			t2 := v_0_0_1.Type
			c := auxIntToInt64(v_0_0_1.AuxInt)
			if v_0_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_0_1.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			e := auxIntToInt64(v_1.AuxInt)
			if !(c >= e) {
				continue
			}
			v.reset(OpAnd16)
			v0 := b.NewValue0(v.Pos, OpRsh16x64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, t2)
			v1.AuxInt = int64ToAuxInt(c - e)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst16, t)
			v2.AuxInt = int16ToAuxInt(d << e)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (Lsh16x64 (And16 (Rsh16Ux64 <t> x (Const64 <t2> [c])) (Const16 [d])) (Const64 [e]))
	// cond: c >= e
	// result: (And16 (Rsh16Ux64 <t> x (Const64 <t2> [c-e])) (Const16 <t> [d<<e]))
	for {
		if v_0.Op != OpAnd16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRsh16Ux64 {
				continue
			}
			t := v_0_0.Type
			_ = v_0_0.Args[1]
			x := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			if v_0_0_1.Op != OpConst64 {
				continue
			}
			t2 := v_0_0_1.Type
			c := auxIntToInt64(v_0_0_1.AuxInt)
			if v_0_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_0_1.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			e := auxIntToInt64(v_1.AuxInt)
			if !(c >= e) {
				continue
			}
			v.reset(OpAnd16)
			v0 := b.NewValue0(v.Pos, OpRsh16Ux64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, t2)
			v1.AuxInt = int64ToAuxInt(c - e)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst16, t)
			v2.AuxInt = int16ToAuxInt(d << e)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (Lsh16x64 (And16 (Rsh16x64 <t> x (Const64 <t2> [c])) (Const16 [d])) (Const64 [e]))
	// cond: c < e
	// result: (And16 (Lsh16x64 <t> x (Const64 <t2> [e-c])) (Const16 <t> [d<<e]))
	for {
		if v_0.Op != OpAnd16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRsh16x64 {
				continue
			}
			t := v_0_0.Type
			_ = v_0_0.Args[1]
			x := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			if v_0_0_1.Op != OpConst64 {
				continue
			}
			t2 := v_0_0_1.Type
			c := auxIntToInt64(v_0_0_1.AuxInt)
			if v_0_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_0_1.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			e := auxIntToInt64(v_1.AuxInt)
			if !(c < e) {
				continue
			}
			v.reset(OpAnd16)
			v0 := b.NewValue0(v.Pos, OpLsh16x64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, t2)
			v1.AuxInt = int64ToAuxInt(e - c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst16, t)
			v2.AuxInt = int16ToAuxInt(d << e)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (Lsh16x64 (And16 (Rsh16Ux64 <t> x (Const64 <t2> [c])) (Const16 [d])) (Const64 [e]))
	// cond: c < e
	// result: (And16 (Lsh16x64 <t> x (Const64 <t2> [e-c])) (Const16 <t> [d<<e]))
	for {
		if v_0.Op != OpAnd16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRsh16Ux64 {
				continue
			}
			t := v_0_0.Type
			_ = v_0_0.Args[1]
			x := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			if v_0_0_1.Op != OpConst64 {
				continue
			}
			t2 := v_0_0_1.Type
			c := auxIntToInt64(v_0_0_1.AuxInt)
			if v_0_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_0_1.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			e := auxIntToInt64(v_1.AuxInt)
			if !(c < e) {
				continue
			}
			v.reset(OpAnd16)
			v0 := b.NewValue0(v.Pos, OpLsh16x64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, t2)
			v1.AuxInt = int64ToAuxInt(e - c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst16, t)
			v2.AuxInt = int16ToAuxInt(d << e)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpLsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh16x8 <t> x (Const8 [c]))
	// result: (Lsh16x64 x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		v.reset(OpLsh16x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint8(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh16x8 (Const16 [0]) _)
	// result: (Const16 [0])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh32x16 <t> x (Const16 [c]))
	// result: (Lsh32x64 x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		v.reset(OpLsh32x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint16(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh32x16 (Const32 [0]) _)
	// result: (Const32 [0])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh32x32 <t> x (Const32 [c]))
	// result: (Lsh32x64 x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpLsh32x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint32(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh32x32 (Const32 [0]) _)
	// result: (Const32 [0])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x64 (Const32 [c]) (Const64 [d]))
	// result: (Const32 [c << uint64(d)])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(c << uint64(d))
		return true
	}
	// match: (Lsh32x64 x (Const64 [0]))
	// result: x
	for {
		x := v_0
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Lsh32x64 (Const32 [0]) _)
	// result: (Const32 [0])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (Lsh32x64 _ (Const64 [c]))
	// cond: uint64(c) >= 32
	// result: (Const32 [0])
	for {
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 32) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (Lsh32x64 <t> (Lsh32x64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Lsh32x64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		if v_0.Op != OpLsh32x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpLsh32x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(c + d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh32x64 i:(Rsh32x64 x (Const64 [c])) (Const64 [c]))
	// cond: c >= 0 && c < 32 && i.Uses == 1
	// result: (And32 x (Const32 <v.Type> [int32(-1) << c]))
	for {
		i := v_0
		if i.Op != OpRsh32x64 {
			break
		}
		_ = i.Args[1]
		x := i.Args[0]
		i_1 := i.Args[1]
		if i_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(i_1.AuxInt)
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != c || !(c >= 0 && c < 32 && i.Uses == 1) {
			break
		}
		v.reset(OpAnd32)
		v0 := b.NewValue0(v.Pos, OpConst32, v.Type)
		v0.AuxInt = int32ToAuxInt(int32(-1) << c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh32x64 i:(Rsh32Ux64 x (Const64 [c])) (Const64 [c]))
	// cond: c >= 0 && c < 32 && i.Uses == 1
	// result: (And32 x (Const32 <v.Type> [int32(-1) << c]))
	for {
		i := v_0
		if i.Op != OpRsh32Ux64 {
			break
		}
		_ = i.Args[1]
		x := i.Args[0]
		i_1 := i.Args[1]
		if i_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(i_1.AuxInt)
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != c || !(c >= 0 && c < 32 && i.Uses == 1) {
			break
		}
		v.reset(OpAnd32)
		v0 := b.NewValue0(v.Pos, OpConst32, v.Type)
		v0.AuxInt = int32ToAuxInt(int32(-1) << c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh32x64 (Rsh32Ux64 (Lsh32x64 x (Const64 [c1])) (Const64 [c2])) (Const64 [c3]))
	// cond: uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)
	// result: (Lsh32x64 x (Const64 <typ.UInt64> [c1-c2+c3]))
	for {
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
		c1 := auxIntToInt64(v_0_0_1.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c2 := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		c3 := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)) {
			break
		}
		v.reset(OpLsh32x64)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c1 - c2 + c3)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh32x64 (And32 (Rsh32x64 <t> x (Const64 <t2> [c])) (Const32 [d])) (Const64 [e]))
	// cond: c >= e
	// result: (And32 (Rsh32x64 <t> x (Const64 <t2> [c-e])) (Const32 <t> [d<<e]))
	for {
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRsh32x64 {
				continue
			}
			t := v_0_0.Type
			_ = v_0_0.Args[1]
			x := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			if v_0_0_1.Op != OpConst64 {
				continue
			}
			t2 := v_0_0_1.Type
			c := auxIntToInt64(v_0_0_1.AuxInt)
			if v_0_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_0_1.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			e := auxIntToInt64(v_1.AuxInt)
			if !(c >= e) {
				continue
			}
			v.reset(OpAnd32)
			v0 := b.NewValue0(v.Pos, OpRsh32x64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, t2)
			v1.AuxInt = int64ToAuxInt(c - e)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst32, t)
			v2.AuxInt = int32ToAuxInt(d << e)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (Lsh32x64 (And32 (Rsh32Ux64 <t> x (Const64 <t2> [c])) (Const32 [d])) (Const64 [e]))
	// cond: c >= e
	// result: (And32 (Rsh32Ux64 <t> x (Const64 <t2> [c-e])) (Const32 <t> [d<<e]))
	for {
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRsh32Ux64 {
				continue
			}
			t := v_0_0.Type
			_ = v_0_0.Args[1]
			x := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			if v_0_0_1.Op != OpConst64 {
				continue
			}
			t2 := v_0_0_1.Type
			c := auxIntToInt64(v_0_0_1.AuxInt)
			if v_0_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_0_1.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			e := auxIntToInt64(v_1.AuxInt)
			if !(c >= e) {
				continue
			}
			v.reset(OpAnd32)
			v0 := b.NewValue0(v.Pos, OpRsh32Ux64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, t2)
			v1.AuxInt = int64ToAuxInt(c - e)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst32, t)
			v2.AuxInt = int32ToAuxInt(d << e)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (Lsh32x64 (And32 (Rsh32x64 <t> x (Const64 <t2> [c])) (Const32 [d])) (Const64 [e]))
	// cond: c < e
	// result: (And32 (Lsh32x64 <t> x (Const64 <t2> [e-c])) (Const32 <t> [d<<e]))
	for {
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRsh32x64 {
				continue
			}
			t := v_0_0.Type
			_ = v_0_0.Args[1]
			x := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			if v_0_0_1.Op != OpConst64 {
				continue
			}
			t2 := v_0_0_1.Type
			c := auxIntToInt64(v_0_0_1.AuxInt)
			if v_0_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_0_1.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			e := auxIntToInt64(v_1.AuxInt)
			if !(c < e) {
				continue
			}
			v.reset(OpAnd32)
			v0 := b.NewValue0(v.Pos, OpLsh32x64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, t2)
			v1.AuxInt = int64ToAuxInt(e - c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst32, t)
			v2.AuxInt = int32ToAuxInt(d << e)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (Lsh32x64 (And32 (Rsh32Ux64 <t> x (Const64 <t2> [c])) (Const32 [d])) (Const64 [e]))
	// cond: c < e
	// result: (And32 (Lsh32x64 <t> x (Const64 <t2> [e-c])) (Const32 <t> [d<<e]))
	for {
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRsh32Ux64 {
				continue
			}
			t := v_0_0.Type
			_ = v_0_0.Args[1]
			x := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			if v_0_0_1.Op != OpConst64 {
				continue
			}
			t2 := v_0_0_1.Type
			c := auxIntToInt64(v_0_0_1.AuxInt)
			if v_0_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_0_1.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			e := auxIntToInt64(v_1.AuxInt)
			if !(c < e) {
				continue
			}
			v.reset(OpAnd32)
			v0 := b.NewValue0(v.Pos, OpLsh32x64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, t2)
			v1.AuxInt = int64ToAuxInt(e - c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst32, t)
			v2.AuxInt = int32ToAuxInt(d << e)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpLsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh32x8 <t> x (Const8 [c]))
	// result: (Lsh32x64 x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		v.reset(OpLsh32x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint8(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh32x8 (Const32 [0]) _)
	// result: (Const32 [0])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh64x16 <t> x (Const16 [c]))
	// result: (Lsh64x64 x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		v.reset(OpLsh64x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint16(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh64x16 (Const64 [0]) _)
	// result: (Const64 [0])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh64x32 <t> x (Const32 [c]))
	// result: (Lsh64x64 x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpLsh64x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint32(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh64x32 (Const64 [0]) _)
	// result: (Const64 [0])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x64 (Const64 [c]) (Const64 [d]))
	// result: (Const64 [c << uint64(d)])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(c << uint64(d))
		return true
	}
	// match: (Lsh64x64 x (Const64 [0]))
	// result: x
	for {
		x := v_0
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Lsh64x64 (Const64 [0]) _)
	// result: (Const64 [0])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (Lsh64x64 _ (Const64 [c]))
	// cond: uint64(c) >= 64
	// result: (Const64 [0])
	for {
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 64) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (Lsh64x64 <t> (Lsh64x64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Lsh64x64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		if v_0.Op != OpLsh64x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpLsh64x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(c + d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh64x64 i:(Rsh64x64 x (Const64 [c])) (Const64 [c]))
	// cond: c >= 0 && c < 64 && i.Uses == 1
	// result: (And64 x (Const64 <v.Type> [int64(-1) << c]))
	for {
		i := v_0
		if i.Op != OpRsh64x64 {
			break
		}
		_ = i.Args[1]
		x := i.Args[0]
		i_1 := i.Args[1]
		if i_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(i_1.AuxInt)
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != c || !(c >= 0 && c < 64 && i.Uses == 1) {
			break
		}
		v.reset(OpAnd64)
		v0 := b.NewValue0(v.Pos, OpConst64, v.Type)
		v0.AuxInt = int64ToAuxInt(int64(-1) << c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh64x64 i:(Rsh64Ux64 x (Const64 [c])) (Const64 [c]))
	// cond: c >= 0 && c < 64 && i.Uses == 1
	// result: (And64 x (Const64 <v.Type> [int64(-1) << c]))
	for {
		i := v_0
		if i.Op != OpRsh64Ux64 {
			break
		}
		_ = i.Args[1]
		x := i.Args[0]
		i_1 := i.Args[1]
		if i_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(i_1.AuxInt)
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != c || !(c >= 0 && c < 64 && i.Uses == 1) {
			break
		}
		v.reset(OpAnd64)
		v0 := b.NewValue0(v.Pos, OpConst64, v.Type)
		v0.AuxInt = int64ToAuxInt(int64(-1) << c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh64x64 (Rsh64Ux64 (Lsh64x64 x (Const64 [c1])) (Const64 [c2])) (Const64 [c3]))
	// cond: uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)
	// result: (Lsh64x64 x (Const64 <typ.UInt64> [c1-c2+c3]))
	for {
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
		c1 := auxIntToInt64(v_0_0_1.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c2 := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		c3 := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)) {
			break
		}
		v.reset(OpLsh64x64)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c1 - c2 + c3)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh64x64 (And64 (Rsh64x64 <t> x (Const64 <t2> [c])) (Const64 [d])) (Const64 [e]))
	// cond: c >= e
	// result: (And64 (Rsh64x64 <t> x (Const64 <t2> [c-e])) (Const64 <t> [d<<e]))
	for {
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRsh64x64 {
				continue
			}
			t := v_0_0.Type
			_ = v_0_0.Args[1]
			x := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			if v_0_0_1.Op != OpConst64 {
				continue
			}
			t2 := v_0_0_1.Type
			c := auxIntToInt64(v_0_0_1.AuxInt)
			if v_0_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_0_1.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			e := auxIntToInt64(v_1.AuxInt)
			if !(c >= e) {
				continue
			}
			v.reset(OpAnd64)
			v0 := b.NewValue0(v.Pos, OpRsh64x64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, t2)
			v1.AuxInt = int64ToAuxInt(c - e)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst64, t)
			v2.AuxInt = int64ToAuxInt(d << e)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (Lsh64x64 (And64 (Rsh64Ux64 <t> x (Const64 <t2> [c])) (Const64 [d])) (Const64 [e]))
	// cond: c >= e
	// result: (And64 (Rsh64Ux64 <t> x (Const64 <t2> [c-e])) (Const64 <t> [d<<e]))
	for {
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRsh64Ux64 {
				continue
			}
			t := v_0_0.Type
			_ = v_0_0.Args[1]
			x := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			if v_0_0_1.Op != OpConst64 {
				continue
			}
			t2 := v_0_0_1.Type
			c := auxIntToInt64(v_0_0_1.AuxInt)
			if v_0_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_0_1.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			e := auxIntToInt64(v_1.AuxInt)
			if !(c >= e) {
				continue
			}
			v.reset(OpAnd64)
			v0 := b.NewValue0(v.Pos, OpRsh64Ux64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, t2)
			v1.AuxInt = int64ToAuxInt(c - e)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst64, t)
			v2.AuxInt = int64ToAuxInt(d << e)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (Lsh64x64 (And64 (Rsh64x64 <t> x (Const64 <t2> [c])) (Const64 [d])) (Const64 [e]))
	// cond: c < e
	// result: (And64 (Lsh64x64 <t> x (Const64 <t2> [e-c])) (Const64 <t> [d<<e]))
	for {
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRsh64x64 {
				continue
			}
			t := v_0_0.Type
			_ = v_0_0.Args[1]
			x := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			if v_0_0_1.Op != OpConst64 {
				continue
			}
			t2 := v_0_0_1.Type
			c := auxIntToInt64(v_0_0_1.AuxInt)
			if v_0_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_0_1.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			e := auxIntToInt64(v_1.AuxInt)
			if !(c < e) {
				continue
			}
			v.reset(OpAnd64)
			v0 := b.NewValue0(v.Pos, OpLsh64x64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, t2)
			v1.AuxInt = int64ToAuxInt(e - c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst64, t)
			v2.AuxInt = int64ToAuxInt(d << e)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (Lsh64x64 (And64 (Rsh64Ux64 <t> x (Const64 <t2> [c])) (Const64 [d])) (Const64 [e]))
	// cond: c < e
	// result: (And64 (Lsh64x64 <t> x (Const64 <t2> [e-c])) (Const64 <t> [d<<e]))
	for {
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRsh64Ux64 {
				continue
			}
			t := v_0_0.Type
			_ = v_0_0.Args[1]
			x := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			if v_0_0_1.Op != OpConst64 {
				continue
			}
			t2 := v_0_0_1.Type
			c := auxIntToInt64(v_0_0_1.AuxInt)
			if v_0_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_0_1.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			e := auxIntToInt64(v_1.AuxInt)
			if !(c < e) {
				continue
			}
			v.reset(OpAnd64)
			v0 := b.NewValue0(v.Pos, OpLsh64x64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, t2)
			v1.AuxInt = int64ToAuxInt(e - c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst64, t)
			v2.AuxInt = int64ToAuxInt(d << e)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpLsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh64x8 <t> x (Const8 [c]))
	// result: (Lsh64x64 x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		v.reset(OpLsh64x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint8(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh64x8 (Const64 [0]) _)
	// result: (Const64 [0])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh8x16 <t> x (Const16 [c]))
	// result: (Lsh8x64 x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		v.reset(OpLsh8x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint16(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh8x16 (Const8 [0]) _)
	// result: (Const8 [0])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh8x32 <t> x (Const32 [c]))
	// result: (Lsh8x64 x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpLsh8x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint32(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh8x32 (Const8 [0]) _)
	// result: (Const8 [0])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x64 (Const8 [c]) (Const64 [d]))
	// result: (Const8 [c << uint64(d)])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(c << uint64(d))
		return true
	}
	// match: (Lsh8x64 x (Const64 [0]))
	// result: x
	for {
		x := v_0
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Lsh8x64 (Const8 [0]) _)
	// result: (Const8 [0])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	// match: (Lsh8x64 _ (Const64 [c]))
	// cond: uint64(c) >= 8
	// result: (Const8 [0])
	for {
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 8) {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	// match: (Lsh8x64 <t> (Lsh8x64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Lsh8x64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		if v_0.Op != OpLsh8x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpLsh8x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(c + d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh8x64 i:(Rsh8x64 x (Const64 [c])) (Const64 [c]))
	// cond: c >= 0 && c < 8 && i.Uses == 1
	// result: (And8 x (Const8 <v.Type> [int8(-1) << c]))
	for {
		i := v_0
		if i.Op != OpRsh8x64 {
			break
		}
		_ = i.Args[1]
		x := i.Args[0]
		i_1 := i.Args[1]
		if i_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(i_1.AuxInt)
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != c || !(c >= 0 && c < 8 && i.Uses == 1) {
			break
		}
		v.reset(OpAnd8)
		v0 := b.NewValue0(v.Pos, OpConst8, v.Type)
		v0.AuxInt = int8ToAuxInt(int8(-1) << c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh8x64 i:(Rsh8Ux64 x (Const64 [c])) (Const64 [c]))
	// cond: c >= 0 && c < 8 && i.Uses == 1
	// result: (And8 x (Const8 <v.Type> [int8(-1) << c]))
	for {
		i := v_0
		if i.Op != OpRsh8Ux64 {
			break
		}
		_ = i.Args[1]
		x := i.Args[0]
		i_1 := i.Args[1]
		if i_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(i_1.AuxInt)
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != c || !(c >= 0 && c < 8 && i.Uses == 1) {
			break
		}
		v.reset(OpAnd8)
		v0 := b.NewValue0(v.Pos, OpConst8, v.Type)
		v0.AuxInt = int8ToAuxInt(int8(-1) << c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh8x64 (Rsh8Ux64 (Lsh8x64 x (Const64 [c1])) (Const64 [c2])) (Const64 [c3]))
	// cond: uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)
	// result: (Lsh8x64 x (Const64 <typ.UInt64> [c1-c2+c3]))
	for {
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
		c1 := auxIntToInt64(v_0_0_1.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c2 := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		c3 := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)) {
			break
		}
		v.reset(OpLsh8x64)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c1 - c2 + c3)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh8x64 (And8 (Rsh8x64 <t> x (Const64 <t2> [c])) (Const8 [d])) (Const64 [e]))
	// cond: c >= e
	// result: (And8 (Rsh8x64 <t> x (Const64 <t2> [c-e])) (Const8 <t> [d<<e]))
	for {
		if v_0.Op != OpAnd8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRsh8x64 {
				continue
			}
			t := v_0_0.Type
			_ = v_0_0.Args[1]
			x := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			if v_0_0_1.Op != OpConst64 {
				continue
			}
			t2 := v_0_0_1.Type
			c := auxIntToInt64(v_0_0_1.AuxInt)
			if v_0_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_0_1.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			e := auxIntToInt64(v_1.AuxInt)
			if !(c >= e) {
				continue
			}
			v.reset(OpAnd8)
			v0 := b.NewValue0(v.Pos, OpRsh8x64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, t2)
			v1.AuxInt = int64ToAuxInt(c - e)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst8, t)
			v2.AuxInt = int8ToAuxInt(d << e)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (Lsh8x64 (And8 (Rsh8Ux64 <t> x (Const64 <t2> [c])) (Const8 [d])) (Const64 [e]))
	// cond: c >= e
	// result: (And8 (Rsh8Ux64 <t> x (Const64 <t2> [c-e])) (Const8 <t> [d<<e]))
	for {
		if v_0.Op != OpAnd8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRsh8Ux64 {
				continue
			}
			t := v_0_0.Type
			_ = v_0_0.Args[1]
			x := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			if v_0_0_1.Op != OpConst64 {
				continue
			}
			t2 := v_0_0_1.Type
			c := auxIntToInt64(v_0_0_1.AuxInt)
			if v_0_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_0_1.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			e := auxIntToInt64(v_1.AuxInt)
			if !(c >= e) {
				continue
			}
			v.reset(OpAnd8)
			v0 := b.NewValue0(v.Pos, OpRsh8Ux64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, t2)
			v1.AuxInt = int64ToAuxInt(c - e)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst8, t)
			v2.AuxInt = int8ToAuxInt(d << e)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (Lsh8x64 (And8 (Rsh8x64 <t> x (Const64 <t2> [c])) (Const8 [d])) (Const64 [e]))
	// cond: c < e
	// result: (And8 (Lsh8x64 <t> x (Const64 <t2> [e-c])) (Const8 <t> [d<<e]))
	for {
		if v_0.Op != OpAnd8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRsh8x64 {
				continue
			}
			t := v_0_0.Type
			_ = v_0_0.Args[1]
			x := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			if v_0_0_1.Op != OpConst64 {
				continue
			}
			t2 := v_0_0_1.Type
			c := auxIntToInt64(v_0_0_1.AuxInt)
			if v_0_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_0_1.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			e := auxIntToInt64(v_1.AuxInt)
			if !(c < e) {
				continue
			}
			v.reset(OpAnd8)
			v0 := b.NewValue0(v.Pos, OpLsh8x64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, t2)
			v1.AuxInt = int64ToAuxInt(e - c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst8, t)
			v2.AuxInt = int8ToAuxInt(d << e)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (Lsh8x64 (And8 (Rsh8Ux64 <t> x (Const64 <t2> [c])) (Const8 [d])) (Const64 [e]))
	// cond: c < e
	// result: (And8 (Lsh8x64 <t> x (Const64 <t2> [e-c])) (Const8 <t> [d<<e]))
	for {
		if v_0.Op != OpAnd8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRsh8Ux64 {
				continue
			}
			t := v_0_0.Type
			_ = v_0_0.Args[1]
			x := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			if v_0_0_1.Op != OpConst64 {
				continue
			}
			t2 := v_0_0_1.Type
			c := auxIntToInt64(v_0_0_1.AuxInt)
			if v_0_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_0_1.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			e := auxIntToInt64(v_1.AuxInt)
			if !(c < e) {
				continue
			}
			v.reset(OpAnd8)
			v0 := b.NewValue0(v.Pos, OpLsh8x64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, t2)
			v1.AuxInt = int64ToAuxInt(e - c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpConst8, t)
			v2.AuxInt = int8ToAuxInt(d << e)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpLsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh8x8 <t> x (Const8 [c]))
	// result: (Lsh8x64 x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		v.reset(OpLsh8x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint8(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh8x8 (Const8 [0]) _)
	// result: (Const8 [0])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMod16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Mod16 (Const16 [c]) (Const16 [d]))
	// cond: d != 0
	// result: (Const16 [c % d])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if v_1.Op != OpConst16 {
			break
		}
		d := auxIntToInt16(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(c % d)
		return true
	}
	// match: (Mod16 <t> n (Const16 [c]))
	// cond: isNonNegative(n) && isPowerOfTwo(c)
	// result: (And16 n (Const16 <t> [c-1]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(isNonNegative(n) && isPowerOfTwo(c)) {
			break
		}
		v.reset(OpAnd16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(c - 1)
		v.AddArg2(n, v0)
		return true
	}
	// match: (Mod16 <t> n (Const16 [c]))
	// cond: c < 0 && c != -1<<15
	// result: (Mod16 <t> n (Const16 <t> [-c]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(c < 0 && c != -1<<15) {
			break
		}
		v.reset(OpMod16)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(-c)
		v.AddArg2(n, v0)
		return true
	}
	// match: (Mod16 <t> x (Const16 [c]))
	// cond: x.Op != OpConst16 && (c > 0 || c == -1<<15)
	// result: (Sub16 x (Mul16 <t> (Div16 <t> x (Const16 <t> [c])) (Const16 <t> [c])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(x.Op != OpConst16 && (c > 0 || c == -1<<15)) {
			break
		}
		v.reset(OpSub16)
		v0 := b.NewValue0(v.Pos, OpMul16, t)
		v1 := b.NewValue0(v.Pos, OpDiv16, t)
		v2 := b.NewValue0(v.Pos, OpConst16, t)
		v2.AuxInt = int16ToAuxInt(c)
		v1.AddArg2(x, v2)
		v0.AddArg2(v1, v2)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMod16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Mod16u (Const16 [c]) (Const16 [d]))
	// cond: d != 0
	// result: (Const16 [int16(uint16(c) % uint16(d))])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if v_1.Op != OpConst16 {
			break
		}
		d := auxIntToInt16(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(int16(uint16(c) % uint16(d)))
		return true
	}
	// match: (Mod16u <t> n (Const16 [c]))
	// cond: isPowerOfTwo(c)
	// result: (And16 n (Const16 <t> [c-1]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpAnd16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(c - 1)
		v.AddArg2(n, v0)
		return true
	}
	// match: (Mod16u <t> x (Const16 [c]))
	// cond: x.Op != OpConst16 && c > 0 && umagicOK16(c)
	// result: (Sub16 x (Mul16 <t> (Div16u <t> x (Const16 <t> [c])) (Const16 <t> [c])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(x.Op != OpConst16 && c > 0 && umagicOK16(c)) {
			break
		}
		v.reset(OpSub16)
		v0 := b.NewValue0(v.Pos, OpMul16, t)
		v1 := b.NewValue0(v.Pos, OpDiv16u, t)
		v2 := b.NewValue0(v.Pos, OpConst16, t)
		v2.AuxInt = int16ToAuxInt(c)
		v1.AddArg2(x, v2)
		v0.AddArg2(v1, v2)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMod32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Mod32 (Const32 [c]) (Const32 [d]))
	// cond: d != 0
	// result: (Const32 [c % d])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if v_1.Op != OpConst32 {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(c % d)
		return true
	}
	// match: (Mod32 <t> n (Const32 [c]))
	// cond: isNonNegative(n) && isPowerOfTwo(c)
	// result: (And32 n (Const32 <t> [c-1]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(isNonNegative(n) && isPowerOfTwo(c)) {
			break
		}
		v.reset(OpAnd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(c - 1)
		v.AddArg2(n, v0)
		return true
	}
	// match: (Mod32 <t> n (Const32 [c]))
	// cond: c < 0 && c != -1<<31
	// result: (Mod32 <t> n (Const32 <t> [-c]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(c < 0 && c != -1<<31) {
			break
		}
		v.reset(OpMod32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(-c)
		v.AddArg2(n, v0)
		return true
	}
	// match: (Mod32 <t> x (Const32 [c]))
	// cond: x.Op != OpConst32 && (c > 0 || c == -1<<31)
	// result: (Sub32 x (Mul32 <t> (Div32 <t> x (Const32 <t> [c])) (Const32 <t> [c])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(x.Op != OpConst32 && (c > 0 || c == -1<<31)) {
			break
		}
		v.reset(OpSub32)
		v0 := b.NewValue0(v.Pos, OpMul32, t)
		v1 := b.NewValue0(v.Pos, OpDiv32, t)
		v2 := b.NewValue0(v.Pos, OpConst32, t)
		v2.AuxInt = int32ToAuxInt(c)
		v1.AddArg2(x, v2)
		v0.AddArg2(v1, v2)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMod32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Mod32u (Const32 [c]) (Const32 [d]))
	// cond: d != 0
	// result: (Const32 [int32(uint32(c) % uint32(d))])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if v_1.Op != OpConst32 {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(uint32(c) % uint32(d)))
		return true
	}
	// match: (Mod32u <t> n (Const32 [c]))
	// cond: isPowerOfTwo(c)
	// result: (And32 n (Const32 <t> [c-1]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpAnd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(c - 1)
		v.AddArg2(n, v0)
		return true
	}
	// match: (Mod32u <t> x (Const32 [c]))
	// cond: x.Op != OpConst32 && c > 0 && umagicOK32(c)
	// result: (Sub32 x (Mul32 <t> (Div32u <t> x (Const32 <t> [c])) (Const32 <t> [c])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(x.Op != OpConst32 && c > 0 && umagicOK32(c)) {
			break
		}
		v.reset(OpSub32)
		v0 := b.NewValue0(v.Pos, OpMul32, t)
		v1 := b.NewValue0(v.Pos, OpDiv32u, t)
		v2 := b.NewValue0(v.Pos, OpConst32, t)
		v2.AuxInt = int32ToAuxInt(c)
		v1.AddArg2(x, v2)
		v0.AddArg2(v1, v2)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMod64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Mod64 (Const64 [c]) (Const64 [d]))
	// cond: d != 0
	// result: (Const64 [c % d])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(c % d)
		return true
	}
	// match: (Mod64 <t> n (Const64 [c]))
	// cond: isNonNegative(n) && isPowerOfTwo(c)
	// result: (And64 n (Const64 <t> [c-1]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isNonNegative(n) && isPowerOfTwo(c)) {
			break
		}
		v.reset(OpAnd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(c - 1)
		v.AddArg2(n, v0)
		return true
	}
	// match: (Mod64 n (Const64 [-1<<63]))
	// cond: isNonNegative(n)
	// result: n
	for {
		n := v_0
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != -1<<63 || !(isNonNegative(n)) {
			break
		}
		v.copyOf(n)
		return true
	}
	// match: (Mod64 <t> n (Const64 [c]))
	// cond: c < 0 && c != -1<<63
	// result: (Mod64 <t> n (Const64 <t> [-c]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(c < 0 && c != -1<<63) {
			break
		}
		v.reset(OpMod64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(-c)
		v.AddArg2(n, v0)
		return true
	}
	// match: (Mod64 <t> x (Const64 [c]))
	// cond: x.Op != OpConst64 && (c > 0 || c == -1<<63)
	// result: (Sub64 x (Mul64 <t> (Div64 <t> x (Const64 <t> [c])) (Const64 <t> [c])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(x.Op != OpConst64 && (c > 0 || c == -1<<63)) {
			break
		}
		v.reset(OpSub64)
		v0 := b.NewValue0(v.Pos, OpMul64, t)
		v1 := b.NewValue0(v.Pos, OpDiv64, t)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(c)
		v1.AddArg2(x, v2)
		v0.AddArg2(v1, v2)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMod64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Mod64u (Const64 [c]) (Const64 [d]))
	// cond: d != 0
	// result: (Const64 [int64(uint64(c) % uint64(d))])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) % uint64(d)))
		return true
	}
	// match: (Mod64u <t> n (Const64 [c]))
	// cond: isPowerOfTwo(c)
	// result: (And64 n (Const64 <t> [c-1]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpAnd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(c - 1)
		v.AddArg2(n, v0)
		return true
	}
	// match: (Mod64u <t> n (Const64 [-1<<63]))
	// result: (And64 n (Const64 <t> [1<<63-1]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != -1<<63 {
			break
		}
		v.reset(OpAnd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(1<<63 - 1)
		v.AddArg2(n, v0)
		return true
	}
	// match: (Mod64u <t> x (Const64 [c]))
	// cond: x.Op != OpConst64 && c > 0 && umagicOK64(c)
	// result: (Sub64 x (Mul64 <t> (Div64u <t> x (Const64 <t> [c])) (Const64 <t> [c])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(x.Op != OpConst64 && c > 0 && umagicOK64(c)) {
			break
		}
		v.reset(OpSub64)
		v0 := b.NewValue0(v.Pos, OpMul64, t)
		v1 := b.NewValue0(v.Pos, OpDiv64u, t)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(c)
		v1.AddArg2(x, v2)
		v0.AddArg2(v1, v2)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMod8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Mod8 (Const8 [c]) (Const8 [d]))
	// cond: d != 0
	// result: (Const8 [c % d])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if v_1.Op != OpConst8 {
			break
		}
		d := auxIntToInt8(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(c % d)
		return true
	}
	// match: (Mod8 <t> n (Const8 [c]))
	// cond: isNonNegative(n) && isPowerOfTwo(c)
	// result: (And8 n (Const8 <t> [c-1]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		if !(isNonNegative(n) && isPowerOfTwo(c)) {
			break
		}
		v.reset(OpAnd8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(c - 1)
		v.AddArg2(n, v0)
		return true
	}
	// match: (Mod8 <t> n (Const8 [c]))
	// cond: c < 0 && c != -1<<7
	// result: (Mod8 <t> n (Const8 <t> [-c]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		if !(c < 0 && c != -1<<7) {
			break
		}
		v.reset(OpMod8)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(-c)
		v.AddArg2(n, v0)
		return true
	}
	// match: (Mod8 <t> x (Const8 [c]))
	// cond: x.Op != OpConst8 && (c > 0 || c == -1<<7)
	// result: (Sub8 x (Mul8 <t> (Div8 <t> x (Const8 <t> [c])) (Const8 <t> [c])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		if !(x.Op != OpConst8 && (c > 0 || c == -1<<7)) {
			break
		}
		v.reset(OpSub8)
		v0 := b.NewValue0(v.Pos, OpMul8, t)
		v1 := b.NewValue0(v.Pos, OpDiv8, t)
		v2 := b.NewValue0(v.Pos, OpConst8, t)
		v2.AuxInt = int8ToAuxInt(c)
		v1.AddArg2(x, v2)
		v0.AddArg2(v1, v2)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMod8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Mod8u (Const8 [c]) (Const8 [d]))
	// cond: d != 0
	// result: (Const8 [int8(uint8(c) % uint8(d))])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if v_1.Op != OpConst8 {
			break
		}
		d := auxIntToInt8(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(int8(uint8(c) % uint8(d)))
		return true
	}
	// match: (Mod8u <t> n (Const8 [c]))
	// cond: isPowerOfTwo(c)
	// result: (And8 n (Const8 <t> [c-1]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpAnd8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(c - 1)
		v.AddArg2(n, v0)
		return true
	}
	// match: (Mod8u <t> x (Const8 [c]))
	// cond: x.Op != OpConst8 && c > 0 && umagicOK8( c)
	// result: (Sub8 x (Mul8 <t> (Div8u <t> x (Const8 <t> [c])) (Const8 <t> [c])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		if !(x.Op != OpConst8 && c > 0 && umagicOK8(c)) {
			break
		}
		v.reset(OpSub8)
		v0 := b.NewValue0(v.Pos, OpMul8, t)
		v1 := b.NewValue0(v.Pos, OpDiv8u, t)
		v2 := b.NewValue0(v.Pos, OpConst8, t)
		v2.AuxInt = int8ToAuxInt(c)
		v1.AddArg2(x, v2)
		v0.AddArg2(v1, v2)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMove(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (Move {t} [n] dst1 src mem:(Zero {t} [n] dst2 _))
	// cond: isSamePtr(src, dst2)
	// result: (Zero {t} [n] dst1 mem)
	for {
		n := auxIntToInt64(v.AuxInt)
		t := auxToType(v.Aux)
		dst1 := v_0
		src := v_1
		mem := v_2
		if mem.Op != OpZero || auxIntToInt64(mem.AuxInt) != n || auxToType(mem.Aux) != t {
			break
		}
		dst2 := mem.Args[0]
		if !(isSamePtr(src, dst2)) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = int64ToAuxInt(n)
		v.Aux = typeToAux(t)
		v.AddArg2(dst1, mem)
		return true
	}
	// match: (Move {t} [n] dst1 src mem:(VarDef (Zero {t} [n] dst0 _)))
	// cond: isSamePtr(src, dst0)
	// result: (Zero {t} [n] dst1 mem)
	for {
		n := auxIntToInt64(v.AuxInt)
		t := auxToType(v.Aux)
		dst1 := v_0
		src := v_1
		mem := v_2
		if mem.Op != OpVarDef {
			break
		}
		mem_0 := mem.Args[0]
		if mem_0.Op != OpZero || auxIntToInt64(mem_0.AuxInt) != n || auxToType(mem_0.Aux) != t {
			break
		}
		dst0 := mem_0.Args[0]
		if !(isSamePtr(src, dst0)) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = int64ToAuxInt(n)
		v.Aux = typeToAux(t)
		v.AddArg2(dst1, mem)
		return true
	}
	// match: (Move {t} [n] dst (Addr {sym} (SB)) mem)
	// cond: symIsROZero(sym)
	// result: (Zero {t} [n] dst mem)
	for {
		n := auxIntToInt64(v.AuxInt)
		t := auxToType(v.Aux)
		dst := v_0
		if v_1.Op != OpAddr {
			break
		}
		sym := auxToSym(v_1.Aux)
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpSB {
			break
		}
		mem := v_2
		if !(symIsROZero(sym)) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = int64ToAuxInt(n)
		v.Aux = typeToAux(t)
		v.AddArg2(dst, mem)
		return true
	}
	// match: (Move {t1} [n] dst1 src1 store:(Store {t2} op:(OffPtr [o2] dst2) _ mem))
	// cond: isSamePtr(dst1, dst2) && store.Uses == 1 && n >= o2 + t2.Size() && disjoint(src1, n, op, t2.Size()) && clobber(store)
	// result: (Move {t1} [n] dst1 src1 mem)
	for {
		n := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		dst1 := v_0
		src1 := v_1
		store := v_2
		if store.Op != OpStore {
			break
		}
		t2 := auxToType(store.Aux)
		mem := store.Args[2]
		op := store.Args[0]
		if op.Op != OpOffPtr {
			break
		}
		o2 := auxIntToInt64(op.AuxInt)
		dst2 := op.Args[0]
		if !(isSamePtr(dst1, dst2) && store.Uses == 1 && n >= o2+t2.Size() && disjoint(src1, n, op, t2.Size()) && clobber(store)) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(n)
		v.Aux = typeToAux(t1)
		v.AddArg3(dst1, src1, mem)
		return true
	}
	// match: (Move {t} [n] dst1 src1 move:(Move {t} [n] dst2 _ mem))
	// cond: move.Uses == 1 && isSamePtr(dst1, dst2) && disjoint(src1, n, dst2, n) && clobber(move)
	// result: (Move {t} [n] dst1 src1 mem)
	for {
		n := auxIntToInt64(v.AuxInt)
		t := auxToType(v.Aux)
		dst1 := v_0
		src1 := v_1
		move := v_2
		if move.Op != OpMove || auxIntToInt64(move.AuxInt) != n || auxToType(move.Aux) != t {
			break
		}
		mem := move.Args[2]
		dst2 := move.Args[0]
		if !(move.Uses == 1 && isSamePtr(dst1, dst2) && disjoint(src1, n, dst2, n) && clobber(move)) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(n)
		v.Aux = typeToAux(t)
		v.AddArg3(dst1, src1, mem)
		return true
	}
	// match: (Move {t} [n] dst1 src1 vardef:(VarDef {x} move:(Move {t} [n] dst2 _ mem)))
	// cond: move.Uses == 1 && vardef.Uses == 1 && isSamePtr(dst1, dst2) && disjoint(src1, n, dst2, n) && clobber(move, vardef)
	// result: (Move {t} [n] dst1 src1 (VarDef {x} mem))
	for {
		n := auxIntToInt64(v.AuxInt)
		t := auxToType(v.Aux)
		dst1 := v_0
		src1 := v_1
		vardef := v_2
		if vardef.Op != OpVarDef {
			break
		}
		x := auxToSym(vardef.Aux)
		move := vardef.Args[0]
		if move.Op != OpMove || auxIntToInt64(move.AuxInt) != n || auxToType(move.Aux) != t {
			break
		}
		mem := move.Args[2]
		dst2 := move.Args[0]
		if !(move.Uses == 1 && vardef.Uses == 1 && isSamePtr(dst1, dst2) && disjoint(src1, n, dst2, n) && clobber(move, vardef)) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(n)
		v.Aux = typeToAux(t)
		v0 := b.NewValue0(v.Pos, OpVarDef, types.TypeMem)
		v0.Aux = symToAux(x)
		v0.AddArg(mem)
		v.AddArg3(dst1, src1, v0)
		return true
	}
	// match: (Move {t} [n] dst1 src1 zero:(Zero {t} [n] dst2 mem))
	// cond: zero.Uses == 1 && isSamePtr(dst1, dst2) && disjoint(src1, n, dst2, n) && clobber(zero)
	// result: (Move {t} [n] dst1 src1 mem)
	for {
		n := auxIntToInt64(v.AuxInt)
		t := auxToType(v.Aux)
		dst1 := v_0
		src1 := v_1
		zero := v_2
		if zero.Op != OpZero || auxIntToInt64(zero.AuxInt) != n || auxToType(zero.Aux) != t {
			break
		}
		mem := zero.Args[1]
		dst2 := zero.Args[0]
		if !(zero.Uses == 1 && isSamePtr(dst1, dst2) && disjoint(src1, n, dst2, n) && clobber(zero)) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(n)
		v.Aux = typeToAux(t)
		v.AddArg3(dst1, src1, mem)
		return true
	}
	// match: (Move {t} [n] dst1 src1 vardef:(VarDef {x} zero:(Zero {t} [n] dst2 mem)))
	// cond: zero.Uses == 1 && vardef.Uses == 1 && isSamePtr(dst1, dst2) && disjoint(src1, n, dst2, n) && clobber(zero, vardef)
	// result: (Move {t} [n] dst1 src1 (VarDef {x} mem))
	for {
		n := auxIntToInt64(v.AuxInt)
		t := auxToType(v.Aux)
		dst1 := v_0
		src1 := v_1
		vardef := v_2
		if vardef.Op != OpVarDef {
			break
		}
		x := auxToSym(vardef.Aux)
		zero := vardef.Args[0]
		if zero.Op != OpZero || auxIntToInt64(zero.AuxInt) != n || auxToType(zero.Aux) != t {
			break
		}
		mem := zero.Args[1]
		dst2 := zero.Args[0]
		if !(zero.Uses == 1 && vardef.Uses == 1 && isSamePtr(dst1, dst2) && disjoint(src1, n, dst2, n) && clobber(zero, vardef)) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(n)
		v.Aux = typeToAux(t)
		v0 := b.NewValue0(v.Pos, OpVarDef, types.TypeMem)
		v0.Aux = symToAux(x)
		v0.AddArg(mem)
		v.AddArg3(dst1, src1, v0)
		return true
	}
	// match: (Move {t1} [n] dst p1 mem:(Store {t2} op2:(OffPtr <tt2> [o2] p2) d1 (Store {t3} op3:(OffPtr <tt3> [0] p3) d2 _)))
	// cond: isSamePtr(p1, p2) && isSamePtr(p2, p3) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && o2 == t3.Size() && n == t2.Size() + t3.Size()
	// result: (Store {t2} (OffPtr <tt2> [o2] dst) d1 (Store {t3} (OffPtr <tt3> [0] dst) d2 mem))
	for {
		n := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		dst := v_0
		p1 := v_1
		mem := v_2
		if mem.Op != OpStore {
			break
		}
		t2 := auxToType(mem.Aux)
		_ = mem.Args[2]
		op2 := mem.Args[0]
		if op2.Op != OpOffPtr {
			break
		}
		tt2 := op2.Type
		o2 := auxIntToInt64(op2.AuxInt)
		p2 := op2.Args[0]
		d1 := mem.Args[1]
		mem_2 := mem.Args[2]
		if mem_2.Op != OpStore {
			break
		}
		t3 := auxToType(mem_2.Aux)
		d2 := mem_2.Args[1]
		op3 := mem_2.Args[0]
		if op3.Op != OpOffPtr {
			break
		}
		tt3 := op3.Type
		if auxIntToInt64(op3.AuxInt) != 0 {
			break
		}
		p3 := op3.Args[0]
		if !(isSamePtr(p1, p2) && isSamePtr(p2, p3) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && o2 == t3.Size() && n == t2.Size()+t3.Size()) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t2)
		v0 := b.NewValue0(v.Pos, OpOffPtr, tt2)
		v0.AuxInt = int64ToAuxInt(o2)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t3)
		v2 := b.NewValue0(v.Pos, OpOffPtr, tt3)
		v2.AuxInt = int64ToAuxInt(0)
		v2.AddArg(dst)
		v1.AddArg3(v2, d2, mem)
		v.AddArg3(v0, d1, v1)
		return true
	}
	// match: (Move {t1} [n] dst p1 mem:(Store {t2} op2:(OffPtr <tt2> [o2] p2) d1 (Store {t3} op3:(OffPtr <tt3> [o3] p3) d2 (Store {t4} op4:(OffPtr <tt4> [0] p4) d3 _))))
	// cond: isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && registerizable(b, t4) && o3 == t4.Size() && o2-o3 == t3.Size() && n == t2.Size() + t3.Size() + t4.Size()
	// result: (Store {t2} (OffPtr <tt2> [o2] dst) d1 (Store {t3} (OffPtr <tt3> [o3] dst) d2 (Store {t4} (OffPtr <tt4> [0] dst) d3 mem)))
	for {
		n := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		dst := v_0
		p1 := v_1
		mem := v_2
		if mem.Op != OpStore {
			break
		}
		t2 := auxToType(mem.Aux)
		_ = mem.Args[2]
		op2 := mem.Args[0]
		if op2.Op != OpOffPtr {
			break
		}
		tt2 := op2.Type
		o2 := auxIntToInt64(op2.AuxInt)
		p2 := op2.Args[0]
		d1 := mem.Args[1]
		mem_2 := mem.Args[2]
		if mem_2.Op != OpStore {
			break
		}
		t3 := auxToType(mem_2.Aux)
		_ = mem_2.Args[2]
		op3 := mem_2.Args[0]
		if op3.Op != OpOffPtr {
			break
		}
		tt3 := op3.Type
		o3 := auxIntToInt64(op3.AuxInt)
		p3 := op3.Args[0]
		d2 := mem_2.Args[1]
		mem_2_2 := mem_2.Args[2]
		if mem_2_2.Op != OpStore {
			break
		}
		t4 := auxToType(mem_2_2.Aux)
		d3 := mem_2_2.Args[1]
		op4 := mem_2_2.Args[0]
		if op4.Op != OpOffPtr {
			break
		}
		tt4 := op4.Type
		if auxIntToInt64(op4.AuxInt) != 0 {
			break
		}
		p4 := op4.Args[0]
		if !(isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && registerizable(b, t4) && o3 == t4.Size() && o2-o3 == t3.Size() && n == t2.Size()+t3.Size()+t4.Size()) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t2)
		v0 := b.NewValue0(v.Pos, OpOffPtr, tt2)
		v0.AuxInt = int64ToAuxInt(o2)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t3)
		v2 := b.NewValue0(v.Pos, OpOffPtr, tt3)
		v2.AuxInt = int64ToAuxInt(o3)
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v3.Aux = typeToAux(t4)
		v4 := b.NewValue0(v.Pos, OpOffPtr, tt4)
		v4.AuxInt = int64ToAuxInt(0)
		v4.AddArg(dst)
		v3.AddArg3(v4, d3, mem)
		v1.AddArg3(v2, d2, v3)
		v.AddArg3(v0, d1, v1)
		return true
	}
	// match: (Move {t1} [n] dst p1 mem:(Store {t2} op2:(OffPtr <tt2> [o2] p2) d1 (Store {t3} op3:(OffPtr <tt3> [o3] p3) d2 (Store {t4} op4:(OffPtr <tt4> [o4] p4) d3 (Store {t5} op5:(OffPtr <tt5> [0] p5) d4 _)))))
	// cond: isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && isSamePtr(p4, p5) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && t5.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && registerizable(b, t4) && registerizable(b, t5) && o4 == t5.Size() && o3-o4 == t4.Size() && o2-o3 == t3.Size() && n == t2.Size() + t3.Size() + t4.Size() + t5.Size()
	// result: (Store {t2} (OffPtr <tt2> [o2] dst) d1 (Store {t3} (OffPtr <tt3> [o3] dst) d2 (Store {t4} (OffPtr <tt4> [o4] dst) d3 (Store {t5} (OffPtr <tt5> [0] dst) d4 mem))))
	for {
		n := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		dst := v_0
		p1 := v_1
		mem := v_2
		if mem.Op != OpStore {
			break
		}
		t2 := auxToType(mem.Aux)
		_ = mem.Args[2]
		op2 := mem.Args[0]
		if op2.Op != OpOffPtr {
			break
		}
		tt2 := op2.Type
		o2 := auxIntToInt64(op2.AuxInt)
		p2 := op2.Args[0]
		d1 := mem.Args[1]
		mem_2 := mem.Args[2]
		if mem_2.Op != OpStore {
			break
		}
		t3 := auxToType(mem_2.Aux)
		_ = mem_2.Args[2]
		op3 := mem_2.Args[0]
		if op3.Op != OpOffPtr {
			break
		}
		tt3 := op3.Type
		o3 := auxIntToInt64(op3.AuxInt)
		p3 := op3.Args[0]
		d2 := mem_2.Args[1]
		mem_2_2 := mem_2.Args[2]
		if mem_2_2.Op != OpStore {
			break
		}
		t4 := auxToType(mem_2_2.Aux)
		_ = mem_2_2.Args[2]
		op4 := mem_2_2.Args[0]
		if op4.Op != OpOffPtr {
			break
		}
		tt4 := op4.Type
		o4 := auxIntToInt64(op4.AuxInt)
		p4 := op4.Args[0]
		d3 := mem_2_2.Args[1]
		mem_2_2_2 := mem_2_2.Args[2]
		if mem_2_2_2.Op != OpStore {
			break
		}
		t5 := auxToType(mem_2_2_2.Aux)
		d4 := mem_2_2_2.Args[1]
		op5 := mem_2_2_2.Args[0]
		if op5.Op != OpOffPtr {
			break
		}
		tt5 := op5.Type
		if auxIntToInt64(op5.AuxInt) != 0 {
			break
		}
		p5 := op5.Args[0]
		if !(isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && isSamePtr(p4, p5) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && t5.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && registerizable(b, t4) && registerizable(b, t5) && o4 == t5.Size() && o3-o4 == t4.Size() && o2-o3 == t3.Size() && n == t2.Size()+t3.Size()+t4.Size()+t5.Size()) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t2)
		v0 := b.NewValue0(v.Pos, OpOffPtr, tt2)
		v0.AuxInt = int64ToAuxInt(o2)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t3)
		v2 := b.NewValue0(v.Pos, OpOffPtr, tt3)
		v2.AuxInt = int64ToAuxInt(o3)
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v3.Aux = typeToAux(t4)
		v4 := b.NewValue0(v.Pos, OpOffPtr, tt4)
		v4.AuxInt = int64ToAuxInt(o4)
		v4.AddArg(dst)
		v5 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v5.Aux = typeToAux(t5)
		v6 := b.NewValue0(v.Pos, OpOffPtr, tt5)
		v6.AuxInt = int64ToAuxInt(0)
		v6.AddArg(dst)
		v5.AddArg3(v6, d4, mem)
		v3.AddArg3(v4, d3, v5)
		v1.AddArg3(v2, d2, v3)
		v.AddArg3(v0, d1, v1)
		return true
	}
	// match: (Move {t1} [n] dst p1 mem:(VarDef (Store {t2} op2:(OffPtr <tt2> [o2] p2) d1 (Store {t3} op3:(OffPtr <tt3> [0] p3) d2 _))))
	// cond: isSamePtr(p1, p2) && isSamePtr(p2, p3) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && o2 == t3.Size() && n == t2.Size() + t3.Size()
	// result: (Store {t2} (OffPtr <tt2> [o2] dst) d1 (Store {t3} (OffPtr <tt3> [0] dst) d2 mem))
	for {
		n := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		dst := v_0
		p1 := v_1
		mem := v_2
		if mem.Op != OpVarDef {
			break
		}
		mem_0 := mem.Args[0]
		if mem_0.Op != OpStore {
			break
		}
		t2 := auxToType(mem_0.Aux)
		_ = mem_0.Args[2]
		op2 := mem_0.Args[0]
		if op2.Op != OpOffPtr {
			break
		}
		tt2 := op2.Type
		o2 := auxIntToInt64(op2.AuxInt)
		p2 := op2.Args[0]
		d1 := mem_0.Args[1]
		mem_0_2 := mem_0.Args[2]
		if mem_0_2.Op != OpStore {
			break
		}
		t3 := auxToType(mem_0_2.Aux)
		d2 := mem_0_2.Args[1]
		op3 := mem_0_2.Args[0]
		if op3.Op != OpOffPtr {
			break
		}
		tt3 := op3.Type
		if auxIntToInt64(op3.AuxInt) != 0 {
			break
		}
		p3 := op3.Args[0]
		if !(isSamePtr(p1, p2) && isSamePtr(p2, p3) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && o2 == t3.Size() && n == t2.Size()+t3.Size()) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t2)
		v0 := b.NewValue0(v.Pos, OpOffPtr, tt2)
		v0.AuxInt = int64ToAuxInt(o2)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t3)
		v2 := b.NewValue0(v.Pos, OpOffPtr, tt3)
		v2.AuxInt = int64ToAuxInt(0)
		v2.AddArg(dst)
		v1.AddArg3(v2, d2, mem)
		v.AddArg3(v0, d1, v1)
		return true
	}
	// match: (Move {t1} [n] dst p1 mem:(VarDef (Store {t2} op2:(OffPtr <tt2> [o2] p2) d1 (Store {t3} op3:(OffPtr <tt3> [o3] p3) d2 (Store {t4} op4:(OffPtr <tt4> [0] p4) d3 _)))))
	// cond: isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && registerizable(b, t4) && o3 == t4.Size() && o2-o3 == t3.Size() && n == t2.Size() + t3.Size() + t4.Size()
	// result: (Store {t2} (OffPtr <tt2> [o2] dst) d1 (Store {t3} (OffPtr <tt3> [o3] dst) d2 (Store {t4} (OffPtr <tt4> [0] dst) d3 mem)))
	for {
		n := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		dst := v_0
		p1 := v_1
		mem := v_2
		if mem.Op != OpVarDef {
			break
		}
		mem_0 := mem.Args[0]
		if mem_0.Op != OpStore {
			break
		}
		t2 := auxToType(mem_0.Aux)
		_ = mem_0.Args[2]
		op2 := mem_0.Args[0]
		if op2.Op != OpOffPtr {
			break
		}
		tt2 := op2.Type
		o2 := auxIntToInt64(op2.AuxInt)
		p2 := op2.Args[0]
		d1 := mem_0.Args[1]
		mem_0_2 := mem_0.Args[2]
		if mem_0_2.Op != OpStore {
			break
		}
		t3 := auxToType(mem_0_2.Aux)
		_ = mem_0_2.Args[2]
		op3 := mem_0_2.Args[0]
		if op3.Op != OpOffPtr {
			break
		}
		tt3 := op3.Type
		o3 := auxIntToInt64(op3.AuxInt)
		p3 := op3.Args[0]
		d2 := mem_0_2.Args[1]
		mem_0_2_2 := mem_0_2.Args[2]
		if mem_0_2_2.Op != OpStore {
			break
		}
		t4 := auxToType(mem_0_2_2.Aux)
		d3 := mem_0_2_2.Args[1]
		op4 := mem_0_2_2.Args[0]
		if op4.Op != OpOffPtr {
			break
		}
		tt4 := op4.Type
		if auxIntToInt64(op4.AuxInt) != 0 {
			break
		}
		p4 := op4.Args[0]
		if !(isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && registerizable(b, t4) && o3 == t4.Size() && o2-o3 == t3.Size() && n == t2.Size()+t3.Size()+t4.Size()) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t2)
		v0 := b.NewValue0(v.Pos, OpOffPtr, tt2)
		v0.AuxInt = int64ToAuxInt(o2)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t3)
		v2 := b.NewValue0(v.Pos, OpOffPtr, tt3)
		v2.AuxInt = int64ToAuxInt(o3)
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v3.Aux = typeToAux(t4)
		v4 := b.NewValue0(v.Pos, OpOffPtr, tt4)
		v4.AuxInt = int64ToAuxInt(0)
		v4.AddArg(dst)
		v3.AddArg3(v4, d3, mem)
		v1.AddArg3(v2, d2, v3)
		v.AddArg3(v0, d1, v1)
		return true
	}
	// match: (Move {t1} [n] dst p1 mem:(VarDef (Store {t2} op2:(OffPtr <tt2> [o2] p2) d1 (Store {t3} op3:(OffPtr <tt3> [o3] p3) d2 (Store {t4} op4:(OffPtr <tt4> [o4] p4) d3 (Store {t5} op5:(OffPtr <tt5> [0] p5) d4 _))))))
	// cond: isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && isSamePtr(p4, p5) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && t5.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && registerizable(b, t4) && registerizable(b, t5) && o4 == t5.Size() && o3-o4 == t4.Size() && o2-o3 == t3.Size() && n == t2.Size() + t3.Size() + t4.Size() + t5.Size()
	// result: (Store {t2} (OffPtr <tt2> [o2] dst) d1 (Store {t3} (OffPtr <tt3> [o3] dst) d2 (Store {t4} (OffPtr <tt4> [o4] dst) d3 (Store {t5} (OffPtr <tt5> [0] dst) d4 mem))))
	for {
		n := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		dst := v_0
		p1 := v_1
		mem := v_2
		if mem.Op != OpVarDef {
			break
		}
		mem_0 := mem.Args[0]
		if mem_0.Op != OpStore {
			break
		}
		t2 := auxToType(mem_0.Aux)
		_ = mem_0.Args[2]
		op2 := mem_0.Args[0]
		if op2.Op != OpOffPtr {
			break
		}
		tt2 := op2.Type
		o2 := auxIntToInt64(op2.AuxInt)
		p2 := op2.Args[0]
		d1 := mem_0.Args[1]
		mem_0_2 := mem_0.Args[2]
		if mem_0_2.Op != OpStore {
			break
		}
		t3 := auxToType(mem_0_2.Aux)
		_ = mem_0_2.Args[2]
		op3 := mem_0_2.Args[0]
		if op3.Op != OpOffPtr {
			break
		}
		tt3 := op3.Type
		o3 := auxIntToInt64(op3.AuxInt)
		p3 := op3.Args[0]
		d2 := mem_0_2.Args[1]
		mem_0_2_2 := mem_0_2.Args[2]
		if mem_0_2_2.Op != OpStore {
			break
		}
		t4 := auxToType(mem_0_2_2.Aux)
		_ = mem_0_2_2.Args[2]
		op4 := mem_0_2_2.Args[0]
		if op4.Op != OpOffPtr {
			break
		}
		tt4 := op4.Type
		o4 := auxIntToInt64(op4.AuxInt)
		p4 := op4.Args[0]
		d3 := mem_0_2_2.Args[1]
		mem_0_2_2_2 := mem_0_2_2.Args[2]
		if mem_0_2_2_2.Op != OpStore {
			break
		}
		t5 := auxToType(mem_0_2_2_2.Aux)
		d4 := mem_0_2_2_2.Args[1]
		op5 := mem_0_2_2_2.Args[0]
		if op5.Op != OpOffPtr {
			break
		}
		tt5 := op5.Type
		if auxIntToInt64(op5.AuxInt) != 0 {
			break
		}
		p5 := op5.Args[0]
		if !(isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && isSamePtr(p4, p5) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && t5.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && registerizable(b, t4) && registerizable(b, t5) && o4 == t5.Size() && o3-o4 == t4.Size() && o2-o3 == t3.Size() && n == t2.Size()+t3.Size()+t4.Size()+t5.Size()) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t2)
		v0 := b.NewValue0(v.Pos, OpOffPtr, tt2)
		v0.AuxInt = int64ToAuxInt(o2)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t3)
		v2 := b.NewValue0(v.Pos, OpOffPtr, tt3)
		v2.AuxInt = int64ToAuxInt(o3)
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v3.Aux = typeToAux(t4)
		v4 := b.NewValue0(v.Pos, OpOffPtr, tt4)
		v4.AuxInt = int64ToAuxInt(o4)
		v4.AddArg(dst)
		v5 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v5.Aux = typeToAux(t5)
		v6 := b.NewValue0(v.Pos, OpOffPtr, tt5)
		v6.AuxInt = int64ToAuxInt(0)
		v6.AddArg(dst)
		v5.AddArg3(v6, d4, mem)
		v3.AddArg3(v4, d3, v5)
		v1.AddArg3(v2, d2, v3)
		v.AddArg3(v0, d1, v1)
		return true
	}
	// match: (Move {t1} [n] dst p1 mem:(Store {t2} op2:(OffPtr <tt2> [o2] p2) d1 (Zero {t3} [n] p3 _)))
	// cond: isSamePtr(p1, p2) && isSamePtr(p2, p3) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && registerizable(b, t2) && n >= o2 + t2.Size()
	// result: (Store {t2} (OffPtr <tt2> [o2] dst) d1 (Zero {t1} [n] dst mem))
	for {
		n := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		dst := v_0
		p1 := v_1
		mem := v_2
		if mem.Op != OpStore {
			break
		}
		t2 := auxToType(mem.Aux)
		_ = mem.Args[2]
		op2 := mem.Args[0]
		if op2.Op != OpOffPtr {
			break
		}
		tt2 := op2.Type
		o2 := auxIntToInt64(op2.AuxInt)
		p2 := op2.Args[0]
		d1 := mem.Args[1]
		mem_2 := mem.Args[2]
		if mem_2.Op != OpZero || auxIntToInt64(mem_2.AuxInt) != n {
			break
		}
		t3 := auxToType(mem_2.Aux)
		p3 := mem_2.Args[0]
		if !(isSamePtr(p1, p2) && isSamePtr(p2, p3) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && registerizable(b, t2) && n >= o2+t2.Size()) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t2)
		v0 := b.NewValue0(v.Pos, OpOffPtr, tt2)
		v0.AuxInt = int64ToAuxInt(o2)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpZero, types.TypeMem)
		v1.AuxInt = int64ToAuxInt(n)
		v1.Aux = typeToAux(t1)
		v1.AddArg2(dst, mem)
		v.AddArg3(v0, d1, v1)
		return true
	}
	// match: (Move {t1} [n] dst p1 mem:(Store {t2} (OffPtr <tt2> [o2] p2) d1 (Store {t3} (OffPtr <tt3> [o3] p3) d2 (Zero {t4} [n] p4 _))))
	// cond: isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && n >= o2 + t2.Size() && n >= o3 + t3.Size()
	// result: (Store {t2} (OffPtr <tt2> [o2] dst) d1 (Store {t3} (OffPtr <tt3> [o3] dst) d2 (Zero {t1} [n] dst mem)))
	for {
		n := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		dst := v_0
		p1 := v_1
		mem := v_2
		if mem.Op != OpStore {
			break
		}
		t2 := auxToType(mem.Aux)
		_ = mem.Args[2]
		mem_0 := mem.Args[0]
		if mem_0.Op != OpOffPtr {
			break
		}
		tt2 := mem_0.Type
		o2 := auxIntToInt64(mem_0.AuxInt)
		p2 := mem_0.Args[0]
		d1 := mem.Args[1]
		mem_2 := mem.Args[2]
		if mem_2.Op != OpStore {
			break
		}
		t3 := auxToType(mem_2.Aux)
		_ = mem_2.Args[2]
		mem_2_0 := mem_2.Args[0]
		if mem_2_0.Op != OpOffPtr {
			break
		}
		tt3 := mem_2_0.Type
		o3 := auxIntToInt64(mem_2_0.AuxInt)
		p3 := mem_2_0.Args[0]
		d2 := mem_2.Args[1]
		mem_2_2 := mem_2.Args[2]
		if mem_2_2.Op != OpZero || auxIntToInt64(mem_2_2.AuxInt) != n {
			break
		}
		t4 := auxToType(mem_2_2.Aux)
		p4 := mem_2_2.Args[0]
		if !(isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && n >= o2+t2.Size() && n >= o3+t3.Size()) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t2)
		v0 := b.NewValue0(v.Pos, OpOffPtr, tt2)
		v0.AuxInt = int64ToAuxInt(o2)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t3)
		v2 := b.NewValue0(v.Pos, OpOffPtr, tt3)
		v2.AuxInt = int64ToAuxInt(o3)
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, OpZero, types.TypeMem)
		v3.AuxInt = int64ToAuxInt(n)
		v3.Aux = typeToAux(t1)
		v3.AddArg2(dst, mem)
		v1.AddArg3(v2, d2, v3)
		v.AddArg3(v0, d1, v1)
		return true
	}
	// match: (Move {t1} [n] dst p1 mem:(Store {t2} (OffPtr <tt2> [o2] p2) d1 (Store {t3} (OffPtr <tt3> [o3] p3) d2 (Store {t4} (OffPtr <tt4> [o4] p4) d3 (Zero {t5} [n] p5 _)))))
	// cond: isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && isSamePtr(p4, p5) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && t5.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && registerizable(b, t4) && n >= o2 + t2.Size() && n >= o3 + t3.Size() && n >= o4 + t4.Size()
	// result: (Store {t2} (OffPtr <tt2> [o2] dst) d1 (Store {t3} (OffPtr <tt3> [o3] dst) d2 (Store {t4} (OffPtr <tt4> [o4] dst) d3 (Zero {t1} [n] dst mem))))
	for {
		n := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		dst := v_0
		p1 := v_1
		mem := v_2
		if mem.Op != OpStore {
			break
		}
		t2 := auxToType(mem.Aux)
		_ = mem.Args[2]
		mem_0 := mem.Args[0]
		if mem_0.Op != OpOffPtr {
			break
		}
		tt2 := mem_0.Type
		o2 := auxIntToInt64(mem_0.AuxInt)
		p2 := mem_0.Args[0]
		d1 := mem.Args[1]
		mem_2 := mem.Args[2]
		if mem_2.Op != OpStore {
			break
		}
		t3 := auxToType(mem_2.Aux)
		_ = mem_2.Args[2]
		mem_2_0 := mem_2.Args[0]
		if mem_2_0.Op != OpOffPtr {
			break
		}
		tt3 := mem_2_0.Type
		o3 := auxIntToInt64(mem_2_0.AuxInt)
		p3 := mem_2_0.Args[0]
		d2 := mem_2.Args[1]
		mem_2_2 := mem_2.Args[2]
		if mem_2_2.Op != OpStore {
			break
		}
		t4 := auxToType(mem_2_2.Aux)
		_ = mem_2_2.Args[2]
		mem_2_2_0 := mem_2_2.Args[0]
		if mem_2_2_0.Op != OpOffPtr {
			break
		}
		tt4 := mem_2_2_0.Type
		o4 := auxIntToInt64(mem_2_2_0.AuxInt)
		p4 := mem_2_2_0.Args[0]
		d3 := mem_2_2.Args[1]
		mem_2_2_2 := mem_2_2.Args[2]
		if mem_2_2_2.Op != OpZero || auxIntToInt64(mem_2_2_2.AuxInt) != n {
			break
		}
		t5 := auxToType(mem_2_2_2.Aux)
		p5 := mem_2_2_2.Args[0]
		if !(isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && isSamePtr(p4, p5) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && t5.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && registerizable(b, t4) && n >= o2+t2.Size() && n >= o3+t3.Size() && n >= o4+t4.Size()) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t2)
		v0 := b.NewValue0(v.Pos, OpOffPtr, tt2)
		v0.AuxInt = int64ToAuxInt(o2)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t3)
		v2 := b.NewValue0(v.Pos, OpOffPtr, tt3)
		v2.AuxInt = int64ToAuxInt(o3)
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v3.Aux = typeToAux(t4)
		v4 := b.NewValue0(v.Pos, OpOffPtr, tt4)
		v4.AuxInt = int64ToAuxInt(o4)
		v4.AddArg(dst)
		v5 := b.NewValue0(v.Pos, OpZero, types.TypeMem)
		v5.AuxInt = int64ToAuxInt(n)
		v5.Aux = typeToAux(t1)
		v5.AddArg2(dst, mem)
		v3.AddArg3(v4, d3, v5)
		v1.AddArg3(v2, d2, v3)
		v.AddArg3(v0, d1, v1)
		return true
	}
	// match: (Move {t1} [n] dst p1 mem:(Store {t2} (OffPtr <tt2> [o2] p2) d1 (Store {t3} (OffPtr <tt3> [o3] p3) d2 (Store {t4} (OffPtr <tt4> [o4] p4) d3 (Store {t5} (OffPtr <tt5> [o5] p5) d4 (Zero {t6} [n] p6 _))))))
	// cond: isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && isSamePtr(p4, p5) && isSamePtr(p5, p6) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && t5.Alignment() <= t1.Alignment() && t6.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && registerizable(b, t4) && registerizable(b, t5) && n >= o2 + t2.Size() && n >= o3 + t3.Size() && n >= o4 + t4.Size() && n >= o5 + t5.Size()
	// result: (Store {t2} (OffPtr <tt2> [o2] dst) d1 (Store {t3} (OffPtr <tt3> [o3] dst) d2 (Store {t4} (OffPtr <tt4> [o4] dst) d3 (Store {t5} (OffPtr <tt5> [o5] dst) d4 (Zero {t1} [n] dst mem)))))
	for {
		n := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		dst := v_0
		p1 := v_1
		mem := v_2
		if mem.Op != OpStore {
			break
		}
		t2 := auxToType(mem.Aux)
		_ = mem.Args[2]
		mem_0 := mem.Args[0]
		if mem_0.Op != OpOffPtr {
			break
		}
		tt2 := mem_0.Type
		o2 := auxIntToInt64(mem_0.AuxInt)
		p2 := mem_0.Args[0]
		d1 := mem.Args[1]
		mem_2 := mem.Args[2]
		if mem_2.Op != OpStore {
			break
		}
		t3 := auxToType(mem_2.Aux)
		_ = mem_2.Args[2]
		mem_2_0 := mem_2.Args[0]
		if mem_2_0.Op != OpOffPtr {
			break
		}
		tt3 := mem_2_0.Type
		o3 := auxIntToInt64(mem_2_0.AuxInt)
		p3 := mem_2_0.Args[0]
		d2 := mem_2.Args[1]
		mem_2_2 := mem_2.Args[2]
		if mem_2_2.Op != OpStore {
			break
		}
		t4 := auxToType(mem_2_2.Aux)
		_ = mem_2_2.Args[2]
		mem_2_2_0 := mem_2_2.Args[0]
		if mem_2_2_0.Op != OpOffPtr {
			break
		}
		tt4 := mem_2_2_0.Type
		o4 := auxIntToInt64(mem_2_2_0.AuxInt)
		p4 := mem_2_2_0.Args[0]
		d3 := mem_2_2.Args[1]
		mem_2_2_2 := mem_2_2.Args[2]
		if mem_2_2_2.Op != OpStore {
			break
		}
		t5 := auxToType(mem_2_2_2.Aux)
		_ = mem_2_2_2.Args[2]
		mem_2_2_2_0 := mem_2_2_2.Args[0]
		if mem_2_2_2_0.Op != OpOffPtr {
			break
		}
		tt5 := mem_2_2_2_0.Type
		o5 := auxIntToInt64(mem_2_2_2_0.AuxInt)
		p5 := mem_2_2_2_0.Args[0]
		d4 := mem_2_2_2.Args[1]
		mem_2_2_2_2 := mem_2_2_2.Args[2]
		if mem_2_2_2_2.Op != OpZero || auxIntToInt64(mem_2_2_2_2.AuxInt) != n {
			break
		}
		t6 := auxToType(mem_2_2_2_2.Aux)
		p6 := mem_2_2_2_2.Args[0]
		if !(isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && isSamePtr(p4, p5) && isSamePtr(p5, p6) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && t5.Alignment() <= t1.Alignment() && t6.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && registerizable(b, t4) && registerizable(b, t5) && n >= o2+t2.Size() && n >= o3+t3.Size() && n >= o4+t4.Size() && n >= o5+t5.Size()) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t2)
		v0 := b.NewValue0(v.Pos, OpOffPtr, tt2)
		v0.AuxInt = int64ToAuxInt(o2)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t3)
		v2 := b.NewValue0(v.Pos, OpOffPtr, tt3)
		v2.AuxInt = int64ToAuxInt(o3)
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v3.Aux = typeToAux(t4)
		v4 := b.NewValue0(v.Pos, OpOffPtr, tt4)
		v4.AuxInt = int64ToAuxInt(o4)
		v4.AddArg(dst)
		v5 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v5.Aux = typeToAux(t5)
		v6 := b.NewValue0(v.Pos, OpOffPtr, tt5)
		v6.AuxInt = int64ToAuxInt(o5)
		v6.AddArg(dst)
		v7 := b.NewValue0(v.Pos, OpZero, types.TypeMem)
		v7.AuxInt = int64ToAuxInt(n)
		v7.Aux = typeToAux(t1)
		v7.AddArg2(dst, mem)
		v5.AddArg3(v6, d4, v7)
		v3.AddArg3(v4, d3, v5)
		v1.AddArg3(v2, d2, v3)
		v.AddArg3(v0, d1, v1)
		return true
	}
	// match: (Move {t1} [n] dst p1 mem:(VarDef (Store {t2} op2:(OffPtr <tt2> [o2] p2) d1 (Zero {t3} [n] p3 _))))
	// cond: isSamePtr(p1, p2) && isSamePtr(p2, p3) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && registerizable(b, t2) && n >= o2 + t2.Size()
	// result: (Store {t2} (OffPtr <tt2> [o2] dst) d1 (Zero {t1} [n] dst mem))
	for {
		n := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		dst := v_0
		p1 := v_1
		mem := v_2
		if mem.Op != OpVarDef {
			break
		}
		mem_0 := mem.Args[0]
		if mem_0.Op != OpStore {
			break
		}
		t2 := auxToType(mem_0.Aux)
		_ = mem_0.Args[2]
		op2 := mem_0.Args[0]
		if op2.Op != OpOffPtr {
			break
		}
		tt2 := op2.Type
		o2 := auxIntToInt64(op2.AuxInt)
		p2 := op2.Args[0]
		d1 := mem_0.Args[1]
		mem_0_2 := mem_0.Args[2]
		if mem_0_2.Op != OpZero || auxIntToInt64(mem_0_2.AuxInt) != n {
			break
		}
		t3 := auxToType(mem_0_2.Aux)
		p3 := mem_0_2.Args[0]
		if !(isSamePtr(p1, p2) && isSamePtr(p2, p3) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && registerizable(b, t2) && n >= o2+t2.Size()) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t2)
		v0 := b.NewValue0(v.Pos, OpOffPtr, tt2)
		v0.AuxInt = int64ToAuxInt(o2)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpZero, types.TypeMem)
		v1.AuxInt = int64ToAuxInt(n)
		v1.Aux = typeToAux(t1)
		v1.AddArg2(dst, mem)
		v.AddArg3(v0, d1, v1)
		return true
	}
	// match: (Move {t1} [n] dst p1 mem:(VarDef (Store {t2} (OffPtr <tt2> [o2] p2) d1 (Store {t3} (OffPtr <tt3> [o3] p3) d2 (Zero {t4} [n] p4 _)))))
	// cond: isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && n >= o2 + t2.Size() && n >= o3 + t3.Size()
	// result: (Store {t2} (OffPtr <tt2> [o2] dst) d1 (Store {t3} (OffPtr <tt3> [o3] dst) d2 (Zero {t1} [n] dst mem)))
	for {
		n := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		dst := v_0
		p1 := v_1
		mem := v_2
		if mem.Op != OpVarDef {
			break
		}
		mem_0 := mem.Args[0]
		if mem_0.Op != OpStore {
			break
		}
		t2 := auxToType(mem_0.Aux)
		_ = mem_0.Args[2]
		mem_0_0 := mem_0.Args[0]
		if mem_0_0.Op != OpOffPtr {
			break
		}
		tt2 := mem_0_0.Type
		o2 := auxIntToInt64(mem_0_0.AuxInt)
		p2 := mem_0_0.Args[0]
		d1 := mem_0.Args[1]
		mem_0_2 := mem_0.Args[2]
		if mem_0_2.Op != OpStore {
			break
		}
		t3 := auxToType(mem_0_2.Aux)
		_ = mem_0_2.Args[2]
		mem_0_2_0 := mem_0_2.Args[0]
		if mem_0_2_0.Op != OpOffPtr {
			break
		}
		tt3 := mem_0_2_0.Type
		o3 := auxIntToInt64(mem_0_2_0.AuxInt)
		p3 := mem_0_2_0.Args[0]
		d2 := mem_0_2.Args[1]
		mem_0_2_2 := mem_0_2.Args[2]
		if mem_0_2_2.Op != OpZero || auxIntToInt64(mem_0_2_2.AuxInt) != n {
			break
		}
		t4 := auxToType(mem_0_2_2.Aux)
		p4 := mem_0_2_2.Args[0]
		if !(isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && n >= o2+t2.Size() && n >= o3+t3.Size()) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t2)
		v0 := b.NewValue0(v.Pos, OpOffPtr, tt2)
		v0.AuxInt = int64ToAuxInt(o2)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t3)
		v2 := b.NewValue0(v.Pos, OpOffPtr, tt3)
		v2.AuxInt = int64ToAuxInt(o3)
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, OpZero, types.TypeMem)
		v3.AuxInt = int64ToAuxInt(n)
		v3.Aux = typeToAux(t1)
		v3.AddArg2(dst, mem)
		v1.AddArg3(v2, d2, v3)
		v.AddArg3(v0, d1, v1)
		return true
	}
	// match: (Move {t1} [n] dst p1 mem:(VarDef (Store {t2} (OffPtr <tt2> [o2] p2) d1 (Store {t3} (OffPtr <tt3> [o3] p3) d2 (Store {t4} (OffPtr <tt4> [o4] p4) d3 (Zero {t5} [n] p5 _))))))
	// cond: isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && isSamePtr(p4, p5) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && t5.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && registerizable(b, t4) && n >= o2 + t2.Size() && n >= o3 + t3.Size() && n >= o4 + t4.Size()
	// result: (Store {t2} (OffPtr <tt2> [o2] dst) d1 (Store {t3} (OffPtr <tt3> [o3] dst) d2 (Store {t4} (OffPtr <tt4> [o4] dst) d3 (Zero {t1} [n] dst mem))))
	for {
		n := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		dst := v_0
		p1 := v_1
		mem := v_2
		if mem.Op != OpVarDef {
			break
		}
		mem_0 := mem.Args[0]
		if mem_0.Op != OpStore {
			break
		}
		t2 := auxToType(mem_0.Aux)
		_ = mem_0.Args[2]
		mem_0_0 := mem_0.Args[0]
		if mem_0_0.Op != OpOffPtr {
			break
		}
		tt2 := mem_0_0.Type
		o2 := auxIntToInt64(mem_0_0.AuxInt)
		p2 := mem_0_0.Args[0]
		d1 := mem_0.Args[1]
		mem_0_2 := mem_0.Args[2]
		if mem_0_2.Op != OpStore {
			break
		}
		t3 := auxToType(mem_0_2.Aux)
		_ = mem_0_2.Args[2]
		mem_0_2_0 := mem_0_2.Args[0]
		if mem_0_2_0.Op != OpOffPtr {
			break
		}
		tt3 := mem_0_2_0.Type
		o3 := auxIntToInt64(mem_0_2_0.AuxInt)
		p3 := mem_0_2_0.Args[0]
		d2 := mem_0_2.Args[1]
		mem_0_2_2 := mem_0_2.Args[2]
		if mem_0_2_2.Op != OpStore {
			break
		}
		t4 := auxToType(mem_0_2_2.Aux)
		_ = mem_0_2_2.Args[2]
		mem_0_2_2_0 := mem_0_2_2.Args[0]
		if mem_0_2_2_0.Op != OpOffPtr {
			break
		}
		tt4 := mem_0_2_2_0.Type
		o4 := auxIntToInt64(mem_0_2_2_0.AuxInt)
		p4 := mem_0_2_2_0.Args[0]
		d3 := mem_0_2_2.Args[1]
		mem_0_2_2_2 := mem_0_2_2.Args[2]
		if mem_0_2_2_2.Op != OpZero || auxIntToInt64(mem_0_2_2_2.AuxInt) != n {
			break
		}
		t5 := auxToType(mem_0_2_2_2.Aux)
		p5 := mem_0_2_2_2.Args[0]
		if !(isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && isSamePtr(p4, p5) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && t5.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && registerizable(b, t4) && n >= o2+t2.Size() && n >= o3+t3.Size() && n >= o4+t4.Size()) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t2)
		v0 := b.NewValue0(v.Pos, OpOffPtr, tt2)
		v0.AuxInt = int64ToAuxInt(o2)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t3)
		v2 := b.NewValue0(v.Pos, OpOffPtr, tt3)
		v2.AuxInt = int64ToAuxInt(o3)
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v3.Aux = typeToAux(t4)
		v4 := b.NewValue0(v.Pos, OpOffPtr, tt4)
		v4.AuxInt = int64ToAuxInt(o4)
		v4.AddArg(dst)
		v5 := b.NewValue0(v.Pos, OpZero, types.TypeMem)
		v5.AuxInt = int64ToAuxInt(n)
		v5.Aux = typeToAux(t1)
		v5.AddArg2(dst, mem)
		v3.AddArg3(v4, d3, v5)
		v1.AddArg3(v2, d2, v3)
		v.AddArg3(v0, d1, v1)
		return true
	}
	// match: (Move {t1} [n] dst p1 mem:(VarDef (Store {t2} (OffPtr <tt2> [o2] p2) d1 (Store {t3} (OffPtr <tt3> [o3] p3) d2 (Store {t4} (OffPtr <tt4> [o4] p4) d3 (Store {t5} (OffPtr <tt5> [o5] p5) d4 (Zero {t6} [n] p6 _)))))))
	// cond: isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && isSamePtr(p4, p5) && isSamePtr(p5, p6) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && t5.Alignment() <= t1.Alignment() && t6.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && registerizable(b, t4) && registerizable(b, t5) && n >= o2 + t2.Size() && n >= o3 + t3.Size() && n >= o4 + t4.Size() && n >= o5 + t5.Size()
	// result: (Store {t2} (OffPtr <tt2> [o2] dst) d1 (Store {t3} (OffPtr <tt3> [o3] dst) d2 (Store {t4} (OffPtr <tt4> [o4] dst) d3 (Store {t5} (OffPtr <tt5> [o5] dst) d4 (Zero {t1} [n] dst mem)))))
	for {
		n := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		dst := v_0
		p1 := v_1
		mem := v_2
		if mem.Op != OpVarDef {
			break
		}
		mem_0 := mem.Args[0]
		if mem_0.Op != OpStore {
			break
		}
		t2 := auxToType(mem_0.Aux)
		_ = mem_0.Args[2]
		mem_0_0 := mem_0.Args[0]
		if mem_0_0.Op != OpOffPtr {
			break
		}
		tt2 := mem_0_0.Type
		o2 := auxIntToInt64(mem_0_0.AuxInt)
		p2 := mem_0_0.Args[0]
		d1 := mem_0.Args[1]
		mem_0_2 := mem_0.Args[2]
		if mem_0_2.Op != OpStore {
			break
		}
		t3 := auxToType(mem_0_2.Aux)
		_ = mem_0_2.Args[2]
		mem_0_2_0 := mem_0_2.Args[0]
		if mem_0_2_0.Op != OpOffPtr {
			break
		}
		tt3 := mem_0_2_0.Type
		o3 := auxIntToInt64(mem_0_2_0.AuxInt)
		p3 := mem_0_2_0.Args[0]
		d2 := mem_0_2.Args[1]
		mem_0_2_2 := mem_0_2.Args[2]
		if mem_0_2_2.Op != OpStore {
			break
		}
		t4 := auxToType(mem_0_2_2.Aux)
		_ = mem_0_2_2.Args[2]
		mem_0_2_2_0 := mem_0_2_2.Args[0]
		if mem_0_2_2_0.Op != OpOffPtr {
			break
		}
		tt4 := mem_0_2_2_0.Type
		o4 := auxIntToInt64(mem_0_2_2_0.AuxInt)
		p4 := mem_0_2_2_0.Args[0]
		d3 := mem_0_2_2.Args[1]
		mem_0_2_2_2 := mem_0_2_2.Args[2]
		if mem_0_2_2_2.Op != OpStore {
			break
		}
		t5 := auxToType(mem_0_2_2_2.Aux)
		_ = mem_0_2_2_2.Args[2]
		mem_0_2_2_2_0 := mem_0_2_2_2.Args[0]
		if mem_0_2_2_2_0.Op != OpOffPtr {
			break
		}
		tt5 := mem_0_2_2_2_0.Type
		o5 := auxIntToInt64(mem_0_2_2_2_0.AuxInt)
		p5 := mem_0_2_2_2_0.Args[0]
		d4 := mem_0_2_2_2.Args[1]
		mem_0_2_2_2_2 := mem_0_2_2_2.Args[2]
		if mem_0_2_2_2_2.Op != OpZero || auxIntToInt64(mem_0_2_2_2_2.AuxInt) != n {
			break
		}
		t6 := auxToType(mem_0_2_2_2_2.Aux)
		p6 := mem_0_2_2_2_2.Args[0]
		if !(isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && isSamePtr(p4, p5) && isSamePtr(p5, p6) && t2.Alignment() <= t1.Alignment() && t3.Alignment() <= t1.Alignment() && t4.Alignment() <= t1.Alignment() && t5.Alignment() <= t1.Alignment() && t6.Alignment() <= t1.Alignment() && registerizable(b, t2) && registerizable(b, t3) && registerizable(b, t4) && registerizable(b, t5) && n >= o2+t2.Size() && n >= o3+t3.Size() && n >= o4+t4.Size() && n >= o5+t5.Size()) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t2)
		v0 := b.NewValue0(v.Pos, OpOffPtr, tt2)
		v0.AuxInt = int64ToAuxInt(o2)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t3)
		v2 := b.NewValue0(v.Pos, OpOffPtr, tt3)
		v2.AuxInt = int64ToAuxInt(o3)
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v3.Aux = typeToAux(t4)
		v4 := b.NewValue0(v.Pos, OpOffPtr, tt4)
		v4.AuxInt = int64ToAuxInt(o4)
		v4.AddArg(dst)
		v5 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v5.Aux = typeToAux(t5)
		v6 := b.NewValue0(v.Pos, OpOffPtr, tt5)
		v6.AuxInt = int64ToAuxInt(o5)
		v6.AddArg(dst)
		v7 := b.NewValue0(v.Pos, OpZero, types.TypeMem)
		v7.AuxInt = int64ToAuxInt(n)
		v7.Aux = typeToAux(t1)
		v7.AddArg2(dst, mem)
		v5.AddArg3(v6, d4, v7)
		v3.AddArg3(v4, d3, v5)
		v1.AddArg3(v2, d2, v3)
		v.AddArg3(v0, d1, v1)
		return true
	}
	// match: (Move {t1} [s] dst tmp1 midmem:(Move {t2} [s] tmp2 src _))
	// cond: t1.Compare(t2) == types.CMPeq && isSamePtr(tmp1, tmp2) && isStackPtr(src) && !isVolatile(src) && disjoint(src, s, tmp2, s) && (disjoint(src, s, dst, s) || isInlinableMemmove(dst, src, s, config))
	// result: (Move {t1} [s] dst src midmem)
	for {
		s := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		dst := v_0
		tmp1 := v_1
		midmem := v_2
		if midmem.Op != OpMove || auxIntToInt64(midmem.AuxInt) != s {
			break
		}
		t2 := auxToType(midmem.Aux)
		src := midmem.Args[1]
		tmp2 := midmem.Args[0]
		if !(t1.Compare(t2) == types.CMPeq && isSamePtr(tmp1, tmp2) && isStackPtr(src) && !isVolatile(src) && disjoint(src, s, tmp2, s) && (disjoint(src, s, dst, s) || isInlinableMemmove(dst, src, s, config))) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(s)
		v.Aux = typeToAux(t1)
		v.AddArg3(dst, src, midmem)
		return true
	}
	// match: (Move {t1} [s] dst tmp1 midmem:(VarDef (Move {t2} [s] tmp2 src _)))
	// cond: t1.Compare(t2) == types.CMPeq && isSamePtr(tmp1, tmp2) && isStackPtr(src) && !isVolatile(src) && disjoint(src, s, tmp2, s) && (disjoint(src, s, dst, s) || isInlinableMemmove(dst, src, s, config))
	// result: (Move {t1} [s] dst src midmem)
	for {
		s := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		dst := v_0
		tmp1 := v_1
		midmem := v_2
		if midmem.Op != OpVarDef {
			break
		}
		midmem_0 := midmem.Args[0]
		if midmem_0.Op != OpMove || auxIntToInt64(midmem_0.AuxInt) != s {
			break
		}
		t2 := auxToType(midmem_0.Aux)
		src := midmem_0.Args[1]
		tmp2 := midmem_0.Args[0]
		if !(t1.Compare(t2) == types.CMPeq && isSamePtr(tmp1, tmp2) && isStackPtr(src) && !isVolatile(src) && disjoint(src, s, tmp2, s) && (disjoint(src, s, dst, s) || isInlinableMemmove(dst, src, s, config))) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(s)
		v.Aux = typeToAux(t1)
		v.AddArg3(dst, src, midmem)
		return true
	}
	// match: (Move dst src mem)
	// cond: isSamePtr(dst, src)
	// result: mem
	for {
		dst := v_0
		src := v_1
		mem := v_2
		if !(isSamePtr(dst, src)) {
			break
		}
		v.copyOf(mem)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMul16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul16 (Const16 [c]) (Const16 [d]))
	// result: (Const16 [c*d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1.AuxInt)
			v.reset(OpConst16)
			v.AuxInt = int16ToAuxInt(c * d)
			return true
		}
		break
	}
	// match: (Mul16 (Const16 [1]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 1 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Mul16 (Const16 [-1]) x)
	// result: (Neg16 x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != -1 {
				continue
			}
			x := v_1
			v.reset(OpNeg16)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Mul16 <t> n (Const16 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Lsh16x64 <t> n (Const64 <typ.UInt64> [log16(c)]))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			n := v_0
			if v_1.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_1.AuxInt)
			if !(isPowerOfTwo(c)) {
				continue
			}
			v.reset(OpLsh16x64)
			v.Type = t
			v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
			v0.AuxInt = int64ToAuxInt(log16(c))
			v.AddArg2(n, v0)
			return true
		}
		break
	}
	// match: (Mul16 <t> n (Const16 [c]))
	// cond: t.IsSigned() && isPowerOfTwo(-c)
	// result: (Neg16 (Lsh16x64 <t> n (Const64 <typ.UInt64> [log16(-c)])))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			n := v_0
			if v_1.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_1.AuxInt)
			if !(t.IsSigned() && isPowerOfTwo(-c)) {
				continue
			}
			v.reset(OpNeg16)
			v0 := b.NewValue0(v.Pos, OpLsh16x64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
			v1.AuxInt = int64ToAuxInt(log16(-c))
			v0.AddArg2(n, v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (Mul16 (Const16 <t> [c]) (Add16 <t> (Const16 <t> [d]) x))
	// result: (Add16 (Const16 <t> [c*d]) (Mul16 <t> (Const16 <t> [c]) x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpAdd16 || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst16 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt16(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpAdd16)
				v0 := b.NewValue0(v.Pos, OpConst16, t)
				v0.AuxInt = int16ToAuxInt(c * d)
				v1 := b.NewValue0(v.Pos, OpMul16, t)
				v2 := b.NewValue0(v.Pos, OpConst16, t)
				v2.AuxInt = int16ToAuxInt(c)
				v1.AddArg2(v2, x)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	// match: (Mul16 (Const16 [0]) _)
	// result: (Const16 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
				continue
			}
			v.reset(OpConst16)
			v.AuxInt = int16ToAuxInt(0)
			return true
		}
		break
	}
	// match: (Mul16 (Mul16 i:(Const16 <t>) z) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Mul16 i (Mul16 <t> x z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpMul16 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst16 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst16 && x.Op != OpConst16) {
					continue
				}
				v.reset(OpMul16)
				v0 := b.NewValue0(v.Pos, OpMul16, t)
				v0.AddArg2(x, z)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (Mul16 (Const16 <t> [c]) (Mul16 (Const16 <t> [d]) x))
	// result: (Mul16 (Const16 <t> [c*d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpMul16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst16 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt16(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpMul16)
				v0 := b.NewValue0(v.Pos, OpConst16, t)
				v0.AuxInt = int16ToAuxInt(c * d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpMul32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul32 (Const32 [c]) (Const32 [d]))
	// result: (Const32 [c*d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			v.reset(OpConst32)
			v.AuxInt = int32ToAuxInt(c * d)
			return true
		}
		break
	}
	// match: (Mul32 (Const32 [1]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 1 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Mul32 (Const32 [-1]) x)
	// result: (Neg32 x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != -1 {
				continue
			}
			x := v_1
			v.reset(OpNeg32)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Mul32 <t> n (Const32 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Lsh32x64 <t> n (Const64 <typ.UInt64> [log32(c)]))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			n := v_0
			if v_1.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			if !(isPowerOfTwo(c)) {
				continue
			}
			v.reset(OpLsh32x64)
			v.Type = t
			v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
			v0.AuxInt = int64ToAuxInt(log32(c))
			v.AddArg2(n, v0)
			return true
		}
		break
	}
	// match: (Mul32 <t> n (Const32 [c]))
	// cond: t.IsSigned() && isPowerOfTwo(-c)
	// result: (Neg32 (Lsh32x64 <t> n (Const64 <typ.UInt64> [log32(-c)])))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			n := v_0
			if v_1.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			if !(t.IsSigned() && isPowerOfTwo(-c)) {
				continue
			}
			v.reset(OpNeg32)
			v0 := b.NewValue0(v.Pos, OpLsh32x64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
			v1.AuxInt = int64ToAuxInt(log32(-c))
			v0.AddArg2(n, v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (Mul32 (Const32 <t> [c]) (Add32 <t> (Const32 <t> [d]) x))
	// result: (Add32 (Const32 <t> [c*d]) (Mul32 <t> (Const32 <t> [c]) x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpAdd32 || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst32 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt32(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpAdd32)
				v0 := b.NewValue0(v.Pos, OpConst32, t)
				v0.AuxInt = int32ToAuxInt(c * d)
				v1 := b.NewValue0(v.Pos, OpMul32, t)
				v2 := b.NewValue0(v.Pos, OpConst32, t)
				v2.AuxInt = int32ToAuxInt(c)
				v1.AddArg2(v2, x)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	// match: (Mul32 (Const32 [0]) _)
	// result: (Const32 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
				continue
			}
			v.reset(OpConst32)
			v.AuxInt = int32ToAuxInt(0)
			return true
		}
		break
	}
	// match: (Mul32 (Mul32 i:(Const32 <t>) z) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Mul32 i (Mul32 <t> x z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpMul32 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst32 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst32 && x.Op != OpConst32) {
					continue
				}
				v.reset(OpMul32)
				v0 := b.NewValue0(v.Pos, OpMul32, t)
				v0.AddArg2(x, z)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (Mul32 (Const32 <t> [c]) (Mul32 (Const32 <t> [d]) x))
	// result: (Mul32 (Const32 <t> [c*d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpMul32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst32 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt32(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpMul32)
				v0 := b.NewValue0(v.Pos, OpConst32, t)
				v0.AuxInt = int32ToAuxInt(c * d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpMul32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mul32F (Const32F [c]) (Const32F [d]))
	// cond: c*d == c*d
	// result: (Const32F [c*d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32F {
				continue
			}
			c := auxIntToFloat32(v_0.AuxInt)
			if v_1.Op != OpConst32F {
				continue
			}
			d := auxIntToFloat32(v_1.AuxInt)
			if !(c*d == c*d) {
				continue
			}
			v.reset(OpConst32F)
			v.AuxInt = float32ToAuxInt(c * d)
			return true
		}
		break
	}
	// match: (Mul32F x (Const32F [1]))
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpConst32F || auxIntToFloat32(v_1.AuxInt) != 1 {
				continue
			}
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Mul32F x (Const32F [-1]))
	// result: (Neg32F x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpConst32F || auxIntToFloat32(v_1.AuxInt) != -1 {
				continue
			}
			v.reset(OpNeg32F)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Mul32F x (Const32F [2]))
	// result: (Add32F x x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpConst32F || auxIntToFloat32(v_1.AuxInt) != 2 {
				continue
			}
			v.reset(OpAdd32F)
			v.AddArg2(x, x)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpMul32uhilo(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul32uhilo (Const32 [c]) (Const32 [d]))
	// result: (MakeTuple (Const32 <typ.UInt32> [bitsMulU32(c, d).hi]) (Const32 <typ.UInt32> [bitsMulU32(c,d).lo]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			v.reset(OpMakeTuple)
			v0 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
			v0.AuxInt = int32ToAuxInt(bitsMulU32(c, d).hi)
			v1 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
			v1.AuxInt = int32ToAuxInt(bitsMulU32(c, d).lo)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpMul32uover(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul32uover (Const32 [c]) (Const32 [d]))
	// result: (MakeTuple (Const32 <typ.UInt32> [bitsMulU32(c, d).lo]) (ConstBool <typ.Bool> [bitsMulU32(c,d).hi != 0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			v.reset(OpMakeTuple)
			v0 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
			v0.AuxInt = int32ToAuxInt(bitsMulU32(c, d).lo)
			v1 := b.NewValue0(v.Pos, OpConstBool, typ.Bool)
			v1.AuxInt = boolToAuxInt(bitsMulU32(c, d).hi != 0)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (Mul32uover <t> (Const32 [1]) x)
	// result: (MakeTuple x (ConstBool <t.FieldType(1)> [false]))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 1 {
				continue
			}
			x := v_1
			v.reset(OpMakeTuple)
			v0 := b.NewValue0(v.Pos, OpConstBool, t.FieldType(1))
			v0.AuxInt = boolToAuxInt(false)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (Mul32uover <t> (Const32 [0]) x)
	// result: (MakeTuple (Const32 <t.FieldType(0)> [0]) (ConstBool <t.FieldType(1)> [false]))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
				continue
			}
			v.reset(OpMakeTuple)
			v0 := b.NewValue0(v.Pos, OpConst32, t.FieldType(0))
			v0.AuxInt = int32ToAuxInt(0)
			v1 := b.NewValue0(v.Pos, OpConstBool, t.FieldType(1))
			v1.AuxInt = boolToAuxInt(false)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpMul64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul64 (Const64 [c]) (Const64 [d]))
	// result: (Const64 [c*d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpConst64)
			v.AuxInt = int64ToAuxInt(c * d)
			return true
		}
		break
	}
	// match: (Mul64 (Const64 [1]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 1 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Mul64 (Const64 [-1]) x)
	// result: (Neg64 x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != -1 {
				continue
			}
			x := v_1
			v.reset(OpNeg64)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Mul64 <t> n (Const64 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Lsh64x64 <t> n (Const64 <typ.UInt64> [log64(c)]))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			n := v_0
			if v_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo(c)) {
				continue
			}
			v.reset(OpLsh64x64)
			v.Type = t
			v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
			v0.AuxInt = int64ToAuxInt(log64(c))
			v.AddArg2(n, v0)
			return true
		}
		break
	}
	// match: (Mul64 <t> n (Const64 [c]))
	// cond: t.IsSigned() && isPowerOfTwo(-c)
	// result: (Neg64 (Lsh64x64 <t> n (Const64 <typ.UInt64> [log64(-c)])))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			n := v_0
			if v_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(t.IsSigned() && isPowerOfTwo(-c)) {
				continue
			}
			v.reset(OpNeg64)
			v0 := b.NewValue0(v.Pos, OpLsh64x64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
			v1.AuxInt = int64ToAuxInt(log64(-c))
			v0.AddArg2(n, v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (Mul64 (Const64 <t> [c]) (Add64 <t> (Const64 <t> [d]) x))
	// result: (Add64 (Const64 <t> [c*d]) (Mul64 <t> (Const64 <t> [c]) x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpAdd64 || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst64 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt64(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpAdd64)
				v0 := b.NewValue0(v.Pos, OpConst64, t)
				v0.AuxInt = int64ToAuxInt(c * d)
				v1 := b.NewValue0(v.Pos, OpMul64, t)
				v2 := b.NewValue0(v.Pos, OpConst64, t)
				v2.AuxInt = int64ToAuxInt(c)
				v1.AddArg2(v2, x)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	// match: (Mul64 (Const64 [0]) _)
	// result: (Const64 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
				continue
			}
			v.reset(OpConst64)
			v.AuxInt = int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (Mul64 (Mul64 i:(Const64 <t>) z) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Mul64 i (Mul64 <t> x z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpMul64 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst64 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst64 && x.Op != OpConst64) {
					continue
				}
				v.reset(OpMul64)
				v0 := b.NewValue0(v.Pos, OpMul64, t)
				v0.AddArg2(x, z)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (Mul64 (Const64 <t> [c]) (Mul64 (Const64 <t> [d]) x))
	// result: (Mul64 (Const64 <t> [c*d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpMul64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst64 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt64(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpMul64)
				v0 := b.NewValue0(v.Pos, OpConst64, t)
				v0.AuxInt = int64ToAuxInt(c * d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpMul64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mul64F (Const64F [c]) (Const64F [d]))
	// cond: c*d == c*d
	// result: (Const64F [c*d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64F {
				continue
			}
			c := auxIntToFloat64(v_0.AuxInt)
			if v_1.Op != OpConst64F {
				continue
			}
			d := auxIntToFloat64(v_1.AuxInt)
			if !(c*d == c*d) {
				continue
			}
			v.reset(OpConst64F)
			v.AuxInt = float64ToAuxInt(c * d)
			return true
		}
		break
	}
	// match: (Mul64F x (Const64F [1]))
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpConst64F || auxIntToFloat64(v_1.AuxInt) != 1 {
				continue
			}
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Mul64F x (Const64F [-1]))
	// result: (Neg64F x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpConst64F || auxIntToFloat64(v_1.AuxInt) != -1 {
				continue
			}
			v.reset(OpNeg64F)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Mul64F x (Const64F [2]))
	// result: (Add64F x x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpConst64F || auxIntToFloat64(v_1.AuxInt) != 2 {
				continue
			}
			v.reset(OpAdd64F)
			v.AddArg2(x, x)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpMul64uhilo(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul64uhilo (Const64 [c]) (Const64 [d]))
	// result: (MakeTuple (Const64 <typ.UInt64> [bitsMulU64(c, d).hi]) (Const64 <typ.UInt64> [bitsMulU64(c,d).lo]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpMakeTuple)
			v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
			v0.AuxInt = int64ToAuxInt(bitsMulU64(c, d).hi)
			v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
			v1.AuxInt = int64ToAuxInt(bitsMulU64(c, d).lo)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpMul64uover(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul64uover (Const64 [c]) (Const64 [d]))
	// result: (MakeTuple (Const64 <typ.UInt64> [bitsMulU64(c, d).lo]) (ConstBool <typ.Bool> [bitsMulU64(c,d).hi != 0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpMakeTuple)
			v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
			v0.AuxInt = int64ToAuxInt(bitsMulU64(c, d).lo)
			v1 := b.NewValue0(v.Pos, OpConstBool, typ.Bool)
			v1.AuxInt = boolToAuxInt(bitsMulU64(c, d).hi != 0)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (Mul64uover <t> (Const64 [1]) x)
	// result: (MakeTuple x (ConstBool <t.FieldType(1)> [false]))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 1 {
				continue
			}
			x := v_1
			v.reset(OpMakeTuple)
			v0 := b.NewValue0(v.Pos, OpConstBool, t.FieldType(1))
			v0.AuxInt = boolToAuxInt(false)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (Mul64uover <t> (Const64 [0]) x)
	// result: (MakeTuple (Const64 <t.FieldType(0)> [0]) (ConstBool <t.FieldType(1)> [false]))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
				continue
			}
			v.reset(OpMakeTuple)
			v0 := b.NewValue0(v.Pos, OpConst64, t.FieldType(0))
			v0.AuxInt = int64ToAuxInt(0)
			v1 := b.NewValue0(v.Pos, OpConstBool, t.FieldType(1))
			v1.AuxInt = boolToAuxInt(false)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpMul8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul8 (Const8 [c]) (Const8 [d]))
	// result: (Const8 [c*d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1.AuxInt)
			v.reset(OpConst8)
			v.AuxInt = int8ToAuxInt(c * d)
			return true
		}
		break
	}
	// match: (Mul8 (Const8 [1]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 1 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Mul8 (Const8 [-1]) x)
	// result: (Neg8 x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != -1 {
				continue
			}
			x := v_1
			v.reset(OpNeg8)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Mul8 <t> n (Const8 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Lsh8x64 <t> n (Const64 <typ.UInt64> [log8(c)]))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			n := v_0
			if v_1.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_1.AuxInt)
			if !(isPowerOfTwo(c)) {
				continue
			}
			v.reset(OpLsh8x64)
			v.Type = t
			v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
			v0.AuxInt = int64ToAuxInt(log8(c))
			v.AddArg2(n, v0)
			return true
		}
		break
	}
	// match: (Mul8 <t> n (Const8 [c]))
	// cond: t.IsSigned() && isPowerOfTwo(-c)
	// result: (Neg8 (Lsh8x64 <t> n (Const64 <typ.UInt64> [log8(-c)])))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			n := v_0
			if v_1.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_1.AuxInt)
			if !(t.IsSigned() && isPowerOfTwo(-c)) {
				continue
			}
			v.reset(OpNeg8)
			v0 := b.NewValue0(v.Pos, OpLsh8x64, t)
			v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
			v1.AuxInt = int64ToAuxInt(log8(-c))
			v0.AddArg2(n, v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (Mul8 (Const8 <t> [c]) (Add8 <t> (Const8 <t> [d]) x))
	// result: (Add8 (Const8 <t> [c*d]) (Mul8 <t> (Const8 <t> [c]) x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpAdd8 || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst8 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt8(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpAdd8)
				v0 := b.NewValue0(v.Pos, OpConst8, t)
				v0.AuxInt = int8ToAuxInt(c * d)
				v1 := b.NewValue0(v.Pos, OpMul8, t)
				v2 := b.NewValue0(v.Pos, OpConst8, t)
				v2.AuxInt = int8ToAuxInt(c)
				v1.AddArg2(v2, x)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	// match: (Mul8 (Const8 [0]) _)
	// result: (Const8 [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
				continue
			}
			v.reset(OpConst8)
			v.AuxInt = int8ToAuxInt(0)
			return true
		}
		break
	}
	// match: (Mul8 (Mul8 i:(Const8 <t>) z) x)
	// cond: (z.Op != OpConst8 && x.Op != OpConst8)
	// result: (Mul8 i (Mul8 <t> x z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpMul8 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst8 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst8 && x.Op != OpConst8) {
					continue
				}
				v.reset(OpMul8)
				v0 := b.NewValue0(v.Pos, OpMul8, t)
				v0.AddArg2(x, z)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (Mul8 (Const8 <t> [c]) (Mul8 (Const8 <t> [d]) x))
	// result: (Mul8 (Const8 <t> [c*d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpMul8 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst8 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt8(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpMul8)
				v0 := b.NewValue0(v.Pos, OpConst8, t)
				v0.AuxInt = int8ToAuxInt(c * d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpNeg16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neg16 (Const16 [c]))
	// result: (Const16 [-c])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(-c)
		return true
	}
	// match: (Neg16 (Sub16 x y))
	// result: (Sub16 y x)
	for {
		if v_0.Op != OpSub16 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpSub16)
		v.AddArg2(y, x)
		return true
	}
	// match: (Neg16 (Neg16 x))
	// result: x
	for {
		if v_0.Op != OpNeg16 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Neg16 <t> (Com16 x))
	// result: (Add16 (Const16 <t> [1]) x)
	for {
		t := v.Type
		if v_0.Op != OpCom16 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAdd16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(1)
		v.AddArg2(v0, x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeg32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neg32 (Const32 [c]))
	// result: (Const32 [-c])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(-c)
		return true
	}
	// match: (Neg32 (Sub32 x y))
	// result: (Sub32 y x)
	for {
		if v_0.Op != OpSub32 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpSub32)
		v.AddArg2(y, x)
		return true
	}
	// match: (Neg32 (Neg32 x))
	// result: x
	for {
		if v_0.Op != OpNeg32 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Neg32 <t> (Com32 x))
	// result: (Add32 (Const32 <t> [1]) x)
	for {
		t := v.Type
		if v_0.Op != OpCom32 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAdd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg2(v0, x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeg32F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Neg32F (Const32F [c]))
	// cond: c != 0
	// result: (Const32F [-c])
	for {
		if v_0.Op != OpConst32F {
			break
		}
		c := auxIntToFloat32(v_0.AuxInt)
		if !(c != 0) {
			break
		}
		v.reset(OpConst32F)
		v.AuxInt = float32ToAuxInt(-c)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeg64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neg64 (Const64 [c]))
	// result: (Const64 [-c])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(-c)
		return true
	}
	// match: (Neg64 (Sub64 x y))
	// result: (Sub64 y x)
	for {
		if v_0.Op != OpSub64 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpSub64)
		v.AddArg2(y, x)
		return true
	}
	// match: (Neg64 (Neg64 x))
	// result: x
	for {
		if v_0.Op != OpNeg64 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Neg64 <t> (Com64 x))
	// result: (Add64 (Const64 <t> [1]) x)
	for {
		t := v.Type
		if v_0.Op != OpCom64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAdd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(1)
		v.AddArg2(v0, x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeg64F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Neg64F (Const64F [c]))
	// cond: c != 0
	// result: (Const64F [-c])
	for {
		if v_0.Op != OpConst64F {
			break
		}
		c := auxIntToFloat64(v_0.AuxInt)
		if !(c != 0) {
			break
		}
		v.reset(OpConst64F)
		v.AuxInt = float64ToAuxInt(-c)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeg8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neg8 (Const8 [c]))
	// result: (Const8 [-c])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(-c)
		return true
	}
	// match: (Neg8 (Sub8 x y))
	// result: (Sub8 y x)
	for {
		if v_0.Op != OpSub8 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpSub8)
		v.AddArg2(y, x)
		return true
	}
	// match: (Neg8 (Neg8 x))
	// result: x
	for {
		if v_0.Op != OpNeg8 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Neg8 <t> (Com8 x))
	// result: (Add8 (Const8 <t> [1]) x)
	for {
		t := v.Type
		if v_0.Op != OpCom8 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAdd8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(1)
		v.AddArg2(v0, x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq16 x x)
	// result: (ConstBool [false])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Neq16 (Const16 <t> [c]) (Add16 (Const16 <t> [d]) x))
	// result: (Neq16 (Const16 <t> [c-d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpAdd16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst16 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt16(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpNeq16)
				v0 := b.NewValue0(v.Pos, OpConst16, t)
				v0.AuxInt = int16ToAuxInt(c - d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Neq16 (Const16 [c]) (Const16 [d]))
	// result: (ConstBool [c != d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c != d)
			return true
		}
		break
	}
	// match: (Neq16 n (Lsh16x64 (Rsh16x64 (Add16 <t> n (Rsh16Ux64 <t> (Rsh16x64 <t> n (Const64 <typ.UInt64> [15])) (Const64 <typ.UInt64> [kbar]))) (Const64 <typ.UInt64> [k])) (Const64 <typ.UInt64> [k])) )
	// cond: k > 0 && k < 15 && kbar == 16 - k
	// result: (Neq16 (And16 <t> n (Const16 <t> [1<<uint(k)-1])) (Const16 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			n := v_0
			if v_1.Op != OpLsh16x64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRsh16x64 {
				continue
			}
			_ = v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			if v_1_0_0.Op != OpAdd16 {
				continue
			}
			t := v_1_0_0.Type
			_ = v_1_0_0.Args[1]
			v_1_0_0_0 := v_1_0_0.Args[0]
			v_1_0_0_1 := v_1_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0_0_0, v_1_0_0_1 = _i1+1, v_1_0_0_1, v_1_0_0_0 {
				if n != v_1_0_0_0 || v_1_0_0_1.Op != OpRsh16Ux64 || v_1_0_0_1.Type != t {
					continue
				}
				_ = v_1_0_0_1.Args[1]
				v_1_0_0_1_0 := v_1_0_0_1.Args[0]
				if v_1_0_0_1_0.Op != OpRsh16x64 || v_1_0_0_1_0.Type != t {
					continue
				}
				_ = v_1_0_0_1_0.Args[1]
				if n != v_1_0_0_1_0.Args[0] {
					continue
				}
				v_1_0_0_1_0_1 := v_1_0_0_1_0.Args[1]
				if v_1_0_0_1_0_1.Op != OpConst64 || v_1_0_0_1_0_1.Type != typ.UInt64 || auxIntToInt64(v_1_0_0_1_0_1.AuxInt) != 15 {
					continue
				}
				v_1_0_0_1_1 := v_1_0_0_1.Args[1]
				if v_1_0_0_1_1.Op != OpConst64 || v_1_0_0_1_1.Type != typ.UInt64 {
					continue
				}
				kbar := auxIntToInt64(v_1_0_0_1_1.AuxInt)
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst64 || v_1_0_1.Type != typ.UInt64 {
					continue
				}
				k := auxIntToInt64(v_1_0_1.AuxInt)
				v_1_1 := v_1.Args[1]
				if v_1_1.Op != OpConst64 || v_1_1.Type != typ.UInt64 || auxIntToInt64(v_1_1.AuxInt) != k || !(k > 0 && k < 15 && kbar == 16-k) {
					continue
				}
				v.reset(OpNeq16)
				v0 := b.NewValue0(v.Pos, OpAnd16, t)
				v1 := b.NewValue0(v.Pos, OpConst16, t)
				v1.AuxInt = int16ToAuxInt(1<<uint(k) - 1)
				v0.AddArg2(n, v1)
				v2 := b.NewValue0(v.Pos, OpConst16, t)
				v2.AuxInt = int16ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq16 s:(Sub16 x y) (Const16 [0]))
	// cond: s.Uses == 1
	// result: (Neq16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s := v_0
			if s.Op != OpSub16 {
				continue
			}
			y := s.Args[1]
			x := s.Args[0]
			if v_1.Op != OpConst16 || auxIntToInt16(v_1.AuxInt) != 0 || !(s.Uses == 1) {
				continue
			}
			v.reset(OpNeq16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Neq16 (And16 <t> x (Const16 <t> [y])) (Const16 <t> [y]))
	// cond: oneBit16(y)
	// result: (Eq16 (And16 <t> x (Const16 <t> [y])) (Const16 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAnd16 {
				continue
			}
			t := v_0.Type
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				if v_0_1.Op != OpConst16 || v_0_1.Type != t {
					continue
				}
				y := auxIntToInt16(v_0_1.AuxInt)
				if v_1.Op != OpConst16 || v_1.Type != t || auxIntToInt16(v_1.AuxInt) != y || !(oneBit16(y)) {
					continue
				}
				v.reset(OpEq16)
				v0 := b.NewValue0(v.Pos, OpAnd16, t)
				v1 := b.NewValue0(v.Pos, OpConst16, t)
				v1.AuxInt = int16ToAuxInt(y)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst16, t)
				v2.AuxInt = int16ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpNeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq32 x x)
	// result: (ConstBool [false])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Neq32 (Const32 <t> [c]) (Add32 (Const32 <t> [d]) x))
	// result: (Neq32 (Const32 <t> [c-d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpAdd32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst32 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt32(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpNeq32)
				v0 := b.NewValue0(v.Pos, OpConst32, t)
				v0.AuxInt = int32ToAuxInt(c - d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Neq32 (Const32 [c]) (Const32 [d]))
	// result: (ConstBool [c != d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c != d)
			return true
		}
		break
	}
	// match: (Neq32 n (Lsh32x64 (Rsh32x64 (Add32 <t> n (Rsh32Ux64 <t> (Rsh32x64 <t> n (Const64 <typ.UInt64> [31])) (Const64 <typ.UInt64> [kbar]))) (Const64 <typ.UInt64> [k])) (Const64 <typ.UInt64> [k])) )
	// cond: k > 0 && k < 31 && kbar == 32 - k
	// result: (Neq32 (And32 <t> n (Const32 <t> [1<<uint(k)-1])) (Const32 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			n := v_0
			if v_1.Op != OpLsh32x64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRsh32x64 {
				continue
			}
			_ = v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			if v_1_0_0.Op != OpAdd32 {
				continue
			}
			t := v_1_0_0.Type
			_ = v_1_0_0.Args[1]
			v_1_0_0_0 := v_1_0_0.Args[0]
			v_1_0_0_1 := v_1_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0_0_0, v_1_0_0_1 = _i1+1, v_1_0_0_1, v_1_0_0_0 {
				if n != v_1_0_0_0 || v_1_0_0_1.Op != OpRsh32Ux64 || v_1_0_0_1.Type != t {
					continue
				}
				_ = v_1_0_0_1.Args[1]
				v_1_0_0_1_0 := v_1_0_0_1.Args[0]
				if v_1_0_0_1_0.Op != OpRsh32x64 || v_1_0_0_1_0.Type != t {
					continue
				}
				_ = v_1_0_0_1_0.Args[1]
				if n != v_1_0_0_1_0.Args[0] {
					continue
				}
				v_1_0_0_1_0_1 := v_1_0_0_1_0.Args[1]
				if v_1_0_0_1_0_1.Op != OpConst64 || v_1_0_0_1_0_1.Type != typ.UInt64 || auxIntToInt64(v_1_0_0_1_0_1.AuxInt) != 31 {
					continue
				}
				v_1_0_0_1_1 := v_1_0_0_1.Args[1]
				if v_1_0_0_1_1.Op != OpConst64 || v_1_0_0_1_1.Type != typ.UInt64 {
					continue
				}
				kbar := auxIntToInt64(v_1_0_0_1_1.AuxInt)
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst64 || v_1_0_1.Type != typ.UInt64 {
					continue
				}
				k := auxIntToInt64(v_1_0_1.AuxInt)
				v_1_1 := v_1.Args[1]
				if v_1_1.Op != OpConst64 || v_1_1.Type != typ.UInt64 || auxIntToInt64(v_1_1.AuxInt) != k || !(k > 0 && k < 31 && kbar == 32-k) {
					continue
				}
				v.reset(OpNeq32)
				v0 := b.NewValue0(v.Pos, OpAnd32, t)
				v1 := b.NewValue0(v.Pos, OpConst32, t)
				v1.AuxInt = int32ToAuxInt(1<<uint(k) - 1)
				v0.AddArg2(n, v1)
				v2 := b.NewValue0(v.Pos, OpConst32, t)
				v2.AuxInt = int32ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq32 s:(Sub32 x y) (Const32 [0]))
	// cond: s.Uses == 1
	// result: (Neq32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s := v_0
			if s.Op != OpSub32 {
				continue
			}
			y := s.Args[1]
			x := s.Args[0]
			if v_1.Op != OpConst32 || auxIntToInt32(v_1.AuxInt) != 0 || !(s.Uses == 1) {
				continue
			}
			v.reset(OpNeq32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Neq32 (And32 <t> x (Const32 <t> [y])) (Const32 <t> [y]))
	// cond: oneBit32(y)
	// result: (Eq32 (And32 <t> x (Const32 <t> [y])) (Const32 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAnd32 {
				continue
			}
			t := v_0.Type
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				if v_0_1.Op != OpConst32 || v_0_1.Type != t {
					continue
				}
				y := auxIntToInt32(v_0_1.AuxInt)
				if v_1.Op != OpConst32 || v_1.Type != t || auxIntToInt32(v_1.AuxInt) != y || !(oneBit32(y)) {
					continue
				}
				v.reset(OpEq32)
				v0 := b.NewValue0(v.Pos, OpAnd32, t)
				v1 := b.NewValue0(v.Pos, OpConst32, t)
				v1.AuxInt = int32ToAuxInt(y)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst32, t)
				v2.AuxInt = int32ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpNeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Neq32F (Const32F [c]) (Const32F [d]))
	// result: (ConstBool [c != d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32F {
				continue
			}
			c := auxIntToFloat32(v_0.AuxInt)
			if v_1.Op != OpConst32F {
				continue
			}
			d := auxIntToFloat32(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c != d)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpNeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq64 x x)
	// result: (ConstBool [false])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Neq64 (Const64 <t> [c]) (Add64 (Const64 <t> [d]) x))
	// result: (Neq64 (Const64 <t> [c-d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpAdd64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst64 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt64(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpNeq64)
				v0 := b.NewValue0(v.Pos, OpConst64, t)
				v0.AuxInt = int64ToAuxInt(c - d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Neq64 (Const64 [c]) (Const64 [d]))
	// result: (ConstBool [c != d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c != d)
			return true
		}
		break
	}
	// match: (Neq64 n (Lsh64x64 (Rsh64x64 (Add64 <t> n (Rsh64Ux64 <t> (Rsh64x64 <t> n (Const64 <typ.UInt64> [63])) (Const64 <typ.UInt64> [kbar]))) (Const64 <typ.UInt64> [k])) (Const64 <typ.UInt64> [k])) )
	// cond: k > 0 && k < 63 && kbar == 64 - k
	// result: (Neq64 (And64 <t> n (Const64 <t> [1<<uint(k)-1])) (Const64 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			n := v_0
			if v_1.Op != OpLsh64x64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRsh64x64 {
				continue
			}
			_ = v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			if v_1_0_0.Op != OpAdd64 {
				continue
			}
			t := v_1_0_0.Type
			_ = v_1_0_0.Args[1]
			v_1_0_0_0 := v_1_0_0.Args[0]
			v_1_0_0_1 := v_1_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0_0_0, v_1_0_0_1 = _i1+1, v_1_0_0_1, v_1_0_0_0 {
				if n != v_1_0_0_0 || v_1_0_0_1.Op != OpRsh64Ux64 || v_1_0_0_1.Type != t {
					continue
				}
				_ = v_1_0_0_1.Args[1]
				v_1_0_0_1_0 := v_1_0_0_1.Args[0]
				if v_1_0_0_1_0.Op != OpRsh64x64 || v_1_0_0_1_0.Type != t {
					continue
				}
				_ = v_1_0_0_1_0.Args[1]
				if n != v_1_0_0_1_0.Args[0] {
					continue
				}
				v_1_0_0_1_0_1 := v_1_0_0_1_0.Args[1]
				if v_1_0_0_1_0_1.Op != OpConst64 || v_1_0_0_1_0_1.Type != typ.UInt64 || auxIntToInt64(v_1_0_0_1_0_1.AuxInt) != 63 {
					continue
				}
				v_1_0_0_1_1 := v_1_0_0_1.Args[1]
				if v_1_0_0_1_1.Op != OpConst64 || v_1_0_0_1_1.Type != typ.UInt64 {
					continue
				}
				kbar := auxIntToInt64(v_1_0_0_1_1.AuxInt)
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst64 || v_1_0_1.Type != typ.UInt64 {
					continue
				}
				k := auxIntToInt64(v_1_0_1.AuxInt)
				v_1_1 := v_1.Args[1]
				if v_1_1.Op != OpConst64 || v_1_1.Type != typ.UInt64 || auxIntToInt64(v_1_1.AuxInt) != k || !(k > 0 && k < 63 && kbar == 64-k) {
					continue
				}
				v.reset(OpNeq64)
				v0 := b.NewValue0(v.Pos, OpAnd64, t)
				v1 := b.NewValue0(v.Pos, OpConst64, t)
				v1.AuxInt = int64ToAuxInt(1<<uint(k) - 1)
				v0.AddArg2(n, v1)
				v2 := b.NewValue0(v.Pos, OpConst64, t)
				v2.AuxInt = int64ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq64 s:(Sub64 x y) (Const64 [0]))
	// cond: s.Uses == 1
	// result: (Neq64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s := v_0
			if s.Op != OpSub64 {
				continue
			}
			y := s.Args[1]
			x := s.Args[0]
			if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 0 || !(s.Uses == 1) {
				continue
			}
			v.reset(OpNeq64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Neq64 (And64 <t> x (Const64 <t> [y])) (Const64 <t> [y]))
	// cond: oneBit64(y)
	// result: (Eq64 (And64 <t> x (Const64 <t> [y])) (Const64 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAnd64 {
				continue
			}
			t := v_0.Type
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				if v_0_1.Op != OpConst64 || v_0_1.Type != t {
					continue
				}
				y := auxIntToInt64(v_0_1.AuxInt)
				if v_1.Op != OpConst64 || v_1.Type != t || auxIntToInt64(v_1.AuxInt) != y || !(oneBit64(y)) {
					continue
				}
				v.reset(OpEq64)
				v0 := b.NewValue0(v.Pos, OpAnd64, t)
				v1 := b.NewValue0(v.Pos, OpConst64, t)
				v1.AuxInt = int64ToAuxInt(y)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst64, t)
				v2.AuxInt = int64ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpNeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Neq64F (Const64F [c]) (Const64F [d]))
	// result: (ConstBool [c != d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64F {
				continue
			}
			c := auxIntToFloat64(v_0.AuxInt)
			if v_1.Op != OpConst64F {
				continue
			}
			d := auxIntToFloat64(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c != d)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpNeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq8 x x)
	// result: (ConstBool [false])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (Neq8 (Const8 <t> [c]) (Add8 (Const8 <t> [d]) x))
	// result: (Neq8 (Const8 <t> [c-d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpAdd8 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst8 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt8(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpNeq8)
				v0 := b.NewValue0(v.Pos, OpConst8, t)
				v0.AuxInt = int8ToAuxInt(c - d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Neq8 (Const8 [c]) (Const8 [d]))
	// result: (ConstBool [c != d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c != d)
			return true
		}
		break
	}
	// match: (Neq8 n (Lsh8x64 (Rsh8x64 (Add8 <t> n (Rsh8Ux64 <t> (Rsh8x64 <t> n (Const64 <typ.UInt64> [ 7])) (Const64 <typ.UInt64> [kbar]))) (Const64 <typ.UInt64> [k])) (Const64 <typ.UInt64> [k])) )
	// cond: k > 0 && k < 7 && kbar == 8 - k
	// result: (Neq8 (And8 <t> n (Const8 <t> [1<<uint(k)-1])) (Const8 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			n := v_0
			if v_1.Op != OpLsh8x64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRsh8x64 {
				continue
			}
			_ = v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			if v_1_0_0.Op != OpAdd8 {
				continue
			}
			t := v_1_0_0.Type
			_ = v_1_0_0.Args[1]
			v_1_0_0_0 := v_1_0_0.Args[0]
			v_1_0_0_1 := v_1_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0_0_0, v_1_0_0_1 = _i1+1, v_1_0_0_1, v_1_0_0_0 {
				if n != v_1_0_0_0 || v_1_0_0_1.Op != OpRsh8Ux64 || v_1_0_0_1.Type != t {
					continue
				}
				_ = v_1_0_0_1.Args[1]
				v_1_0_0_1_0 := v_1_0_0_1.Args[0]
				if v_1_0_0_1_0.Op != OpRsh8x64 || v_1_0_0_1_0.Type != t {
					continue
				}
				_ = v_1_0_0_1_0.Args[1]
				if n != v_1_0_0_1_0.Args[0] {
					continue
				}
				v_1_0_0_1_0_1 := v_1_0_0_1_0.Args[1]
				if v_1_0_0_1_0_1.Op != OpConst64 || v_1_0_0_1_0_1.Type != typ.UInt64 || auxIntToInt64(v_1_0_0_1_0_1.AuxInt) != 7 {
					continue
				}
				v_1_0_0_1_1 := v_1_0_0_1.Args[1]
				if v_1_0_0_1_1.Op != OpConst64 || v_1_0_0_1_1.Type != typ.UInt64 {
					continue
				}
				kbar := auxIntToInt64(v_1_0_0_1_1.AuxInt)
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst64 || v_1_0_1.Type != typ.UInt64 {
					continue
				}
				k := auxIntToInt64(v_1_0_1.AuxInt)
				v_1_1 := v_1.Args[1]
				if v_1_1.Op != OpConst64 || v_1_1.Type != typ.UInt64 || auxIntToInt64(v_1_1.AuxInt) != k || !(k > 0 && k < 7 && kbar == 8-k) {
					continue
				}
				v.reset(OpNeq8)
				v0 := b.NewValue0(v.Pos, OpAnd8, t)
				v1 := b.NewValue0(v.Pos, OpConst8, t)
				v1.AuxInt = int8ToAuxInt(1<<uint(k) - 1)
				v0.AddArg2(n, v1)
				v2 := b.NewValue0(v.Pos, OpConst8, t)
				v2.AuxInt = int8ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq8 s:(Sub8 x y) (Const8 [0]))
	// cond: s.Uses == 1
	// result: (Neq8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s := v_0
			if s.Op != OpSub8 {
				continue
			}
			y := s.Args[1]
			x := s.Args[0]
			if v_1.Op != OpConst8 || auxIntToInt8(v_1.AuxInt) != 0 || !(s.Uses == 1) {
				continue
			}
			v.reset(OpNeq8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Neq8 (And8 <t> x (Const8 <t> [y])) (Const8 <t> [y]))
	// cond: oneBit8(y)
	// result: (Eq8 (And8 <t> x (Const8 <t> [y])) (Const8 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAnd8 {
				continue
			}
			t := v_0.Type
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				if v_0_1.Op != OpConst8 || v_0_1.Type != t {
					continue
				}
				y := auxIntToInt8(v_0_1.AuxInt)
				if v_1.Op != OpConst8 || v_1.Type != t || auxIntToInt8(v_1.AuxInt) != y || !(oneBit8(y)) {
					continue
				}
				v.reset(OpEq8)
				v0 := b.NewValue0(v.Pos, OpAnd8, t)
				v1 := b.NewValue0(v.Pos, OpConst8, t)
				v1.AuxInt = int8ToAuxInt(y)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst8, t)
				v2.AuxInt = int8ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpNeqB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (NeqB (ConstBool [c]) (ConstBool [d]))
	// result: (ConstBool [c != d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConstBool {
				continue
			}
			c := auxIntToBool(v_0.AuxInt)
			if v_1.Op != OpConstBool {
				continue
			}
			d := auxIntToBool(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c != d)
			return true
		}
		break
	}
	// match: (NeqB (ConstBool [false]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConstBool || auxIntToBool(v_0.AuxInt) != false {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (NeqB (ConstBool [true]) x)
	// result: (Not x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConstBool || auxIntToBool(v_0.AuxInt) != true {
				continue
			}
			x := v_1
			v.reset(OpNot)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (NeqB (Not x) (Not y))
	// result: (NeqB x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpNot {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != OpNot {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpNeqB)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpNeqInter(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NeqInter x y)
	// result: (NeqPtr (ITab x) (ITab y))
	for {
		x := v_0
		y := v_1
		v.reset(OpNeqPtr)
		v0 := b.NewValue0(v.Pos, OpITab, typ.Uintptr)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpITab, typ.Uintptr)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuegeneric_OpNeqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (NeqPtr x x)
	// result: (ConstBool [false])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (NeqPtr (Addr {x} _) (Addr {y} _))
	// result: (ConstBool [x != y])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAddr {
				continue
			}
			x := auxToSym(v_0.Aux)
			if v_1.Op != OpAddr {
				continue
			}
			y := auxToSym(v_1.Aux)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(x != y)
			return true
		}
		break
	}
	// match: (NeqPtr (Addr {x} _) (OffPtr [o] (Addr {y} _)))
	// result: (ConstBool [x != y || o != 0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAddr {
				continue
			}
			x := auxToSym(v_0.Aux)
			if v_1.Op != OpOffPtr {
				continue
			}
			o := auxIntToInt64(v_1.AuxInt)
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpAddr {
				continue
			}
			y := auxToSym(v_1_0.Aux)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(x != y || o != 0)
			return true
		}
		break
	}
	// match: (NeqPtr (OffPtr [o1] (Addr {x} _)) (OffPtr [o2] (Addr {y} _)))
	// result: (ConstBool [x != y || o1 != o2])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpOffPtr {
				continue
			}
			o1 := auxIntToInt64(v_0.AuxInt)
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAddr {
				continue
			}
			x := auxToSym(v_0_0.Aux)
			if v_1.Op != OpOffPtr {
				continue
			}
			o2 := auxIntToInt64(v_1.AuxInt)
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpAddr {
				continue
			}
			y := auxToSym(v_1_0.Aux)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(x != y || o1 != o2)
			return true
		}
		break
	}
	// match: (NeqPtr (LocalAddr {x} _ _) (LocalAddr {y} _ _))
	// result: (ConstBool [x != y])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLocalAddr {
				continue
			}
			x := auxToSym(v_0.Aux)
			if v_1.Op != OpLocalAddr {
				continue
			}
			y := auxToSym(v_1.Aux)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(x != y)
			return true
		}
		break
	}
	// match: (NeqPtr (LocalAddr {x} _ _) (OffPtr [o] (LocalAddr {y} _ _)))
	// result: (ConstBool [x != y || o != 0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLocalAddr {
				continue
			}
			x := auxToSym(v_0.Aux)
			if v_1.Op != OpOffPtr {
				continue
			}
			o := auxIntToInt64(v_1.AuxInt)
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpLocalAddr {
				continue
			}
			y := auxToSym(v_1_0.Aux)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(x != y || o != 0)
			return true
		}
		break
	}
	// match: (NeqPtr (OffPtr [o1] (LocalAddr {x} _ _)) (OffPtr [o2] (LocalAddr {y} _ _)))
	// result: (ConstBool [x != y || o1 != o2])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpOffPtr {
				continue
			}
			o1 := auxIntToInt64(v_0.AuxInt)
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpLocalAddr {
				continue
			}
			x := auxToSym(v_0_0.Aux)
			if v_1.Op != OpOffPtr {
				continue
			}
			o2 := auxIntToInt64(v_1.AuxInt)
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpLocalAddr {
				continue
			}
			y := auxToSym(v_1_0.Aux)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(x != y || o1 != o2)
			return true
		}
		break
	}
	// match: (NeqPtr (OffPtr [o1] p1) p2)
	// cond: isSamePtr(p1, p2)
	// result: (ConstBool [o1 != 0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpOffPtr {
				continue
			}
			o1 := auxIntToInt64(v_0.AuxInt)
			p1 := v_0.Args[0]
			p2 := v_1
			if !(isSamePtr(p1, p2)) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(o1 != 0)
			return true
		}
		break
	}
	// match: (NeqPtr (OffPtr [o1] p1) (OffPtr [o2] p2))
	// cond: isSamePtr(p1, p2)
	// result: (ConstBool [o1 != o2])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpOffPtr {
				continue
			}
			o1 := auxIntToInt64(v_0.AuxInt)
			p1 := v_0.Args[0]
			if v_1.Op != OpOffPtr {
				continue
			}
			o2 := auxIntToInt64(v_1.AuxInt)
			p2 := v_1.Args[0]
			if !(isSamePtr(p1, p2)) {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(o1 != o2)
			return true
		}
		break
	}
	// match: (NeqPtr (Const32 [c]) (Const32 [d]))
	// result: (ConstBool [c != d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c != d)
			return true
		}
		break
	}
	// match: (NeqPtr (Const64 [c]) (Const64 [d]))
	// result: (ConstBool [c != d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(c != d)
			return true
		}
		break
	}
	// match: (NeqPtr (Convert (Addr {x} _) _) (Addr {y} _))
	// result: (ConstBool [x!=y])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConvert {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAddr {
				continue
			}
			x := auxToSym(v_0_0.Aux)
			if v_1.Op != OpAddr {
				continue
			}
			y := auxToSym(v_1.Aux)
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(x != y)
			return true
		}
		break
	}
	// match: (NeqPtr (LocalAddr _ _) (Addr _))
	// result: (ConstBool [true])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLocalAddr || v_1.Op != OpAddr {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (NeqPtr (OffPtr (LocalAddr _ _)) (Addr _))
	// result: (ConstBool [true])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpOffPtr {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpLocalAddr || v_1.Op != OpAddr {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (NeqPtr (LocalAddr _ _) (OffPtr (Addr _)))
	// result: (ConstBool [true])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLocalAddr || v_1.Op != OpOffPtr {
				continue
			}
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpAddr {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (NeqPtr (OffPtr (LocalAddr _ _)) (OffPtr (Addr _)))
	// result: (ConstBool [true])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpOffPtr {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpLocalAddr || v_1.Op != OpOffPtr {
				continue
			}
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpAddr {
				continue
			}
			v.reset(OpConstBool)
			v.AuxInt = boolToAuxInt(true)
			return true
		}
		break
	}
	// match: (NeqPtr (AddPtr p1 o1) p2)
	// cond: isSamePtr(p1, p2)
	// result: (IsNonNil o1)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAddPtr {
				continue
			}
			o1 := v_0.Args[1]
			p1 := v_0.Args[0]
			p2 := v_1
			if !(isSamePtr(p1, p2)) {
				continue
			}
			v.reset(OpIsNonNil)
			v.AddArg(o1)
			return true
		}
		break
	}
	// match: (NeqPtr (Const32 [0]) p)
	// result: (IsNonNil p)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
				continue
			}
			p := v_1
			v.reset(OpIsNonNil)
			v.AddArg(p)
			return true
		}
		break
	}
	// match: (NeqPtr (Const64 [0]) p)
	// result: (IsNonNil p)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
				continue
			}
			p := v_1
			v.reset(OpIsNonNil)
			v.AddArg(p)
			return true
		}
		break
	}
	// match: (NeqPtr (ConstNil) p)
	// result: (IsNonNil p)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConstNil {
				continue
			}
			p := v_1
			v.reset(OpIsNonNil)
			v.AddArg(p)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpNeqSlice(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NeqSlice x y)
	// result: (NeqPtr (SlicePtr x) (SlicePtr y))
	for {
		x := v_0
		y := v_1
		v.reset(OpNeqPtr)
		v0 := b.NewValue0(v.Pos, OpSlicePtr, typ.BytePtr)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSlicePtr, typ.BytePtr)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuegeneric_OpNilCheck(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	fe := b.Func.fe
	// match: (NilCheck ptr:(GetG mem) mem)
	// result: ptr
	for {
		ptr := v_0
		if ptr.Op != OpGetG {
			break
		}
		mem := ptr.Args[0]
		if mem != v_1 {
			break
		}
		v.copyOf(ptr)
		return true
	}
	// match: (NilCheck ptr:(SelectN [0] call:(StaticLECall _ _)) _)
	// cond: isSameCall(call.Aux, "runtime.newobject") && warnRule(fe.Debug_checknil(), v, "removed nil check")
	// result: ptr
	for {
		ptr := v_0
		if ptr.Op != OpSelectN || auxIntToInt64(ptr.AuxInt) != 0 {
			break
		}
		call := ptr.Args[0]
		if call.Op != OpStaticLECall || len(call.Args) != 2 || !(isSameCall(call.Aux, "runtime.newobject") && warnRule(fe.Debug_checknil(), v, "removed nil check")) {
			break
		}
		v.copyOf(ptr)
		return true
	}
	// match: (NilCheck ptr:(OffPtr (SelectN [0] call:(StaticLECall _ _))) _)
	// cond: isSameCall(call.Aux, "runtime.newobject") && warnRule(fe.Debug_checknil(), v, "removed nil check")
	// result: ptr
	for {
		ptr := v_0
		if ptr.Op != OpOffPtr {
			break
		}
		ptr_0 := ptr.Args[0]
		if ptr_0.Op != OpSelectN || auxIntToInt64(ptr_0.AuxInt) != 0 {
			break
		}
		call := ptr_0.Args[0]
		if call.Op != OpStaticLECall || len(call.Args) != 2 || !(isSameCall(call.Aux, "runtime.newobject") && warnRule(fe.Debug_checknil(), v, "removed nil check")) {
			break
		}
		v.copyOf(ptr)
		return true
	}
	// match: (NilCheck ptr:(Addr {_} (SB)) _)
	// result: ptr
	for {
		ptr := v_0
		if ptr.Op != OpAddr {
			break
		}
		ptr_0 := ptr.Args[0]
		if ptr_0.Op != OpSB {
			break
		}
		v.copyOf(ptr)
		return true
	}
	// match: (NilCheck ptr:(Convert (Addr {_} (SB)) _) _)
	// result: ptr
	for {
		ptr := v_0
		if ptr.Op != OpConvert {
			break
		}
		ptr_0 := ptr.Args[0]
		if ptr_0.Op != OpAddr {
			break
		}
		ptr_0_0 := ptr_0.Args[0]
		if ptr_0_0.Op != OpSB {
			break
		}
		v.copyOf(ptr)
		return true
	}
	// match: (NilCheck ptr:(LocalAddr _ _) _)
	// cond: warnRule(fe.Debug_checknil(), v, "removed nil check")
	// result: ptr
	for {
		ptr := v_0
		if ptr.Op != OpLocalAddr || !(warnRule(fe.Debug_checknil(), v, "removed nil check")) {
			break
		}
		v.copyOf(ptr)
		return true
	}
	// match: (NilCheck ptr:(NilCheck _ _) _ )
	// result: ptr
	for {
		ptr := v_0
		if ptr.Op != OpNilCheck {
			break
		}
		v.copyOf(ptr)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNot(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Not (ConstBool [c]))
	// result: (ConstBool [!c])
	for {
		if v_0.Op != OpConstBool {
			break
		}
		c := auxIntToBool(v_0.AuxInt)
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(!c)
		return true
	}
	// match: (Not (Eq64 x y))
	// result: (Neq64 x y)
	for {
		if v_0.Op != OpEq64 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpNeq64)
		v.AddArg2(x, y)
		return true
	}
	// match: (Not (Eq32 x y))
	// result: (Neq32 x y)
	for {
		if v_0.Op != OpEq32 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpNeq32)
		v.AddArg2(x, y)
		return true
	}
	// match: (Not (Eq16 x y))
	// result: (Neq16 x y)
	for {
		if v_0.Op != OpEq16 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpNeq16)
		v.AddArg2(x, y)
		return true
	}
	// match: (Not (Eq8 x y))
	// result: (Neq8 x y)
	for {
		if v_0.Op != OpEq8 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpNeq8)
		v.AddArg2(x, y)
		return true
	}
	// match: (Not (EqB x y))
	// result: (NeqB x y)
	for {
		if v_0.Op != OpEqB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpNeqB)
		v.AddArg2(x, y)
		return true
	}
	// match: (Not (EqPtr x y))
	// result: (NeqPtr x y)
	for {
		if v_0.Op != OpEqPtr {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpNeqPtr)
		v.AddArg2(x, y)
		return true
	}
	// match: (Not (Eq64F x y))
	// result: (Neq64F x y)
	for {
		if v_0.Op != OpEq64F {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpNeq64F)
		v.AddArg2(x, y)
		return true
	}
	// match: (Not (Eq32F x y))
	// result: (Neq32F x y)
	for {
		if v_0.Op != OpEq32F {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpNeq32F)
		v.AddArg2(x, y)
		return true
	}
	// match: (Not (Neq64 x y))
	// result: (Eq64 x y)
	for {
		if v_0.Op != OpNeq64 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpEq64)
		v.AddArg2(x, y)
		return true
	}
	// match: (Not (Neq32 x y))
	// result: (Eq32 x y)
	for {
		if v_0.Op != OpNeq32 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpEq32)
		v.AddArg2(x, y)
		return true
	}
	// match: (Not (Neq16 x y))
	// result: (Eq16 x y)
	for {
		if v_0.Op != OpNeq16 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpEq16)
		v.AddArg2(x, y)
		return true
	}
	// match: (Not (Neq8 x y))
	// result: (Eq8 x y)
	for {
		if v_0.Op != OpNeq8 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpEq8)
		v.AddArg2(x, y)
		return true
	}
	// match: (Not (NeqB x y))
	// result: (EqB x y)
	for {
		if v_0.Op != OpNeqB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpEqB)
		v.AddArg2(x, y)
		return true
	}
	// match: (Not (NeqPtr x y))
	// result: (EqPtr x y)
	for {
		if v_0.Op != OpNeqPtr {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpEqPtr)
		v.AddArg2(x, y)
		return true
	}
	// match: (Not (Neq64F x y))
	// result: (Eq64F x y)
	for {
		if v_0.Op != OpNeq64F {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpEq64F)
		v.AddArg2(x, y)
		return true
	}
	// match: (Not (Neq32F x y))
	// result: (Eq32F x y)
	for {
		if v_0.Op != OpNeq32F {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpEq32F)
		v.AddArg2(x, y)
		return true
	}
	// match: (Not (Less64 x y))
	// result: (Leq64 y x)
	for {
		if v_0.Op != OpLess64 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLeq64)
		v.AddArg2(y, x)
		return true
	}
	// match: (Not (Less32 x y))
	// result: (Leq32 y x)
	for {
		if v_0.Op != OpLess32 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLeq32)
		v.AddArg2(y, x)
		return true
	}
	// match: (Not (Less16 x y))
	// result: (Leq16 y x)
	for {
		if v_0.Op != OpLess16 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLeq16)
		v.AddArg2(y, x)
		return true
	}
	// match: (Not (Less8 x y))
	// result: (Leq8 y x)
	for {
		if v_0.Op != OpLess8 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLeq8)
		v.AddArg2(y, x)
		return true
	}
	// match: (Not (Less64U x y))
	// result: (Leq64U y x)
	for {
		if v_0.Op != OpLess64U {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLeq64U)
		v.AddArg2(y, x)
		return true
	}
	// match: (Not (Less32U x y))
	// result: (Leq32U y x)
	for {
		if v_0.Op != OpLess32U {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLeq32U)
		v.AddArg2(y, x)
		return true
	}
	// match: (Not (Less16U x y))
	// result: (Leq16U y x)
	for {
		if v_0.Op != OpLess16U {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLeq16U)
		v.AddArg2(y, x)
		return true
	}
	// match: (Not (Less8U x y))
	// result: (Leq8U y x)
	for {
		if v_0.Op != OpLess8U {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLeq8U)
		v.AddArg2(y, x)
		return true
	}
	// match: (Not (Leq64 x y))
	// result: (Less64 y x)
	for {
		if v_0.Op != OpLeq64 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLess64)
		v.AddArg2(y, x)
		return true
	}
	// match: (Not (Leq32 x y))
	// result: (Less32 y x)
	for {
		if v_0.Op != OpLeq32 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLess32)
		v.AddArg2(y, x)
		return true
	}
	// match: (Not (Leq16 x y))
	// result: (Less16 y x)
	for {
		if v_0.Op != OpLeq16 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLess16)
		v.AddArg2(y, x)
		return true
	}
	// match: (Not (Leq8 x y))
	// result: (Less8 y x)
	for {
		if v_0.Op != OpLeq8 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLess8)
		v.AddArg2(y, x)
		return true
	}
	// match: (Not (Leq64U x y))
	// result: (Less64U y x)
	for {
		if v_0.Op != OpLeq64U {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLess64U)
		v.AddArg2(y, x)
		return true
	}
	// match: (Not (Leq32U x y))
	// result: (Less32U y x)
	for {
		if v_0.Op != OpLeq32U {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLess32U)
		v.AddArg2(y, x)
		return true
	}
	// match: (Not (Leq16U x y))
	// result: (Less16U y x)
	for {
		if v_0.Op != OpLeq16U {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLess16U)
		v.AddArg2(y, x)
		return true
	}
	// match: (Not (Leq8U x y))
	// result: (Less8U y x)
	for {
		if v_0.Op != OpLeq8U {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLess8U)
		v.AddArg2(y, x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpOffPtr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (OffPtr (OffPtr p [y]) [x])
	// result: (OffPtr p [x+y])
	for {
		x := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpOffPtr {
			break
		}
		y := auxIntToInt64(v_0.AuxInt)
		p := v_0.Args[0]
		v.reset(OpOffPtr)
		v.AuxInt = int64ToAuxInt(x + y)
		v.AddArg(p)
		return true
	}
	// match: (OffPtr p [0])
	// cond: v.Type.Compare(p.Type) == types.CMPeq
	// result: p
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		p := v_0
		if !(v.Type.Compare(p.Type) == types.CMPeq) {
			break
		}
		v.copyOf(p)
		return true
	}
	return false
}
func rewriteValuegeneric_OpOr16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (Or16 (Const16 [c]) (Const16 [d]))
	// result: (Const16 [c|d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1.AuxInt)
			v.reset(OpConst16)
			v.AuxInt = int16ToAuxInt(c | d)
			return true
		}
		break
	}
	// match: (Or16 <t> (Com16 x) (Com16 y))
	// result: (Com16 (And16 <t> x y))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom16 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != OpCom16 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpCom16)
			v0 := b.NewValue0(v.Pos, OpAnd16, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (Or16 x x)
	// result: x
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Or16 (Const16 [0]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Or16 (Const16 [-1]) _)
	// result: (Const16 [-1])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != -1 {
				continue
			}
			v.reset(OpConst16)
			v.AuxInt = int16ToAuxInt(-1)
			return true
		}
		break
	}
	// match: (Or16 (Com16 x) x)
	// result: (Const16 [-1])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom16 {
				continue
			}
			x := v_0.Args[0]
			if x != v_1 {
				continue
			}
			v.reset(OpConst16)
			v.AuxInt = int16ToAuxInt(-1)
			return true
		}
		break
	}
	// match: (Or16 x (Or16 x y))
	// result: (Or16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpOr16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				y := v_1_1
				v.reset(OpOr16)
				v.AddArg2(x, y)
				return true
			}
		}
		break
	}
	// match: (Or16 (And16 x (Const16 [c2])) (Const16 <t> [c1]))
	// cond: ^(c1 | c2) == 0
	// result: (Or16 (Const16 <t> [c1]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAnd16 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				if v_0_1.Op != OpConst16 {
					continue
				}
				c2 := auxIntToInt16(v_0_1.AuxInt)
				if v_1.Op != OpConst16 {
					continue
				}
				t := v_1.Type
				c1 := auxIntToInt16(v_1.AuxInt)
				if !(^(c1 | c2) == 0) {
					continue
				}
				v.reset(OpOr16)
				v0 := b.NewValue0(v.Pos, OpConst16, t)
				v0.AuxInt = int16ToAuxInt(c1)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Or16 (Or16 i:(Const16 <t>) z) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Or16 i (Or16 <t> z x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpOr16 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst16 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst16 && x.Op != OpConst16) {
					continue
				}
				v.reset(OpOr16)
				v0 := b.NewValue0(v.Pos, OpOr16, t)
				v0.AddArg2(z, x)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (Or16 (Const16 <t> [c]) (Or16 (Const16 <t> [d]) x))
	// result: (Or16 (Const16 <t> [c|d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpOr16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst16 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt16(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpOr16)
				v0 := b.NewValue0(v.Pos, OpConst16, t)
				v0.AuxInt = int16ToAuxInt(c | d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Or16 (Lsh16x64 x z:(Const64 <t> [c])) (Rsh16Ux64 x (Const64 [d])))
	// cond: c < 16 && d == 16-c && canRotate(config, 16)
	// result: (RotateLeft16 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLsh16x64 {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(z.AuxInt)
			if v_1.Op != OpRsh16Ux64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(c < 16 && d == 16-c && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Or16 left:(Lsh16x64 x y) right:(Rsh16Ux64 x (Sub64 (Const64 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh16x64 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh16Ux64 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub64 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst64 || auxIntToInt64(right_1_0.AuxInt) != 16 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Or16 left:(Lsh16x32 x y) right:(Rsh16Ux32 x (Sub32 (Const32 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh16x32 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh16Ux32 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub32 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst32 || auxIntToInt32(right_1_0.AuxInt) != 16 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Or16 left:(Lsh16x16 x y) right:(Rsh16Ux16 x (Sub16 (Const16 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh16x16 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh16Ux16 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub16 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst16 || auxIntToInt16(right_1_0.AuxInt) != 16 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Or16 left:(Lsh16x8 x y) right:(Rsh16Ux8 x (Sub8 (Const8 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh16x8 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh16Ux8 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub8 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst8 || auxIntToInt8(right_1_0.AuxInt) != 16 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Or16 right:(Rsh16Ux64 x y) left:(Lsh16x64 x z:(Sub64 (Const64 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh16Ux64 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh16x64 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub64 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst64 || auxIntToInt64(z_0.AuxInt) != 16 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Or16 right:(Rsh16Ux32 x y) left:(Lsh16x32 x z:(Sub32 (Const32 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh16Ux32 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh16x32 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub32 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst32 || auxIntToInt32(z_0.AuxInt) != 16 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Or16 right:(Rsh16Ux16 x y) left:(Lsh16x16 x z:(Sub16 (Const16 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh16Ux16 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh16x16 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub16 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst16 || auxIntToInt16(z_0.AuxInt) != 16 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Or16 right:(Rsh16Ux8 x y) left:(Lsh16x8 x z:(Sub8 (Const8 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh16Ux8 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh16x8 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub8 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst8 || auxIntToInt8(z_0.AuxInt) != 16 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpOr32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (Or32 (Const32 [c]) (Const32 [d]))
	// result: (Const32 [c|d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			v.reset(OpConst32)
			v.AuxInt = int32ToAuxInt(c | d)
			return true
		}
		break
	}
	// match: (Or32 <t> (Com32 x) (Com32 y))
	// result: (Com32 (And32 <t> x y))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom32 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != OpCom32 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpCom32)
			v0 := b.NewValue0(v.Pos, OpAnd32, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (Or32 x x)
	// result: x
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Or32 (Const32 [0]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Or32 (Const32 [-1]) _)
	// result: (Const32 [-1])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != -1 {
				continue
			}
			v.reset(OpConst32)
			v.AuxInt = int32ToAuxInt(-1)
			return true
		}
		break
	}
	// match: (Or32 (Com32 x) x)
	// result: (Const32 [-1])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom32 {
				continue
			}
			x := v_0.Args[0]
			if x != v_1 {
				continue
			}
			v.reset(OpConst32)
			v.AuxInt = int32ToAuxInt(-1)
			return true
		}
		break
	}
	// match: (Or32 x (Or32 x y))
	// result: (Or32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpOr32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				y := v_1_1
				v.reset(OpOr32)
				v.AddArg2(x, y)
				return true
			}
		}
		break
	}
	// match: (Or32 (And32 x (Const32 [c2])) (Const32 <t> [c1]))
	// cond: ^(c1 | c2) == 0
	// result: (Or32 (Const32 <t> [c1]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAnd32 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				if v_0_1.Op != OpConst32 {
					continue
				}
				c2 := auxIntToInt32(v_0_1.AuxInt)
				if v_1.Op != OpConst32 {
					continue
				}
				t := v_1.Type
				c1 := auxIntToInt32(v_1.AuxInt)
				if !(^(c1 | c2) == 0) {
					continue
				}
				v.reset(OpOr32)
				v0 := b.NewValue0(v.Pos, OpConst32, t)
				v0.AuxInt = int32ToAuxInt(c1)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Or32 (Or32 i:(Const32 <t>) z) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Or32 i (Or32 <t> z x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpOr32 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst32 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst32 && x.Op != OpConst32) {
					continue
				}
				v.reset(OpOr32)
				v0 := b.NewValue0(v.Pos, OpOr32, t)
				v0.AddArg2(z, x)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (Or32 (Const32 <t> [c]) (Or32 (Const32 <t> [d]) x))
	// result: (Or32 (Const32 <t> [c|d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpOr32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst32 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt32(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpOr32)
				v0 := b.NewValue0(v.Pos, OpConst32, t)
				v0.AuxInt = int32ToAuxInt(c | d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Or32 (Lsh32x64 x z:(Const64 <t> [c])) (Rsh32Ux64 x (Const64 [d])))
	// cond: c < 32 && d == 32-c && canRotate(config, 32)
	// result: (RotateLeft32 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLsh32x64 {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(z.AuxInt)
			if v_1.Op != OpRsh32Ux64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(c < 32 && d == 32-c && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Or32 left:(Lsh32x64 x y) right:(Rsh32Ux64 x (Sub64 (Const64 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh32x64 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh32Ux64 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub64 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst64 || auxIntToInt64(right_1_0.AuxInt) != 32 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Or32 left:(Lsh32x32 x y) right:(Rsh32Ux32 x (Sub32 (Const32 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh32x32 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh32Ux32 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub32 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst32 || auxIntToInt32(right_1_0.AuxInt) != 32 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Or32 left:(Lsh32x16 x y) right:(Rsh32Ux16 x (Sub16 (Const16 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh32x16 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh32Ux16 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub16 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst16 || auxIntToInt16(right_1_0.AuxInt) != 32 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Or32 left:(Lsh32x8 x y) right:(Rsh32Ux8 x (Sub8 (Const8 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh32x8 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh32Ux8 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub8 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst8 || auxIntToInt8(right_1_0.AuxInt) != 32 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Or32 right:(Rsh32Ux64 x y) left:(Lsh32x64 x z:(Sub64 (Const64 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh32Ux64 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh32x64 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub64 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst64 || auxIntToInt64(z_0.AuxInt) != 32 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Or32 right:(Rsh32Ux32 x y) left:(Lsh32x32 x z:(Sub32 (Const32 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh32Ux32 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh32x32 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub32 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst32 || auxIntToInt32(z_0.AuxInt) != 32 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Or32 right:(Rsh32Ux16 x y) left:(Lsh32x16 x z:(Sub16 (Const16 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh32Ux16 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh32x16 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub16 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst16 || auxIntToInt16(z_0.AuxInt) != 32 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Or32 right:(Rsh32Ux8 x y) left:(Lsh32x8 x z:(Sub8 (Const8 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh32Ux8 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh32x8 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub8 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst8 || auxIntToInt8(z_0.AuxInt) != 32 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpOr64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (Or64 (Const64 [c]) (Const64 [d]))
	// result: (Const64 [c|d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpConst64)
			v.AuxInt = int64ToAuxInt(c | d)
			return true
		}
		break
	}
	// match: (Or64 <t> (Com64 x) (Com64 y))
	// result: (Com64 (And64 <t> x y))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom64 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != OpCom64 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpCom64)
			v0 := b.NewValue0(v.Pos, OpAnd64, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (Or64 x x)
	// result: x
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Or64 (Const64 [0]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Or64 (Const64 [-1]) _)
	// result: (Const64 [-1])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != -1 {
				continue
			}
			v.reset(OpConst64)
			v.AuxInt = int64ToAuxInt(-1)
			return true
		}
		break
	}
	// match: (Or64 (Com64 x) x)
	// result: (Const64 [-1])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom64 {
				continue
			}
			x := v_0.Args[0]
			if x != v_1 {
				continue
			}
			v.reset(OpConst64)
			v.AuxInt = int64ToAuxInt(-1)
			return true
		}
		break
	}
	// match: (Or64 x (Or64 x y))
	// result: (Or64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpOr64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				y := v_1_1
				v.reset(OpOr64)
				v.AddArg2(x, y)
				return true
			}
		}
		break
	}
	// match: (Or64 (And64 x (Const64 [c2])) (Const64 <t> [c1]))
	// cond: ^(c1 | c2) == 0
	// result: (Or64 (Const64 <t> [c1]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAnd64 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				if v_0_1.Op != OpConst64 {
					continue
				}
				c2 := auxIntToInt64(v_0_1.AuxInt)
				if v_1.Op != OpConst64 {
					continue
				}
				t := v_1.Type
				c1 := auxIntToInt64(v_1.AuxInt)
				if !(^(c1 | c2) == 0) {
					continue
				}
				v.reset(OpOr64)
				v0 := b.NewValue0(v.Pos, OpConst64, t)
				v0.AuxInt = int64ToAuxInt(c1)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Or64 (Or64 i:(Const64 <t>) z) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Or64 i (Or64 <t> z x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpOr64 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst64 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst64 && x.Op != OpConst64) {
					continue
				}
				v.reset(OpOr64)
				v0 := b.NewValue0(v.Pos, OpOr64, t)
				v0.AddArg2(z, x)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (Or64 (Const64 <t> [c]) (Or64 (Const64 <t> [d]) x))
	// result: (Or64 (Const64 <t> [c|d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpOr64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst64 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt64(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpOr64)
				v0 := b.NewValue0(v.Pos, OpConst64, t)
				v0.AuxInt = int64ToAuxInt(c | d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Or64 (Lsh64x64 x z:(Const64 <t> [c])) (Rsh64Ux64 x (Const64 [d])))
	// cond: c < 64 && d == 64-c && canRotate(config, 64)
	// result: (RotateLeft64 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLsh64x64 {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(z.AuxInt)
			if v_1.Op != OpRsh64Ux64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(c < 64 && d == 64-c && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Or64 left:(Lsh64x64 x y) right:(Rsh64Ux64 x (Sub64 (Const64 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh64x64 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh64Ux64 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub64 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst64 || auxIntToInt64(right_1_0.AuxInt) != 64 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Or64 left:(Lsh64x32 x y) right:(Rsh64Ux32 x (Sub32 (Const32 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh64x32 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh64Ux32 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub32 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst32 || auxIntToInt32(right_1_0.AuxInt) != 64 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Or64 left:(Lsh64x16 x y) right:(Rsh64Ux16 x (Sub16 (Const16 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh64x16 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh64Ux16 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub16 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst16 || auxIntToInt16(right_1_0.AuxInt) != 64 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Or64 left:(Lsh64x8 x y) right:(Rsh64Ux8 x (Sub8 (Const8 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh64x8 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh64Ux8 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub8 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst8 || auxIntToInt8(right_1_0.AuxInt) != 64 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Or64 right:(Rsh64Ux64 x y) left:(Lsh64x64 x z:(Sub64 (Const64 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh64Ux64 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh64x64 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub64 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst64 || auxIntToInt64(z_0.AuxInt) != 64 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Or64 right:(Rsh64Ux32 x y) left:(Lsh64x32 x z:(Sub32 (Const32 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh64Ux32 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh64x32 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub32 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst32 || auxIntToInt32(z_0.AuxInt) != 64 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Or64 right:(Rsh64Ux16 x y) left:(Lsh64x16 x z:(Sub16 (Const16 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh64Ux16 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh64x16 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub16 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst16 || auxIntToInt16(z_0.AuxInt) != 64 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Or64 right:(Rsh64Ux8 x y) left:(Lsh64x8 x z:(Sub8 (Const8 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh64Ux8 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh64x8 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub8 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst8 || auxIntToInt8(z_0.AuxInt) != 64 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpOr8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (Or8 (Const8 [c]) (Const8 [d]))
	// result: (Const8 [c|d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1.AuxInt)
			v.reset(OpConst8)
			v.AuxInt = int8ToAuxInt(c | d)
			return true
		}
		break
	}
	// match: (Or8 <t> (Com8 x) (Com8 y))
	// result: (Com8 (And8 <t> x y))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom8 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != OpCom8 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpCom8)
			v0 := b.NewValue0(v.Pos, OpAnd8, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (Or8 x x)
	// result: x
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Or8 (Const8 [0]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Or8 (Const8 [-1]) _)
	// result: (Const8 [-1])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != -1 {
				continue
			}
			v.reset(OpConst8)
			v.AuxInt = int8ToAuxInt(-1)
			return true
		}
		break
	}
	// match: (Or8 (Com8 x) x)
	// result: (Const8 [-1])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom8 {
				continue
			}
			x := v_0.Args[0]
			if x != v_1 {
				continue
			}
			v.reset(OpConst8)
			v.AuxInt = int8ToAuxInt(-1)
			return true
		}
		break
	}
	// match: (Or8 x (Or8 x y))
	// result: (Or8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpOr8 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				y := v_1_1
				v.reset(OpOr8)
				v.AddArg2(x, y)
				return true
			}
		}
		break
	}
	// match: (Or8 (And8 x (Const8 [c2])) (Const8 <t> [c1]))
	// cond: ^(c1 | c2) == 0
	// result: (Or8 (Const8 <t> [c1]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAnd8 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				if v_0_1.Op != OpConst8 {
					continue
				}
				c2 := auxIntToInt8(v_0_1.AuxInt)
				if v_1.Op != OpConst8 {
					continue
				}
				t := v_1.Type
				c1 := auxIntToInt8(v_1.AuxInt)
				if !(^(c1 | c2) == 0) {
					continue
				}
				v.reset(OpOr8)
				v0 := b.NewValue0(v.Pos, OpConst8, t)
				v0.AuxInt = int8ToAuxInt(c1)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Or8 (Or8 i:(Const8 <t>) z) x)
	// cond: (z.Op != OpConst8 && x.Op != OpConst8)
	// result: (Or8 i (Or8 <t> z x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpOr8 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst8 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst8 && x.Op != OpConst8) {
					continue
				}
				v.reset(OpOr8)
				v0 := b.NewValue0(v.Pos, OpOr8, t)
				v0.AddArg2(z, x)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (Or8 (Const8 <t> [c]) (Or8 (Const8 <t> [d]) x))
	// result: (Or8 (Const8 <t> [c|d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpOr8 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst8 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt8(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpOr8)
				v0 := b.NewValue0(v.Pos, OpConst8, t)
				v0.AuxInt = int8ToAuxInt(c | d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Or8 (Lsh8x64 x z:(Const64 <t> [c])) (Rsh8Ux64 x (Const64 [d])))
	// cond: c < 8 && d == 8-c && canRotate(config, 8)
	// result: (RotateLeft8 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLsh8x64 {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(z.AuxInt)
			if v_1.Op != OpRsh8Ux64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(c < 8 && d == 8-c && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Or8 left:(Lsh8x64 x y) right:(Rsh8Ux64 x (Sub64 (Const64 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh8x64 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh8Ux64 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub64 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst64 || auxIntToInt64(right_1_0.AuxInt) != 8 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Or8 left:(Lsh8x32 x y) right:(Rsh8Ux32 x (Sub32 (Const32 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh8x32 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh8Ux32 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub32 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst32 || auxIntToInt32(right_1_0.AuxInt) != 8 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Or8 left:(Lsh8x16 x y) right:(Rsh8Ux16 x (Sub16 (Const16 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh8x16 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh8Ux16 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub16 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst16 || auxIntToInt16(right_1_0.AuxInt) != 8 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Or8 left:(Lsh8x8 x y) right:(Rsh8Ux8 x (Sub8 (Const8 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh8x8 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh8Ux8 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub8 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst8 || auxIntToInt8(right_1_0.AuxInt) != 8 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Or8 right:(Rsh8Ux64 x y) left:(Lsh8x64 x z:(Sub64 (Const64 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh8Ux64 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh8x64 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub64 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst64 || auxIntToInt64(z_0.AuxInt) != 8 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Or8 right:(Rsh8Ux32 x y) left:(Lsh8x32 x z:(Sub32 (Const32 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh8Ux32 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh8x32 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub32 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst32 || auxIntToInt32(z_0.AuxInt) != 8 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Or8 right:(Rsh8Ux16 x y) left:(Lsh8x16 x z:(Sub16 (Const16 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh8Ux16 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh8x16 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub16 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst16 || auxIntToInt16(z_0.AuxInt) != 8 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Or8 right:(Rsh8Ux8 x y) left:(Lsh8x8 x z:(Sub8 (Const8 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh8Ux8 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh8x8 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub8 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst8 || auxIntToInt8(z_0.AuxInt) != 8 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpOrB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (OrB (Less64 (Const64 [c]) x) (Less64 x (Const64 [d])))
	// cond: c >= d
	// result: (Less64U (Const64 <x.Type> [c-d]) (Sub64 <x.Type> x (Const64 <x.Type> [d])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess64 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpLess64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(c >= d) {
				continue
			}
			v.reset(OpLess64U)
			v0 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v0.AuxInt = int64ToAuxInt(c - d)
			v1 := b.NewValue0(v.Pos, OpSub64, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v2.AuxInt = int64ToAuxInt(d)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Leq64 (Const64 [c]) x) (Less64 x (Const64 [d])))
	// cond: c >= d
	// result: (Leq64U (Const64 <x.Type> [c-d]) (Sub64 <x.Type> x (Const64 <x.Type> [d])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq64 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpLess64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(c >= d) {
				continue
			}
			v.reset(OpLeq64U)
			v0 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v0.AuxInt = int64ToAuxInt(c - d)
			v1 := b.NewValue0(v.Pos, OpSub64, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v2.AuxInt = int64ToAuxInt(d)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Less32 (Const32 [c]) x) (Less32 x (Const32 [d])))
	// cond: c >= d
	// result: (Less32U (Const32 <x.Type> [c-d]) (Sub32 <x.Type> x (Const32 <x.Type> [d])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess32 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpLess32 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1_1.AuxInt)
			if !(c >= d) {
				continue
			}
			v.reset(OpLess32U)
			v0 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v0.AuxInt = int32ToAuxInt(c - d)
			v1 := b.NewValue0(v.Pos, OpSub32, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v2.AuxInt = int32ToAuxInt(d)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Leq32 (Const32 [c]) x) (Less32 x (Const32 [d])))
	// cond: c >= d
	// result: (Leq32U (Const32 <x.Type> [c-d]) (Sub32 <x.Type> x (Const32 <x.Type> [d])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq32 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpLess32 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1_1.AuxInt)
			if !(c >= d) {
				continue
			}
			v.reset(OpLeq32U)
			v0 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v0.AuxInt = int32ToAuxInt(c - d)
			v1 := b.NewValue0(v.Pos, OpSub32, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v2.AuxInt = int32ToAuxInt(d)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Less16 (Const16 [c]) x) (Less16 x (Const16 [d])))
	// cond: c >= d
	// result: (Less16U (Const16 <x.Type> [c-d]) (Sub16 <x.Type> x (Const16 <x.Type> [d])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess16 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0.AuxInt)
			if v_1.Op != OpLess16 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1_1.AuxInt)
			if !(c >= d) {
				continue
			}
			v.reset(OpLess16U)
			v0 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v0.AuxInt = int16ToAuxInt(c - d)
			v1 := b.NewValue0(v.Pos, OpSub16, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v2.AuxInt = int16ToAuxInt(d)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Leq16 (Const16 [c]) x) (Less16 x (Const16 [d])))
	// cond: c >= d
	// result: (Leq16U (Const16 <x.Type> [c-d]) (Sub16 <x.Type> x (Const16 <x.Type> [d])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq16 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0.AuxInt)
			if v_1.Op != OpLess16 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1_1.AuxInt)
			if !(c >= d) {
				continue
			}
			v.reset(OpLeq16U)
			v0 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v0.AuxInt = int16ToAuxInt(c - d)
			v1 := b.NewValue0(v.Pos, OpSub16, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v2.AuxInt = int16ToAuxInt(d)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Less8 (Const8 [c]) x) (Less8 x (Const8 [d])))
	// cond: c >= d
	// result: (Less8U (Const8 <x.Type> [c-d]) (Sub8 <x.Type> x (Const8 <x.Type> [d])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess8 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0.AuxInt)
			if v_1.Op != OpLess8 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1_1.AuxInt)
			if !(c >= d) {
				continue
			}
			v.reset(OpLess8U)
			v0 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v0.AuxInt = int8ToAuxInt(c - d)
			v1 := b.NewValue0(v.Pos, OpSub8, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v2.AuxInt = int8ToAuxInt(d)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Leq8 (Const8 [c]) x) (Less8 x (Const8 [d])))
	// cond: c >= d
	// result: (Leq8U (Const8 <x.Type> [c-d]) (Sub8 <x.Type> x (Const8 <x.Type> [d])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq8 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0.AuxInt)
			if v_1.Op != OpLess8 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1_1.AuxInt)
			if !(c >= d) {
				continue
			}
			v.reset(OpLeq8U)
			v0 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v0.AuxInt = int8ToAuxInt(c - d)
			v1 := b.NewValue0(v.Pos, OpSub8, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v2.AuxInt = int8ToAuxInt(d)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Less64 (Const64 [c]) x) (Leq64 x (Const64 [d])))
	// cond: c >= d+1 && d+1 > d
	// result: (Less64U (Const64 <x.Type> [c-d-1]) (Sub64 <x.Type> x (Const64 <x.Type> [d+1])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess64 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpLeq64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(c >= d+1 && d+1 > d) {
				continue
			}
			v.reset(OpLess64U)
			v0 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v0.AuxInt = int64ToAuxInt(c - d - 1)
			v1 := b.NewValue0(v.Pos, OpSub64, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v2.AuxInt = int64ToAuxInt(d + 1)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Leq64 (Const64 [c]) x) (Leq64 x (Const64 [d])))
	// cond: c >= d+1 && d+1 > d
	// result: (Leq64U (Const64 <x.Type> [c-d-1]) (Sub64 <x.Type> x (Const64 <x.Type> [d+1])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq64 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpLeq64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(c >= d+1 && d+1 > d) {
				continue
			}
			v.reset(OpLeq64U)
			v0 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v0.AuxInt = int64ToAuxInt(c - d - 1)
			v1 := b.NewValue0(v.Pos, OpSub64, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v2.AuxInt = int64ToAuxInt(d + 1)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Less32 (Const32 [c]) x) (Leq32 x (Const32 [d])))
	// cond: c >= d+1 && d+1 > d
	// result: (Less32U (Const32 <x.Type> [c-d-1]) (Sub32 <x.Type> x (Const32 <x.Type> [d+1])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess32 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpLeq32 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1_1.AuxInt)
			if !(c >= d+1 && d+1 > d) {
				continue
			}
			v.reset(OpLess32U)
			v0 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v0.AuxInt = int32ToAuxInt(c - d - 1)
			v1 := b.NewValue0(v.Pos, OpSub32, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v2.AuxInt = int32ToAuxInt(d + 1)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Leq32 (Const32 [c]) x) (Leq32 x (Const32 [d])))
	// cond: c >= d+1 && d+1 > d
	// result: (Leq32U (Const32 <x.Type> [c-d-1]) (Sub32 <x.Type> x (Const32 <x.Type> [d+1])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq32 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpLeq32 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1_1.AuxInt)
			if !(c >= d+1 && d+1 > d) {
				continue
			}
			v.reset(OpLeq32U)
			v0 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v0.AuxInt = int32ToAuxInt(c - d - 1)
			v1 := b.NewValue0(v.Pos, OpSub32, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v2.AuxInt = int32ToAuxInt(d + 1)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Less16 (Const16 [c]) x) (Leq16 x (Const16 [d])))
	// cond: c >= d+1 && d+1 > d
	// result: (Less16U (Const16 <x.Type> [c-d-1]) (Sub16 <x.Type> x (Const16 <x.Type> [d+1])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess16 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0.AuxInt)
			if v_1.Op != OpLeq16 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1_1.AuxInt)
			if !(c >= d+1 && d+1 > d) {
				continue
			}
			v.reset(OpLess16U)
			v0 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v0.AuxInt = int16ToAuxInt(c - d - 1)
			v1 := b.NewValue0(v.Pos, OpSub16, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v2.AuxInt = int16ToAuxInt(d + 1)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Leq16 (Const16 [c]) x) (Leq16 x (Const16 [d])))
	// cond: c >= d+1 && d+1 > d
	// result: (Leq16U (Const16 <x.Type> [c-d-1]) (Sub16 <x.Type> x (Const16 <x.Type> [d+1])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq16 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0.AuxInt)
			if v_1.Op != OpLeq16 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1_1.AuxInt)
			if !(c >= d+1 && d+1 > d) {
				continue
			}
			v.reset(OpLeq16U)
			v0 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v0.AuxInt = int16ToAuxInt(c - d - 1)
			v1 := b.NewValue0(v.Pos, OpSub16, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v2.AuxInt = int16ToAuxInt(d + 1)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Less8 (Const8 [c]) x) (Leq8 x (Const8 [d])))
	// cond: c >= d+1 && d+1 > d
	// result: (Less8U (Const8 <x.Type> [c-d-1]) (Sub8 <x.Type> x (Const8 <x.Type> [d+1])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess8 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0.AuxInt)
			if v_1.Op != OpLeq8 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1_1.AuxInt)
			if !(c >= d+1 && d+1 > d) {
				continue
			}
			v.reset(OpLess8U)
			v0 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v0.AuxInt = int8ToAuxInt(c - d - 1)
			v1 := b.NewValue0(v.Pos, OpSub8, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v2.AuxInt = int8ToAuxInt(d + 1)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Leq8 (Const8 [c]) x) (Leq8 x (Const8 [d])))
	// cond: c >= d+1 && d+1 > d
	// result: (Leq8U (Const8 <x.Type> [c-d-1]) (Sub8 <x.Type> x (Const8 <x.Type> [d+1])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq8 {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0.AuxInt)
			if v_1.Op != OpLeq8 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1_1.AuxInt)
			if !(c >= d+1 && d+1 > d) {
				continue
			}
			v.reset(OpLeq8U)
			v0 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v0.AuxInt = int8ToAuxInt(c - d - 1)
			v1 := b.NewValue0(v.Pos, OpSub8, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v2.AuxInt = int8ToAuxInt(d + 1)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Less64U (Const64 [c]) x) (Less64U x (Const64 [d])))
	// cond: uint64(c) >= uint64(d)
	// result: (Less64U (Const64 <x.Type> [c-d]) (Sub64 <x.Type> x (Const64 <x.Type> [d])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess64U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpLess64U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(uint64(c) >= uint64(d)) {
				continue
			}
			v.reset(OpLess64U)
			v0 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v0.AuxInt = int64ToAuxInt(c - d)
			v1 := b.NewValue0(v.Pos, OpSub64, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v2.AuxInt = int64ToAuxInt(d)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Leq64U (Const64 [c]) x) (Less64U x (Const64 [d])))
	// cond: uint64(c) >= uint64(d)
	// result: (Leq64U (Const64 <x.Type> [c-d]) (Sub64 <x.Type> x (Const64 <x.Type> [d])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq64U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpLess64U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(uint64(c) >= uint64(d)) {
				continue
			}
			v.reset(OpLeq64U)
			v0 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v0.AuxInt = int64ToAuxInt(c - d)
			v1 := b.NewValue0(v.Pos, OpSub64, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v2.AuxInt = int64ToAuxInt(d)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Less32U (Const32 [c]) x) (Less32U x (Const32 [d])))
	// cond: uint32(c) >= uint32(d)
	// result: (Less32U (Const32 <x.Type> [c-d]) (Sub32 <x.Type> x (Const32 <x.Type> [d])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess32U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpLess32U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1_1.AuxInt)
			if !(uint32(c) >= uint32(d)) {
				continue
			}
			v.reset(OpLess32U)
			v0 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v0.AuxInt = int32ToAuxInt(c - d)
			v1 := b.NewValue0(v.Pos, OpSub32, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v2.AuxInt = int32ToAuxInt(d)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Leq32U (Const32 [c]) x) (Less32U x (Const32 [d])))
	// cond: uint32(c) >= uint32(d)
	// result: (Leq32U (Const32 <x.Type> [c-d]) (Sub32 <x.Type> x (Const32 <x.Type> [d])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq32U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpLess32U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1_1.AuxInt)
			if !(uint32(c) >= uint32(d)) {
				continue
			}
			v.reset(OpLeq32U)
			v0 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v0.AuxInt = int32ToAuxInt(c - d)
			v1 := b.NewValue0(v.Pos, OpSub32, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v2.AuxInt = int32ToAuxInt(d)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Less16U (Const16 [c]) x) (Less16U x (Const16 [d])))
	// cond: uint16(c) >= uint16(d)
	// result: (Less16U (Const16 <x.Type> [c-d]) (Sub16 <x.Type> x (Const16 <x.Type> [d])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess16U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0.AuxInt)
			if v_1.Op != OpLess16U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1_1.AuxInt)
			if !(uint16(c) >= uint16(d)) {
				continue
			}
			v.reset(OpLess16U)
			v0 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v0.AuxInt = int16ToAuxInt(c - d)
			v1 := b.NewValue0(v.Pos, OpSub16, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v2.AuxInt = int16ToAuxInt(d)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Leq16U (Const16 [c]) x) (Less16U x (Const16 [d])))
	// cond: uint16(c) >= uint16(d)
	// result: (Leq16U (Const16 <x.Type> [c-d]) (Sub16 <x.Type> x (Const16 <x.Type> [d])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq16U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0.AuxInt)
			if v_1.Op != OpLess16U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1_1.AuxInt)
			if !(uint16(c) >= uint16(d)) {
				continue
			}
			v.reset(OpLeq16U)
			v0 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v0.AuxInt = int16ToAuxInt(c - d)
			v1 := b.NewValue0(v.Pos, OpSub16, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v2.AuxInt = int16ToAuxInt(d)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Less8U (Const8 [c]) x) (Less8U x (Const8 [d])))
	// cond: uint8(c) >= uint8(d)
	// result: (Less8U (Const8 <x.Type> [c-d]) (Sub8 <x.Type> x (Const8 <x.Type> [d])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess8U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0.AuxInt)
			if v_1.Op != OpLess8U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1_1.AuxInt)
			if !(uint8(c) >= uint8(d)) {
				continue
			}
			v.reset(OpLess8U)
			v0 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v0.AuxInt = int8ToAuxInt(c - d)
			v1 := b.NewValue0(v.Pos, OpSub8, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v2.AuxInt = int8ToAuxInt(d)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Leq8U (Const8 [c]) x) (Less8U x (Const8 [d])))
	// cond: uint8(c) >= uint8(d)
	// result: (Leq8U (Const8 <x.Type> [c-d]) (Sub8 <x.Type> x (Const8 <x.Type> [d])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq8U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0.AuxInt)
			if v_1.Op != OpLess8U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1_1.AuxInt)
			if !(uint8(c) >= uint8(d)) {
				continue
			}
			v.reset(OpLeq8U)
			v0 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v0.AuxInt = int8ToAuxInt(c - d)
			v1 := b.NewValue0(v.Pos, OpSub8, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v2.AuxInt = int8ToAuxInt(d)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Less64U (Const64 [c]) x) (Leq64U x (Const64 [d])))
	// cond: uint64(c) >= uint64(d+1) && uint64(d+1) > uint64(d)
	// result: (Less64U (Const64 <x.Type> [c-d-1]) (Sub64 <x.Type> x (Const64 <x.Type> [d+1])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess64U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpLeq64U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(uint64(c) >= uint64(d+1) && uint64(d+1) > uint64(d)) {
				continue
			}
			v.reset(OpLess64U)
			v0 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v0.AuxInt = int64ToAuxInt(c - d - 1)
			v1 := b.NewValue0(v.Pos, OpSub64, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v2.AuxInt = int64ToAuxInt(d + 1)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Leq64U (Const64 [c]) x) (Leq64U x (Const64 [d])))
	// cond: uint64(c) >= uint64(d+1) && uint64(d+1) > uint64(d)
	// result: (Leq64U (Const64 <x.Type> [c-d-1]) (Sub64 <x.Type> x (Const64 <x.Type> [d+1])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq64U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if v_1.Op != OpLeq64U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(uint64(c) >= uint64(d+1) && uint64(d+1) > uint64(d)) {
				continue
			}
			v.reset(OpLeq64U)
			v0 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v0.AuxInt = int64ToAuxInt(c - d - 1)
			v1 := b.NewValue0(v.Pos, OpSub64, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst64, x.Type)
			v2.AuxInt = int64ToAuxInt(d + 1)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Less32U (Const32 [c]) x) (Leq32U x (Const32 [d])))
	// cond: uint32(c) >= uint32(d+1) && uint32(d+1) > uint32(d)
	// result: (Less32U (Const32 <x.Type> [c-d-1]) (Sub32 <x.Type> x (Const32 <x.Type> [d+1])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess32U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpLeq32U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1_1.AuxInt)
			if !(uint32(c) >= uint32(d+1) && uint32(d+1) > uint32(d)) {
				continue
			}
			v.reset(OpLess32U)
			v0 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v0.AuxInt = int32ToAuxInt(c - d - 1)
			v1 := b.NewValue0(v.Pos, OpSub32, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v2.AuxInt = int32ToAuxInt(d + 1)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Leq32U (Const32 [c]) x) (Leq32U x (Const32 [d])))
	// cond: uint32(c) >= uint32(d+1) && uint32(d+1) > uint32(d)
	// result: (Leq32U (Const32 <x.Type> [c-d-1]) (Sub32 <x.Type> x (Const32 <x.Type> [d+1])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq32U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0_0.AuxInt)
			if v_1.Op != OpLeq32U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1_1.AuxInt)
			if !(uint32(c) >= uint32(d+1) && uint32(d+1) > uint32(d)) {
				continue
			}
			v.reset(OpLeq32U)
			v0 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v0.AuxInt = int32ToAuxInt(c - d - 1)
			v1 := b.NewValue0(v.Pos, OpSub32, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst32, x.Type)
			v2.AuxInt = int32ToAuxInt(d + 1)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Less16U (Const16 [c]) x) (Leq16U x (Const16 [d])))
	// cond: uint16(c) >= uint16(d+1) && uint16(d+1) > uint16(d)
	// result: (Less16U (Const16 <x.Type> [c-d-1]) (Sub16 <x.Type> x (Const16 <x.Type> [d+1])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess16U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0.AuxInt)
			if v_1.Op != OpLeq16U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1_1.AuxInt)
			if !(uint16(c) >= uint16(d+1) && uint16(d+1) > uint16(d)) {
				continue
			}
			v.reset(OpLess16U)
			v0 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v0.AuxInt = int16ToAuxInt(c - d - 1)
			v1 := b.NewValue0(v.Pos, OpSub16, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v2.AuxInt = int16ToAuxInt(d + 1)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Leq16U (Const16 [c]) x) (Leq16U x (Const16 [d])))
	// cond: uint16(c) >= uint16(d+1) && uint16(d+1) > uint16(d)
	// result: (Leq16U (Const16 <x.Type> [c-d-1]) (Sub16 <x.Type> x (Const16 <x.Type> [d+1])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq16U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0_0.AuxInt)
			if v_1.Op != OpLeq16U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1_1.AuxInt)
			if !(uint16(c) >= uint16(d+1) && uint16(d+1) > uint16(d)) {
				continue
			}
			v.reset(OpLeq16U)
			v0 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v0.AuxInt = int16ToAuxInt(c - d - 1)
			v1 := b.NewValue0(v.Pos, OpSub16, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst16, x.Type)
			v2.AuxInt = int16ToAuxInt(d + 1)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Less8U (Const8 [c]) x) (Leq8U x (Const8 [d])))
	// cond: uint8(c) >= uint8(d+1) && uint8(d+1) > uint8(d)
	// result: (Less8U (Const8 <x.Type> [c-d-1]) (Sub8 <x.Type> x (Const8 <x.Type> [d+1])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLess8U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0.AuxInt)
			if v_1.Op != OpLeq8U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1_1.AuxInt)
			if !(uint8(c) >= uint8(d+1) && uint8(d+1) > uint8(d)) {
				continue
			}
			v.reset(OpLess8U)
			v0 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v0.AuxInt = int8ToAuxInt(c - d - 1)
			v1 := b.NewValue0(v.Pos, OpSub8, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v2.AuxInt = int8ToAuxInt(d + 1)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	// match: (OrB (Leq8U (Const8 [c]) x) (Leq8U x (Const8 [d])))
	// cond: uint8(c) >= uint8(d+1) && uint8(d+1) > uint8(d)
	// result: (Leq8U (Const8 <x.Type> [c-d-1]) (Sub8 <x.Type> x (Const8 <x.Type> [d+1])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLeq8U {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0_0.AuxInt)
			if v_1.Op != OpLeq8U {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1_1.AuxInt)
			if !(uint8(c) >= uint8(d+1) && uint8(d+1) > uint8(d)) {
				continue
			}
			v.reset(OpLeq8U)
			v0 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v0.AuxInt = int8ToAuxInt(c - d - 1)
			v1 := b.NewValue0(v.Pos, OpSub8, x.Type)
			v2 := b.NewValue0(v.Pos, OpConst8, x.Type)
			v2.AuxInt = int8ToAuxInt(d + 1)
			v1.AddArg2(x, v2)
			v.AddArg2(v0, v1)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpPhi(v *Value) bool {
	b := v.Block
	// match: (Phi (Const8 [c]) (Const8 [c]))
	// result: (Const8 [c])
	for {
		if len(v.Args) != 2 {
			break
		}
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 || auxIntToInt8(v_1.AuxInt) != c {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(c)
		return true
	}
	// match: (Phi (Const16 [c]) (Const16 [c]))
	// result: (Const16 [c])
	for {
		if len(v.Args) != 2 {
			break
		}
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 || auxIntToInt16(v_1.AuxInt) != c {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(c)
		return true
	}
	// match: (Phi (Const32 [c]) (Const32 [c]))
	// result: (Const32 [c])
	for {
		if len(v.Args) != 2 {
			break
		}
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 || auxIntToInt32(v_1.AuxInt) != c {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(c)
		return true
	}
	// match: (Phi (Const64 [c]) (Const64 [c]))
	// result: (Const64 [c])
	for {
		if len(v.Args) != 2 {
			break
		}
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != c {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(c)
		return true
	}
	// match: (Phi <t> nx:(Not x) ny:(Not y))
	// cond: nx.Uses == 1 && ny.Uses == 1
	// result: (Not (Phi <t> x y))
	for {
		if len(v.Args) != 2 {
			break
		}
		t := v.Type
		_ = v.Args[1]
		nx := v.Args[0]
		if nx.Op != OpNot {
			break
		}
		x := nx.Args[0]
		ny := v.Args[1]
		if ny.Op != OpNot {
			break
		}
		y := ny.Args[0]
		if !(nx.Uses == 1 && ny.Uses == 1) {
			break
		}
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpPhi, t)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpPopCount16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (PopCount16 (Const16 [c]))
	// cond: config.PtrSize == 8
	// result: (Const64 [int64(bits.OnesCount16(uint16(c)))])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(bits.OnesCount16(uint16(c))))
		return true
	}
	// match: (PopCount16 (Const16 [c]))
	// cond: config.PtrSize == 4
	// result: (Const32 [int32(bits.OnesCount16(uint16(c)))])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(bits.OnesCount16(uint16(c))))
		return true
	}
	return false
}
func rewriteValuegeneric_OpPopCount32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (PopCount32 (Const32 [c]))
	// cond: config.PtrSize == 8
	// result: (Const64 [int64(bits.OnesCount32(uint32(c)))])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(bits.OnesCount32(uint32(c))))
		return true
	}
	// match: (PopCount32 (Const32 [c]))
	// cond: config.PtrSize == 4
	// result: (Const32 [int32(bits.OnesCount32(uint32(c)))])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(bits.OnesCount32(uint32(c))))
		return true
	}
	return false
}
func rewriteValuegeneric_OpPopCount64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (PopCount64 (Const64 [c]))
	// cond: config.PtrSize == 8
	// result: (Const64 [int64(bits.OnesCount64(uint64(c)))])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(bits.OnesCount64(uint64(c))))
		return true
	}
	// match: (PopCount64 (Const64 [c]))
	// cond: config.PtrSize == 4
	// result: (Const32 [int32(bits.OnesCount64(uint64(c)))])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(bits.OnesCount64(uint64(c))))
		return true
	}
	return false
}
func rewriteValuegeneric_OpPopCount8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (PopCount8 (Const8 [c]))
	// cond: config.PtrSize == 8
	// result: (Const64 [int64(bits.OnesCount8(uint8(c)))])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(bits.OnesCount8(uint8(c))))
		return true
	}
	// match: (PopCount8 (Const8 [c]))
	// cond: config.PtrSize == 4
	// result: (Const32 [int32(bits.OnesCount8(uint8(c)))])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(bits.OnesCount8(uint8(c))))
		return true
	}
	return false
}
func rewriteValuegeneric_OpPtrIndex(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (PtrIndex <t> ptr idx)
	// cond: config.PtrSize == 4 && is32Bit(t.Elem().Size())
	// result: (AddPtr ptr (Mul32 <typ.Int> idx (Const32 <typ.Int> [int32(t.Elem().Size())])))
	for {
		t := v.Type
		ptr := v_0
		idx := v_1
		if !(config.PtrSize == 4 && is32Bit(t.Elem().Size())) {
			break
		}
		v.reset(OpAddPtr)
		v0 := b.NewValue0(v.Pos, OpMul32, typ.Int)
		v1 := b.NewValue0(v.Pos, OpConst32, typ.Int)
		v1.AuxInt = int32ToAuxInt(int32(t.Elem().Size()))
		v0.AddArg2(idx, v1)
		v.AddArg2(ptr, v0)
		return true
	}
	// match: (PtrIndex <t> ptr idx)
	// cond: config.PtrSize == 8
	// result: (AddPtr ptr (Mul64 <typ.Int> idx (Const64 <typ.Int> [t.Elem().Size()])))
	for {
		t := v.Type
		ptr := v_0
		idx := v_1
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpAddPtr)
		v0 := b.NewValue0(v.Pos, OpMul64, typ.Int)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.Int)
		v1.AuxInt = int64ToAuxInt(t.Elem().Size())
		v0.AddArg2(idx, v1)
		v.AddArg2(ptr, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRotateLeft16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (RotateLeft16 x (Const16 [c]))
	// cond: c%16 == 0
	// result: x
	for {
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(c%16 == 0) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (RotateLeft16 x (And64 y (Const64 [c])))
	// cond: c&15 == 15
	// result: (RotateLeft16 x y)
	for {
		x := v_0
		if v_1.Op != OpAnd64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_1.AuxInt)
			if !(c&15 == 15) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft16 x (And32 y (Const32 [c])))
	// cond: c&15 == 15
	// result: (RotateLeft16 x y)
	for {
		x := v_0
		if v_1.Op != OpAnd32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_1_1.AuxInt)
			if !(c&15 == 15) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft16 x (And16 y (Const16 [c])))
	// cond: c&15 == 15
	// result: (RotateLeft16 x y)
	for {
		x := v_0
		if v_1.Op != OpAnd16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_1_1.AuxInt)
			if !(c&15 == 15) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft16 x (And8 y (Const8 [c])))
	// cond: c&15 == 15
	// result: (RotateLeft16 x y)
	for {
		x := v_0
		if v_1.Op != OpAnd8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_1_1.AuxInt)
			if !(c&15 == 15) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft16 x (Neg64 (And64 y (Const64 [c]))))
	// cond: c&15 == 15
	// result: (RotateLeft16 x (Neg64 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpNeg64 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAnd64 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		v_1_0_1 := v_1_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0_0, v_1_0_1 = _i0+1, v_1_0_1, v_1_0_0 {
			y := v_1_0_0
			if v_1_0_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_0_1.AuxInt)
			if !(c&15 == 15) {
				continue
			}
			v.reset(OpRotateLeft16)
			v0 := b.NewValue0(v.Pos, OpNeg64, y.Type)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (RotateLeft16 x (Neg32 (And32 y (Const32 [c]))))
	// cond: c&15 == 15
	// result: (RotateLeft16 x (Neg32 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpNeg32 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAnd32 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		v_1_0_1 := v_1_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0_0, v_1_0_1 = _i0+1, v_1_0_1, v_1_0_0 {
			y := v_1_0_0
			if v_1_0_1.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_1_0_1.AuxInt)
			if !(c&15 == 15) {
				continue
			}
			v.reset(OpRotateLeft16)
			v0 := b.NewValue0(v.Pos, OpNeg32, y.Type)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (RotateLeft16 x (Neg16 (And16 y (Const16 [c]))))
	// cond: c&15 == 15
	// result: (RotateLeft16 x (Neg16 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpNeg16 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAnd16 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		v_1_0_1 := v_1_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0_0, v_1_0_1 = _i0+1, v_1_0_1, v_1_0_0 {
			y := v_1_0_0
			if v_1_0_1.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_1_0_1.AuxInt)
			if !(c&15 == 15) {
				continue
			}
			v.reset(OpRotateLeft16)
			v0 := b.NewValue0(v.Pos, OpNeg16, y.Type)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (RotateLeft16 x (Neg8 (And8 y (Const8 [c]))))
	// cond: c&15 == 15
	// result: (RotateLeft16 x (Neg8 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpNeg8 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAnd8 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		v_1_0_1 := v_1_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0_0, v_1_0_1 = _i0+1, v_1_0_1, v_1_0_0 {
			y := v_1_0_0
			if v_1_0_1.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_1_0_1.AuxInt)
			if !(c&15 == 15) {
				continue
			}
			v.reset(OpRotateLeft16)
			v0 := b.NewValue0(v.Pos, OpNeg8, y.Type)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (RotateLeft16 x (Add64 y (Const64 [c])))
	// cond: c&15 == 0
	// result: (RotateLeft16 x y)
	for {
		x := v_0
		if v_1.Op != OpAdd64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_1.AuxInt)
			if !(c&15 == 0) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft16 x (Add32 y (Const32 [c])))
	// cond: c&15 == 0
	// result: (RotateLeft16 x y)
	for {
		x := v_0
		if v_1.Op != OpAdd32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_1_1.AuxInt)
			if !(c&15 == 0) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft16 x (Add16 y (Const16 [c])))
	// cond: c&15 == 0
	// result: (RotateLeft16 x y)
	for {
		x := v_0
		if v_1.Op != OpAdd16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_1_1.AuxInt)
			if !(c&15 == 0) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft16 x (Add8 y (Const8 [c])))
	// cond: c&15 == 0
	// result: (RotateLeft16 x y)
	for {
		x := v_0
		if v_1.Op != OpAdd8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_1_1.AuxInt)
			if !(c&15 == 0) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft16 x (Sub64 (Const64 [c]) y))
	// cond: c&15 == 0
	// result: (RotateLeft16 x (Neg64 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpSub64 {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1_0.AuxInt)
		if !(c&15 == 0) {
			break
		}
		v.reset(OpRotateLeft16)
		v0 := b.NewValue0(v.Pos, OpNeg64, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft16 x (Sub32 (Const32 [c]) y))
	// cond: c&15 == 0
	// result: (RotateLeft16 x (Neg32 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpSub32 {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		if !(c&15 == 0) {
			break
		}
		v.reset(OpRotateLeft16)
		v0 := b.NewValue0(v.Pos, OpNeg32, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft16 x (Sub16 (Const16 [c]) y))
	// cond: c&15 == 0
	// result: (RotateLeft16 x (Neg16 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpSub16 {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1_0.AuxInt)
		if !(c&15 == 0) {
			break
		}
		v.reset(OpRotateLeft16)
		v0 := b.NewValue0(v.Pos, OpNeg16, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft16 x (Sub8 (Const8 [c]) y))
	// cond: c&15 == 0
	// result: (RotateLeft16 x (Neg8 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpSub8 {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1_0.AuxInt)
		if !(c&15 == 0) {
			break
		}
		v.reset(OpRotateLeft16)
		v0 := b.NewValue0(v.Pos, OpNeg8, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft16 x (Const64 <t> [c]))
	// cond: config.PtrSize == 4
	// result: (RotateLeft16 x (Const32 <t> [int32(c)]))
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		c := auxIntToInt64(v_1.AuxInt)
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpRotateLeft16)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft16 (RotateLeft16 x c) d)
	// cond: c.Type.Size() == 8 && d.Type.Size() == 8
	// result: (RotateLeft16 x (Add64 <c.Type> c d))
	for {
		if v_0.Op != OpRotateLeft16 {
			break
		}
		c := v_0.Args[1]
		x := v_0.Args[0]
		d := v_1
		if !(c.Type.Size() == 8 && d.Type.Size() == 8) {
			break
		}
		v.reset(OpRotateLeft16)
		v0 := b.NewValue0(v.Pos, OpAdd64, c.Type)
		v0.AddArg2(c, d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft16 (RotateLeft16 x c) d)
	// cond: c.Type.Size() == 4 && d.Type.Size() == 4
	// result: (RotateLeft16 x (Add32 <c.Type> c d))
	for {
		if v_0.Op != OpRotateLeft16 {
			break
		}
		c := v_0.Args[1]
		x := v_0.Args[0]
		d := v_1
		if !(c.Type.Size() == 4 && d.Type.Size() == 4) {
			break
		}
		v.reset(OpRotateLeft16)
		v0 := b.NewValue0(v.Pos, OpAdd32, c.Type)
		v0.AddArg2(c, d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft16 (RotateLeft16 x c) d)
	// cond: c.Type.Size() == 2 && d.Type.Size() == 2
	// result: (RotateLeft16 x (Add16 <c.Type> c d))
	for {
		if v_0.Op != OpRotateLeft16 {
			break
		}
		c := v_0.Args[1]
		x := v_0.Args[0]
		d := v_1
		if !(c.Type.Size() == 2 && d.Type.Size() == 2) {
			break
		}
		v.reset(OpRotateLeft16)
		v0 := b.NewValue0(v.Pos, OpAdd16, c.Type)
		v0.AddArg2(c, d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft16 (RotateLeft16 x c) d)
	// cond: c.Type.Size() == 1 && d.Type.Size() == 1
	// result: (RotateLeft16 x (Add8 <c.Type> c d))
	for {
		if v_0.Op != OpRotateLeft16 {
			break
		}
		c := v_0.Args[1]
		x := v_0.Args[0]
		d := v_1
		if !(c.Type.Size() == 1 && d.Type.Size() == 1) {
			break
		}
		v.reset(OpRotateLeft16)
		v0 := b.NewValue0(v.Pos, OpAdd8, c.Type)
		v0.AddArg2(c, d)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRotateLeft32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (RotateLeft32 x (Const32 [c]))
	// cond: c%32 == 0
	// result: x
	for {
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(c%32 == 0) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (RotateLeft32 x (And64 y (Const64 [c])))
	// cond: c&31 == 31
	// result: (RotateLeft32 x y)
	for {
		x := v_0
		if v_1.Op != OpAnd64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_1.AuxInt)
			if !(c&31 == 31) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft32 x (And32 y (Const32 [c])))
	// cond: c&31 == 31
	// result: (RotateLeft32 x y)
	for {
		x := v_0
		if v_1.Op != OpAnd32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_1_1.AuxInt)
			if !(c&31 == 31) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft32 x (And16 y (Const16 [c])))
	// cond: c&31 == 31
	// result: (RotateLeft32 x y)
	for {
		x := v_0
		if v_1.Op != OpAnd16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_1_1.AuxInt)
			if !(c&31 == 31) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft32 x (And8 y (Const8 [c])))
	// cond: c&31 == 31
	// result: (RotateLeft32 x y)
	for {
		x := v_0
		if v_1.Op != OpAnd8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_1_1.AuxInt)
			if !(c&31 == 31) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft32 x (Neg64 (And64 y (Const64 [c]))))
	// cond: c&31 == 31
	// result: (RotateLeft32 x (Neg64 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpNeg64 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAnd64 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		v_1_0_1 := v_1_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0_0, v_1_0_1 = _i0+1, v_1_0_1, v_1_0_0 {
			y := v_1_0_0
			if v_1_0_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_0_1.AuxInt)
			if !(c&31 == 31) {
				continue
			}
			v.reset(OpRotateLeft32)
			v0 := b.NewValue0(v.Pos, OpNeg64, y.Type)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (RotateLeft32 x (Neg32 (And32 y (Const32 [c]))))
	// cond: c&31 == 31
	// result: (RotateLeft32 x (Neg32 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpNeg32 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAnd32 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		v_1_0_1 := v_1_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0_0, v_1_0_1 = _i0+1, v_1_0_1, v_1_0_0 {
			y := v_1_0_0
			if v_1_0_1.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_1_0_1.AuxInt)
			if !(c&31 == 31) {
				continue
			}
			v.reset(OpRotateLeft32)
			v0 := b.NewValue0(v.Pos, OpNeg32, y.Type)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (RotateLeft32 x (Neg16 (And16 y (Const16 [c]))))
	// cond: c&31 == 31
	// result: (RotateLeft32 x (Neg16 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpNeg16 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAnd16 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		v_1_0_1 := v_1_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0_0, v_1_0_1 = _i0+1, v_1_0_1, v_1_0_0 {
			y := v_1_0_0
			if v_1_0_1.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_1_0_1.AuxInt)
			if !(c&31 == 31) {
				continue
			}
			v.reset(OpRotateLeft32)
			v0 := b.NewValue0(v.Pos, OpNeg16, y.Type)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (RotateLeft32 x (Neg8 (And8 y (Const8 [c]))))
	// cond: c&31 == 31
	// result: (RotateLeft32 x (Neg8 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpNeg8 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAnd8 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		v_1_0_1 := v_1_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0_0, v_1_0_1 = _i0+1, v_1_0_1, v_1_0_0 {
			y := v_1_0_0
			if v_1_0_1.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_1_0_1.AuxInt)
			if !(c&31 == 31) {
				continue
			}
			v.reset(OpRotateLeft32)
			v0 := b.NewValue0(v.Pos, OpNeg8, y.Type)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (RotateLeft32 x (Add64 y (Const64 [c])))
	// cond: c&31 == 0
	// result: (RotateLeft32 x y)
	for {
		x := v_0
		if v_1.Op != OpAdd64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_1.AuxInt)
			if !(c&31 == 0) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft32 x (Add32 y (Const32 [c])))
	// cond: c&31 == 0
	// result: (RotateLeft32 x y)
	for {
		x := v_0
		if v_1.Op != OpAdd32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_1_1.AuxInt)
			if !(c&31 == 0) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft32 x (Add16 y (Const16 [c])))
	// cond: c&31 == 0
	// result: (RotateLeft32 x y)
	for {
		x := v_0
		if v_1.Op != OpAdd16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_1_1.AuxInt)
			if !(c&31 == 0) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft32 x (Add8 y (Const8 [c])))
	// cond: c&31 == 0
	// result: (RotateLeft32 x y)
	for {
		x := v_0
		if v_1.Op != OpAdd8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_1_1.AuxInt)
			if !(c&31 == 0) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft32 x (Sub64 (Const64 [c]) y))
	// cond: c&31 == 0
	// result: (RotateLeft32 x (Neg64 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpSub64 {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1_0.AuxInt)
		if !(c&31 == 0) {
			break
		}
		v.reset(OpRotateLeft32)
		v0 := b.NewValue0(v.Pos, OpNeg64, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft32 x (Sub32 (Const32 [c]) y))
	// cond: c&31 == 0
	// result: (RotateLeft32 x (Neg32 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpSub32 {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		if !(c&31 == 0) {
			break
		}
		v.reset(OpRotateLeft32)
		v0 := b.NewValue0(v.Pos, OpNeg32, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft32 x (Sub16 (Const16 [c]) y))
	// cond: c&31 == 0
	// result: (RotateLeft32 x (Neg16 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpSub16 {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1_0.AuxInt)
		if !(c&31 == 0) {
			break
		}
		v.reset(OpRotateLeft32)
		v0 := b.NewValue0(v.Pos, OpNeg16, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft32 x (Sub8 (Const8 [c]) y))
	// cond: c&31 == 0
	// result: (RotateLeft32 x (Neg8 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpSub8 {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1_0.AuxInt)
		if !(c&31 == 0) {
			break
		}
		v.reset(OpRotateLeft32)
		v0 := b.NewValue0(v.Pos, OpNeg8, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft32 x (Const64 <t> [c]))
	// cond: config.PtrSize == 4
	// result: (RotateLeft32 x (Const32 <t> [int32(c)]))
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		c := auxIntToInt64(v_1.AuxInt)
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpRotateLeft32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft32 (RotateLeft32 x c) d)
	// cond: c.Type.Size() == 8 && d.Type.Size() == 8
	// result: (RotateLeft32 x (Add64 <c.Type> c d))
	for {
		if v_0.Op != OpRotateLeft32 {
			break
		}
		c := v_0.Args[1]
		x := v_0.Args[0]
		d := v_1
		if !(c.Type.Size() == 8 && d.Type.Size() == 8) {
			break
		}
		v.reset(OpRotateLeft32)
		v0 := b.NewValue0(v.Pos, OpAdd64, c.Type)
		v0.AddArg2(c, d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft32 (RotateLeft32 x c) d)
	// cond: c.Type.Size() == 4 && d.Type.Size() == 4
	// result: (RotateLeft32 x (Add32 <c.Type> c d))
	for {
		if v_0.Op != OpRotateLeft32 {
			break
		}
		c := v_0.Args[1]
		x := v_0.Args[0]
		d := v_1
		if !(c.Type.Size() == 4 && d.Type.Size() == 4) {
			break
		}
		v.reset(OpRotateLeft32)
		v0 := b.NewValue0(v.Pos, OpAdd32, c.Type)
		v0.AddArg2(c, d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft32 (RotateLeft32 x c) d)
	// cond: c.Type.Size() == 2 && d.Type.Size() == 2
	// result: (RotateLeft32 x (Add16 <c.Type> c d))
	for {
		if v_0.Op != OpRotateLeft32 {
			break
		}
		c := v_0.Args[1]
		x := v_0.Args[0]
		d := v_1
		if !(c.Type.Size() == 2 && d.Type.Size() == 2) {
			break
		}
		v.reset(OpRotateLeft32)
		v0 := b.NewValue0(v.Pos, OpAdd16, c.Type)
		v0.AddArg2(c, d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft32 (RotateLeft32 x c) d)
	// cond: c.Type.Size() == 1 && d.Type.Size() == 1
	// result: (RotateLeft32 x (Add8 <c.Type> c d))
	for {
		if v_0.Op != OpRotateLeft32 {
			break
		}
		c := v_0.Args[1]
		x := v_0.Args[0]
		d := v_1
		if !(c.Type.Size() == 1 && d.Type.Size() == 1) {
			break
		}
		v.reset(OpRotateLeft32)
		v0 := b.NewValue0(v.Pos, OpAdd8, c.Type)
		v0.AddArg2(c, d)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRotateLeft64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (RotateLeft64 x (Const64 [c]))
	// cond: c%64 == 0
	// result: x
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(c%64 == 0) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (RotateLeft64 x (And64 y (Const64 [c])))
	// cond: c&63 == 63
	// result: (RotateLeft64 x y)
	for {
		x := v_0
		if v_1.Op != OpAnd64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_1.AuxInt)
			if !(c&63 == 63) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft64 x (And32 y (Const32 [c])))
	// cond: c&63 == 63
	// result: (RotateLeft64 x y)
	for {
		x := v_0
		if v_1.Op != OpAnd32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_1_1.AuxInt)
			if !(c&63 == 63) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft64 x (And16 y (Const16 [c])))
	// cond: c&63 == 63
	// result: (RotateLeft64 x y)
	for {
		x := v_0
		if v_1.Op != OpAnd16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_1_1.AuxInt)
			if !(c&63 == 63) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft64 x (And8 y (Const8 [c])))
	// cond: c&63 == 63
	// result: (RotateLeft64 x y)
	for {
		x := v_0
		if v_1.Op != OpAnd8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_1_1.AuxInt)
			if !(c&63 == 63) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft64 x (Neg64 (And64 y (Const64 [c]))))
	// cond: c&63 == 63
	// result: (RotateLeft64 x (Neg64 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpNeg64 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAnd64 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		v_1_0_1 := v_1_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0_0, v_1_0_1 = _i0+1, v_1_0_1, v_1_0_0 {
			y := v_1_0_0
			if v_1_0_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_0_1.AuxInt)
			if !(c&63 == 63) {
				continue
			}
			v.reset(OpRotateLeft64)
			v0 := b.NewValue0(v.Pos, OpNeg64, y.Type)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (RotateLeft64 x (Neg32 (And32 y (Const32 [c]))))
	// cond: c&63 == 63
	// result: (RotateLeft64 x (Neg32 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpNeg32 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAnd32 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		v_1_0_1 := v_1_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0_0, v_1_0_1 = _i0+1, v_1_0_1, v_1_0_0 {
			y := v_1_0_0
			if v_1_0_1.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_1_0_1.AuxInt)
			if !(c&63 == 63) {
				continue
			}
			v.reset(OpRotateLeft64)
			v0 := b.NewValue0(v.Pos, OpNeg32, y.Type)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (RotateLeft64 x (Neg16 (And16 y (Const16 [c]))))
	// cond: c&63 == 63
	// result: (RotateLeft64 x (Neg16 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpNeg16 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAnd16 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		v_1_0_1 := v_1_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0_0, v_1_0_1 = _i0+1, v_1_0_1, v_1_0_0 {
			y := v_1_0_0
			if v_1_0_1.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_1_0_1.AuxInt)
			if !(c&63 == 63) {
				continue
			}
			v.reset(OpRotateLeft64)
			v0 := b.NewValue0(v.Pos, OpNeg16, y.Type)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (RotateLeft64 x (Neg8 (And8 y (Const8 [c]))))
	// cond: c&63 == 63
	// result: (RotateLeft64 x (Neg8 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpNeg8 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAnd8 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		v_1_0_1 := v_1_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0_0, v_1_0_1 = _i0+1, v_1_0_1, v_1_0_0 {
			y := v_1_0_0
			if v_1_0_1.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_1_0_1.AuxInt)
			if !(c&63 == 63) {
				continue
			}
			v.reset(OpRotateLeft64)
			v0 := b.NewValue0(v.Pos, OpNeg8, y.Type)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (RotateLeft64 x (Add64 y (Const64 [c])))
	// cond: c&63 == 0
	// result: (RotateLeft64 x y)
	for {
		x := v_0
		if v_1.Op != OpAdd64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_1.AuxInt)
			if !(c&63 == 0) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft64 x (Add32 y (Const32 [c])))
	// cond: c&63 == 0
	// result: (RotateLeft64 x y)
	for {
		x := v_0
		if v_1.Op != OpAdd32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_1_1.AuxInt)
			if !(c&63 == 0) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft64 x (Add16 y (Const16 [c])))
	// cond: c&63 == 0
	// result: (RotateLeft64 x y)
	for {
		x := v_0
		if v_1.Op != OpAdd16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_1_1.AuxInt)
			if !(c&63 == 0) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft64 x (Add8 y (Const8 [c])))
	// cond: c&63 == 0
	// result: (RotateLeft64 x y)
	for {
		x := v_0
		if v_1.Op != OpAdd8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_1_1.AuxInt)
			if !(c&63 == 0) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft64 x (Sub64 (Const64 [c]) y))
	// cond: c&63 == 0
	// result: (RotateLeft64 x (Neg64 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpSub64 {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1_0.AuxInt)
		if !(c&63 == 0) {
			break
		}
		v.reset(OpRotateLeft64)
		v0 := b.NewValue0(v.Pos, OpNeg64, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft64 x (Sub32 (Const32 [c]) y))
	// cond: c&63 == 0
	// result: (RotateLeft64 x (Neg32 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpSub32 {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		if !(c&63 == 0) {
			break
		}
		v.reset(OpRotateLeft64)
		v0 := b.NewValue0(v.Pos, OpNeg32, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft64 x (Sub16 (Const16 [c]) y))
	// cond: c&63 == 0
	// result: (RotateLeft64 x (Neg16 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpSub16 {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1_0.AuxInt)
		if !(c&63 == 0) {
			break
		}
		v.reset(OpRotateLeft64)
		v0 := b.NewValue0(v.Pos, OpNeg16, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft64 x (Sub8 (Const8 [c]) y))
	// cond: c&63 == 0
	// result: (RotateLeft64 x (Neg8 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpSub8 {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1_0.AuxInt)
		if !(c&63 == 0) {
			break
		}
		v.reset(OpRotateLeft64)
		v0 := b.NewValue0(v.Pos, OpNeg8, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft64 x (Const64 <t> [c]))
	// cond: config.PtrSize == 4
	// result: (RotateLeft64 x (Const32 <t> [int32(c)]))
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		c := auxIntToInt64(v_1.AuxInt)
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpRotateLeft64)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft64 (RotateLeft64 x c) d)
	// cond: c.Type.Size() == 8 && d.Type.Size() == 8
	// result: (RotateLeft64 x (Add64 <c.Type> c d))
	for {
		if v_0.Op != OpRotateLeft64 {
			break
		}
		c := v_0.Args[1]
		x := v_0.Args[0]
		d := v_1
		if !(c.Type.Size() == 8 && d.Type.Size() == 8) {
			break
		}
		v.reset(OpRotateLeft64)
		v0 := b.NewValue0(v.Pos, OpAdd64, c.Type)
		v0.AddArg2(c, d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft64 (RotateLeft64 x c) d)
	// cond: c.Type.Size() == 4 && d.Type.Size() == 4
	// result: (RotateLeft64 x (Add32 <c.Type> c d))
	for {
		if v_0.Op != OpRotateLeft64 {
			break
		}
		c := v_0.Args[1]
		x := v_0.Args[0]
		d := v_1
		if !(c.Type.Size() == 4 && d.Type.Size() == 4) {
			break
		}
		v.reset(OpRotateLeft64)
		v0 := b.NewValue0(v.Pos, OpAdd32, c.Type)
		v0.AddArg2(c, d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft64 (RotateLeft64 x c) d)
	// cond: c.Type.Size() == 2 && d.Type.Size() == 2
	// result: (RotateLeft64 x (Add16 <c.Type> c d))
	for {
		if v_0.Op != OpRotateLeft64 {
			break
		}
		c := v_0.Args[1]
		x := v_0.Args[0]
		d := v_1
		if !(c.Type.Size() == 2 && d.Type.Size() == 2) {
			break
		}
		v.reset(OpRotateLeft64)
		v0 := b.NewValue0(v.Pos, OpAdd16, c.Type)
		v0.AddArg2(c, d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft64 (RotateLeft64 x c) d)
	// cond: c.Type.Size() == 1 && d.Type.Size() == 1
	// result: (RotateLeft64 x (Add8 <c.Type> c d))
	for {
		if v_0.Op != OpRotateLeft64 {
			break
		}
		c := v_0.Args[1]
		x := v_0.Args[0]
		d := v_1
		if !(c.Type.Size() == 1 && d.Type.Size() == 1) {
			break
		}
		v.reset(OpRotateLeft64)
		v0 := b.NewValue0(v.Pos, OpAdd8, c.Type)
		v0.AddArg2(c, d)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRotateLeft8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (RotateLeft8 x (Const8 [c]))
	// cond: c%8 == 0
	// result: x
	for {
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		if !(c%8 == 0) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (RotateLeft8 x (And64 y (Const64 [c])))
	// cond: c&7 == 7
	// result: (RotateLeft8 x y)
	for {
		x := v_0
		if v_1.Op != OpAnd64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_1.AuxInt)
			if !(c&7 == 7) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft8 x (And32 y (Const32 [c])))
	// cond: c&7 == 7
	// result: (RotateLeft8 x y)
	for {
		x := v_0
		if v_1.Op != OpAnd32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_1_1.AuxInt)
			if !(c&7 == 7) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft8 x (And16 y (Const16 [c])))
	// cond: c&7 == 7
	// result: (RotateLeft8 x y)
	for {
		x := v_0
		if v_1.Op != OpAnd16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_1_1.AuxInt)
			if !(c&7 == 7) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft8 x (And8 y (Const8 [c])))
	// cond: c&7 == 7
	// result: (RotateLeft8 x y)
	for {
		x := v_0
		if v_1.Op != OpAnd8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_1_1.AuxInt)
			if !(c&7 == 7) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft8 x (Neg64 (And64 y (Const64 [c]))))
	// cond: c&7 == 7
	// result: (RotateLeft8 x (Neg64 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpNeg64 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAnd64 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		v_1_0_1 := v_1_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0_0, v_1_0_1 = _i0+1, v_1_0_1, v_1_0_0 {
			y := v_1_0_0
			if v_1_0_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_0_1.AuxInt)
			if !(c&7 == 7) {
				continue
			}
			v.reset(OpRotateLeft8)
			v0 := b.NewValue0(v.Pos, OpNeg64, y.Type)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (RotateLeft8 x (Neg32 (And32 y (Const32 [c]))))
	// cond: c&7 == 7
	// result: (RotateLeft8 x (Neg32 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpNeg32 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAnd32 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		v_1_0_1 := v_1_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0_0, v_1_0_1 = _i0+1, v_1_0_1, v_1_0_0 {
			y := v_1_0_0
			if v_1_0_1.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_1_0_1.AuxInt)
			if !(c&7 == 7) {
				continue
			}
			v.reset(OpRotateLeft8)
			v0 := b.NewValue0(v.Pos, OpNeg32, y.Type)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (RotateLeft8 x (Neg16 (And16 y (Const16 [c]))))
	// cond: c&7 == 7
	// result: (RotateLeft8 x (Neg16 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpNeg16 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAnd16 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		v_1_0_1 := v_1_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0_0, v_1_0_1 = _i0+1, v_1_0_1, v_1_0_0 {
			y := v_1_0_0
			if v_1_0_1.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_1_0_1.AuxInt)
			if !(c&7 == 7) {
				continue
			}
			v.reset(OpRotateLeft8)
			v0 := b.NewValue0(v.Pos, OpNeg16, y.Type)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (RotateLeft8 x (Neg8 (And8 y (Const8 [c]))))
	// cond: c&7 == 7
	// result: (RotateLeft8 x (Neg8 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpNeg8 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAnd8 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		v_1_0_1 := v_1_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0_0, v_1_0_1 = _i0+1, v_1_0_1, v_1_0_0 {
			y := v_1_0_0
			if v_1_0_1.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_1_0_1.AuxInt)
			if !(c&7 == 7) {
				continue
			}
			v.reset(OpRotateLeft8)
			v0 := b.NewValue0(v.Pos, OpNeg8, y.Type)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (RotateLeft8 x (Add64 y (Const64 [c])))
	// cond: c&7 == 0
	// result: (RotateLeft8 x y)
	for {
		x := v_0
		if v_1.Op != OpAdd64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_1_1.AuxInt)
			if !(c&7 == 0) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft8 x (Add32 y (Const32 [c])))
	// cond: c&7 == 0
	// result: (RotateLeft8 x y)
	for {
		x := v_0
		if v_1.Op != OpAdd32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_1_1.AuxInt)
			if !(c&7 == 0) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft8 x (Add16 y (Const16 [c])))
	// cond: c&7 == 0
	// result: (RotateLeft8 x y)
	for {
		x := v_0
		if v_1.Op != OpAdd16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_1_1.AuxInt)
			if !(c&7 == 0) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft8 x (Add8 y (Const8 [c])))
	// cond: c&7 == 0
	// result: (RotateLeft8 x y)
	for {
		x := v_0
		if v_1.Op != OpAdd8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			y := v_1_0
			if v_1_1.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_1_1.AuxInt)
			if !(c&7 == 0) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (RotateLeft8 x (Sub64 (Const64 [c]) y))
	// cond: c&7 == 0
	// result: (RotateLeft8 x (Neg64 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpSub64 {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1_0.AuxInt)
		if !(c&7 == 0) {
			break
		}
		v.reset(OpRotateLeft8)
		v0 := b.NewValue0(v.Pos, OpNeg64, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft8 x (Sub32 (Const32 [c]) y))
	// cond: c&7 == 0
	// result: (RotateLeft8 x (Neg32 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpSub32 {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		if !(c&7 == 0) {
			break
		}
		v.reset(OpRotateLeft8)
		v0 := b.NewValue0(v.Pos, OpNeg32, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft8 x (Sub16 (Const16 [c]) y))
	// cond: c&7 == 0
	// result: (RotateLeft8 x (Neg16 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpSub16 {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1_0.AuxInt)
		if !(c&7 == 0) {
			break
		}
		v.reset(OpRotateLeft8)
		v0 := b.NewValue0(v.Pos, OpNeg16, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft8 x (Sub8 (Const8 [c]) y))
	// cond: c&7 == 0
	// result: (RotateLeft8 x (Neg8 <y.Type> y))
	for {
		x := v_0
		if v_1.Op != OpSub8 {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1_0.AuxInt)
		if !(c&7 == 0) {
			break
		}
		v.reset(OpRotateLeft8)
		v0 := b.NewValue0(v.Pos, OpNeg8, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft8 x (Const64 <t> [c]))
	// cond: config.PtrSize == 4
	// result: (RotateLeft8 x (Const32 <t> [int32(c)]))
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		c := auxIntToInt64(v_1.AuxInt)
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpRotateLeft8)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft8 (RotateLeft8 x c) d)
	// cond: c.Type.Size() == 8 && d.Type.Size() == 8
	// result: (RotateLeft8 x (Add64 <c.Type> c d))
	for {
		if v_0.Op != OpRotateLeft8 {
			break
		}
		c := v_0.Args[1]
		x := v_0.Args[0]
		d := v_1
		if !(c.Type.Size() == 8 && d.Type.Size() == 8) {
			break
		}
		v.reset(OpRotateLeft8)
		v0 := b.NewValue0(v.Pos, OpAdd64, c.Type)
		v0.AddArg2(c, d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft8 (RotateLeft8 x c) d)
	// cond: c.Type.Size() == 4 && d.Type.Size() == 4
	// result: (RotateLeft8 x (Add32 <c.Type> c d))
	for {
		if v_0.Op != OpRotateLeft8 {
			break
		}
		c := v_0.Args[1]
		x := v_0.Args[0]
		d := v_1
		if !(c.Type.Size() == 4 && d.Type.Size() == 4) {
			break
		}
		v.reset(OpRotateLeft8)
		v0 := b.NewValue0(v.Pos, OpAdd32, c.Type)
		v0.AddArg2(c, d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft8 (RotateLeft8 x c) d)
	// cond: c.Type.Size() == 2 && d.Type.Size() == 2
	// result: (RotateLeft8 x (Add16 <c.Type> c d))
	for {
		if v_0.Op != OpRotateLeft8 {
			break
		}
		c := v_0.Args[1]
		x := v_0.Args[0]
		d := v_1
		if !(c.Type.Size() == 2 && d.Type.Size() == 2) {
			break
		}
		v.reset(OpRotateLeft8)
		v0 := b.NewValue0(v.Pos, OpAdd16, c.Type)
		v0.AddArg2(c, d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (RotateLeft8 (RotateLeft8 x c) d)
	// cond: c.Type.Size() == 1 && d.Type.Size() == 1
	// result: (RotateLeft8 x (Add8 <c.Type> c d))
	for {
		if v_0.Op != OpRotateLeft8 {
			break
		}
		c := v_0.Args[1]
		x := v_0.Args[0]
		d := v_1
		if !(c.Type.Size() == 1 && d.Type.Size() == 1) {
			break
		}
		v.reset(OpRotateLeft8)
		v0 := b.NewValue0(v.Pos, OpAdd8, c.Type)
		v0.AddArg2(c, d)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRound32F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Round32F x:(Const32F))
	// result: x
	for {
		x := v_0
		if x.Op != OpConst32F {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRound64F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Round64F x:(Const64F))
	// result: x
	for {
		x := v_0
		if x.Op != OpConst64F {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRoundToEven(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEven (Const64F [c]))
	// result: (Const64F [math.RoundToEven(c)])
	for {
		if v_0.Op != OpConst64F {
			break
		}
		c := auxIntToFloat64(v_0.AuxInt)
		v.reset(OpConst64F)
		v.AuxInt = float64ToAuxInt(math.RoundToEven(c))
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh16Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16Ux16 <t> x (Const16 [c]))
	// result: (Rsh16Ux64 x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		v.reset(OpRsh16Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint16(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16Ux16 (Const16 [0]) _)
	// result: (Const16 [0])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh16Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16Ux32 <t> x (Const32 [c]))
	// result: (Rsh16Ux64 x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpRsh16Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint32(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16Ux32 (Const16 [0]) _)
	// result: (Const16 [0])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh16Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 (Const16 [c]) (Const64 [d]))
	// result: (Const16 [int16(uint16(c) >> uint64(d))])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(int16(uint16(c) >> uint64(d)))
		return true
	}
	// match: (Rsh16Ux64 x (Const64 [0]))
	// result: x
	for {
		x := v_0
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Rsh16Ux64 (Const16 [0]) _)
	// result: (Const16 [0])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	// match: (Rsh16Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 16
	// result: (Const16 [0])
	for {
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 16) {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	// match: (Rsh16Ux64 <t> (Rsh16Ux64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Rsh16Ux64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		if v_0.Op != OpRsh16Ux64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpRsh16Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(c + d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16Ux64 (Rsh16x64 x _) (Const64 <t> [15]))
	// result: (Rsh16Ux64 x (Const64 <t> [15]))
	for {
		if v_0.Op != OpRsh16x64 {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		if auxIntToInt64(v_1.AuxInt) != 15 {
			break
		}
		v.reset(OpRsh16Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(15)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16Ux64 i:(Lsh16x64 x (Const64 [c])) (Const64 [c]))
	// cond: c >= 0 && c < 16 && i.Uses == 1
	// result: (And16 x (Const16 <v.Type> [int16(^uint16(0)>>c)]))
	for {
		i := v_0
		if i.Op != OpLsh16x64 {
			break
		}
		_ = i.Args[1]
		x := i.Args[0]
		i_1 := i.Args[1]
		if i_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(i_1.AuxInt)
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != c || !(c >= 0 && c < 16 && i.Uses == 1) {
			break
		}
		v.reset(OpAnd16)
		v0 := b.NewValue0(v.Pos, OpConst16, v.Type)
		v0.AuxInt = int16ToAuxInt(int16(^uint16(0) >> c))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16Ux64 (Lsh16x64 (Rsh16Ux64 x (Const64 [c1])) (Const64 [c2])) (Const64 [c3]))
	// cond: uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)
	// result: (Rsh16Ux64 x (Const64 <typ.UInt64> [c1-c2+c3]))
	for {
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
		c1 := auxIntToInt64(v_0_0_1.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c2 := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		c3 := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)) {
			break
		}
		v.reset(OpRsh16Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c1 - c2 + c3)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16Ux64 (Lsh16x64 x (Const64 [8])) (Const64 [8]))
	// result: (ZeroExt8to16 (Trunc16to8 <typ.UInt8> x))
	for {
		if v_0.Op != OpLsh16x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 || auxIntToInt64(v_0_1.AuxInt) != 8 || v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 8 {
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
func rewriteValuegeneric_OpRsh16Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16Ux8 <t> x (Const8 [c]))
	// result: (Rsh16Ux64 x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		v.reset(OpRsh16Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint8(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16Ux8 (Const16 [0]) _)
	// result: (Const16 [0])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16x16 <t> x (Const16 [c]))
	// result: (Rsh16x64 x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		v.reset(OpRsh16x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint16(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16x16 (Const16 [0]) _)
	// result: (Const16 [0])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16x32 <t> x (Const32 [c]))
	// result: (Rsh16x64 x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpRsh16x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint32(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16x32 (Const16 [0]) _)
	// result: (Const16 [0])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 (Const16 [c]) (Const64 [d]))
	// result: (Const16 [c >> uint64(d)])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(c >> uint64(d))
		return true
	}
	// match: (Rsh16x64 x (Const64 [0]))
	// result: x
	for {
		x := v_0
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Rsh16x64 (Const16 [0]) _)
	// result: (Const16 [0])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	// match: (Rsh16x64 <t> (Rsh16x64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Rsh16x64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		if v_0.Op != OpRsh16x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpRsh16x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(c + d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16x64 (Lsh16x64 x (Const64 [8])) (Const64 [8]))
	// result: (SignExt8to16 (Trunc16to8 <typ.Int8> x))
	for {
		if v_0.Op != OpLsh16x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 || auxIntToInt64(v_0_1.AuxInt) != 8 || v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 8 {
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
func rewriteValuegeneric_OpRsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16x8 <t> x (Const8 [c]))
	// result: (Rsh16x64 x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		v.reset(OpRsh16x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint8(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16x8 (Const16 [0]) _)
	// result: (Const16 [0])
	for {
		if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh32Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32Ux16 <t> x (Const16 [c]))
	// result: (Rsh32Ux64 x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		v.reset(OpRsh32Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint16(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32Ux16 (Const32 [0]) _)
	// result: (Const32 [0])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh32Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32Ux32 <t> x (Const32 [c]))
	// result: (Rsh32Ux64 x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpRsh32Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint32(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32Ux32 (Const32 [0]) _)
	// result: (Const32 [0])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh32Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux64 (Const32 [c]) (Const64 [d]))
	// result: (Const32 [int32(uint32(c) >> uint64(d))])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		return true
	}
	// match: (Rsh32Ux64 x (Const64 [0]))
	// result: x
	for {
		x := v_0
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Rsh32Ux64 (Const32 [0]) _)
	// result: (Const32 [0])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (Rsh32Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 32
	// result: (Const32 [0])
	for {
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 32) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (Rsh32Ux64 <t> (Rsh32Ux64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Rsh32Ux64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		if v_0.Op != OpRsh32Ux64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpRsh32Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(c + d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32Ux64 (Rsh32x64 x _) (Const64 <t> [31]))
	// result: (Rsh32Ux64 x (Const64 <t> [31]))
	for {
		if v_0.Op != OpRsh32x64 {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		if auxIntToInt64(v_1.AuxInt) != 31 {
			break
		}
		v.reset(OpRsh32Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(31)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32Ux64 i:(Lsh32x64 x (Const64 [c])) (Const64 [c]))
	// cond: c >= 0 && c < 32 && i.Uses == 1
	// result: (And32 x (Const32 <v.Type> [int32(^uint32(0)>>c)]))
	for {
		i := v_0
		if i.Op != OpLsh32x64 {
			break
		}
		_ = i.Args[1]
		x := i.Args[0]
		i_1 := i.Args[1]
		if i_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(i_1.AuxInt)
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != c || !(c >= 0 && c < 32 && i.Uses == 1) {
			break
		}
		v.reset(OpAnd32)
		v0 := b.NewValue0(v.Pos, OpConst32, v.Type)
		v0.AuxInt = int32ToAuxInt(int32(^uint32(0) >> c))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32Ux64 (Lsh32x64 (Rsh32Ux64 x (Const64 [c1])) (Const64 [c2])) (Const64 [c3]))
	// cond: uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)
	// result: (Rsh32Ux64 x (Const64 <typ.UInt64> [c1-c2+c3]))
	for {
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
		c1 := auxIntToInt64(v_0_0_1.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c2 := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		c3 := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)) {
			break
		}
		v.reset(OpRsh32Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c1 - c2 + c3)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32Ux64 (Lsh32x64 x (Const64 [24])) (Const64 [24]))
	// result: (ZeroExt8to32 (Trunc32to8 <typ.UInt8> x))
	for {
		if v_0.Op != OpLsh32x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 || auxIntToInt64(v_0_1.AuxInt) != 24 || v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 24 {
			break
		}
		v.reset(OpZeroExt8to32)
		v0 := b.NewValue0(v.Pos, OpTrunc32to8, typ.UInt8)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh32Ux64 (Lsh32x64 x (Const64 [16])) (Const64 [16]))
	// result: (ZeroExt16to32 (Trunc32to16 <typ.UInt16> x))
	for {
		if v_0.Op != OpLsh32x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 || auxIntToInt64(v_0_1.AuxInt) != 16 || v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 16 {
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
func rewriteValuegeneric_OpRsh32Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32Ux8 <t> x (Const8 [c]))
	// result: (Rsh32Ux64 x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		v.reset(OpRsh32Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint8(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32Ux8 (Const32 [0]) _)
	// result: (Const32 [0])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32x16 <t> x (Const16 [c]))
	// result: (Rsh32x64 x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		v.reset(OpRsh32x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint16(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x16 (Const32 [0]) _)
	// result: (Const32 [0])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32x32 <t> x (Const32 [c]))
	// result: (Rsh32x64 x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpRsh32x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint32(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x32 (Const32 [0]) _)
	// result: (Const32 [0])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x64 (Const32 [c]) (Const64 [d]))
	// result: (Const32 [c >> uint64(d)])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(c >> uint64(d))
		return true
	}
	// match: (Rsh32x64 x (Const64 [0]))
	// result: x
	for {
		x := v_0
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Rsh32x64 (Const32 [0]) _)
	// result: (Const32 [0])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (Rsh32x64 <t> (Rsh32x64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Rsh32x64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		if v_0.Op != OpRsh32x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpRsh32x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(c + d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x64 (Lsh32x64 x (Const64 [24])) (Const64 [24]))
	// result: (SignExt8to32 (Trunc32to8 <typ.Int8> x))
	for {
		if v_0.Op != OpLsh32x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 || auxIntToInt64(v_0_1.AuxInt) != 24 || v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 24 {
			break
		}
		v.reset(OpSignExt8to32)
		v0 := b.NewValue0(v.Pos, OpTrunc32to8, typ.Int8)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh32x64 (Lsh32x64 x (Const64 [16])) (Const64 [16]))
	// result: (SignExt16to32 (Trunc32to16 <typ.Int16> x))
	for {
		if v_0.Op != OpLsh32x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 || auxIntToInt64(v_0_1.AuxInt) != 16 || v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 16 {
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
func rewriteValuegeneric_OpRsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32x8 <t> x (Const8 [c]))
	// result: (Rsh32x64 x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		v.reset(OpRsh32x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint8(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x8 (Const32 [0]) _)
	// result: (Const32 [0])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh64Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64Ux16 <t> x (Const16 [c]))
	// result: (Rsh64Ux64 x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		v.reset(OpRsh64Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint16(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64Ux16 (Const64 [0]) _)
	// result: (Const64 [0])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh64Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64Ux32 <t> x (Const32 [c]))
	// result: (Rsh64Ux64 x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpRsh64Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint32(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64Ux32 (Const64 [0]) _)
	// result: (Const64 [0])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh64Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux64 (Const64 [c]) (Const64 [d]))
	// result: (Const64 [int64(uint64(c) >> uint64(d))])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) >> uint64(d)))
		return true
	}
	// match: (Rsh64Ux64 x (Const64 [0]))
	// result: x
	for {
		x := v_0
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Rsh64Ux64 (Const64 [0]) _)
	// result: (Const64 [0])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (Rsh64Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 64
	// result: (Const64 [0])
	for {
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 64) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (Rsh64Ux64 <t> (Rsh64Ux64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Rsh64Ux64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		if v_0.Op != OpRsh64Ux64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpRsh64Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(c + d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64Ux64 (Rsh64x64 x _) (Const64 <t> [63]))
	// result: (Rsh64Ux64 x (Const64 <t> [63]))
	for {
		if v_0.Op != OpRsh64x64 {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		if auxIntToInt64(v_1.AuxInt) != 63 {
			break
		}
		v.reset(OpRsh64Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(63)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64Ux64 i:(Lsh64x64 x (Const64 [c])) (Const64 [c]))
	// cond: c >= 0 && c < 64 && i.Uses == 1
	// result: (And64 x (Const64 <v.Type> [int64(^uint64(0)>>c)]))
	for {
		i := v_0
		if i.Op != OpLsh64x64 {
			break
		}
		_ = i.Args[1]
		x := i.Args[0]
		i_1 := i.Args[1]
		if i_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(i_1.AuxInt)
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != c || !(c >= 0 && c < 64 && i.Uses == 1) {
			break
		}
		v.reset(OpAnd64)
		v0 := b.NewValue0(v.Pos, OpConst64, v.Type)
		v0.AuxInt = int64ToAuxInt(int64(^uint64(0) >> c))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64Ux64 (Lsh64x64 (Rsh64Ux64 x (Const64 [c1])) (Const64 [c2])) (Const64 [c3]))
	// cond: uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)
	// result: (Rsh64Ux64 x (Const64 <typ.UInt64> [c1-c2+c3]))
	for {
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
		c1 := auxIntToInt64(v_0_0_1.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c2 := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		c3 := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)) {
			break
		}
		v.reset(OpRsh64Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c1 - c2 + c3)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64Ux64 (Lsh64x64 x (Const64 [56])) (Const64 [56]))
	// result: (ZeroExt8to64 (Trunc64to8 <typ.UInt8> x))
	for {
		if v_0.Op != OpLsh64x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 || auxIntToInt64(v_0_1.AuxInt) != 56 || v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 56 {
			break
		}
		v.reset(OpZeroExt8to64)
		v0 := b.NewValue0(v.Pos, OpTrunc64to8, typ.UInt8)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64Ux64 (Lsh64x64 x (Const64 [48])) (Const64 [48]))
	// result: (ZeroExt16to64 (Trunc64to16 <typ.UInt16> x))
	for {
		if v_0.Op != OpLsh64x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 || auxIntToInt64(v_0_1.AuxInt) != 48 || v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 48 {
			break
		}
		v.reset(OpZeroExt16to64)
		v0 := b.NewValue0(v.Pos, OpTrunc64to16, typ.UInt16)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64Ux64 (Lsh64x64 x (Const64 [32])) (Const64 [32]))
	// result: (ZeroExt32to64 (Trunc64to32 <typ.UInt32> x))
	for {
		if v_0.Op != OpLsh64x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 || auxIntToInt64(v_0_1.AuxInt) != 32 || v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 32 {
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
func rewriteValuegeneric_OpRsh64Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64Ux8 <t> x (Const8 [c]))
	// result: (Rsh64Ux64 x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		v.reset(OpRsh64Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint8(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64Ux8 (Const64 [0]) _)
	// result: (Const64 [0])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x16 <t> x (Const16 [c]))
	// result: (Rsh64x64 x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		v.reset(OpRsh64x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint16(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x16 (Const64 [0]) _)
	// result: (Const64 [0])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x32 <t> x (Const32 [c]))
	// result: (Rsh64x64 x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpRsh64x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint32(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x32 (Const64 [0]) _)
	// result: (Const64 [0])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x64 (Const64 [c]) (Const64 [d]))
	// result: (Const64 [c >> uint64(d)])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(c >> uint64(d))
		return true
	}
	// match: (Rsh64x64 x (Const64 [0]))
	// result: x
	for {
		x := v_0
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Rsh64x64 (Const64 [0]) _)
	// result: (Const64 [0])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (Rsh64x64 <t> (Rsh64x64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Rsh64x64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		if v_0.Op != OpRsh64x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpRsh64x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(c + d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x64 (Lsh64x64 x (Const64 [56])) (Const64 [56]))
	// result: (SignExt8to64 (Trunc64to8 <typ.Int8> x))
	for {
		if v_0.Op != OpLsh64x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 || auxIntToInt64(v_0_1.AuxInt) != 56 || v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 56 {
			break
		}
		v.reset(OpSignExt8to64)
		v0 := b.NewValue0(v.Pos, OpTrunc64to8, typ.Int8)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64x64 (Lsh64x64 x (Const64 [48])) (Const64 [48]))
	// result: (SignExt16to64 (Trunc64to16 <typ.Int16> x))
	for {
		if v_0.Op != OpLsh64x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 || auxIntToInt64(v_0_1.AuxInt) != 48 || v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 48 {
			break
		}
		v.reset(OpSignExt16to64)
		v0 := b.NewValue0(v.Pos, OpTrunc64to16, typ.Int16)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64x64 (Lsh64x64 x (Const64 [32])) (Const64 [32]))
	// result: (SignExt32to64 (Trunc64to32 <typ.Int32> x))
	for {
		if v_0.Op != OpLsh64x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 || auxIntToInt64(v_0_1.AuxInt) != 32 || v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 32 {
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
func rewriteValuegeneric_OpRsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x8 <t> x (Const8 [c]))
	// result: (Rsh64x64 x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		v.reset(OpRsh64x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint8(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x8 (Const64 [0]) _)
	// result: (Const64 [0])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh8Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8Ux16 <t> x (Const16 [c]))
	// result: (Rsh8Ux64 x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		v.reset(OpRsh8Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint16(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8Ux16 (Const8 [0]) _)
	// result: (Const8 [0])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh8Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8Ux32 <t> x (Const32 [c]))
	// result: (Rsh8Ux64 x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpRsh8Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint32(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8Ux32 (Const8 [0]) _)
	// result: (Const8 [0])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh8Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 (Const8 [c]) (Const64 [d]))
	// result: (Const8 [int8(uint8(c) >> uint64(d))])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(int8(uint8(c) >> uint64(d)))
		return true
	}
	// match: (Rsh8Ux64 x (Const64 [0]))
	// result: x
	for {
		x := v_0
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Rsh8Ux64 (Const8 [0]) _)
	// result: (Const8 [0])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	// match: (Rsh8Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 8
	// result: (Const8 [0])
	for {
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 8) {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	// match: (Rsh8Ux64 <t> (Rsh8Ux64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Rsh8Ux64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		if v_0.Op != OpRsh8Ux64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpRsh8Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(c + d)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8Ux64 (Rsh8x64 x _) (Const64 <t> [7] ))
	// result: (Rsh8Ux64 x (Const64 <t> [7] ))
	for {
		if v_0.Op != OpRsh8x64 {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		if auxIntToInt64(v_1.AuxInt) != 7 {
			break
		}
		v.reset(OpRsh8Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(7)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8Ux64 i:(Lsh8x64 x (Const64 [c])) (Const64 [c]))
	// cond: c >= 0 && c < 8 && i.Uses == 1
	// result: (And8 x (Const8 <v.Type> [int8 (^uint8 (0)>>c)]))
	for {
		i := v_0
		if i.Op != OpLsh8x64 {
			break
		}
		_ = i.Args[1]
		x := i.Args[0]
		i_1 := i.Args[1]
		if i_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(i_1.AuxInt)
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != c || !(c >= 0 && c < 8 && i.Uses == 1) {
			break
		}
		v.reset(OpAnd8)
		v0 := b.NewValue0(v.Pos, OpConst8, v.Type)
		v0.AuxInt = int8ToAuxInt(int8(^uint8(0) >> c))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8Ux64 (Lsh8x64 (Rsh8Ux64 x (Const64 [c1])) (Const64 [c2])) (Const64 [c3]))
	// cond: uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)
	// result: (Rsh8Ux64 x (Const64 <typ.UInt64> [c1-c2+c3]))
	for {
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
		c1 := auxIntToInt64(v_0_0_1.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c2 := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		c3 := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)) {
			break
		}
		v.reset(OpRsh8Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c1 - c2 + c3)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh8Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8Ux8 <t> x (Const8 [c]))
	// result: (Rsh8Ux64 x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		v.reset(OpRsh8Ux64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint8(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8Ux8 (Const8 [0]) _)
	// result: (Const8 [0])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8x16 <t> x (Const16 [c]))
	// result: (Rsh8x64 x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		v.reset(OpRsh8x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint16(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8x16 (Const8 [0]) _)
	// result: (Const8 [0])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8x32 <t> x (Const32 [c]))
	// result: (Rsh8x64 x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpRsh8x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint32(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8x32 (Const8 [0]) _)
	// result: (Const8 [0])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8x64 (Const8 [c]) (Const64 [d]))
	// result: (Const8 [c >> uint64(d)])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(c >> uint64(d))
		return true
	}
	// match: (Rsh8x64 x (Const64 [0]))
	// result: x
	for {
		x := v_0
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Rsh8x64 (Const8 [0]) _)
	// result: (Const8 [0])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	// match: (Rsh8x64 <t> (Rsh8x64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Rsh8x64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		if v_0.Op != OpRsh8x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpRsh8x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(c + d)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8x8 <t> x (Const8 [c]))
	// result: (Rsh8x64 x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		v.reset(OpRsh8x64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(int64(uint8(c)))
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8x8 (Const8 [0]) _)
	// result: (Const8 [0])
	for {
		if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSelect0(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Select0 (MakeTuple x y))
	// result: x
	for {
		if v_0.Op != OpMakeTuple {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSelect1(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Select1 (MakeTuple x y))
	// result: y
	for {
		if v_0.Op != OpMakeTuple {
			break
		}
		y := v_0.Args[1]
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSelectN(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (SelectN [0] (MakeResult x ___))
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpMakeResult || len(v_0.Args) < 1 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (SelectN [1] (MakeResult x y ___))
	// result: y
	for {
		if auxIntToInt64(v.AuxInt) != 1 || v_0.Op != OpMakeResult || len(v_0.Args) < 2 {
			break
		}
		y := v_0.Args[1]
		v.copyOf(y)
		return true
	}
	// match: (SelectN [2] (MakeResult x y z ___))
	// result: z
	for {
		if auxIntToInt64(v.AuxInt) != 2 || v_0.Op != OpMakeResult || len(v_0.Args) < 3 {
			break
		}
		z := v_0.Args[2]
		v.copyOf(z)
		return true
	}
	// match: (SelectN [0] call:(StaticCall {sym} sptr (Const64 [c]) mem))
	// cond: isInlinableMemclr(config, int64(c)) && isSameCall(sym, "runtime.memclrNoHeapPointers") && call.Uses == 1 && clobber(call)
	// result: (Zero {types.Types[types.TUINT8]} [int64(c)] sptr mem)
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != OpStaticCall || len(call.Args) != 3 {
			break
		}
		sym := auxToCall(call.Aux)
		mem := call.Args[2]
		sptr := call.Args[0]
		call_1 := call.Args[1]
		if call_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(call_1.AuxInt)
		if !(isInlinableMemclr(config, int64(c)) && isSameCall(sym, "runtime.memclrNoHeapPointers") && call.Uses == 1 && clobber(call)) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = int64ToAuxInt(int64(c))
		v.Aux = typeToAux(types.Types[types.TUINT8])
		v.AddArg2(sptr, mem)
		return true
	}
	// match: (SelectN [0] call:(StaticCall {sym} sptr (Const32 [c]) mem))
	// cond: isInlinableMemclr(config, int64(c)) && isSameCall(sym, "runtime.memclrNoHeapPointers") && call.Uses == 1 && clobber(call)
	// result: (Zero {types.Types[types.TUINT8]} [int64(c)] sptr mem)
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != OpStaticCall || len(call.Args) != 3 {
			break
		}
		sym := auxToCall(call.Aux)
		mem := call.Args[2]
		sptr := call.Args[0]
		call_1 := call.Args[1]
		if call_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(call_1.AuxInt)
		if !(isInlinableMemclr(config, int64(c)) && isSameCall(sym, "runtime.memclrNoHeapPointers") && call.Uses == 1 && clobber(call)) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = int64ToAuxInt(int64(c))
		v.Aux = typeToAux(types.Types[types.TUINT8])
		v.AddArg2(sptr, mem)
		return true
	}
	// match: (SelectN [0] call:(StaticCall {sym} s1:(Store _ (Const64 [sz]) s2:(Store _ src s3:(Store {t} _ dst mem)))))
	// cond: sz >= 0 && isSameCall(sym, "runtime.memmove") && s1.Uses == 1 && s2.Uses == 1 && s3.Uses == 1 && isInlinableMemmove(dst, src, int64(sz), config) && clobber(s1, s2, s3, call)
	// result: (Move {types.Types[types.TUINT8]} [int64(sz)] dst src mem)
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != OpStaticCall || len(call.Args) != 1 {
			break
		}
		sym := auxToCall(call.Aux)
		s1 := call.Args[0]
		if s1.Op != OpStore {
			break
		}
		_ = s1.Args[2]
		s1_1 := s1.Args[1]
		if s1_1.Op != OpConst64 {
			break
		}
		sz := auxIntToInt64(s1_1.AuxInt)
		s2 := s1.Args[2]
		if s2.Op != OpStore {
			break
		}
		_ = s2.Args[2]
		src := s2.Args[1]
		s3 := s2.Args[2]
		if s3.Op != OpStore {
			break
		}
		mem := s3.Args[2]
		dst := s3.Args[1]
		if !(sz >= 0 && isSameCall(sym, "runtime.memmove") && s1.Uses == 1 && s2.Uses == 1 && s3.Uses == 1 && isInlinableMemmove(dst, src, int64(sz), config) && clobber(s1, s2, s3, call)) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(int64(sz))
		v.Aux = typeToAux(types.Types[types.TUINT8])
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (SelectN [0] call:(StaticCall {sym} s1:(Store _ (Const32 [sz]) s2:(Store _ src s3:(Store {t} _ dst mem)))))
	// cond: sz >= 0 && isSameCall(sym, "runtime.memmove") && s1.Uses == 1 && s2.Uses == 1 && s3.Uses == 1 && isInlinableMemmove(dst, src, int64(sz), config) && clobber(s1, s2, s3, call)
	// result: (Move {types.Types[types.TUINT8]} [int64(sz)] dst src mem)
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != OpStaticCall || len(call.Args) != 1 {
			break
		}
		sym := auxToCall(call.Aux)
		s1 := call.Args[0]
		if s1.Op != OpStore {
			break
		}
		_ = s1.Args[2]
		s1_1 := s1.Args[1]
		if s1_1.Op != OpConst32 {
			break
		}
		sz := auxIntToInt32(s1_1.AuxInt)
		s2 := s1.Args[2]
		if s2.Op != OpStore {
			break
		}
		_ = s2.Args[2]
		src := s2.Args[1]
		s3 := s2.Args[2]
		if s3.Op != OpStore {
			break
		}
		mem := s3.Args[2]
		dst := s3.Args[1]
		if !(sz >= 0 && isSameCall(sym, "runtime.memmove") && s1.Uses == 1 && s2.Uses == 1 && s3.Uses == 1 && isInlinableMemmove(dst, src, int64(sz), config) && clobber(s1, s2, s3, call)) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(int64(sz))
		v.Aux = typeToAux(types.Types[types.TUINT8])
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (SelectN [0] call:(StaticCall {sym} dst src (Const64 [sz]) mem))
	// cond: sz >= 0 && call.Uses == 1 && isSameCall(sym, "runtime.memmove") && isInlinableMemmove(dst, src, int64(sz), config) && clobber(call)
	// result: (Move {types.Types[types.TUINT8]} [int64(sz)] dst src mem)
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != OpStaticCall || len(call.Args) != 4 {
			break
		}
		sym := auxToCall(call.Aux)
		mem := call.Args[3]
		dst := call.Args[0]
		src := call.Args[1]
		call_2 := call.Args[2]
		if call_2.Op != OpConst64 {
			break
		}
		sz := auxIntToInt64(call_2.AuxInt)
		if !(sz >= 0 && call.Uses == 1 && isSameCall(sym, "runtime.memmove") && isInlinableMemmove(dst, src, int64(sz), config) && clobber(call)) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(int64(sz))
		v.Aux = typeToAux(types.Types[types.TUINT8])
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (SelectN [0] call:(StaticCall {sym} dst src (Const32 [sz]) mem))
	// cond: sz >= 0 && call.Uses == 1 && isSameCall(sym, "runtime.memmove") && isInlinableMemmove(dst, src, int64(sz), config) && clobber(call)
	// result: (Move {types.Types[types.TUINT8]} [int64(sz)] dst src mem)
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != OpStaticCall || len(call.Args) != 4 {
			break
		}
		sym := auxToCall(call.Aux)
		mem := call.Args[3]
		dst := call.Args[0]
		src := call.Args[1]
		call_2 := call.Args[2]
		if call_2.Op != OpConst32 {
			break
		}
		sz := auxIntToInt32(call_2.AuxInt)
		if !(sz >= 0 && call.Uses == 1 && isSameCall(sym, "runtime.memmove") && isInlinableMemmove(dst, src, int64(sz), config) && clobber(call)) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(int64(sz))
		v.Aux = typeToAux(types.Types[types.TUINT8])
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (SelectN [0] call:(StaticLECall {sym} dst src (Const64 [sz]) mem))
	// cond: sz >= 0 && call.Uses == 1 && isSameCall(sym, "runtime.memmove") && isInlinableMemmove(dst, src, int64(sz), config) && clobber(call)
	// result: (Move {types.Types[types.TUINT8]} [int64(sz)] dst src mem)
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != OpStaticLECall || len(call.Args) != 4 {
			break
		}
		sym := auxToCall(call.Aux)
		mem := call.Args[3]
		dst := call.Args[0]
		src := call.Args[1]
		call_2 := call.Args[2]
		if call_2.Op != OpConst64 {
			break
		}
		sz := auxIntToInt64(call_2.AuxInt)
		if !(sz >= 0 && call.Uses == 1 && isSameCall(sym, "runtime.memmove") && isInlinableMemmove(dst, src, int64(sz), config) && clobber(call)) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(int64(sz))
		v.Aux = typeToAux(types.Types[types.TUINT8])
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (SelectN [0] call:(StaticLECall {sym} dst src (Const32 [sz]) mem))
	// cond: sz >= 0 && call.Uses == 1 && isSameCall(sym, "runtime.memmove") && isInlinableMemmove(dst, src, int64(sz), config) && clobber(call)
	// result: (Move {types.Types[types.TUINT8]} [int64(sz)] dst src mem)
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != OpStaticLECall || len(call.Args) != 4 {
			break
		}
		sym := auxToCall(call.Aux)
		mem := call.Args[3]
		dst := call.Args[0]
		src := call.Args[1]
		call_2 := call.Args[2]
		if call_2.Op != OpConst32 {
			break
		}
		sz := auxIntToInt32(call_2.AuxInt)
		if !(sz >= 0 && call.Uses == 1 && isSameCall(sym, "runtime.memmove") && isInlinableMemmove(dst, src, int64(sz), config) && clobber(call)) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(int64(sz))
		v.Aux = typeToAux(types.Types[types.TUINT8])
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (SelectN [0] call:(StaticLECall {sym} a x))
	// cond: needRaceCleanup(sym, call) && clobber(call)
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != OpStaticLECall || len(call.Args) != 2 {
			break
		}
		sym := auxToCall(call.Aux)
		x := call.Args[1]
		if !(needRaceCleanup(sym, call) && clobber(call)) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (SelectN [0] call:(StaticLECall {sym} x))
	// cond: needRaceCleanup(sym, call) && clobber(call)
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != OpStaticLECall || len(call.Args) != 1 {
			break
		}
		sym := auxToCall(call.Aux)
		x := call.Args[0]
		if !(needRaceCleanup(sym, call) && clobber(call)) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (SelectN [1] (StaticCall {sym} _ newLen:(Const64) _ _ _ _))
	// cond: v.Type.IsInteger() && isSameCall(sym, "runtime.growslice")
	// result: newLen
	for {
		if auxIntToInt64(v.AuxInt) != 1 || v_0.Op != OpStaticCall || len(v_0.Args) != 6 {
			break
		}
		sym := auxToCall(v_0.Aux)
		_ = v_0.Args[1]
		newLen := v_0.Args[1]
		if newLen.Op != OpConst64 || !(v.Type.IsInteger() && isSameCall(sym, "runtime.growslice")) {
			break
		}
		v.copyOf(newLen)
		return true
	}
	// match: (SelectN [1] (StaticCall {sym} _ newLen:(Const32) _ _ _ _))
	// cond: v.Type.IsInteger() && isSameCall(sym, "runtime.growslice")
	// result: newLen
	for {
		if auxIntToInt64(v.AuxInt) != 1 || v_0.Op != OpStaticCall || len(v_0.Args) != 6 {
			break
		}
		sym := auxToCall(v_0.Aux)
		_ = v_0.Args[1]
		newLen := v_0.Args[1]
		if newLen.Op != OpConst32 || !(v.Type.IsInteger() && isSameCall(sym, "runtime.growslice")) {
			break
		}
		v.copyOf(newLen)
		return true
	}
	// match: (SelectN [0] (StaticLECall {f} x y (SelectN [1] c:(StaticLECall {g} x y mem))))
	// cond: isSameCall(f, "runtime.cmpstring") && isSameCall(g, "runtime.cmpstring")
	// result: @c.Block (SelectN [0] <typ.Int> c)
	for {
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpStaticLECall || len(v_0.Args) != 3 {
			break
		}
		f := auxToCall(v_0.Aux)
		_ = v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v_0_2 := v_0.Args[2]
		if v_0_2.Op != OpSelectN || auxIntToInt64(v_0_2.AuxInt) != 1 {
			break
		}
		c := v_0_2.Args[0]
		if c.Op != OpStaticLECall || len(c.Args) != 3 {
			break
		}
		g := auxToCall(c.Aux)
		if x != c.Args[0] || y != c.Args[1] || !(isSameCall(f, "runtime.cmpstring") && isSameCall(g, "runtime.cmpstring")) {
			break
		}
		b = c.Block
		v0 := b.NewValue0(v.Pos, OpSelectN, typ.Int)
		v.copyOf(v0)
		v0.AuxInt = int64ToAuxInt(0)
		v0.AddArg(c)
		return true
	}
	// match: (SelectN [1] c:(StaticLECall {f} _ _ mem))
	// cond: c.Uses == 1 && isSameCall(f, "runtime.cmpstring") && clobber(c)
	// result: mem
	for {
		if auxIntToInt64(v.AuxInt) != 1 {
			break
		}
		c := v_0
		if c.Op != OpStaticLECall || len(c.Args) != 3 {
			break
		}
		f := auxToCall(c.Aux)
		mem := c.Args[2]
		if !(c.Uses == 1 && isSameCall(f, "runtime.cmpstring") && clobber(c)) {
			break
		}
		v.copyOf(mem)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSignExt16to32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SignExt16to32 (Const16 [c]))
	// result: (Const32 [int32(c)])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(c))
		return true
	}
	// match: (SignExt16to32 (Trunc32to16 x:(Rsh32x64 _ (Const64 [s]))))
	// cond: s >= 16
	// result: x
	for {
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
		s := auxIntToInt64(x_1.AuxInt)
		if !(s >= 16) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSignExt16to64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SignExt16to64 (Const16 [c]))
	// result: (Const64 [int64(c)])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(c))
		return true
	}
	// match: (SignExt16to64 (Trunc64to16 x:(Rsh64x64 _ (Const64 [s]))))
	// cond: s >= 48
	// result: x
	for {
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
		s := auxIntToInt64(x_1.AuxInt)
		if !(s >= 48) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSignExt32to64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SignExt32to64 (Const32 [c]))
	// result: (Const64 [int64(c)])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(c))
		return true
	}
	// match: (SignExt32to64 (Trunc64to32 x:(Rsh64x64 _ (Const64 [s]))))
	// cond: s >= 32
	// result: x
	for {
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
		s := auxIntToInt64(x_1.AuxInt)
		if !(s >= 32) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSignExt8to16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SignExt8to16 (Const8 [c]))
	// result: (Const16 [int16(c)])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(int16(c))
		return true
	}
	// match: (SignExt8to16 (Trunc16to8 x:(Rsh16x64 _ (Const64 [s]))))
	// cond: s >= 8
	// result: x
	for {
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
		s := auxIntToInt64(x_1.AuxInt)
		if !(s >= 8) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSignExt8to32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SignExt8to32 (Const8 [c]))
	// result: (Const32 [int32(c)])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(c))
		return true
	}
	// match: (SignExt8to32 (Trunc32to8 x:(Rsh32x64 _ (Const64 [s]))))
	// cond: s >= 24
	// result: x
	for {
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
		s := auxIntToInt64(x_1.AuxInt)
		if !(s >= 24) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSignExt8to64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SignExt8to64 (Const8 [c]))
	// result: (Const64 [int64(c)])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(c))
		return true
	}
	// match: (SignExt8to64 (Trunc64to8 x:(Rsh64x64 _ (Const64 [s]))))
	// cond: s >= 56
	// result: x
	for {
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
		s := auxIntToInt64(x_1.AuxInt)
		if !(s >= 56) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSliceCap(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SliceCap (SliceMake _ _ (Const64 <t> [c])))
	// result: (Const64 <t> [c])
	for {
		if v_0.Op != OpSliceMake {
			break
		}
		_ = v_0.Args[2]
		v_0_2 := v_0.Args[2]
		if v_0_2.Op != OpConst64 {
			break
		}
		t := v_0_2.Type
		c := auxIntToInt64(v_0_2.AuxInt)
		v.reset(OpConst64)
		v.Type = t
		v.AuxInt = int64ToAuxInt(c)
		return true
	}
	// match: (SliceCap (SliceMake _ _ (Const32 <t> [c])))
	// result: (Const32 <t> [c])
	for {
		if v_0.Op != OpSliceMake {
			break
		}
		_ = v_0.Args[2]
		v_0_2 := v_0.Args[2]
		if v_0_2.Op != OpConst32 {
			break
		}
		t := v_0_2.Type
		c := auxIntToInt32(v_0_2.AuxInt)
		v.reset(OpConst32)
		v.Type = t
		v.AuxInt = int32ToAuxInt(c)
		return true
	}
	// match: (SliceCap (SliceMake _ _ (SliceCap x)))
	// result: (SliceCap x)
	for {
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
	// result: (SliceLen x)
	for {
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
func rewriteValuegeneric_OpSliceLen(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SliceLen (SliceMake _ (Const64 <t> [c]) _))
	// result: (Const64 <t> [c])
	for {
		if v_0.Op != OpSliceMake {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		t := v_0_1.Type
		c := auxIntToInt64(v_0_1.AuxInt)
		v.reset(OpConst64)
		v.Type = t
		v.AuxInt = int64ToAuxInt(c)
		return true
	}
	// match: (SliceLen (SliceMake _ (Const32 <t> [c]) _))
	// result: (Const32 <t> [c])
	for {
		if v_0.Op != OpSliceMake {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst32 {
			break
		}
		t := v_0_1.Type
		c := auxIntToInt32(v_0_1.AuxInt)
		v.reset(OpConst32)
		v.Type = t
		v.AuxInt = int32ToAuxInt(c)
		return true
	}
	// match: (SliceLen (SliceMake _ (SliceLen x) _))
	// result: (SliceLen x)
	for {
		if v_0.Op != OpSliceMake {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpSliceLen {
			break
		}
		x := v_0_1.Args[0]
		v.reset(OpSliceLen)
		v.AddArg(x)
		return true
	}
	// match: (SliceLen (SelectN [0] (StaticLECall {sym} _ newLen:(Const64) _ _ _ _)))
	// cond: isSameCall(sym, "runtime.growslice")
	// result: newLen
	for {
		if v_0.Op != OpSelectN || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpStaticLECall || len(v_0_0.Args) != 6 {
			break
		}
		sym := auxToCall(v_0_0.Aux)
		_ = v_0_0.Args[1]
		newLen := v_0_0.Args[1]
		if newLen.Op != OpConst64 || !(isSameCall(sym, "runtime.growslice")) {
			break
		}
		v.copyOf(newLen)
		return true
	}
	// match: (SliceLen (SelectN [0] (StaticLECall {sym} _ newLen:(Const32) _ _ _ _)))
	// cond: isSameCall(sym, "runtime.growslice")
	// result: newLen
	for {
		if v_0.Op != OpSelectN || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpStaticLECall || len(v_0_0.Args) != 6 {
			break
		}
		sym := auxToCall(v_0_0.Aux)
		_ = v_0_0.Args[1]
		newLen := v_0_0.Args[1]
		if newLen.Op != OpConst32 || !(isSameCall(sym, "runtime.growslice")) {
			break
		}
		v.copyOf(newLen)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSlicePtr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SlicePtr (SliceMake (SlicePtr x) _ _))
	// result: (SlicePtr x)
	for {
		if v_0.Op != OpSliceMake {
			break
		}
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
func rewriteValuegeneric_OpSlicemask(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Slicemask (Const32 [x]))
	// cond: x > 0
	// result: (Const32 [-1])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(x > 0) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(-1)
		return true
	}
	// match: (Slicemask (Const32 [0]))
	// result: (Const32 [0])
	for {
		if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (Slicemask (Const64 [x]))
	// cond: x > 0
	// result: (Const64 [-1])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(x > 0) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	// match: (Slicemask (Const64 [0]))
	// result: (Const64 [0])
	for {
		if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSqrt(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Sqrt (Const64F [c]))
	// cond: !math.IsNaN(math.Sqrt(c))
	// result: (Const64F [math.Sqrt(c)])
	for {
		if v_0.Op != OpConst64F {
			break
		}
		c := auxIntToFloat64(v_0.AuxInt)
		if !(!math.IsNaN(math.Sqrt(c))) {
			break
		}
		v.reset(OpConst64F)
		v.AuxInt = float64ToAuxInt(math.Sqrt(c))
		return true
	}
	return false
}
func rewriteValuegeneric_OpStaticCall(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (StaticCall {callAux} p q _ mem)
	// cond: isSameCall(callAux, "runtime.memequal") && isSamePtr(p, q)
	// result: (MakeResult (ConstBool <typ.Bool> [true]) mem)
	for {
		if len(v.Args) != 4 {
			break
		}
		callAux := auxToCall(v.Aux)
		mem := v.Args[3]
		p := v.Args[0]
		q := v.Args[1]
		if !(isSameCall(callAux, "runtime.memequal") && isSamePtr(p, q)) {
			break
		}
		v.reset(OpMakeResult)
		v0 := b.NewValue0(v.Pos, OpConstBool, typ.Bool)
		v0.AuxInt = boolToAuxInt(true)
		v.AddArg2(v0, mem)
		return true
	}
	return false
}
func rewriteValuegeneric_OpStaticLECall(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (StaticLECall {callAux} sptr (Addr {scon} (SB)) (Const64 [1]) mem)
	// cond: isSameCall(callAux, "runtime.memequal") && symIsRO(scon)
	// result: (MakeResult (Eq8 (Load <typ.Int8> sptr mem) (Const8 <typ.Int8> [int8(read8(scon,0))])) mem)
	for {
		if len(v.Args) != 4 {
			break
		}
		callAux := auxToCall(v.Aux)
		mem := v.Args[3]
		sptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAddr {
			break
		}
		scon := auxToSym(v_1.Aux)
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpSB {
			break
		}
		v_2 := v.Args[2]
		if v_2.Op != OpConst64 || auxIntToInt64(v_2.AuxInt) != 1 || !(isSameCall(callAux, "runtime.memequal") && symIsRO(scon)) {
			break
		}
		v.reset(OpMakeResult)
		v0 := b.NewValue0(v.Pos, OpEq8, typ.Bool)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.Int8)
		v1.AddArg2(sptr, mem)
		v2 := b.NewValue0(v.Pos, OpConst8, typ.Int8)
		v2.AuxInt = int8ToAuxInt(int8(read8(scon, 0)))
		v0.AddArg2(v1, v2)
		v.AddArg2(v0, mem)
		return true
	}
	// match: (StaticLECall {callAux} (Addr {scon} (SB)) sptr (Const64 [1]) mem)
	// cond: isSameCall(callAux, "runtime.memequal") && symIsRO(scon)
	// result: (MakeResult (Eq8 (Load <typ.Int8> sptr mem) (Const8 <typ.Int8> [int8(read8(scon,0))])) mem)
	for {
		if len(v.Args) != 4 {
			break
		}
		callAux := auxToCall(v.Aux)
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpAddr {
			break
		}
		scon := auxToSym(v_0.Aux)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpSB {
			break
		}
		sptr := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpConst64 || auxIntToInt64(v_2.AuxInt) != 1 || !(isSameCall(callAux, "runtime.memequal") && symIsRO(scon)) {
			break
		}
		v.reset(OpMakeResult)
		v0 := b.NewValue0(v.Pos, OpEq8, typ.Bool)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.Int8)
		v1.AddArg2(sptr, mem)
		v2 := b.NewValue0(v.Pos, OpConst8, typ.Int8)
		v2.AuxInt = int8ToAuxInt(int8(read8(scon, 0)))
		v0.AddArg2(v1, v2)
		v.AddArg2(v0, mem)
		return true
	}
	// match: (StaticLECall {callAux} sptr (Addr {scon} (SB)) (Const64 [2]) mem)
	// cond: isSameCall(callAux, "runtime.memequal") && symIsRO(scon) && canLoadUnaligned(config)
	// result: (MakeResult (Eq16 (Load <typ.Int16> sptr mem) (Const16 <typ.Int16> [int16(read16(scon,0,config.ctxt.Arch.ByteOrder))])) mem)
	for {
		if len(v.Args) != 4 {
			break
		}
		callAux := auxToCall(v.Aux)
		mem := v.Args[3]
		sptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAddr {
			break
		}
		scon := auxToSym(v_1.Aux)
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpSB {
			break
		}
		v_2 := v.Args[2]
		if v_2.Op != OpConst64 || auxIntToInt64(v_2.AuxInt) != 2 || !(isSameCall(callAux, "runtime.memequal") && symIsRO(scon) && canLoadUnaligned(config)) {
			break
		}
		v.reset(OpMakeResult)
		v0 := b.NewValue0(v.Pos, OpEq16, typ.Bool)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.Int16)
		v1.AddArg2(sptr, mem)
		v2 := b.NewValue0(v.Pos, OpConst16, typ.Int16)
		v2.AuxInt = int16ToAuxInt(int16(read16(scon, 0, config.ctxt.Arch.ByteOrder)))
		v0.AddArg2(v1, v2)
		v.AddArg2(v0, mem)
		return true
	}
	// match: (StaticLECall {callAux} (Addr {scon} (SB)) sptr (Const64 [2]) mem)
	// cond: isSameCall(callAux, "runtime.memequal") && symIsRO(scon) && canLoadUnaligned(config)
	// result: (MakeResult (Eq16 (Load <typ.Int16> sptr mem) (Const16 <typ.Int16> [int16(read16(scon,0,config.ctxt.Arch.ByteOrder))])) mem)
	for {
		if len(v.Args) != 4 {
			break
		}
		callAux := auxToCall(v.Aux)
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpAddr {
			break
		}
		scon := auxToSym(v_0.Aux)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpSB {
			break
		}
		sptr := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpConst64 || auxIntToInt64(v_2.AuxInt) != 2 || !(isSameCall(callAux, "runtime.memequal") && symIsRO(scon) && canLoadUnaligned(config)) {
			break
		}
		v.reset(OpMakeResult)
		v0 := b.NewValue0(v.Pos, OpEq16, typ.Bool)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.Int16)
		v1.AddArg2(sptr, mem)
		v2 := b.NewValue0(v.Pos, OpConst16, typ.Int16)
		v2.AuxInt = int16ToAuxInt(int16(read16(scon, 0, config.ctxt.Arch.ByteOrder)))
		v0.AddArg2(v1, v2)
		v.AddArg2(v0, mem)
		return true
	}
	// match: (StaticLECall {callAux} sptr (Addr {scon} (SB)) (Const64 [4]) mem)
	// cond: isSameCall(callAux, "runtime.memequal") && symIsRO(scon) && canLoadUnaligned(config)
	// result: (MakeResult (Eq32 (Load <typ.Int32> sptr mem) (Const32 <typ.Int32> [int32(read32(scon,0,config.ctxt.Arch.ByteOrder))])) mem)
	for {
		if len(v.Args) != 4 {
			break
		}
		callAux := auxToCall(v.Aux)
		mem := v.Args[3]
		sptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAddr {
			break
		}
		scon := auxToSym(v_1.Aux)
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpSB {
			break
		}
		v_2 := v.Args[2]
		if v_2.Op != OpConst64 || auxIntToInt64(v_2.AuxInt) != 4 || !(isSameCall(callAux, "runtime.memequal") && symIsRO(scon) && canLoadUnaligned(config)) {
			break
		}
		v.reset(OpMakeResult)
		v0 := b.NewValue0(v.Pos, OpEq32, typ.Bool)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.Int32)
		v1.AddArg2(sptr, mem)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.Int32)
		v2.AuxInt = int32ToAuxInt(int32(read32(scon, 0, config.ctxt.Arch.ByteOrder)))
		v0.AddArg2(v1, v2)
		v.AddArg2(v0, mem)
		return true
	}
	// match: (StaticLECall {callAux} (Addr {scon} (SB)) sptr (Const64 [4]) mem)
	// cond: isSameCall(callAux, "runtime.memequal") && symIsRO(scon) && canLoadUnaligned(config)
	// result: (MakeResult (Eq32 (Load <typ.Int32> sptr mem) (Const32 <typ.Int32> [int32(read32(scon,0,config.ctxt.Arch.ByteOrder))])) mem)
	for {
		if len(v.Args) != 4 {
			break
		}
		callAux := auxToCall(v.Aux)
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpAddr {
			break
		}
		scon := auxToSym(v_0.Aux)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpSB {
			break
		}
		sptr := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpConst64 || auxIntToInt64(v_2.AuxInt) != 4 || !(isSameCall(callAux, "runtime.memequal") && symIsRO(scon) && canLoadUnaligned(config)) {
			break
		}
		v.reset(OpMakeResult)
		v0 := b.NewValue0(v.Pos, OpEq32, typ.Bool)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.Int32)
		v1.AddArg2(sptr, mem)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.Int32)
		v2.AuxInt = int32ToAuxInt(int32(read32(scon, 0, config.ctxt.Arch.ByteOrder)))
		v0.AddArg2(v1, v2)
		v.AddArg2(v0, mem)
		return true
	}
	// match: (StaticLECall {callAux} sptr (Addr {scon} (SB)) (Const64 [8]) mem)
	// cond: isSameCall(callAux, "runtime.memequal") && symIsRO(scon) && canLoadUnaligned(config) && config.PtrSize == 8
	// result: (MakeResult (Eq64 (Load <typ.Int64> sptr mem) (Const64 <typ.Int64> [int64(read64(scon,0,config.ctxt.Arch.ByteOrder))])) mem)
	for {
		if len(v.Args) != 4 {
			break
		}
		callAux := auxToCall(v.Aux)
		mem := v.Args[3]
		sptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAddr {
			break
		}
		scon := auxToSym(v_1.Aux)
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpSB {
			break
		}
		v_2 := v.Args[2]
		if v_2.Op != OpConst64 || auxIntToInt64(v_2.AuxInt) != 8 || !(isSameCall(callAux, "runtime.memequal") && symIsRO(scon) && canLoadUnaligned(config) && config.PtrSize == 8) {
			break
		}
		v.reset(OpMakeResult)
		v0 := b.NewValue0(v.Pos, OpEq64, typ.Bool)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.Int64)
		v1.AddArg2(sptr, mem)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.Int64)
		v2.AuxInt = int64ToAuxInt(int64(read64(scon, 0, config.ctxt.Arch.ByteOrder)))
		v0.AddArg2(v1, v2)
		v.AddArg2(v0, mem)
		return true
	}
	// match: (StaticLECall {callAux} (Addr {scon} (SB)) sptr (Const64 [8]) mem)
	// cond: isSameCall(callAux, "runtime.memequal") && symIsRO(scon) && canLoadUnaligned(config) && config.PtrSize == 8
	// result: (MakeResult (Eq64 (Load <typ.Int64> sptr mem) (Const64 <typ.Int64> [int64(read64(scon,0,config.ctxt.Arch.ByteOrder))])) mem)
	for {
		if len(v.Args) != 4 {
			break
		}
		callAux := auxToCall(v.Aux)
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpAddr {
			break
		}
		scon := auxToSym(v_0.Aux)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpSB {
			break
		}
		sptr := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpConst64 || auxIntToInt64(v_2.AuxInt) != 8 || !(isSameCall(callAux, "runtime.memequal") && symIsRO(scon) && canLoadUnaligned(config) && config.PtrSize == 8) {
			break
		}
		v.reset(OpMakeResult)
		v0 := b.NewValue0(v.Pos, OpEq64, typ.Bool)
		v1 := b.NewValue0(v.Pos, OpLoad, typ.Int64)
		v1.AddArg2(sptr, mem)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.Int64)
		v2.AuxInt = int64ToAuxInt(int64(read64(scon, 0, config.ctxt.Arch.ByteOrder)))
		v0.AddArg2(v1, v2)
		v.AddArg2(v0, mem)
		return true
	}
	// match: (StaticLECall {callAux} _ _ (Const64 [0]) mem)
	// cond: isSameCall(callAux, "runtime.memequal")
	// result: (MakeResult (ConstBool <typ.Bool> [true]) mem)
	for {
		if len(v.Args) != 4 {
			break
		}
		callAux := auxToCall(v.Aux)
		mem := v.Args[3]
		v_2 := v.Args[2]
		if v_2.Op != OpConst64 || auxIntToInt64(v_2.AuxInt) != 0 || !(isSameCall(callAux, "runtime.memequal")) {
			break
		}
		v.reset(OpMakeResult)
		v0 := b.NewValue0(v.Pos, OpConstBool, typ.Bool)
		v0.AuxInt = boolToAuxInt(true)
		v.AddArg2(v0, mem)
		return true
	}
	// match: (StaticLECall {callAux} p q _ mem)
	// cond: isSameCall(callAux, "runtime.memequal") && isSamePtr(p, q)
	// result: (MakeResult (ConstBool <typ.Bool> [true]) mem)
	for {
		if len(v.Args) != 4 {
			break
		}
		callAux := auxToCall(v.Aux)
		mem := v.Args[3]
		p := v.Args[0]
		q := v.Args[1]
		if !(isSameCall(callAux, "runtime.memequal") && isSamePtr(p, q)) {
			break
		}
		v.reset(OpMakeResult)
		v0 := b.NewValue0(v.Pos, OpConstBool, typ.Bool)
		v0.AuxInt = boolToAuxInt(true)
		v.AddArg2(v0, mem)
		return true
	}
	// match: (StaticLECall {callAux} _ (Const64 [0]) (Const64 [0]) mem)
	// cond: isSameCall(callAux, "runtime.makeslice")
	// result: (MakeResult (Addr <v.Type.FieldType(0)> {ir.Syms.Zerobase} (SB)) mem)
	for {
		if len(v.Args) != 4 {
			break
		}
		callAux := auxToCall(v.Aux)
		mem := v.Args[3]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v_2 := v.Args[2]
		if v_2.Op != OpConst64 || auxIntToInt64(v_2.AuxInt) != 0 || !(isSameCall(callAux, "runtime.makeslice")) {
			break
		}
		v.reset(OpMakeResult)
		v0 := b.NewValue0(v.Pos, OpAddr, v.Type.FieldType(0))
		v0.Aux = symToAux(ir.Syms.Zerobase)
		v1 := b.NewValue0(v.Pos, OpSB, typ.Uintptr)
		v0.AddArg(v1)
		v.AddArg2(v0, mem)
		return true
	}
	// match: (StaticLECall {callAux} _ (Const32 [0]) (Const32 [0]) mem)
	// cond: isSameCall(callAux, "runtime.makeslice")
	// result: (MakeResult (Addr <v.Type.FieldType(0)> {ir.Syms.Zerobase} (SB)) mem)
	for {
		if len(v.Args) != 4 {
			break
		}
		callAux := auxToCall(v.Aux)
		mem := v.Args[3]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 || auxIntToInt32(v_1.AuxInt) != 0 {
			break
		}
		v_2 := v.Args[2]
		if v_2.Op != OpConst32 || auxIntToInt32(v_2.AuxInt) != 0 || !(isSameCall(callAux, "runtime.makeslice")) {
			break
		}
		v.reset(OpMakeResult)
		v0 := b.NewValue0(v.Pos, OpAddr, v.Type.FieldType(0))
		v0.Aux = symToAux(ir.Syms.Zerobase)
		v1 := b.NewValue0(v.Pos, OpSB, typ.Uintptr)
		v0.AddArg(v1)
		v.AddArg2(v0, mem)
		return true
	}
	// match: (StaticLECall {f} typ_ x y mem)
	// cond: isSameCall(f, "runtime.efaceeq") && isDirectType(typ_) && clobber(v)
	// result: (MakeResult (EqPtr x y) mem)
	for {
		if len(v.Args) != 4 {
			break
		}
		f := auxToCall(v.Aux)
		mem := v.Args[3]
		typ_ := v.Args[0]
		x := v.Args[1]
		y := v.Args[2]
		if !(isSameCall(f, "runtime.efaceeq") && isDirectType(typ_) && clobber(v)) {
			break
		}
		v.reset(OpMakeResult)
		v0 := b.NewValue0(v.Pos, OpEqPtr, typ.Bool)
		v0.AddArg2(x, y)
		v.AddArg2(v0, mem)
		return true
	}
	// match: (StaticLECall {f} itab x y mem)
	// cond: isSameCall(f, "runtime.ifaceeq") && isDirectIface(itab) && clobber(v)
	// result: (MakeResult (EqPtr x y) mem)
	for {
		if len(v.Args) != 4 {
			break
		}
		f := auxToCall(v.Aux)
		mem := v.Args[3]
		itab := v.Args[0]
		x := v.Args[1]
		y := v.Args[2]
		if !(isSameCall(f, "runtime.ifaceeq") && isDirectIface(itab) && clobber(v)) {
			break
		}
		v.reset(OpMakeResult)
		v0 := b.NewValue0(v.Pos, OpEqPtr, typ.Bool)
		v0.AddArg2(x, y)
		v.AddArg2(v0, mem)
		return true
	}
	// match: (StaticLECall {f} [argsize] typ_ map_ key:(SelectN [0] sbts:(StaticLECall {g} _ ptr len mem)) m:(SelectN [1] sbts))
	// cond: (isSameCall(f, "runtime.mapaccess1_faststr") || isSameCall(f, "runtime.mapaccess2_faststr") || isSameCall(f, "runtime.mapdelete_faststr")) && isSameCall(g, "runtime.slicebytetostring") && key.Uses == 1 && sbts.Uses == 2 && resetCopy(m, mem) && clobber(sbts) && clobber(key)
	// result: (StaticLECall {f} [argsize] typ_ map_ (StringMake <typ.String> ptr len) mem)
	for {
		if len(v.Args) != 4 {
			break
		}
		argsize := auxIntToInt32(v.AuxInt)
		f := auxToCall(v.Aux)
		_ = v.Args[3]
		typ_ := v.Args[0]
		map_ := v.Args[1]
		key := v.Args[2]
		if key.Op != OpSelectN || auxIntToInt64(key.AuxInt) != 0 {
			break
		}
		sbts := key.Args[0]
		if sbts.Op != OpStaticLECall || len(sbts.Args) != 4 {
			break
		}
		g := auxToCall(sbts.Aux)
		mem := sbts.Args[3]
		ptr := sbts.Args[1]
		len := sbts.Args[2]
		m := v.Args[3]
		if m.Op != OpSelectN || auxIntToInt64(m.AuxInt) != 1 || sbts != m.Args[0] || !((isSameCall(f, "runtime.mapaccess1_faststr") || isSameCall(f, "runtime.mapaccess2_faststr") || isSameCall(f, "runtime.mapdelete_faststr")) && isSameCall(g, "runtime.slicebytetostring") && key.Uses == 1 && sbts.Uses == 2 && resetCopy(m, mem) && clobber(sbts) && clobber(key)) {
			break
		}
		v.reset(OpStaticLECall)
		v.AuxInt = int32ToAuxInt(argsize)
		v.Aux = callToAux(f)
		v0 := b.NewValue0(v.Pos, OpStringMake, typ.String)
		v0.AddArg2(ptr, len)
		v.AddArg4(typ_, map_, v0, mem)
		return true
	}
	// match: (StaticLECall {f} [argsize] dict_ key:(SelectN [0] sbts:(StaticLECall {g} _ ptr len mem)) m:(SelectN [1] sbts))
	// cond: isSameCall(f, "unique.Make[go.shape.string]") && isSameCall(g, "runtime.slicebytetostring") && key.Uses == 1 && sbts.Uses == 2 && resetCopy(m, mem) && clobber(sbts) && clobber(key)
	// result: (StaticLECall {f} [argsize] dict_ (StringMake <typ.String> ptr len) mem)
	for {
		if len(v.Args) != 3 {
			break
		}
		argsize := auxIntToInt32(v.AuxInt)
		f := auxToCall(v.Aux)
		_ = v.Args[2]
		dict_ := v.Args[0]
		key := v.Args[1]
		if key.Op != OpSelectN || auxIntToInt64(key.AuxInt) != 0 {
			break
		}
		sbts := key.Args[0]
		if sbts.Op != OpStaticLECall || len(sbts.Args) != 4 {
			break
		}
		g := auxToCall(sbts.Aux)
		mem := sbts.Args[3]
		ptr := sbts.Args[1]
		len := sbts.Args[2]
		m := v.Args[2]
		if m.Op != OpSelectN || auxIntToInt64(m.AuxInt) != 1 || sbts != m.Args[0] || !(isSameCall(f, "unique.Make[go.shape.string]") && isSameCall(g, "runtime.slicebytetostring") && key.Uses == 1 && sbts.Uses == 2 && resetCopy(m, mem) && clobber(sbts) && clobber(key)) {
			break
		}
		v.reset(OpStaticLECall)
		v.AuxInt = int32ToAuxInt(argsize)
		v.Aux = callToAux(f)
		v0 := b.NewValue0(v.Pos, OpStringMake, typ.String)
		v0.AddArg2(ptr, len)
		v.AddArg3(dict_, v0, mem)
		return true
	}
	return false
}
func rewriteValuegeneric_OpStore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Store {t1} p1 (Load <t2> p2 mem) mem)
	// cond: isSamePtr(p1, p2) && t2.Size() == t1.Size()
	// result: mem
	for {
		t1 := auxToType(v.Aux)
		p1 := v_0
		if v_1.Op != OpLoad {
			break
		}
		t2 := v_1.Type
		mem := v_1.Args[1]
		p2 := v_1.Args[0]
		if mem != v_2 || !(isSamePtr(p1, p2) && t2.Size() == t1.Size()) {
			break
		}
		v.copyOf(mem)
		return true
	}
	// match: (Store {t1} p1 (Load <t2> p2 oldmem) mem:(Store {t3} p3 _ oldmem))
	// cond: isSamePtr(p1, p2) && t2.Size() == t1.Size() && disjoint(p1, t1.Size(), p3, t3.Size())
	// result: mem
	for {
		t1 := auxToType(v.Aux)
		p1 := v_0
		if v_1.Op != OpLoad {
			break
		}
		t2 := v_1.Type
		oldmem := v_1.Args[1]
		p2 := v_1.Args[0]
		mem := v_2
		if mem.Op != OpStore {
			break
		}
		t3 := auxToType(mem.Aux)
		_ = mem.Args[2]
		p3 := mem.Args[0]
		if oldmem != mem.Args[2] || !(isSamePtr(p1, p2) && t2.Size() == t1.Size() && disjoint(p1, t1.Size(), p3, t3.Size())) {
			break
		}
		v.copyOf(mem)
		return true
	}
	// match: (Store {t1} p1 (Load <t2> p2 oldmem) mem:(Store {t3} p3 _ (Store {t4} p4 _ oldmem)))
	// cond: isSamePtr(p1, p2) && t2.Size() == t1.Size() && disjoint(p1, t1.Size(), p3, t3.Size()) && disjoint(p1, t1.Size(), p4, t4.Size())
	// result: mem
	for {
		t1 := auxToType(v.Aux)
		p1 := v_0
		if v_1.Op != OpLoad {
			break
		}
		t2 := v_1.Type
		oldmem := v_1.Args[1]
		p2 := v_1.Args[0]
		mem := v_2
		if mem.Op != OpStore {
			break
		}
		t3 := auxToType(mem.Aux)
		_ = mem.Args[2]
		p3 := mem.Args[0]
		mem_2 := mem.Args[2]
		if mem_2.Op != OpStore {
			break
		}
		t4 := auxToType(mem_2.Aux)
		_ = mem_2.Args[2]
		p4 := mem_2.Args[0]
		if oldmem != mem_2.Args[2] || !(isSamePtr(p1, p2) && t2.Size() == t1.Size() && disjoint(p1, t1.Size(), p3, t3.Size()) && disjoint(p1, t1.Size(), p4, t4.Size())) {
			break
		}
		v.copyOf(mem)
		return true
	}
	// match: (Store {t1} p1 (Load <t2> p2 oldmem) mem:(Store {t3} p3 _ (Store {t4} p4 _ (Store {t5} p5 _ oldmem))))
	// cond: isSamePtr(p1, p2) && t2.Size() == t1.Size() && disjoint(p1, t1.Size(), p3, t3.Size()) && disjoint(p1, t1.Size(), p4, t4.Size()) && disjoint(p1, t1.Size(), p5, t5.Size())
	// result: mem
	for {
		t1 := auxToType(v.Aux)
		p1 := v_0
		if v_1.Op != OpLoad {
			break
		}
		t2 := v_1.Type
		oldmem := v_1.Args[1]
		p2 := v_1.Args[0]
		mem := v_2
		if mem.Op != OpStore {
			break
		}
		t3 := auxToType(mem.Aux)
		_ = mem.Args[2]
		p3 := mem.Args[0]
		mem_2 := mem.Args[2]
		if mem_2.Op != OpStore {
			break
		}
		t4 := auxToType(mem_2.Aux)
		_ = mem_2.Args[2]
		p4 := mem_2.Args[0]
		mem_2_2 := mem_2.Args[2]
		if mem_2_2.Op != OpStore {
			break
		}
		t5 := auxToType(mem_2_2.Aux)
		_ = mem_2_2.Args[2]
		p5 := mem_2_2.Args[0]
		if oldmem != mem_2_2.Args[2] || !(isSamePtr(p1, p2) && t2.Size() == t1.Size() && disjoint(p1, t1.Size(), p3, t3.Size()) && disjoint(p1, t1.Size(), p4, t4.Size()) && disjoint(p1, t1.Size(), p5, t5.Size())) {
			break
		}
		v.copyOf(mem)
		return true
	}
	// match: (Store {t} (OffPtr [o] p1) x mem:(Zero [n] p2 _))
	// cond: isConstZero(x) && o >= 0 && t.Size() + o <= n && isSamePtr(p1, p2)
	// result: mem
	for {
		t := auxToType(v.Aux)
		if v_0.Op != OpOffPtr {
			break
		}
		o := auxIntToInt64(v_0.AuxInt)
		p1 := v_0.Args[0]
		x := v_1
		mem := v_2
		if mem.Op != OpZero {
			break
		}
		n := auxIntToInt64(mem.AuxInt)
		p2 := mem.Args[0]
		if !(isConstZero(x) && o >= 0 && t.Size()+o <= n && isSamePtr(p1, p2)) {
			break
		}
		v.copyOf(mem)
		return true
	}
	// match: (Store {t1} op:(OffPtr [o1] p1) x mem:(Store {t2} p2 _ (Zero [n] p3 _)))
	// cond: isConstZero(x) && o1 >= 0 && t1.Size() + o1 <= n && isSamePtr(p1, p3) && disjoint(op, t1.Size(), p2, t2.Size())
	// result: mem
	for {
		t1 := auxToType(v.Aux)
		op := v_0
		if op.Op != OpOffPtr {
			break
		}
		o1 := auxIntToInt64(op.AuxInt)
		p1 := op.Args[0]
		x := v_1
		mem := v_2
		if mem.Op != OpStore {
			break
		}
		t2 := auxToType(mem.Aux)
		_ = mem.Args[2]
		p2 := mem.Args[0]
		mem_2 := mem.Args[2]
		if mem_2.Op != OpZero {
			break
		}
		n := auxIntToInt64(mem_2.AuxInt)
		p3 := mem_2.Args[0]
		if !(isConstZero(x) && o1 >= 0 && t1.Size()+o1 <= n && isSamePtr(p1, p3) && disjoint(op, t1.Size(), p2, t2.Size())) {
			break
		}
		v.copyOf(mem)
		return true
	}
	// match: (Store {t1} op:(OffPtr [o1] p1) x mem:(Store {t2} p2 _ (Store {t3} p3 _ (Zero [n] p4 _))))
	// cond: isConstZero(x) && o1 >= 0 && t1.Size() + o1 <= n && isSamePtr(p1, p4) && disjoint(op, t1.Size(), p2, t2.Size()) && disjoint(op, t1.Size(), p3, t3.Size())
	// result: mem
	for {
		t1 := auxToType(v.Aux)
		op := v_0
		if op.Op != OpOffPtr {
			break
		}
		o1 := auxIntToInt64(op.AuxInt)
		p1 := op.Args[0]
		x := v_1
		mem := v_2
		if mem.Op != OpStore {
			break
		}
		t2 := auxToType(mem.Aux)
		_ = mem.Args[2]
		p2 := mem.Args[0]
		mem_2 := mem.Args[2]
		if mem_2.Op != OpStore {
			break
		}
		t3 := auxToType(mem_2.Aux)
		_ = mem_2.Args[2]
		p3 := mem_2.Args[0]
		mem_2_2 := mem_2.Args[2]
		if mem_2_2.Op != OpZero {
			break
		}
		n := auxIntToInt64(mem_2_2.AuxInt)
		p4 := mem_2_2.Args[0]
		if !(isConstZero(x) && o1 >= 0 && t1.Size()+o1 <= n && isSamePtr(p1, p4) && disjoint(op, t1.Size(), p2, t2.Size()) && disjoint(op, t1.Size(), p3, t3.Size())) {
			break
		}
		v.copyOf(mem)
		return true
	}
	// match: (Store {t1} op:(OffPtr [o1] p1) x mem:(Store {t2} p2 _ (Store {t3} p3 _ (Store {t4} p4 _ (Zero [n] p5 _)))))
	// cond: isConstZero(x) && o1 >= 0 && t1.Size() + o1 <= n && isSamePtr(p1, p5) && disjoint(op, t1.Size(), p2, t2.Size()) && disjoint(op, t1.Size(), p3, t3.Size()) && disjoint(op, t1.Size(), p4, t4.Size())
	// result: mem
	for {
		t1 := auxToType(v.Aux)
		op := v_0
		if op.Op != OpOffPtr {
			break
		}
		o1 := auxIntToInt64(op.AuxInt)
		p1 := op.Args[0]
		x := v_1
		mem := v_2
		if mem.Op != OpStore {
			break
		}
		t2 := auxToType(mem.Aux)
		_ = mem.Args[2]
		p2 := mem.Args[0]
		mem_2 := mem.Args[2]
		if mem_2.Op != OpStore {
			break
		}
		t3 := auxToType(mem_2.Aux)
		_ = mem_2.Args[2]
		p3 := mem_2.Args[0]
		mem_2_2 := mem_2.Args[2]
		if mem_2_2.Op != OpStore {
			break
		}
		t4 := auxToType(mem_2_2.Aux)
		_ = mem_2_2.Args[2]
		p4 := mem_2_2.Args[0]
		mem_2_2_2 := mem_2_2.Args[2]
		if mem_2_2_2.Op != OpZero {
			break
		}
		n := auxIntToInt64(mem_2_2_2.AuxInt)
		p5 := mem_2_2_2.Args[0]
		if !(isConstZero(x) && o1 >= 0 && t1.Size()+o1 <= n && isSamePtr(p1, p5) && disjoint(op, t1.Size(), p2, t2.Size()) && disjoint(op, t1.Size(), p3, t3.Size()) && disjoint(op, t1.Size(), p4, t4.Size())) {
			break
		}
		v.copyOf(mem)
		return true
	}
	// match: (Store _ (StructMake ___) _)
	// result: rewriteStructStore(v)
	for {
		if v_1.Op != OpStructMake {
			break
		}
		v.copyOf(rewriteStructStore(v))
		return true
	}
	// match: (Store {t} dst (Load src mem) mem)
	// cond: !CanSSA(t)
	// result: (Move {t} [t.Size()] dst src mem)
	for {
		t := auxToType(v.Aux)
		dst := v_0
		if v_1.Op != OpLoad {
			break
		}
		mem := v_1.Args[1]
		src := v_1.Args[0]
		if mem != v_2 || !(!CanSSA(t)) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(t.Size())
		v.Aux = typeToAux(t)
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Store {t} dst (Load src mem) (VarDef {x} mem))
	// cond: !CanSSA(t)
	// result: (Move {t} [t.Size()] dst src (VarDef {x} mem))
	for {
		t := auxToType(v.Aux)
		dst := v_0
		if v_1.Op != OpLoad {
			break
		}
		mem := v_1.Args[1]
		src := v_1.Args[0]
		if v_2.Op != OpVarDef {
			break
		}
		x := auxToSym(v_2.Aux)
		if mem != v_2.Args[0] || !(!CanSSA(t)) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(t.Size())
		v.Aux = typeToAux(t)
		v0 := b.NewValue0(v.Pos, OpVarDef, types.TypeMem)
		v0.Aux = symToAux(x)
		v0.AddArg(mem)
		v.AddArg3(dst, src, v0)
		return true
	}
	// match: (Store _ (ArrayMake0) mem)
	// result: mem
	for {
		if v_1.Op != OpArrayMake0 {
			break
		}
		mem := v_2
		v.copyOf(mem)
		return true
	}
	// match: (Store dst (ArrayMake1 e) mem)
	// result: (Store {e.Type} dst e mem)
	for {
		dst := v_0
		if v_1.Op != OpArrayMake1 {
			break
		}
		e := v_1.Args[0]
		mem := v_2
		v.reset(OpStore)
		v.Aux = typeToAux(e.Type)
		v.AddArg3(dst, e, mem)
		return true
	}
	// match: (Store (SelectN [0] call:(StaticLECall _ _)) x mem:(SelectN [1] call))
	// cond: isConstZero(x) && isSameCall(call.Aux, "runtime.newobject")
	// result: mem
	for {
		if v_0.Op != OpSelectN || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		call := v_0.Args[0]
		if call.Op != OpStaticLECall || len(call.Args) != 2 {
			break
		}
		x := v_1
		mem := v_2
		if mem.Op != OpSelectN || auxIntToInt64(mem.AuxInt) != 1 || call != mem.Args[0] || !(isConstZero(x) && isSameCall(call.Aux, "runtime.newobject")) {
			break
		}
		v.copyOf(mem)
		return true
	}
	// match: (Store (OffPtr (SelectN [0] call:(StaticLECall _ _))) x mem:(SelectN [1] call))
	// cond: isConstZero(x) && isSameCall(call.Aux, "runtime.newobject")
	// result: mem
	for {
		if v_0.Op != OpOffPtr {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpSelectN || auxIntToInt64(v_0_0.AuxInt) != 0 {
			break
		}
		call := v_0_0.Args[0]
		if call.Op != OpStaticLECall || len(call.Args) != 2 {
			break
		}
		x := v_1
		mem := v_2
		if mem.Op != OpSelectN || auxIntToInt64(mem.AuxInt) != 1 || call != mem.Args[0] || !(isConstZero(x) && isSameCall(call.Aux, "runtime.newobject")) {
			break
		}
		v.copyOf(mem)
		return true
	}
	// match: (Store {t1} op1:(OffPtr [o1] p1) d1 m2:(Store {t2} op2:(OffPtr [0] p2) d2 m3:(Move [n] p3 _ mem)))
	// cond: m2.Uses == 1 && m3.Uses == 1 && o1 == t2.Size() && n == t2.Size() + t1.Size() && isSamePtr(p1, p2) && isSamePtr(p2, p3) && clobber(m2, m3)
	// result: (Store {t1} op1 d1 (Store {t2} op2 d2 mem))
	for {
		t1 := auxToType(v.Aux)
		op1 := v_0
		if op1.Op != OpOffPtr {
			break
		}
		o1 := auxIntToInt64(op1.AuxInt)
		p1 := op1.Args[0]
		d1 := v_1
		m2 := v_2
		if m2.Op != OpStore {
			break
		}
		t2 := auxToType(m2.Aux)
		_ = m2.Args[2]
		op2 := m2.Args[0]
		if op2.Op != OpOffPtr || auxIntToInt64(op2.AuxInt) != 0 {
			break
		}
		p2 := op2.Args[0]
		d2 := m2.Args[1]
		m3 := m2.Args[2]
		if m3.Op != OpMove {
			break
		}
		n := auxIntToInt64(m3.AuxInt)
		mem := m3.Args[2]
		p3 := m3.Args[0]
		if !(m2.Uses == 1 && m3.Uses == 1 && o1 == t2.Size() && n == t2.Size()+t1.Size() && isSamePtr(p1, p2) && isSamePtr(p2, p3) && clobber(m2, m3)) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t1)
		v0 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v0.Aux = typeToAux(t2)
		v0.AddArg3(op2, d2, mem)
		v.AddArg3(op1, d1, v0)
		return true
	}
	// match: (Store {t1} op1:(OffPtr [o1] p1) d1 m2:(Store {t2} op2:(OffPtr [o2] p2) d2 m3:(Store {t3} op3:(OffPtr [0] p3) d3 m4:(Move [n] p4 _ mem))))
	// cond: m2.Uses == 1 && m3.Uses == 1 && m4.Uses == 1 && o2 == t3.Size() && o1-o2 == t2.Size() && n == t3.Size() + t2.Size() + t1.Size() && isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && clobber(m2, m3, m4)
	// result: (Store {t1} op1 d1 (Store {t2} op2 d2 (Store {t3} op3 d3 mem)))
	for {
		t1 := auxToType(v.Aux)
		op1 := v_0
		if op1.Op != OpOffPtr {
			break
		}
		o1 := auxIntToInt64(op1.AuxInt)
		p1 := op1.Args[0]
		d1 := v_1
		m2 := v_2
		if m2.Op != OpStore {
			break
		}
		t2 := auxToType(m2.Aux)
		_ = m2.Args[2]
		op2 := m2.Args[0]
		if op2.Op != OpOffPtr {
			break
		}
		o2 := auxIntToInt64(op2.AuxInt)
		p2 := op2.Args[0]
		d2 := m2.Args[1]
		m3 := m2.Args[2]
		if m3.Op != OpStore {
			break
		}
		t3 := auxToType(m3.Aux)
		_ = m3.Args[2]
		op3 := m3.Args[0]
		if op3.Op != OpOffPtr || auxIntToInt64(op3.AuxInt) != 0 {
			break
		}
		p3 := op3.Args[0]
		d3 := m3.Args[1]
		m4 := m3.Args[2]
		if m4.Op != OpMove {
			break
		}
		n := auxIntToInt64(m4.AuxInt)
		mem := m4.Args[2]
		p4 := m4.Args[0]
		if !(m2.Uses == 1 && m3.Uses == 1 && m4.Uses == 1 && o2 == t3.Size() && o1-o2 == t2.Size() && n == t3.Size()+t2.Size()+t1.Size() && isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && clobber(m2, m3, m4)) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t1)
		v0 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v0.Aux = typeToAux(t2)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t3)
		v1.AddArg3(op3, d3, mem)
		v0.AddArg3(op2, d2, v1)
		v.AddArg3(op1, d1, v0)
		return true
	}
	// match: (Store {t1} op1:(OffPtr [o1] p1) d1 m2:(Store {t2} op2:(OffPtr [o2] p2) d2 m3:(Store {t3} op3:(OffPtr [o3] p3) d3 m4:(Store {t4} op4:(OffPtr [0] p4) d4 m5:(Move [n] p5 _ mem)))))
	// cond: m2.Uses == 1 && m3.Uses == 1 && m4.Uses == 1 && m5.Uses == 1 && o3 == t4.Size() && o2-o3 == t3.Size() && o1-o2 == t2.Size() && n == t4.Size() + t3.Size() + t2.Size() + t1.Size() && isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && isSamePtr(p4, p5) && clobber(m2, m3, m4, m5)
	// result: (Store {t1} op1 d1 (Store {t2} op2 d2 (Store {t3} op3 d3 (Store {t4} op4 d4 mem))))
	for {
		t1 := auxToType(v.Aux)
		op1 := v_0
		if op1.Op != OpOffPtr {
			break
		}
		o1 := auxIntToInt64(op1.AuxInt)
		p1 := op1.Args[0]
		d1 := v_1
		m2 := v_2
		if m2.Op != OpStore {
			break
		}
		t2 := auxToType(m2.Aux)
		_ = m2.Args[2]
		op2 := m2.Args[0]
		if op2.Op != OpOffPtr {
			break
		}
		o2 := auxIntToInt64(op2.AuxInt)
		p2 := op2.Args[0]
		d2 := m2.Args[1]
		m3 := m2.Args[2]
		if m3.Op != OpStore {
			break
		}
		t3 := auxToType(m3.Aux)
		_ = m3.Args[2]
		op3 := m3.Args[0]
		if op3.Op != OpOffPtr {
			break
		}
		o3 := auxIntToInt64(op3.AuxInt)
		p3 := op3.Args[0]
		d3 := m3.Args[1]
		m4 := m3.Args[2]
		if m4.Op != OpStore {
			break
		}
		t4 := auxToType(m4.Aux)
		_ = m4.Args[2]
		op4 := m4.Args[0]
		if op4.Op != OpOffPtr || auxIntToInt64(op4.AuxInt) != 0 {
			break
		}
		p4 := op4.Args[0]
		d4 := m4.Args[1]
		m5 := m4.Args[2]
		if m5.Op != OpMove {
			break
		}
		n := auxIntToInt64(m5.AuxInt)
		mem := m5.Args[2]
		p5 := m5.Args[0]
		if !(m2.Uses == 1 && m3.Uses == 1 && m4.Uses == 1 && m5.Uses == 1 && o3 == t4.Size() && o2-o3 == t3.Size() && o1-o2 == t2.Size() && n == t4.Size()+t3.Size()+t2.Size()+t1.Size() && isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && isSamePtr(p4, p5) && clobber(m2, m3, m4, m5)) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t1)
		v0 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v0.Aux = typeToAux(t2)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t3)
		v2 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v2.Aux = typeToAux(t4)
		v2.AddArg3(op4, d4, mem)
		v1.AddArg3(op3, d3, v2)
		v0.AddArg3(op2, d2, v1)
		v.AddArg3(op1, d1, v0)
		return true
	}
	// match: (Store {t1} op1:(OffPtr [o1] p1) d1 m2:(Store {t2} op2:(OffPtr [0] p2) d2 m3:(Zero [n] p3 mem)))
	// cond: m2.Uses == 1 && m3.Uses == 1 && o1 == t2.Size() && n == t2.Size() + t1.Size() && isSamePtr(p1, p2) && isSamePtr(p2, p3) && clobber(m2, m3)
	// result: (Store {t1} op1 d1 (Store {t2} op2 d2 mem))
	for {
		t1 := auxToType(v.Aux)
		op1 := v_0
		if op1.Op != OpOffPtr {
			break
		}
		o1 := auxIntToInt64(op1.AuxInt)
		p1 := op1.Args[0]
		d1 := v_1
		m2 := v_2
		if m2.Op != OpStore {
			break
		}
		t2 := auxToType(m2.Aux)
		_ = m2.Args[2]
		op2 := m2.Args[0]
		if op2.Op != OpOffPtr || auxIntToInt64(op2.AuxInt) != 0 {
			break
		}
		p2 := op2.Args[0]
		d2 := m2.Args[1]
		m3 := m2.Args[2]
		if m3.Op != OpZero {
			break
		}
		n := auxIntToInt64(m3.AuxInt)
		mem := m3.Args[1]
		p3 := m3.Args[0]
		if !(m2.Uses == 1 && m3.Uses == 1 && o1 == t2.Size() && n == t2.Size()+t1.Size() && isSamePtr(p1, p2) && isSamePtr(p2, p3) && clobber(m2, m3)) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t1)
		v0 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v0.Aux = typeToAux(t2)
		v0.AddArg3(op2, d2, mem)
		v.AddArg3(op1, d1, v0)
		return true
	}
	// match: (Store {t1} op1:(OffPtr [o1] p1) d1 m2:(Store {t2} op2:(OffPtr [o2] p2) d2 m3:(Store {t3} op3:(OffPtr [0] p3) d3 m4:(Zero [n] p4 mem))))
	// cond: m2.Uses == 1 && m3.Uses == 1 && m4.Uses == 1 && o2 == t3.Size() && o1-o2 == t2.Size() && n == t3.Size() + t2.Size() + t1.Size() && isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && clobber(m2, m3, m4)
	// result: (Store {t1} op1 d1 (Store {t2} op2 d2 (Store {t3} op3 d3 mem)))
	for {
		t1 := auxToType(v.Aux)
		op1 := v_0
		if op1.Op != OpOffPtr {
			break
		}
		o1 := auxIntToInt64(op1.AuxInt)
		p1 := op1.Args[0]
		d1 := v_1
		m2 := v_2
		if m2.Op != OpStore {
			break
		}
		t2 := auxToType(m2.Aux)
		_ = m2.Args[2]
		op2 := m2.Args[0]
		if op2.Op != OpOffPtr {
			break
		}
		o2 := auxIntToInt64(op2.AuxInt)
		p2 := op2.Args[0]
		d2 := m2.Args[1]
		m3 := m2.Args[2]
		if m3.Op != OpStore {
			break
		}
		t3 := auxToType(m3.Aux)
		_ = m3.Args[2]
		op3 := m3.Args[0]
		if op3.Op != OpOffPtr || auxIntToInt64(op3.AuxInt) != 0 {
			break
		}
		p3 := op3.Args[0]
		d3 := m3.Args[1]
		m4 := m3.Args[2]
		if m4.Op != OpZero {
			break
		}
		n := auxIntToInt64(m4.AuxInt)
		mem := m4.Args[1]
		p4 := m4.Args[0]
		if !(m2.Uses == 1 && m3.Uses == 1 && m4.Uses == 1 && o2 == t3.Size() && o1-o2 == t2.Size() && n == t3.Size()+t2.Size()+t1.Size() && isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && clobber(m2, m3, m4)) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t1)
		v0 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v0.Aux = typeToAux(t2)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t3)
		v1.AddArg3(op3, d3, mem)
		v0.AddArg3(op2, d2, v1)
		v.AddArg3(op1, d1, v0)
		return true
	}
	// match: (Store {t1} op1:(OffPtr [o1] p1) d1 m2:(Store {t2} op2:(OffPtr [o2] p2) d2 m3:(Store {t3} op3:(OffPtr [o3] p3) d3 m4:(Store {t4} op4:(OffPtr [0] p4) d4 m5:(Zero [n] p5 mem)))))
	// cond: m2.Uses == 1 && m3.Uses == 1 && m4.Uses == 1 && m5.Uses == 1 && o3 == t4.Size() && o2-o3 == t3.Size() && o1-o2 == t2.Size() && n == t4.Size() + t3.Size() + t2.Size() + t1.Size() && isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && isSamePtr(p4, p5) && clobber(m2, m3, m4, m5)
	// result: (Store {t1} op1 d1 (Store {t2} op2 d2 (Store {t3} op3 d3 (Store {t4} op4 d4 mem))))
	for {
		t1 := auxToType(v.Aux)
		op1 := v_0
		if op1.Op != OpOffPtr {
			break
		}
		o1 := auxIntToInt64(op1.AuxInt)
		p1 := op1.Args[0]
		d1 := v_1
		m2 := v_2
		if m2.Op != OpStore {
			break
		}
		t2 := auxToType(m2.Aux)
		_ = m2.Args[2]
		op2 := m2.Args[0]
		if op2.Op != OpOffPtr {
			break
		}
		o2 := auxIntToInt64(op2.AuxInt)
		p2 := op2.Args[0]
		d2 := m2.Args[1]
		m3 := m2.Args[2]
		if m3.Op != OpStore {
			break
		}
		t3 := auxToType(m3.Aux)
		_ = m3.Args[2]
		op3 := m3.Args[0]
		if op3.Op != OpOffPtr {
			break
		}
		o3 := auxIntToInt64(op3.AuxInt)
		p3 := op3.Args[0]
		d3 := m3.Args[1]
		m4 := m3.Args[2]
		if m4.Op != OpStore {
			break
		}
		t4 := auxToType(m4.Aux)
		_ = m4.Args[2]
		op4 := m4.Args[0]
		if op4.Op != OpOffPtr || auxIntToInt64(op4.AuxInt) != 0 {
			break
		}
		p4 := op4.Args[0]
		d4 := m4.Args[1]
		m5 := m4.Args[2]
		if m5.Op != OpZero {
			break
		}
		n := auxIntToInt64(m5.AuxInt)
		mem := m5.Args[1]
		p5 := m5.Args[0]
		if !(m2.Uses == 1 && m3.Uses == 1 && m4.Uses == 1 && m5.Uses == 1 && o3 == t4.Size() && o2-o3 == t3.Size() && o1-o2 == t2.Size() && n == t4.Size()+t3.Size()+t2.Size()+t1.Size() && isSamePtr(p1, p2) && isSamePtr(p2, p3) && isSamePtr(p3, p4) && isSamePtr(p4, p5) && clobber(m2, m3, m4, m5)) {
			break
		}
		v.reset(OpStore)
		v.Aux = typeToAux(t1)
		v0 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v0.Aux = typeToAux(t2)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = typeToAux(t3)
		v2 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v2.Aux = typeToAux(t4)
		v2.AddArg3(op4, d4, mem)
		v1.AddArg3(op3, d3, v2)
		v0.AddArg3(op2, d2, v1)
		v.AddArg3(op1, d1, v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpStringLen(v *Value) bool {
	v_0 := v.Args[0]
	// match: (StringLen (StringMake _ (Const64 <t> [c])))
	// result: (Const64 <t> [c])
	for {
		if v_0.Op != OpStringMake {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		t := v_0_1.Type
		c := auxIntToInt64(v_0_1.AuxInt)
		v.reset(OpConst64)
		v.Type = t
		v.AuxInt = int64ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValuegeneric_OpStringPtr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (StringPtr (StringMake (Addr <t> {s} base) _))
	// result: (Addr <t> {s} base)
	for {
		if v_0.Op != OpStringMake {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAddr {
			break
		}
		t := v_0_0.Type
		s := auxToSym(v_0_0.Aux)
		base := v_0_0.Args[0]
		v.reset(OpAddr)
		v.Type = t
		v.Aux = symToAux(s)
		v.AddArg(base)
		return true
	}
	return false
}
func rewriteValuegeneric_OpStructSelect(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (StructSelect [i] x:(StructMake ___))
	// result: x.Args[i]
	for {
		i := auxIntToInt64(v.AuxInt)
		x := v_0
		if x.Op != OpStructMake {
			break
		}
		v.copyOf(x.Args[i])
		return true
	}
	// match: (StructSelect [i] x:(Load <t> ptr mem))
	// cond: !CanSSA(t)
	// result: @x.Block (Load <v.Type> (OffPtr <v.Type.PtrTo()> [t.FieldOff(int(i))] ptr) mem)
	for {
		i := auxIntToInt64(v.AuxInt)
		x := v_0
		if x.Op != OpLoad {
			break
		}
		t := x.Type
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(!CanSSA(t)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpLoad, v.Type)
		v.copyOf(v0)
		v1 := b.NewValue0(v.Pos, OpOffPtr, v.Type.PtrTo())
		v1.AuxInt = int64ToAuxInt(t.FieldOff(int(i)))
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (StructSelect [0] (IData x))
	// result: (IData x)
	for {
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpIData {
			break
		}
		x := v_0.Args[0]
		v.reset(OpIData)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSub16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Sub16 (Const16 [c]) (Const16 [d]))
	// result: (Const16 [c-d])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		if v_1.Op != OpConst16 {
			break
		}
		d := auxIntToInt16(v_1.AuxInt)
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(c - d)
		return true
	}
	// match: (Sub16 x (Const16 <t> [c]))
	// cond: x.Op != OpConst16
	// result: (Add16 (Const16 <t> [-c]) x)
	for {
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		t := v_1.Type
		c := auxIntToInt16(v_1.AuxInt)
		if !(x.Op != OpConst16) {
			break
		}
		v.reset(OpAdd16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(-c)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Sub16 <t> (Mul16 x y) (Mul16 x z))
	// result: (Mul16 x (Sub16 <t> y z))
	for {
		t := v.Type
		if v_0.Op != OpMul16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if v_1.Op != OpMul16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				z := v_1_1
				v.reset(OpMul16)
				v0 := b.NewValue0(v.Pos, OpSub16, t)
				v0.AddArg2(y, z)
				v.AddArg2(x, v0)
				return true
			}
		}
		break
	}
	// match: (Sub16 x x)
	// result: (Const16 [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	// match: (Sub16 (Neg16 x) (Com16 x))
	// result: (Const16 [1])
	for {
		if v_0.Op != OpNeg16 {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpCom16 || x != v_1.Args[0] {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(1)
		return true
	}
	// match: (Sub16 (Com16 x) (Neg16 x))
	// result: (Const16 [-1])
	for {
		if v_0.Op != OpCom16 {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpNeg16 || x != v_1.Args[0] {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(-1)
		return true
	}
	// match: (Sub16 (Add16 t x) (Add16 t y))
	// result: (Sub16 x y)
	for {
		if v_0.Op != OpAdd16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			t := v_0_0
			x := v_0_1
			if v_1.Op != OpAdd16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if t != v_1_0 {
					continue
				}
				y := v_1_1
				v.reset(OpSub16)
				v.AddArg2(x, y)
				return true
			}
		}
		break
	}
	// match: (Sub16 (Add16 x y) x)
	// result: y
	for {
		if v_0.Op != OpAdd16 {
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
			v.copyOf(y)
			return true
		}
		break
	}
	// match: (Sub16 (Add16 x y) y)
	// result: x
	for {
		if v_0.Op != OpAdd16 {
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
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Sub16 (Sub16 x y) x)
	// result: (Neg16 y)
	for {
		if v_0.Op != OpSub16 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpNeg16)
		v.AddArg(y)
		return true
	}
	// match: (Sub16 x (Add16 x y))
	// result: (Neg16 y)
	for {
		x := v_0
		if v_1.Op != OpAdd16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if x != v_1_0 {
				continue
			}
			y := v_1_1
			v.reset(OpNeg16)
			v.AddArg(y)
			return true
		}
		break
	}
	// match: (Sub16 x (Sub16 i:(Const16 <t>) z))
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Sub16 (Add16 <t> x z) i)
	for {
		x := v_0
		if v_1.Op != OpSub16 {
			break
		}
		z := v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpSub16)
		v0 := b.NewValue0(v.Pos, OpAdd16, t)
		v0.AddArg2(x, z)
		v.AddArg2(v0, i)
		return true
	}
	// match: (Sub16 x (Add16 z i:(Const16 <t>)))
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Sub16 (Sub16 <t> x z) i)
	for {
		x := v_0
		if v_1.Op != OpAdd16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			z := v_1_0
			i := v_1_1
			if i.Op != OpConst16 {
				continue
			}
			t := i.Type
			if !(z.Op != OpConst16 && x.Op != OpConst16) {
				continue
			}
			v.reset(OpSub16)
			v0 := b.NewValue0(v.Pos, OpSub16, t)
			v0.AddArg2(x, z)
			v.AddArg2(v0, i)
			return true
		}
		break
	}
	// match: (Sub16 (Sub16 i:(Const16 <t>) z) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Sub16 i (Add16 <t> z x))
	for {
		if v_0.Op != OpSub16 {
			break
		}
		z := v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		x := v_1
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpSub16)
		v0 := b.NewValue0(v.Pos, OpAdd16, t)
		v0.AddArg2(z, x)
		v.AddArg2(i, v0)
		return true
	}
	// match: (Sub16 (Add16 z i:(Const16 <t>)) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Add16 i (Sub16 <t> z x))
	for {
		if v_0.Op != OpAdd16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			z := v_0_0
			i := v_0_1
			if i.Op != OpConst16 {
				continue
			}
			t := i.Type
			x := v_1
			if !(z.Op != OpConst16 && x.Op != OpConst16) {
				continue
			}
			v.reset(OpAdd16)
			v0 := b.NewValue0(v.Pos, OpSub16, t)
			v0.AddArg2(z, x)
			v.AddArg2(i, v0)
			return true
		}
		break
	}
	// match: (Sub16 (Const16 <t> [c]) (Sub16 (Const16 <t> [d]) x))
	// result: (Add16 (Const16 <t> [c-d]) x)
	for {
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := auxIntToInt16(v_0.AuxInt)
		if v_1.Op != OpSub16 {
			break
		}
		x := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst16 || v_1_0.Type != t {
			break
		}
		d := auxIntToInt16(v_1_0.AuxInt)
		v.reset(OpAdd16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int16ToAuxInt(c - d)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Sub16 (Const16 <t> [c]) (Add16 (Const16 <t> [d]) x))
	// result: (Sub16 (Const16 <t> [c-d]) x)
	for {
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := auxIntToInt16(v_0.AuxInt)
		if v_1.Op != OpAdd16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if v_1_0.Op != OpConst16 || v_1_0.Type != t {
				continue
			}
			d := auxIntToInt16(v_1_0.AuxInt)
			x := v_1_1
			v.reset(OpSub16)
			v0 := b.NewValue0(v.Pos, OpConst16, t)
			v0.AuxInt = int16ToAuxInt(c - d)
			v.AddArg2(v0, x)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpSub32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Sub32 (Const32 [c]) (Const32 [d]))
	// result: (Const32 [c-d])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if v_1.Op != OpConst32 {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(c - d)
		return true
	}
	// match: (Sub32 x (Const32 <t> [c]))
	// cond: x.Op != OpConst32
	// result: (Add32 (Const32 <t> [-c]) x)
	for {
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		t := v_1.Type
		c := auxIntToInt32(v_1.AuxInt)
		if !(x.Op != OpConst32) {
			break
		}
		v.reset(OpAdd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(-c)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Sub32 <t> (Mul32 x y) (Mul32 x z))
	// result: (Mul32 x (Sub32 <t> y z))
	for {
		t := v.Type
		if v_0.Op != OpMul32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if v_1.Op != OpMul32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				z := v_1_1
				v.reset(OpMul32)
				v0 := b.NewValue0(v.Pos, OpSub32, t)
				v0.AddArg2(y, z)
				v.AddArg2(x, v0)
				return true
			}
		}
		break
	}
	// match: (Sub32 x x)
	// result: (Const32 [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (Sub32 (Neg32 x) (Com32 x))
	// result: (Const32 [1])
	for {
		if v_0.Op != OpNeg32 {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpCom32 || x != v_1.Args[0] {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (Sub32 (Com32 x) (Neg32 x))
	// result: (Const32 [-1])
	for {
		if v_0.Op != OpCom32 {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpNeg32 || x != v_1.Args[0] {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(-1)
		return true
	}
	// match: (Sub32 (Add32 t x) (Add32 t y))
	// result: (Sub32 x y)
	for {
		if v_0.Op != OpAdd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			t := v_0_0
			x := v_0_1
			if v_1.Op != OpAdd32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if t != v_1_0 {
					continue
				}
				y := v_1_1
				v.reset(OpSub32)
				v.AddArg2(x, y)
				return true
			}
		}
		break
	}
	// match: (Sub32 (Add32 x y) x)
	// result: y
	for {
		if v_0.Op != OpAdd32 {
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
			v.copyOf(y)
			return true
		}
		break
	}
	// match: (Sub32 (Add32 x y) y)
	// result: x
	for {
		if v_0.Op != OpAdd32 {
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
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Sub32 (Sub32 x y) x)
	// result: (Neg32 y)
	for {
		if v_0.Op != OpSub32 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpNeg32)
		v.AddArg(y)
		return true
	}
	// match: (Sub32 x (Add32 x y))
	// result: (Neg32 y)
	for {
		x := v_0
		if v_1.Op != OpAdd32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if x != v_1_0 {
				continue
			}
			y := v_1_1
			v.reset(OpNeg32)
			v.AddArg(y)
			return true
		}
		break
	}
	// match: (Sub32 x (Sub32 i:(Const32 <t>) z))
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Sub32 (Add32 <t> x z) i)
	for {
		x := v_0
		if v_1.Op != OpSub32 {
			break
		}
		z := v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpSub32)
		v0 := b.NewValue0(v.Pos, OpAdd32, t)
		v0.AddArg2(x, z)
		v.AddArg2(v0, i)
		return true
	}
	// match: (Sub32 x (Add32 z i:(Const32 <t>)))
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Sub32 (Sub32 <t> x z) i)
	for {
		x := v_0
		if v_1.Op != OpAdd32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			z := v_1_0
			i := v_1_1
			if i.Op != OpConst32 {
				continue
			}
			t := i.Type
			if !(z.Op != OpConst32 && x.Op != OpConst32) {
				continue
			}
			v.reset(OpSub32)
			v0 := b.NewValue0(v.Pos, OpSub32, t)
			v0.AddArg2(x, z)
			v.AddArg2(v0, i)
			return true
		}
		break
	}
	// match: (Sub32 (Sub32 i:(Const32 <t>) z) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Sub32 i (Add32 <t> z x))
	for {
		if v_0.Op != OpSub32 {
			break
		}
		z := v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		x := v_1
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpSub32)
		v0 := b.NewValue0(v.Pos, OpAdd32, t)
		v0.AddArg2(z, x)
		v.AddArg2(i, v0)
		return true
	}
	// match: (Sub32 (Add32 z i:(Const32 <t>)) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Add32 i (Sub32 <t> z x))
	for {
		if v_0.Op != OpAdd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			z := v_0_0
			i := v_0_1
			if i.Op != OpConst32 {
				continue
			}
			t := i.Type
			x := v_1
			if !(z.Op != OpConst32 && x.Op != OpConst32) {
				continue
			}
			v.reset(OpAdd32)
			v0 := b.NewValue0(v.Pos, OpSub32, t)
			v0.AddArg2(z, x)
			v.AddArg2(i, v0)
			return true
		}
		break
	}
	// match: (Sub32 (Const32 <t> [c]) (Sub32 (Const32 <t> [d]) x))
	// result: (Add32 (Const32 <t> [c-d]) x)
	for {
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := auxIntToInt32(v_0.AuxInt)
		if v_1.Op != OpSub32 {
			break
		}
		x := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst32 || v_1_0.Type != t {
			break
		}
		d := auxIntToInt32(v_1_0.AuxInt)
		v.reset(OpAdd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int32ToAuxInt(c - d)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Sub32 (Const32 <t> [c]) (Add32 (Const32 <t> [d]) x))
	// result: (Sub32 (Const32 <t> [c-d]) x)
	for {
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := auxIntToInt32(v_0.AuxInt)
		if v_1.Op != OpAdd32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if v_1_0.Op != OpConst32 || v_1_0.Type != t {
				continue
			}
			d := auxIntToInt32(v_1_0.AuxInt)
			x := v_1_1
			v.reset(OpSub32)
			v0 := b.NewValue0(v.Pos, OpConst32, t)
			v0.AuxInt = int32ToAuxInt(c - d)
			v.AddArg2(v0, x)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpSub32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Sub32F (Const32F [c]) (Const32F [d]))
	// cond: c-d == c-d
	// result: (Const32F [c-d])
	for {
		if v_0.Op != OpConst32F {
			break
		}
		c := auxIntToFloat32(v_0.AuxInt)
		if v_1.Op != OpConst32F {
			break
		}
		d := auxIntToFloat32(v_1.AuxInt)
		if !(c-d == c-d) {
			break
		}
		v.reset(OpConst32F)
		v.AuxInt = float32ToAuxInt(c - d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSub64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Sub64 (Const64 [c]) (Const64 [d]))
	// result: (Const64 [c-d])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpConst64 {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(c - d)
		return true
	}
	// match: (Sub64 x (Const64 <t> [c]))
	// cond: x.Op != OpConst64
	// result: (Add64 (Const64 <t> [-c]) x)
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		c := auxIntToInt64(v_1.AuxInt)
		if !(x.Op != OpConst64) {
			break
		}
		v.reset(OpAdd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(-c)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Sub64 <t> (Mul64 x y) (Mul64 x z))
	// result: (Mul64 x (Sub64 <t> y z))
	for {
		t := v.Type
		if v_0.Op != OpMul64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if v_1.Op != OpMul64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				z := v_1_1
				v.reset(OpMul64)
				v0 := b.NewValue0(v.Pos, OpSub64, t)
				v0.AddArg2(y, z)
				v.AddArg2(x, v0)
				return true
			}
		}
		break
	}
	// match: (Sub64 x x)
	// result: (Const64 [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (Sub64 (Neg64 x) (Com64 x))
	// result: (Const64 [1])
	for {
		if v_0.Op != OpNeg64 {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpCom64 || x != v_1.Args[0] {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (Sub64 (Com64 x) (Neg64 x))
	// result: (Const64 [-1])
	for {
		if v_0.Op != OpCom64 {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpNeg64 || x != v_1.Args[0] {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	// match: (Sub64 (Add64 t x) (Add64 t y))
	// result: (Sub64 x y)
	for {
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			t := v_0_0
			x := v_0_1
			if v_1.Op != OpAdd64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if t != v_1_0 {
					continue
				}
				y := v_1_1
				v.reset(OpSub64)
				v.AddArg2(x, y)
				return true
			}
		}
		break
	}
	// match: (Sub64 (Add64 x y) x)
	// result: y
	for {
		if v_0.Op != OpAdd64 {
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
			v.copyOf(y)
			return true
		}
		break
	}
	// match: (Sub64 (Add64 x y) y)
	// result: x
	for {
		if v_0.Op != OpAdd64 {
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
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Sub64 (Sub64 x y) x)
	// result: (Neg64 y)
	for {
		if v_0.Op != OpSub64 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpNeg64)
		v.AddArg(y)
		return true
	}
	// match: (Sub64 x (Add64 x y))
	// result: (Neg64 y)
	for {
		x := v_0
		if v_1.Op != OpAdd64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if x != v_1_0 {
				continue
			}
			y := v_1_1
			v.reset(OpNeg64)
			v.AddArg(y)
			return true
		}
		break
	}
	// match: (Sub64 x (Sub64 i:(Const64 <t>) z))
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Sub64 (Add64 <t> x z) i)
	for {
		x := v_0
		if v_1.Op != OpSub64 {
			break
		}
		z := v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpSub64)
		v0 := b.NewValue0(v.Pos, OpAdd64, t)
		v0.AddArg2(x, z)
		v.AddArg2(v0, i)
		return true
	}
	// match: (Sub64 x (Add64 z i:(Const64 <t>)))
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Sub64 (Sub64 <t> x z) i)
	for {
		x := v_0
		if v_1.Op != OpAdd64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			z := v_1_0
			i := v_1_1
			if i.Op != OpConst64 {
				continue
			}
			t := i.Type
			if !(z.Op != OpConst64 && x.Op != OpConst64) {
				continue
			}
			v.reset(OpSub64)
			v0 := b.NewValue0(v.Pos, OpSub64, t)
			v0.AddArg2(x, z)
			v.AddArg2(v0, i)
			return true
		}
		break
	}
	// match: (Sub64 (Sub64 i:(Const64 <t>) z) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Sub64 i (Add64 <t> z x))
	for {
		if v_0.Op != OpSub64 {
			break
		}
		z := v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		x := v_1
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpSub64)
		v0 := b.NewValue0(v.Pos, OpAdd64, t)
		v0.AddArg2(z, x)
		v.AddArg2(i, v0)
		return true
	}
	// match: (Sub64 (Add64 z i:(Const64 <t>)) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Add64 i (Sub64 <t> z x))
	for {
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			z := v_0_0
			i := v_0_1
			if i.Op != OpConst64 {
				continue
			}
			t := i.Type
			x := v_1
			if !(z.Op != OpConst64 && x.Op != OpConst64) {
				continue
			}
			v.reset(OpAdd64)
			v0 := b.NewValue0(v.Pos, OpSub64, t)
			v0.AddArg2(z, x)
			v.AddArg2(i, v0)
			return true
		}
		break
	}
	// match: (Sub64 (Const64 <t> [c]) (Sub64 (Const64 <t> [d]) x))
	// result: (Add64 (Const64 <t> [c-d]) x)
	for {
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpSub64 {
			break
		}
		x := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst64 || v_1_0.Type != t {
			break
		}
		d := auxIntToInt64(v_1_0.AuxInt)
		v.reset(OpAdd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64ToAuxInt(c - d)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Sub64 (Const64 <t> [c]) (Add64 (Const64 <t> [d]) x))
	// result: (Sub64 (Const64 <t> [c-d]) x)
	for {
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpAdd64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if v_1_0.Op != OpConst64 || v_1_0.Type != t {
				continue
			}
			d := auxIntToInt64(v_1_0.AuxInt)
			x := v_1_1
			v.reset(OpSub64)
			v0 := b.NewValue0(v.Pos, OpConst64, t)
			v0.AuxInt = int64ToAuxInt(c - d)
			v.AddArg2(v0, x)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpSub64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Sub64F (Const64F [c]) (Const64F [d]))
	// cond: c-d == c-d
	// result: (Const64F [c-d])
	for {
		if v_0.Op != OpConst64F {
			break
		}
		c := auxIntToFloat64(v_0.AuxInt)
		if v_1.Op != OpConst64F {
			break
		}
		d := auxIntToFloat64(v_1.AuxInt)
		if !(c-d == c-d) {
			break
		}
		v.reset(OpConst64F)
		v.AuxInt = float64ToAuxInt(c - d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSub8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Sub8 (Const8 [c]) (Const8 [d]))
	// result: (Const8 [c-d])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if v_1.Op != OpConst8 {
			break
		}
		d := auxIntToInt8(v_1.AuxInt)
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(c - d)
		return true
	}
	// match: (Sub8 x (Const8 <t> [c]))
	// cond: x.Op != OpConst8
	// result: (Add8 (Const8 <t> [-c]) x)
	for {
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		t := v_1.Type
		c := auxIntToInt8(v_1.AuxInt)
		if !(x.Op != OpConst8) {
			break
		}
		v.reset(OpAdd8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(-c)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Sub8 <t> (Mul8 x y) (Mul8 x z))
	// result: (Mul8 x (Sub8 <t> y z))
	for {
		t := v.Type
		if v_0.Op != OpMul8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if v_1.Op != OpMul8 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				z := v_1_1
				v.reset(OpMul8)
				v0 := b.NewValue0(v.Pos, OpSub8, t)
				v0.AddArg2(y, z)
				v.AddArg2(x, v0)
				return true
			}
		}
		break
	}
	// match: (Sub8 x x)
	// result: (Const8 [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	// match: (Sub8 (Neg8 x) (Com8 x))
	// result: (Const8 [1])
	for {
		if v_0.Op != OpNeg8 {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpCom8 || x != v_1.Args[0] {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(1)
		return true
	}
	// match: (Sub8 (Com8 x) (Neg8 x))
	// result: (Const8 [-1])
	for {
		if v_0.Op != OpCom8 {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpNeg8 || x != v_1.Args[0] {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(-1)
		return true
	}
	// match: (Sub8 (Add8 t x) (Add8 t y))
	// result: (Sub8 x y)
	for {
		if v_0.Op != OpAdd8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			t := v_0_0
			x := v_0_1
			if v_1.Op != OpAdd8 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if t != v_1_0 {
					continue
				}
				y := v_1_1
				v.reset(OpSub8)
				v.AddArg2(x, y)
				return true
			}
		}
		break
	}
	// match: (Sub8 (Add8 x y) x)
	// result: y
	for {
		if v_0.Op != OpAdd8 {
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
			v.copyOf(y)
			return true
		}
		break
	}
	// match: (Sub8 (Add8 x y) y)
	// result: x
	for {
		if v_0.Op != OpAdd8 {
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
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Sub8 (Sub8 x y) x)
	// result: (Neg8 y)
	for {
		if v_0.Op != OpSub8 {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpNeg8)
		v.AddArg(y)
		return true
	}
	// match: (Sub8 x (Add8 x y))
	// result: (Neg8 y)
	for {
		x := v_0
		if v_1.Op != OpAdd8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if x != v_1_0 {
				continue
			}
			y := v_1_1
			v.reset(OpNeg8)
			v.AddArg(y)
			return true
		}
		break
	}
	// match: (Sub8 x (Sub8 i:(Const8 <t>) z))
	// cond: (z.Op != OpConst8 && x.Op != OpConst8)
	// result: (Sub8 (Add8 <t> x z) i)
	for {
		x := v_0
		if v_1.Op != OpSub8 {
			break
		}
		z := v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpSub8)
		v0 := b.NewValue0(v.Pos, OpAdd8, t)
		v0.AddArg2(x, z)
		v.AddArg2(v0, i)
		return true
	}
	// match: (Sub8 x (Add8 z i:(Const8 <t>)))
	// cond: (z.Op != OpConst8 && x.Op != OpConst8)
	// result: (Sub8 (Sub8 <t> x z) i)
	for {
		x := v_0
		if v_1.Op != OpAdd8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			z := v_1_0
			i := v_1_1
			if i.Op != OpConst8 {
				continue
			}
			t := i.Type
			if !(z.Op != OpConst8 && x.Op != OpConst8) {
				continue
			}
			v.reset(OpSub8)
			v0 := b.NewValue0(v.Pos, OpSub8, t)
			v0.AddArg2(x, z)
			v.AddArg2(v0, i)
			return true
		}
		break
	}
	// match: (Sub8 (Sub8 i:(Const8 <t>) z) x)
	// cond: (z.Op != OpConst8 && x.Op != OpConst8)
	// result: (Sub8 i (Add8 <t> z x))
	for {
		if v_0.Op != OpSub8 {
			break
		}
		z := v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		x := v_1
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpSub8)
		v0 := b.NewValue0(v.Pos, OpAdd8, t)
		v0.AddArg2(z, x)
		v.AddArg2(i, v0)
		return true
	}
	// match: (Sub8 (Add8 z i:(Const8 <t>)) x)
	// cond: (z.Op != OpConst8 && x.Op != OpConst8)
	// result: (Add8 i (Sub8 <t> z x))
	for {
		if v_0.Op != OpAdd8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			z := v_0_0
			i := v_0_1
			if i.Op != OpConst8 {
				continue
			}
			t := i.Type
			x := v_1
			if !(z.Op != OpConst8 && x.Op != OpConst8) {
				continue
			}
			v.reset(OpAdd8)
			v0 := b.NewValue0(v.Pos, OpSub8, t)
			v0.AddArg2(z, x)
			v.AddArg2(i, v0)
			return true
		}
		break
	}
	// match: (Sub8 (Const8 <t> [c]) (Sub8 (Const8 <t> [d]) x))
	// result: (Add8 (Const8 <t> [c-d]) x)
	for {
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := auxIntToInt8(v_0.AuxInt)
		if v_1.Op != OpSub8 {
			break
		}
		x := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst8 || v_1_0.Type != t {
			break
		}
		d := auxIntToInt8(v_1_0.AuxInt)
		v.reset(OpAdd8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int8ToAuxInt(c - d)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Sub8 (Const8 <t> [c]) (Add8 (Const8 <t> [d]) x))
	// result: (Sub8 (Const8 <t> [c-d]) x)
	for {
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := auxIntToInt8(v_0.AuxInt)
		if v_1.Op != OpAdd8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if v_1_0.Op != OpConst8 || v_1_0.Type != t {
				continue
			}
			d := auxIntToInt8(v_1_0.AuxInt)
			x := v_1_1
			v.reset(OpSub8)
			v0 := b.NewValue0(v.Pos, OpConst8, t)
			v0.AuxInt = int8ToAuxInt(c - d)
			v.AddArg2(v0, x)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpTrunc(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc (Const64F [c]))
	// result: (Const64F [math.Trunc(c)])
	for {
		if v_0.Op != OpConst64F {
			break
		}
		c := auxIntToFloat64(v_0.AuxInt)
		v.reset(OpConst64F)
		v.AuxInt = float64ToAuxInt(math.Trunc(c))
		return true
	}
	return false
}
func rewriteValuegeneric_OpTrunc16to8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc16to8 (Const16 [c]))
	// result: (Const8 [int8(c)])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(int8(c))
		return true
	}
	// match: (Trunc16to8 (ZeroExt8to16 x))
	// result: x
	for {
		if v_0.Op != OpZeroExt8to16 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Trunc16to8 (SignExt8to16 x))
	// result: x
	for {
		if v_0.Op != OpSignExt8to16 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Trunc16to8 (And16 (Const16 [y]) x))
	// cond: y&0xFF == 0xFF
	// result: (Trunc16to8 x)
	for {
		if v_0.Op != OpAnd16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConst16 {
				continue
			}
			y := auxIntToInt16(v_0_0.AuxInt)
			x := v_0_1
			if !(y&0xFF == 0xFF) {
				continue
			}
			v.reset(OpTrunc16to8)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpTrunc32to16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc32to16 (Const32 [c]))
	// result: (Const16 [int16(c)])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(int16(c))
		return true
	}
	// match: (Trunc32to16 (ZeroExt8to32 x))
	// result: (ZeroExt8to16 x)
	for {
		if v_0.Op != OpZeroExt8to32 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpZeroExt8to16)
		v.AddArg(x)
		return true
	}
	// match: (Trunc32to16 (ZeroExt16to32 x))
	// result: x
	for {
		if v_0.Op != OpZeroExt16to32 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Trunc32to16 (SignExt8to32 x))
	// result: (SignExt8to16 x)
	for {
		if v_0.Op != OpSignExt8to32 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpSignExt8to16)
		v.AddArg(x)
		return true
	}
	// match: (Trunc32to16 (SignExt16to32 x))
	// result: x
	for {
		if v_0.Op != OpSignExt16to32 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Trunc32to16 (And32 (Const32 [y]) x))
	// cond: y&0xFFFF == 0xFFFF
	// result: (Trunc32to16 x)
	for {
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConst32 {
				continue
			}
			y := auxIntToInt32(v_0_0.AuxInt)
			x := v_0_1
			if !(y&0xFFFF == 0xFFFF) {
				continue
			}
			v.reset(OpTrunc32to16)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpTrunc32to8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc32to8 (Const32 [c]))
	// result: (Const8 [int8(c)])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(int8(c))
		return true
	}
	// match: (Trunc32to8 (ZeroExt8to32 x))
	// result: x
	for {
		if v_0.Op != OpZeroExt8to32 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Trunc32to8 (SignExt8to32 x))
	// result: x
	for {
		if v_0.Op != OpSignExt8to32 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Trunc32to8 (And32 (Const32 [y]) x))
	// cond: y&0xFF == 0xFF
	// result: (Trunc32to8 x)
	for {
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConst32 {
				continue
			}
			y := auxIntToInt32(v_0_0.AuxInt)
			x := v_0_1
			if !(y&0xFF == 0xFF) {
				continue
			}
			v.reset(OpTrunc32to8)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpTrunc64to16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc64to16 (Const64 [c]))
	// result: (Const16 [int16(c)])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(int16(c))
		return true
	}
	// match: (Trunc64to16 (ZeroExt8to64 x))
	// result: (ZeroExt8to16 x)
	for {
		if v_0.Op != OpZeroExt8to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpZeroExt8to16)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to16 (ZeroExt16to64 x))
	// result: x
	for {
		if v_0.Op != OpZeroExt16to64 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Trunc64to16 (SignExt8to64 x))
	// result: (SignExt8to16 x)
	for {
		if v_0.Op != OpSignExt8to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpSignExt8to16)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to16 (SignExt16to64 x))
	// result: x
	for {
		if v_0.Op != OpSignExt16to64 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Trunc64to16 (And64 (Const64 [y]) x))
	// cond: y&0xFFFF == 0xFFFF
	// result: (Trunc64to16 x)
	for {
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConst64 {
				continue
			}
			y := auxIntToInt64(v_0_0.AuxInt)
			x := v_0_1
			if !(y&0xFFFF == 0xFFFF) {
				continue
			}
			v.reset(OpTrunc64to16)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpTrunc64to32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc64to32 (Const64 [c]))
	// result: (Const32 [int32(c)])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(c))
		return true
	}
	// match: (Trunc64to32 (ZeroExt8to64 x))
	// result: (ZeroExt8to32 x)
	for {
		if v_0.Op != OpZeroExt8to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpZeroExt8to32)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to32 (ZeroExt16to64 x))
	// result: (ZeroExt16to32 x)
	for {
		if v_0.Op != OpZeroExt16to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpZeroExt16to32)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to32 (ZeroExt32to64 x))
	// result: x
	for {
		if v_0.Op != OpZeroExt32to64 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Trunc64to32 (SignExt8to64 x))
	// result: (SignExt8to32 x)
	for {
		if v_0.Op != OpSignExt8to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpSignExt8to32)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to32 (SignExt16to64 x))
	// result: (SignExt16to32 x)
	for {
		if v_0.Op != OpSignExt16to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpSignExt16to32)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to32 (SignExt32to64 x))
	// result: x
	for {
		if v_0.Op != OpSignExt32to64 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Trunc64to32 (And64 (Const64 [y]) x))
	// cond: y&0xFFFFFFFF == 0xFFFFFFFF
	// result: (Trunc64to32 x)
	for {
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConst64 {
				continue
			}
			y := auxIntToInt64(v_0_0.AuxInt)
			x := v_0_1
			if !(y&0xFFFFFFFF == 0xFFFFFFFF) {
				continue
			}
			v.reset(OpTrunc64to32)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpTrunc64to8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc64to8 (Const64 [c]))
	// result: (Const8 [int8(c)])
	for {
		if v_0.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(int8(c))
		return true
	}
	// match: (Trunc64to8 (ZeroExt8to64 x))
	// result: x
	for {
		if v_0.Op != OpZeroExt8to64 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Trunc64to8 (SignExt8to64 x))
	// result: x
	for {
		if v_0.Op != OpSignExt8to64 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Trunc64to8 (And64 (Const64 [y]) x))
	// cond: y&0xFF == 0xFF
	// result: (Trunc64to8 x)
	for {
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpConst64 {
				continue
			}
			y := auxIntToInt64(v_0_0.AuxInt)
			x := v_0_1
			if !(y&0xFF == 0xFF) {
				continue
			}
			v.reset(OpTrunc64to8)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpXor16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (Xor16 (Const16 [c]) (Const16 [d]))
	// result: (Const16 [c^d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			c := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpConst16 {
				continue
			}
			d := auxIntToInt16(v_1.AuxInt)
			v.reset(OpConst16)
			v.AuxInt = int16ToAuxInt(c ^ d)
			return true
		}
		break
	}
	// match: (Xor16 x x)
	// result: (Const16 [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	// match: (Xor16 (Const16 [0]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != 0 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Xor16 (Com16 x) x)
	// result: (Const16 [-1])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom16 {
				continue
			}
			x := v_0.Args[0]
			if x != v_1 {
				continue
			}
			v.reset(OpConst16)
			v.AuxInt = int16ToAuxInt(-1)
			return true
		}
		break
	}
	// match: (Xor16 (Const16 [-1]) x)
	// result: (Com16 x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 || auxIntToInt16(v_0.AuxInt) != -1 {
				continue
			}
			x := v_1
			v.reset(OpCom16)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Xor16 x (Xor16 x y))
	// result: y
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpXor16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				y := v_1_1
				v.copyOf(y)
				return true
			}
		}
		break
	}
	// match: (Xor16 (Xor16 i:(Const16 <t>) z) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Xor16 i (Xor16 <t> z x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpXor16 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst16 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst16 && x.Op != OpConst16) {
					continue
				}
				v.reset(OpXor16)
				v0 := b.NewValue0(v.Pos, OpXor16, t)
				v0.AddArg2(z, x)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (Xor16 (Const16 <t> [c]) (Xor16 (Const16 <t> [d]) x))
	// result: (Xor16 (Const16 <t> [c^d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst16 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt16(v_0.AuxInt)
			if v_1.Op != OpXor16 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst16 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt16(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpXor16)
				v0 := b.NewValue0(v.Pos, OpConst16, t)
				v0.AuxInt = int16ToAuxInt(c ^ d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Xor16 (Lsh16x64 x z:(Const64 <t> [c])) (Rsh16Ux64 x (Const64 [d])))
	// cond: c < 16 && d == 16-c && canRotate(config, 16)
	// result: (RotateLeft16 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLsh16x64 {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(z.AuxInt)
			if v_1.Op != OpRsh16Ux64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(c < 16 && d == 16-c && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Xor16 left:(Lsh16x64 x y) right:(Rsh16Ux64 x (Sub64 (Const64 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh16x64 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh16Ux64 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub64 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst64 || auxIntToInt64(right_1_0.AuxInt) != 16 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Xor16 left:(Lsh16x32 x y) right:(Rsh16Ux32 x (Sub32 (Const32 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh16x32 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh16Ux32 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub32 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst32 || auxIntToInt32(right_1_0.AuxInt) != 16 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Xor16 left:(Lsh16x16 x y) right:(Rsh16Ux16 x (Sub16 (Const16 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh16x16 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh16Ux16 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub16 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst16 || auxIntToInt16(right_1_0.AuxInt) != 16 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Xor16 left:(Lsh16x8 x y) right:(Rsh16Ux8 x (Sub8 (Const8 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh16x8 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh16Ux8 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub8 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst8 || auxIntToInt8(right_1_0.AuxInt) != 16 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Xor16 right:(Rsh16Ux64 x y) left:(Lsh16x64 x z:(Sub64 (Const64 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh16Ux64 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh16x64 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub64 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst64 || auxIntToInt64(z_0.AuxInt) != 16 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Xor16 right:(Rsh16Ux32 x y) left:(Lsh16x32 x z:(Sub32 (Const32 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh16Ux32 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh16x32 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub32 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst32 || auxIntToInt32(z_0.AuxInt) != 16 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Xor16 right:(Rsh16Ux16 x y) left:(Lsh16x16 x z:(Sub16 (Const16 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh16Ux16 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh16x16 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub16 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst16 || auxIntToInt16(z_0.AuxInt) != 16 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Xor16 right:(Rsh16Ux8 x y) left:(Lsh16x8 x z:(Sub8 (Const8 [16]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)
	// result: (RotateLeft16 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh16Ux8 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh16x8 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub8 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst8 || auxIntToInt8(z_0.AuxInt) != 16 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 16)) {
				continue
			}
			v.reset(OpRotateLeft16)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpXor32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (Xor32 (Const32 [c]) (Const32 [d]))
	// result: (Const32 [c^d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpConst32 {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			v.reset(OpConst32)
			v.AuxInt = int32ToAuxInt(c ^ d)
			return true
		}
		break
	}
	// match: (Xor32 x x)
	// result: (Const32 [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (Xor32 (Const32 [0]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != 0 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Xor32 (Com32 x) x)
	// result: (Const32 [-1])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom32 {
				continue
			}
			x := v_0.Args[0]
			if x != v_1 {
				continue
			}
			v.reset(OpConst32)
			v.AuxInt = int32ToAuxInt(-1)
			return true
		}
		break
	}
	// match: (Xor32 (Const32 [-1]) x)
	// result: (Com32 x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 || auxIntToInt32(v_0.AuxInt) != -1 {
				continue
			}
			x := v_1
			v.reset(OpCom32)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Xor32 x (Xor32 x y))
	// result: y
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpXor32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				y := v_1_1
				v.copyOf(y)
				return true
			}
		}
		break
	}
	// match: (Xor32 (Xor32 i:(Const32 <t>) z) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Xor32 i (Xor32 <t> z x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpXor32 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst32 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst32 && x.Op != OpConst32) {
					continue
				}
				v.reset(OpXor32)
				v0 := b.NewValue0(v.Pos, OpXor32, t)
				v0.AddArg2(z, x)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (Xor32 (Const32 <t> [c]) (Xor32 (Const32 <t> [d]) x))
	// result: (Xor32 (Const32 <t> [c^d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst32 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt32(v_0.AuxInt)
			if v_1.Op != OpXor32 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst32 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt32(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpXor32)
				v0 := b.NewValue0(v.Pos, OpConst32, t)
				v0.AuxInt = int32ToAuxInt(c ^ d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Xor32 (Lsh32x64 x z:(Const64 <t> [c])) (Rsh32Ux64 x (Const64 [d])))
	// cond: c < 32 && d == 32-c && canRotate(config, 32)
	// result: (RotateLeft32 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLsh32x64 {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(z.AuxInt)
			if v_1.Op != OpRsh32Ux64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(c < 32 && d == 32-c && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Xor32 left:(Lsh32x64 x y) right:(Rsh32Ux64 x (Sub64 (Const64 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh32x64 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh32Ux64 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub64 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst64 || auxIntToInt64(right_1_0.AuxInt) != 32 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Xor32 left:(Lsh32x32 x y) right:(Rsh32Ux32 x (Sub32 (Const32 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh32x32 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh32Ux32 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub32 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst32 || auxIntToInt32(right_1_0.AuxInt) != 32 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Xor32 left:(Lsh32x16 x y) right:(Rsh32Ux16 x (Sub16 (Const16 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh32x16 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh32Ux16 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub16 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst16 || auxIntToInt16(right_1_0.AuxInt) != 32 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Xor32 left:(Lsh32x8 x y) right:(Rsh32Ux8 x (Sub8 (Const8 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh32x8 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh32Ux8 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub8 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst8 || auxIntToInt8(right_1_0.AuxInt) != 32 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Xor32 right:(Rsh32Ux64 x y) left:(Lsh32x64 x z:(Sub64 (Const64 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh32Ux64 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh32x64 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub64 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst64 || auxIntToInt64(z_0.AuxInt) != 32 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Xor32 right:(Rsh32Ux32 x y) left:(Lsh32x32 x z:(Sub32 (Const32 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh32Ux32 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh32x32 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub32 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst32 || auxIntToInt32(z_0.AuxInt) != 32 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Xor32 right:(Rsh32Ux16 x y) left:(Lsh32x16 x z:(Sub16 (Const16 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh32Ux16 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh32x16 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub16 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst16 || auxIntToInt16(z_0.AuxInt) != 32 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Xor32 right:(Rsh32Ux8 x y) left:(Lsh32x8 x z:(Sub8 (Const8 [32]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)
	// result: (RotateLeft32 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh32Ux8 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh32x8 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub8 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst8 || auxIntToInt8(z_0.AuxInt) != 32 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 32)) {
				continue
			}
			v.reset(OpRotateLeft32)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpXor64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (Xor64 (Const64 [c]) (Const64 [d]))
	// result: (Const64 [c^d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpConst64)
			v.AuxInt = int64ToAuxInt(c ^ d)
			return true
		}
		break
	}
	// match: (Xor64 x x)
	// result: (Const64 [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (Xor64 (Const64 [0]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != 0 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Xor64 (Com64 x) x)
	// result: (Const64 [-1])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom64 {
				continue
			}
			x := v_0.Args[0]
			if x != v_1 {
				continue
			}
			v.reset(OpConst64)
			v.AuxInt = int64ToAuxInt(-1)
			return true
		}
		break
	}
	// match: (Xor64 (Const64 [-1]) x)
	// result: (Com64 x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 || auxIntToInt64(v_0.AuxInt) != -1 {
				continue
			}
			x := v_1
			v.reset(OpCom64)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Xor64 x (Xor64 x y))
	// result: y
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpXor64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				y := v_1_1
				v.copyOf(y)
				return true
			}
		}
		break
	}
	// match: (Xor64 (Xor64 i:(Const64 <t>) z) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Xor64 i (Xor64 <t> z x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpXor64 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst64 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst64 && x.Op != OpConst64) {
					continue
				}
				v.reset(OpXor64)
				v0 := b.NewValue0(v.Pos, OpXor64, t)
				v0.AddArg2(z, x)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (Xor64 (Const64 <t> [c]) (Xor64 (Const64 <t> [d]) x))
	// result: (Xor64 (Const64 <t> [c^d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst64 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpXor64 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst64 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt64(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpXor64)
				v0 := b.NewValue0(v.Pos, OpConst64, t)
				v0.AuxInt = int64ToAuxInt(c ^ d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Xor64 (Lsh64x64 x z:(Const64 <t> [c])) (Rsh64Ux64 x (Const64 [d])))
	// cond: c < 64 && d == 64-c && canRotate(config, 64)
	// result: (RotateLeft64 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLsh64x64 {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(z.AuxInt)
			if v_1.Op != OpRsh64Ux64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(c < 64 && d == 64-c && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Xor64 left:(Lsh64x64 x y) right:(Rsh64Ux64 x (Sub64 (Const64 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh64x64 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh64Ux64 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub64 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst64 || auxIntToInt64(right_1_0.AuxInt) != 64 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Xor64 left:(Lsh64x32 x y) right:(Rsh64Ux32 x (Sub32 (Const32 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh64x32 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh64Ux32 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub32 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst32 || auxIntToInt32(right_1_0.AuxInt) != 64 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Xor64 left:(Lsh64x16 x y) right:(Rsh64Ux16 x (Sub16 (Const16 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh64x16 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh64Ux16 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub16 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst16 || auxIntToInt16(right_1_0.AuxInt) != 64 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Xor64 left:(Lsh64x8 x y) right:(Rsh64Ux8 x (Sub8 (Const8 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh64x8 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh64Ux8 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub8 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst8 || auxIntToInt8(right_1_0.AuxInt) != 64 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Xor64 right:(Rsh64Ux64 x y) left:(Lsh64x64 x z:(Sub64 (Const64 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh64Ux64 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh64x64 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub64 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst64 || auxIntToInt64(z_0.AuxInt) != 64 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Xor64 right:(Rsh64Ux32 x y) left:(Lsh64x32 x z:(Sub32 (Const32 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh64Ux32 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh64x32 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub32 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst32 || auxIntToInt32(z_0.AuxInt) != 64 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Xor64 right:(Rsh64Ux16 x y) left:(Lsh64x16 x z:(Sub16 (Const16 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh64Ux16 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh64x16 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub16 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst16 || auxIntToInt16(z_0.AuxInt) != 64 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Xor64 right:(Rsh64Ux8 x y) left:(Lsh64x8 x z:(Sub8 (Const8 [64]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)
	// result: (RotateLeft64 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh64Ux8 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh64x8 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub8 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst8 || auxIntToInt8(z_0.AuxInt) != 64 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 64)) {
				continue
			}
			v.reset(OpRotateLeft64)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpXor8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (Xor8 (Const8 [c]) (Const8 [d]))
	// result: (Const8 [c^d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			c := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpConst8 {
				continue
			}
			d := auxIntToInt8(v_1.AuxInt)
			v.reset(OpConst8)
			v.AuxInt = int8ToAuxInt(c ^ d)
			return true
		}
		break
	}
	// match: (Xor8 x x)
	// result: (Const8 [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	// match: (Xor8 (Const8 [0]) x)
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != 0 {
				continue
			}
			x := v_1
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (Xor8 (Com8 x) x)
	// result: (Const8 [-1])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpCom8 {
				continue
			}
			x := v_0.Args[0]
			if x != v_1 {
				continue
			}
			v.reset(OpConst8)
			v.AuxInt = int8ToAuxInt(-1)
			return true
		}
		break
	}
	// match: (Xor8 (Const8 [-1]) x)
	// result: (Com8 x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 || auxIntToInt8(v_0.AuxInt) != -1 {
				continue
			}
			x := v_1
			v.reset(OpCom8)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Xor8 x (Xor8 x y))
	// result: y
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpXor8 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				y := v_1_1
				v.copyOf(y)
				return true
			}
		}
		break
	}
	// match: (Xor8 (Xor8 i:(Const8 <t>) z) x)
	// cond: (z.Op != OpConst8 && x.Op != OpConst8)
	// result: (Xor8 i (Xor8 <t> z x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpXor8 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				i := v_0_0
				if i.Op != OpConst8 {
					continue
				}
				t := i.Type
				z := v_0_1
				x := v_1
				if !(z.Op != OpConst8 && x.Op != OpConst8) {
					continue
				}
				v.reset(OpXor8)
				v0 := b.NewValue0(v.Pos, OpXor8, t)
				v0.AddArg2(z, x)
				v.AddArg2(i, v0)
				return true
			}
		}
		break
	}
	// match: (Xor8 (Const8 <t> [c]) (Xor8 (Const8 <t> [d]) x))
	// result: (Xor8 (Const8 <t> [c^d]) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpConst8 {
				continue
			}
			t := v_0.Type
			c := auxIntToInt8(v_0.AuxInt)
			if v_1.Op != OpXor8 {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpConst8 || v_1_0.Type != t {
					continue
				}
				d := auxIntToInt8(v_1_0.AuxInt)
				x := v_1_1
				v.reset(OpXor8)
				v0 := b.NewValue0(v.Pos, OpConst8, t)
				v0.AuxInt = int8ToAuxInt(c ^ d)
				v.AddArg2(v0, x)
				return true
			}
		}
		break
	}
	// match: (Xor8 (Lsh8x64 x z:(Const64 <t> [c])) (Rsh8Ux64 x (Const64 [d])))
	// cond: c < 8 && d == 8-c && canRotate(config, 8)
	// result: (RotateLeft8 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLsh8x64 {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpConst64 {
				continue
			}
			c := auxIntToInt64(z.AuxInt)
			if v_1.Op != OpRsh8Ux64 {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpConst64 {
				continue
			}
			d := auxIntToInt64(v_1_1.AuxInt)
			if !(c < 8 && d == 8-c && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Xor8 left:(Lsh8x64 x y) right:(Rsh8Ux64 x (Sub64 (Const64 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh8x64 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh8Ux64 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub64 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst64 || auxIntToInt64(right_1_0.AuxInt) != 8 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Xor8 left:(Lsh8x32 x y) right:(Rsh8Ux32 x (Sub32 (Const32 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh8x32 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh8Ux32 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub32 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst32 || auxIntToInt32(right_1_0.AuxInt) != 8 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Xor8 left:(Lsh8x16 x y) right:(Rsh8Ux16 x (Sub16 (Const16 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh8x16 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh8Ux16 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub16 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst16 || auxIntToInt16(right_1_0.AuxInt) != 8 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Xor8 left:(Lsh8x8 x y) right:(Rsh8Ux8 x (Sub8 (Const8 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			left := v_0
			if left.Op != OpLsh8x8 {
				continue
			}
			y := left.Args[1]
			x := left.Args[0]
			right := v_1
			if right.Op != OpRsh8Ux8 {
				continue
			}
			_ = right.Args[1]
			if x != right.Args[0] {
				continue
			}
			right_1 := right.Args[1]
			if right_1.Op != OpSub8 {
				continue
			}
			_ = right_1.Args[1]
			right_1_0 := right_1.Args[0]
			if right_1_0.Op != OpConst8 || auxIntToInt8(right_1_0.AuxInt) != 8 || y != right_1.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (Xor8 right:(Rsh8Ux64 x y) left:(Lsh8x64 x z:(Sub64 (Const64 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh8Ux64 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh8x64 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub64 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst64 || auxIntToInt64(z_0.AuxInt) != 8 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Xor8 right:(Rsh8Ux32 x y) left:(Lsh8x32 x z:(Sub32 (Const32 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh8Ux32 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh8x32 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub32 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst32 || auxIntToInt32(z_0.AuxInt) != 8 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Xor8 right:(Rsh8Ux16 x y) left:(Lsh8x16 x z:(Sub16 (Const16 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh8Ux16 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh8x16 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub16 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst16 || auxIntToInt16(z_0.AuxInt) != 8 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	// match: (Xor8 right:(Rsh8Ux8 x y) left:(Lsh8x8 x z:(Sub8 (Const8 [8]) y)))
	// cond: (shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)
	// result: (RotateLeft8 x z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			right := v_0
			if right.Op != OpRsh8Ux8 {
				continue
			}
			y := right.Args[1]
			x := right.Args[0]
			left := v_1
			if left.Op != OpLsh8x8 {
				continue
			}
			_ = left.Args[1]
			if x != left.Args[0] {
				continue
			}
			z := left.Args[1]
			if z.Op != OpSub8 {
				continue
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			if z_0.Op != OpConst8 || auxIntToInt8(z_0.AuxInt) != 8 || y != z.Args[1] || !((shiftIsBounded(left) || shiftIsBounded(right)) && canRotate(config, 8)) {
				continue
			}
			v.reset(OpRotateLeft8)
			v.AddArg2(x, z)
			return true
		}
		break
	}
	return false
}
func rewriteValuegeneric_OpZero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Zero (SelectN [0] call:(StaticLECall _ _)) mem:(SelectN [1] call))
	// cond: isSameCall(call.Aux, "runtime.newobject")
	// result: mem
	for {
		if v_0.Op != OpSelectN || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		call := v_0.Args[0]
		if call.Op != OpStaticLECall || len(call.Args) != 2 {
			break
		}
		mem := v_1
		if mem.Op != OpSelectN || auxIntToInt64(mem.AuxInt) != 1 || call != mem.Args[0] || !(isSameCall(call.Aux, "runtime.newobject")) {
			break
		}
		v.copyOf(mem)
		return true
	}
	// match: (Zero {t1} [n] p1 store:(Store {t2} (OffPtr [o2] p2) _ mem))
	// cond: isSamePtr(p1, p2) && store.Uses == 1 && n >= o2 + t2.Size() && clobber(store)
	// result: (Zero {t1} [n] p1 mem)
	for {
		n := auxIntToInt64(v.AuxInt)
		t1 := auxToType(v.Aux)
		p1 := v_0
		store := v_1
		if store.Op != OpStore {
			break
		}
		t2 := auxToType(store.Aux)
		mem := store.Args[2]
		store_0 := store.Args[0]
		if store_0.Op != OpOffPtr {
			break
		}
		o2 := auxIntToInt64(store_0.AuxInt)
		p2 := store_0.Args[0]
		if !(isSamePtr(p1, p2) && store.Uses == 1 && n >= o2+t2.Size() && clobber(store)) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = int64ToAuxInt(n)
		v.Aux = typeToAux(t1)
		v.AddArg2(p1, mem)
		return true
	}
	// match: (Zero {t} [n] dst1 move:(Move {t} [n] dst2 _ mem))
	// cond: move.Uses == 1 && isSamePtr(dst1, dst2) && clobber(move)
	// result: (Zero {t} [n] dst1 mem)
	for {
		n := auxIntToInt64(v.AuxInt)
		t := auxToType(v.Aux)
		dst1 := v_0
		move := v_1
		if move.Op != OpMove || auxIntToInt64(move.AuxInt) != n || auxToType(move.Aux) != t {
			break
		}
		mem := move.Args[2]
		dst2 := move.Args[0]
		if !(move.Uses == 1 && isSamePtr(dst1, dst2) && clobber(move)) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = int64ToAuxInt(n)
		v.Aux = typeToAux(t)
		v.AddArg2(dst1, mem)
		return true
	}
	// match: (Zero {t} [n] dst1 vardef:(VarDef {x} move:(Move {t} [n] dst2 _ mem)))
	// cond: move.Uses == 1 && vardef.Uses == 1 && isSamePtr(dst1, dst2) && clobber(move, vardef)
	// result: (Zero {t} [n] dst1 (VarDef {x} mem))
	for {
		n := auxIntToInt64(v.AuxInt)
		t := auxToType(v.Aux)
		dst1 := v_0
		vardef := v_1
		if vardef.Op != OpVarDef {
			break
		}
		x := auxToSym(vardef.Aux)
		move := vardef.Args[0]
		if move.Op != OpMove || auxIntToInt64(move.AuxInt) != n || auxToType(move.Aux) != t {
			break
		}
		mem := move.Args[2]
		dst2 := move.Args[0]
		if !(move.Uses == 1 && vardef.Uses == 1 && isSamePtr(dst1, dst2) && clobber(move, vardef)) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = int64ToAuxInt(n)
		v.Aux = typeToAux(t)
		v0 := b.NewValue0(v.Pos, OpVarDef, types.TypeMem)
		v0.Aux = symToAux(x)
		v0.AddArg(mem)
		v.AddArg2(dst1, v0)
		return true
	}
	// match: (Zero {t} [s] dst1 zero:(Zero {t} [s] dst2 _))
	// cond: isSamePtr(dst1, dst2)
	// result: zero
	for {
		s := auxIntToInt64(v.AuxInt)
		t := auxToType(v.Aux)
		dst1 := v_0
		zero := v_1
		if zero.Op != OpZero || auxIntToInt64(zero.AuxInt) != s || auxToType(zero.Aux) != t {
			break
		}
		dst2 := zero.Args[0]
		if !(isSamePtr(dst1, dst2)) {
			break
		}
		v.copyOf(zero)
		return true
	}
	// match: (Zero {t} [s] dst1 vardef:(VarDef (Zero {t} [s] dst2 _)))
	// cond: isSamePtr(dst1, dst2)
	// result: vardef
	for {
		s := auxIntToInt64(v.AuxInt)
		t := auxToType(v.Aux)
		dst1 := v_0
		vardef := v_1
		if vardef.Op != OpVarDef {
			break
		}
		vardef_0 := vardef.Args[0]
		if vardef_0.Op != OpZero || auxIntToInt64(vardef_0.AuxInt) != s || auxToType(vardef_0.Aux) != t {
			break
		}
		dst2 := vardef_0.Args[0]
		if !(isSamePtr(dst1, dst2)) {
			break
		}
		v.copyOf(vardef)
		return true
	}
	return false
}
func rewriteValuegeneric_OpZeroExt16to32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ZeroExt16to32 (Const16 [c]))
	// result: (Const32 [int32(uint16(c))])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(uint16(c)))
		return true
	}
	// match: (ZeroExt16to32 (Trunc32to16 x:(Rsh32Ux64 _ (Const64 [s]))))
	// cond: s >= 16
	// result: x
	for {
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
		s := auxIntToInt64(x_1.AuxInt)
		if !(s >= 16) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpZeroExt16to64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ZeroExt16to64 (Const16 [c]))
	// result: (Const64 [int64(uint16(c))])
	for {
		if v_0.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_0.AuxInt)
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(uint16(c)))
		return true
	}
	// match: (ZeroExt16to64 (Trunc64to16 x:(Rsh64Ux64 _ (Const64 [s]))))
	// cond: s >= 48
	// result: x
	for {
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
		s := auxIntToInt64(x_1.AuxInt)
		if !(s >= 48) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpZeroExt32to64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ZeroExt32to64 (Const32 [c]))
	// result: (Const64 [int64(uint32(c))])
	for {
		if v_0.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(uint32(c)))
		return true
	}
	// match: (ZeroExt32to64 (Trunc64to32 x:(Rsh64Ux64 _ (Const64 [s]))))
	// cond: s >= 32
	// result: x
	for {
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
		s := auxIntToInt64(x_1.AuxInt)
		if !(s >= 32) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpZeroExt8to16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ZeroExt8to16 (Const8 [c]))
	// result: (Const16 [int16( uint8(c))])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(int16(uint8(c)))
		return true
	}
	// match: (ZeroExt8to16 (Trunc16to8 x:(Rsh16Ux64 _ (Const64 [s]))))
	// cond: s >= 8
	// result: x
	for {
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
		s := auxIntToInt64(x_1.AuxInt)
		if !(s >= 8) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpZeroExt8to32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ZeroExt8to32 (Const8 [c]))
	// result: (Const32 [int32( uint8(c))])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(int32(uint8(c)))
		return true
	}
	// match: (ZeroExt8to32 (Trunc32to8 x:(Rsh32Ux64 _ (Const64 [s]))))
	// cond: s >= 24
	// result: x
	for {
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
		s := auxIntToInt64(x_1.AuxInt)
		if !(s >= 24) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpZeroExt8to64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ZeroExt8to64 (Const8 [c]))
	// result: (Const64 [int64( uint8(c))])
	for {
		if v_0.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		v.reset(OpConst64)
		v.AuxInt = int64ToAuxInt(int64(uint8(c)))
		return true
	}
	// match: (ZeroExt8to64 (Trunc64to8 x:(Rsh64Ux64 _ (Const64 [s]))))
	// cond: s >= 56
	// result: x
	for {
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
		s := auxIntToInt64(x_1.AuxInt)
		if !(s >= 56) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteBlockgeneric(b *Block) bool {
	switch b.Kind {
	case BlockIf:
		// match: (If (Not cond) yes no)
		// result: (If cond no yes)
		for b.Controls[0].Op == OpNot {
			v_0 := b.Controls[0]
			cond := v_0.Args[0]
			b.resetWithControl(BlockIf, cond)
			b.swapSuccessors()
			return true
		}
		// match: (If (ConstBool [c]) yes no)
		// cond: c
		// result: (First yes no)
		for b.Controls[0].Op == OpConstBool {
			v_0 := b.Controls[0]
			c := auxIntToBool(v_0.AuxInt)
			if !(c) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (If (ConstBool [c]) yes no)
		// cond: !c
		// result: (First no yes)
		for b.Controls[0].Op == OpConstBool {
			v_0 := b.Controls[0]
			c := auxIntToBool(v_0.AuxInt)
			if !(!c) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
	}
	return false
}
