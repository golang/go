// Code generated from gen/RISCV64.rules; DO NOT EDIT.
// generated with: cd gen; go run *.go

package ssa

import "math"
import "cmd/compile/internal/types"

func rewriteValueRISCV64(v *Value) bool {
	switch v.Op {
	case OpAdd16:
		return rewriteValueRISCV64_OpAdd16(v)
	case OpAdd32:
		return rewriteValueRISCV64_OpAdd32(v)
	case OpAdd32F:
		return rewriteValueRISCV64_OpAdd32F(v)
	case OpAdd64:
		return rewriteValueRISCV64_OpAdd64(v)
	case OpAdd64F:
		return rewriteValueRISCV64_OpAdd64F(v)
	case OpAdd8:
		return rewriteValueRISCV64_OpAdd8(v)
	case OpAddPtr:
		return rewriteValueRISCV64_OpAddPtr(v)
	case OpAddr:
		return rewriteValueRISCV64_OpAddr(v)
	case OpAnd16:
		return rewriteValueRISCV64_OpAnd16(v)
	case OpAnd32:
		return rewriteValueRISCV64_OpAnd32(v)
	case OpAnd64:
		return rewriteValueRISCV64_OpAnd64(v)
	case OpAnd8:
		return rewriteValueRISCV64_OpAnd8(v)
	case OpAndB:
		return rewriteValueRISCV64_OpAndB(v)
	case OpAvg64u:
		return rewriteValueRISCV64_OpAvg64u(v)
	case OpClosureCall:
		return rewriteValueRISCV64_OpClosureCall(v)
	case OpCom16:
		return rewriteValueRISCV64_OpCom16(v)
	case OpCom32:
		return rewriteValueRISCV64_OpCom32(v)
	case OpCom64:
		return rewriteValueRISCV64_OpCom64(v)
	case OpCom8:
		return rewriteValueRISCV64_OpCom8(v)
	case OpConst16:
		return rewriteValueRISCV64_OpConst16(v)
	case OpConst32:
		return rewriteValueRISCV64_OpConst32(v)
	case OpConst32F:
		return rewriteValueRISCV64_OpConst32F(v)
	case OpConst64:
		return rewriteValueRISCV64_OpConst64(v)
	case OpConst64F:
		return rewriteValueRISCV64_OpConst64F(v)
	case OpConst8:
		return rewriteValueRISCV64_OpConst8(v)
	case OpConstBool:
		return rewriteValueRISCV64_OpConstBool(v)
	case OpConstNil:
		return rewriteValueRISCV64_OpConstNil(v)
	case OpConvert:
		return rewriteValueRISCV64_OpConvert(v)
	case OpCvt32Fto32:
		return rewriteValueRISCV64_OpCvt32Fto32(v)
	case OpCvt32Fto64:
		return rewriteValueRISCV64_OpCvt32Fto64(v)
	case OpCvt32Fto64F:
		return rewriteValueRISCV64_OpCvt32Fto64F(v)
	case OpCvt32to32F:
		return rewriteValueRISCV64_OpCvt32to32F(v)
	case OpCvt32to64F:
		return rewriteValueRISCV64_OpCvt32to64F(v)
	case OpCvt64Fto32:
		return rewriteValueRISCV64_OpCvt64Fto32(v)
	case OpCvt64Fto32F:
		return rewriteValueRISCV64_OpCvt64Fto32F(v)
	case OpCvt64Fto64:
		return rewriteValueRISCV64_OpCvt64Fto64(v)
	case OpCvt64to32F:
		return rewriteValueRISCV64_OpCvt64to32F(v)
	case OpCvt64to64F:
		return rewriteValueRISCV64_OpCvt64to64F(v)
	case OpDiv16:
		return rewriteValueRISCV64_OpDiv16(v)
	case OpDiv16u:
		return rewriteValueRISCV64_OpDiv16u(v)
	case OpDiv32:
		return rewriteValueRISCV64_OpDiv32(v)
	case OpDiv32F:
		return rewriteValueRISCV64_OpDiv32F(v)
	case OpDiv32u:
		return rewriteValueRISCV64_OpDiv32u(v)
	case OpDiv64:
		return rewriteValueRISCV64_OpDiv64(v)
	case OpDiv64F:
		return rewriteValueRISCV64_OpDiv64F(v)
	case OpDiv64u:
		return rewriteValueRISCV64_OpDiv64u(v)
	case OpDiv8:
		return rewriteValueRISCV64_OpDiv8(v)
	case OpDiv8u:
		return rewriteValueRISCV64_OpDiv8u(v)
	case OpEq16:
		return rewriteValueRISCV64_OpEq16(v)
	case OpEq32:
		return rewriteValueRISCV64_OpEq32(v)
	case OpEq32F:
		return rewriteValueRISCV64_OpEq32F(v)
	case OpEq64:
		return rewriteValueRISCV64_OpEq64(v)
	case OpEq64F:
		return rewriteValueRISCV64_OpEq64F(v)
	case OpEq8:
		return rewriteValueRISCV64_OpEq8(v)
	case OpEqB:
		return rewriteValueRISCV64_OpEqB(v)
	case OpEqPtr:
		return rewriteValueRISCV64_OpEqPtr(v)
	case OpGeq16:
		return rewriteValueRISCV64_OpGeq16(v)
	case OpGeq16U:
		return rewriteValueRISCV64_OpGeq16U(v)
	case OpGeq32:
		return rewriteValueRISCV64_OpGeq32(v)
	case OpGeq32F:
		return rewriteValueRISCV64_OpGeq32F(v)
	case OpGeq32U:
		return rewriteValueRISCV64_OpGeq32U(v)
	case OpGeq64:
		return rewriteValueRISCV64_OpGeq64(v)
	case OpGeq64F:
		return rewriteValueRISCV64_OpGeq64F(v)
	case OpGeq64U:
		return rewriteValueRISCV64_OpGeq64U(v)
	case OpGeq8:
		return rewriteValueRISCV64_OpGeq8(v)
	case OpGeq8U:
		return rewriteValueRISCV64_OpGeq8U(v)
	case OpGetCallerPC:
		return rewriteValueRISCV64_OpGetCallerPC(v)
	case OpGetCallerSP:
		return rewriteValueRISCV64_OpGetCallerSP(v)
	case OpGetClosurePtr:
		return rewriteValueRISCV64_OpGetClosurePtr(v)
	case OpGreater16:
		return rewriteValueRISCV64_OpGreater16(v)
	case OpGreater16U:
		return rewriteValueRISCV64_OpGreater16U(v)
	case OpGreater32:
		return rewriteValueRISCV64_OpGreater32(v)
	case OpGreater32F:
		return rewriteValueRISCV64_OpGreater32F(v)
	case OpGreater32U:
		return rewriteValueRISCV64_OpGreater32U(v)
	case OpGreater64:
		return rewriteValueRISCV64_OpGreater64(v)
	case OpGreater64F:
		return rewriteValueRISCV64_OpGreater64F(v)
	case OpGreater64U:
		return rewriteValueRISCV64_OpGreater64U(v)
	case OpGreater8:
		return rewriteValueRISCV64_OpGreater8(v)
	case OpGreater8U:
		return rewriteValueRISCV64_OpGreater8U(v)
	case OpHmul32:
		return rewriteValueRISCV64_OpHmul32(v)
	case OpHmul32u:
		return rewriteValueRISCV64_OpHmul32u(v)
	case OpHmul64:
		return rewriteValueRISCV64_OpHmul64(v)
	case OpHmul64u:
		return rewriteValueRISCV64_OpHmul64u(v)
	case OpInterCall:
		return rewriteValueRISCV64_OpInterCall(v)
	case OpIsInBounds:
		return rewriteValueRISCV64_OpIsInBounds(v)
	case OpIsNonNil:
		return rewriteValueRISCV64_OpIsNonNil(v)
	case OpIsSliceInBounds:
		return rewriteValueRISCV64_OpIsSliceInBounds(v)
	case OpLeq16:
		return rewriteValueRISCV64_OpLeq16(v)
	case OpLeq16U:
		return rewriteValueRISCV64_OpLeq16U(v)
	case OpLeq32:
		return rewriteValueRISCV64_OpLeq32(v)
	case OpLeq32F:
		return rewriteValueRISCV64_OpLeq32F(v)
	case OpLeq32U:
		return rewriteValueRISCV64_OpLeq32U(v)
	case OpLeq64:
		return rewriteValueRISCV64_OpLeq64(v)
	case OpLeq64F:
		return rewriteValueRISCV64_OpLeq64F(v)
	case OpLeq64U:
		return rewriteValueRISCV64_OpLeq64U(v)
	case OpLeq8:
		return rewriteValueRISCV64_OpLeq8(v)
	case OpLeq8U:
		return rewriteValueRISCV64_OpLeq8U(v)
	case OpLess16:
		return rewriteValueRISCV64_OpLess16(v)
	case OpLess16U:
		return rewriteValueRISCV64_OpLess16U(v)
	case OpLess32:
		return rewriteValueRISCV64_OpLess32(v)
	case OpLess32F:
		return rewriteValueRISCV64_OpLess32F(v)
	case OpLess32U:
		return rewriteValueRISCV64_OpLess32U(v)
	case OpLess64:
		return rewriteValueRISCV64_OpLess64(v)
	case OpLess64F:
		return rewriteValueRISCV64_OpLess64F(v)
	case OpLess64U:
		return rewriteValueRISCV64_OpLess64U(v)
	case OpLess8:
		return rewriteValueRISCV64_OpLess8(v)
	case OpLess8U:
		return rewriteValueRISCV64_OpLess8U(v)
	case OpLoad:
		return rewriteValueRISCV64_OpLoad(v)
	case OpLocalAddr:
		return rewriteValueRISCV64_OpLocalAddr(v)
	case OpLsh16x16:
		return rewriteValueRISCV64_OpLsh16x16(v)
	case OpLsh16x32:
		return rewriteValueRISCV64_OpLsh16x32(v)
	case OpLsh16x64:
		return rewriteValueRISCV64_OpLsh16x64(v)
	case OpLsh16x8:
		return rewriteValueRISCV64_OpLsh16x8(v)
	case OpLsh32x16:
		return rewriteValueRISCV64_OpLsh32x16(v)
	case OpLsh32x32:
		return rewriteValueRISCV64_OpLsh32x32(v)
	case OpLsh32x64:
		return rewriteValueRISCV64_OpLsh32x64(v)
	case OpLsh32x8:
		return rewriteValueRISCV64_OpLsh32x8(v)
	case OpLsh64x16:
		return rewriteValueRISCV64_OpLsh64x16(v)
	case OpLsh64x32:
		return rewriteValueRISCV64_OpLsh64x32(v)
	case OpLsh64x64:
		return rewriteValueRISCV64_OpLsh64x64(v)
	case OpLsh64x8:
		return rewriteValueRISCV64_OpLsh64x8(v)
	case OpLsh8x16:
		return rewriteValueRISCV64_OpLsh8x16(v)
	case OpLsh8x32:
		return rewriteValueRISCV64_OpLsh8x32(v)
	case OpLsh8x64:
		return rewriteValueRISCV64_OpLsh8x64(v)
	case OpLsh8x8:
		return rewriteValueRISCV64_OpLsh8x8(v)
	case OpMod16:
		return rewriteValueRISCV64_OpMod16(v)
	case OpMod16u:
		return rewriteValueRISCV64_OpMod16u(v)
	case OpMod32:
		return rewriteValueRISCV64_OpMod32(v)
	case OpMod32u:
		return rewriteValueRISCV64_OpMod32u(v)
	case OpMod64:
		return rewriteValueRISCV64_OpMod64(v)
	case OpMod64u:
		return rewriteValueRISCV64_OpMod64u(v)
	case OpMod8:
		return rewriteValueRISCV64_OpMod8(v)
	case OpMod8u:
		return rewriteValueRISCV64_OpMod8u(v)
	case OpMove:
		return rewriteValueRISCV64_OpMove(v)
	case OpMul16:
		return rewriteValueRISCV64_OpMul16(v)
	case OpMul32:
		return rewriteValueRISCV64_OpMul32(v)
	case OpMul32F:
		return rewriteValueRISCV64_OpMul32F(v)
	case OpMul64:
		return rewriteValueRISCV64_OpMul64(v)
	case OpMul64F:
		return rewriteValueRISCV64_OpMul64F(v)
	case OpMul8:
		return rewriteValueRISCV64_OpMul8(v)
	case OpNeg16:
		return rewriteValueRISCV64_OpNeg16(v)
	case OpNeg32:
		return rewriteValueRISCV64_OpNeg32(v)
	case OpNeg32F:
		return rewriteValueRISCV64_OpNeg32F(v)
	case OpNeg64:
		return rewriteValueRISCV64_OpNeg64(v)
	case OpNeg64F:
		return rewriteValueRISCV64_OpNeg64F(v)
	case OpNeg8:
		return rewriteValueRISCV64_OpNeg8(v)
	case OpNeq16:
		return rewriteValueRISCV64_OpNeq16(v)
	case OpNeq32:
		return rewriteValueRISCV64_OpNeq32(v)
	case OpNeq32F:
		return rewriteValueRISCV64_OpNeq32F(v)
	case OpNeq64:
		return rewriteValueRISCV64_OpNeq64(v)
	case OpNeq64F:
		return rewriteValueRISCV64_OpNeq64F(v)
	case OpNeq8:
		return rewriteValueRISCV64_OpNeq8(v)
	case OpNeqB:
		return rewriteValueRISCV64_OpNeqB(v)
	case OpNeqPtr:
		return rewriteValueRISCV64_OpNeqPtr(v)
	case OpNilCheck:
		return rewriteValueRISCV64_OpNilCheck(v)
	case OpNot:
		return rewriteValueRISCV64_OpNot(v)
	case OpOffPtr:
		return rewriteValueRISCV64_OpOffPtr(v)
	case OpOr16:
		return rewriteValueRISCV64_OpOr16(v)
	case OpOr32:
		return rewriteValueRISCV64_OpOr32(v)
	case OpOr64:
		return rewriteValueRISCV64_OpOr64(v)
	case OpOr8:
		return rewriteValueRISCV64_OpOr8(v)
	case OpOrB:
		return rewriteValueRISCV64_OpOrB(v)
	case OpPanicBounds:
		return rewriteValueRISCV64_OpPanicBounds(v)
	case OpRISCV64ADD:
		return rewriteValueRISCV64_OpRISCV64ADD(v)
	case OpRISCV64ADDI:
		return rewriteValueRISCV64_OpRISCV64ADDI(v)
	case OpRISCV64MOVBUload:
		return rewriteValueRISCV64_OpRISCV64MOVBUload(v)
	case OpRISCV64MOVBload:
		return rewriteValueRISCV64_OpRISCV64MOVBload(v)
	case OpRISCV64MOVBstore:
		return rewriteValueRISCV64_OpRISCV64MOVBstore(v)
	case OpRISCV64MOVDconst:
		return rewriteValueRISCV64_OpRISCV64MOVDconst(v)
	case OpRISCV64MOVDload:
		return rewriteValueRISCV64_OpRISCV64MOVDload(v)
	case OpRISCV64MOVDstore:
		return rewriteValueRISCV64_OpRISCV64MOVDstore(v)
	case OpRISCV64MOVHUload:
		return rewriteValueRISCV64_OpRISCV64MOVHUload(v)
	case OpRISCV64MOVHload:
		return rewriteValueRISCV64_OpRISCV64MOVHload(v)
	case OpRISCV64MOVHstore:
		return rewriteValueRISCV64_OpRISCV64MOVHstore(v)
	case OpRISCV64MOVWUload:
		return rewriteValueRISCV64_OpRISCV64MOVWUload(v)
	case OpRISCV64MOVWload:
		return rewriteValueRISCV64_OpRISCV64MOVWload(v)
	case OpRISCV64MOVWstore:
		return rewriteValueRISCV64_OpRISCV64MOVWstore(v)
	case OpRotateLeft16:
		return rewriteValueRISCV64_OpRotateLeft16(v)
	case OpRotateLeft32:
		return rewriteValueRISCV64_OpRotateLeft32(v)
	case OpRotateLeft64:
		return rewriteValueRISCV64_OpRotateLeft64(v)
	case OpRotateLeft8:
		return rewriteValueRISCV64_OpRotateLeft8(v)
	case OpRound32F:
		return rewriteValueRISCV64_OpRound32F(v)
	case OpRound64F:
		return rewriteValueRISCV64_OpRound64F(v)
	case OpRsh16Ux16:
		return rewriteValueRISCV64_OpRsh16Ux16(v)
	case OpRsh16Ux32:
		return rewriteValueRISCV64_OpRsh16Ux32(v)
	case OpRsh16Ux64:
		return rewriteValueRISCV64_OpRsh16Ux64(v)
	case OpRsh16Ux8:
		return rewriteValueRISCV64_OpRsh16Ux8(v)
	case OpRsh16x16:
		return rewriteValueRISCV64_OpRsh16x16(v)
	case OpRsh16x32:
		return rewriteValueRISCV64_OpRsh16x32(v)
	case OpRsh16x64:
		return rewriteValueRISCV64_OpRsh16x64(v)
	case OpRsh16x8:
		return rewriteValueRISCV64_OpRsh16x8(v)
	case OpRsh32Ux16:
		return rewriteValueRISCV64_OpRsh32Ux16(v)
	case OpRsh32Ux32:
		return rewriteValueRISCV64_OpRsh32Ux32(v)
	case OpRsh32Ux64:
		return rewriteValueRISCV64_OpRsh32Ux64(v)
	case OpRsh32Ux8:
		return rewriteValueRISCV64_OpRsh32Ux8(v)
	case OpRsh32x16:
		return rewriteValueRISCV64_OpRsh32x16(v)
	case OpRsh32x32:
		return rewriteValueRISCV64_OpRsh32x32(v)
	case OpRsh32x64:
		return rewriteValueRISCV64_OpRsh32x64(v)
	case OpRsh32x8:
		return rewriteValueRISCV64_OpRsh32x8(v)
	case OpRsh64Ux16:
		return rewriteValueRISCV64_OpRsh64Ux16(v)
	case OpRsh64Ux32:
		return rewriteValueRISCV64_OpRsh64Ux32(v)
	case OpRsh64Ux64:
		return rewriteValueRISCV64_OpRsh64Ux64(v)
	case OpRsh64Ux8:
		return rewriteValueRISCV64_OpRsh64Ux8(v)
	case OpRsh64x16:
		return rewriteValueRISCV64_OpRsh64x16(v)
	case OpRsh64x32:
		return rewriteValueRISCV64_OpRsh64x32(v)
	case OpRsh64x64:
		return rewriteValueRISCV64_OpRsh64x64(v)
	case OpRsh64x8:
		return rewriteValueRISCV64_OpRsh64x8(v)
	case OpRsh8Ux16:
		return rewriteValueRISCV64_OpRsh8Ux16(v)
	case OpRsh8Ux32:
		return rewriteValueRISCV64_OpRsh8Ux32(v)
	case OpRsh8Ux64:
		return rewriteValueRISCV64_OpRsh8Ux64(v)
	case OpRsh8Ux8:
		return rewriteValueRISCV64_OpRsh8Ux8(v)
	case OpRsh8x16:
		return rewriteValueRISCV64_OpRsh8x16(v)
	case OpRsh8x32:
		return rewriteValueRISCV64_OpRsh8x32(v)
	case OpRsh8x64:
		return rewriteValueRISCV64_OpRsh8x64(v)
	case OpRsh8x8:
		return rewriteValueRISCV64_OpRsh8x8(v)
	case OpSignExt16to32:
		return rewriteValueRISCV64_OpSignExt16to32(v)
	case OpSignExt16to64:
		return rewriteValueRISCV64_OpSignExt16to64(v)
	case OpSignExt32to64:
		return rewriteValueRISCV64_OpSignExt32to64(v)
	case OpSignExt8to16:
		return rewriteValueRISCV64_OpSignExt8to16(v)
	case OpSignExt8to32:
		return rewriteValueRISCV64_OpSignExt8to32(v)
	case OpSignExt8to64:
		return rewriteValueRISCV64_OpSignExt8to64(v)
	case OpSlicemask:
		return rewriteValueRISCV64_OpSlicemask(v)
	case OpSqrt:
		return rewriteValueRISCV64_OpSqrt(v)
	case OpStaticCall:
		return rewriteValueRISCV64_OpStaticCall(v)
	case OpStore:
		return rewriteValueRISCV64_OpStore(v)
	case OpSub16:
		return rewriteValueRISCV64_OpSub16(v)
	case OpSub32:
		return rewriteValueRISCV64_OpSub32(v)
	case OpSub32F:
		return rewriteValueRISCV64_OpSub32F(v)
	case OpSub64:
		return rewriteValueRISCV64_OpSub64(v)
	case OpSub64F:
		return rewriteValueRISCV64_OpSub64F(v)
	case OpSub8:
		return rewriteValueRISCV64_OpSub8(v)
	case OpSubPtr:
		return rewriteValueRISCV64_OpSubPtr(v)
	case OpTrunc16to8:
		return rewriteValueRISCV64_OpTrunc16to8(v)
	case OpTrunc32to16:
		return rewriteValueRISCV64_OpTrunc32to16(v)
	case OpTrunc32to8:
		return rewriteValueRISCV64_OpTrunc32to8(v)
	case OpTrunc64to16:
		return rewriteValueRISCV64_OpTrunc64to16(v)
	case OpTrunc64to32:
		return rewriteValueRISCV64_OpTrunc64to32(v)
	case OpTrunc64to8:
		return rewriteValueRISCV64_OpTrunc64to8(v)
	case OpWB:
		return rewriteValueRISCV64_OpWB(v)
	case OpXor16:
		return rewriteValueRISCV64_OpXor16(v)
	case OpXor32:
		return rewriteValueRISCV64_OpXor32(v)
	case OpXor64:
		return rewriteValueRISCV64_OpXor64(v)
	case OpXor8:
		return rewriteValueRISCV64_OpXor8(v)
	case OpZero:
		return rewriteValueRISCV64_OpZero(v)
	case OpZeroExt16to32:
		return rewriteValueRISCV64_OpZeroExt16to32(v)
	case OpZeroExt16to64:
		return rewriteValueRISCV64_OpZeroExt16to64(v)
	case OpZeroExt32to64:
		return rewriteValueRISCV64_OpZeroExt32to64(v)
	case OpZeroExt8to16:
		return rewriteValueRISCV64_OpZeroExt8to16(v)
	case OpZeroExt8to32:
		return rewriteValueRISCV64_OpZeroExt8to32(v)
	case OpZeroExt8to64:
		return rewriteValueRISCV64_OpZeroExt8to64(v)
	}
	return false
}
func rewriteValueRISCV64_OpAdd16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Add16 x y)
	// result: (ADD x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64ADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAdd32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Add32 x y)
	// result: (ADD x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64ADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAdd32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Add32F x y)
	// result: (FADDS x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FADDS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAdd64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Add64 x y)
	// result: (ADD x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64ADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAdd64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Add64F x y)
	// result: (FADDD x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FADDD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAdd8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Add8 x y)
	// result: (ADD x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64ADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAddPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AddPtr x y)
	// result: (ADD x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64ADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAddr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Addr {sym} base)
	// result: (MOVaddr {sym} base)
	for {
		sym := v.Aux
		base := v_0
		v.reset(OpRISCV64MOVaddr)
		v.Aux = sym
		v.AddArg(base)
		return true
	}
}
func rewriteValueRISCV64_OpAnd16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (And16 x y)
	// result: (AND x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64AND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAnd32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (And32 x y)
	// result: (AND x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64AND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAnd64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (And64 x y)
	// result: (AND x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64AND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAnd8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (And8 x y)
	// result: (AND x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64AND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAndB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AndB x y)
	// result: (AND x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64AND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAvg64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Avg64u <t> x y)
	// result: (ADD (ADD <t> (SRLI <t> [1] x) (SRLI <t> [1] y)) (ANDI <t> [1] (AND <t> x y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpClosureCall(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ClosureCall [argwid] entry closure mem)
	// result: (CALLclosure [argwid] entry closure mem)
	for {
		argwid := v.AuxInt
		entry := v_0
		closure := v_1
		mem := v_2
		v.reset(OpRISCV64CALLclosure)
		v.AuxInt = argwid
		v.AddArg(entry)
		v.AddArg(closure)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueRISCV64_OpCom16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Com16 x)
	// result: (XORI [int64(-1)] x)
	for {
		x := v_0
		v.reset(OpRISCV64XORI)
		v.AuxInt = int64(-1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCom32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Com32 x)
	// result: (XORI [int64(-1)] x)
	for {
		x := v_0
		v.reset(OpRISCV64XORI)
		v.AuxInt = int64(-1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCom64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Com64 x)
	// result: (XORI [int64(-1)] x)
	for {
		x := v_0
		v.reset(OpRISCV64XORI)
		v.AuxInt = int64(-1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCom8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Com8 x)
	// result: (XORI [int64(-1)] x)
	for {
		x := v_0
		v.reset(OpRISCV64XORI)
		v.AuxInt = int64(-1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpConst16(v *Value) bool {
	// match: (Const16 [val])
	// result: (MOVHconst [val])
	for {
		val := v.AuxInt
		v.reset(OpRISCV64MOVHconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueRISCV64_OpConst32(v *Value) bool {
	// match: (Const32 [val])
	// result: (MOVWconst [val])
	for {
		val := v.AuxInt
		v.reset(OpRISCV64MOVWconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueRISCV64_OpConst32F(v *Value) bool {
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
func rewriteValueRISCV64_OpConst64(v *Value) bool {
	// match: (Const64 [val])
	// result: (MOVDconst [val])
	for {
		val := v.AuxInt
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueRISCV64_OpConst64F(v *Value) bool {
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
func rewriteValueRISCV64_OpConst8(v *Value) bool {
	// match: (Const8 [val])
	// result: (MOVBconst [val])
	for {
		val := v.AuxInt
		v.reset(OpRISCV64MOVBconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueRISCV64_OpConstBool(v *Value) bool {
	// match: (ConstBool [b])
	// result: (MOVBconst [b])
	for {
		b := v.AuxInt
		v.reset(OpRISCV64MOVBconst)
		v.AuxInt = b
		return true
	}
}
func rewriteValueRISCV64_OpConstNil(v *Value) bool {
	// match: (ConstNil)
	// result: (MOVDconst [0])
	for {
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = 0
		return true
	}
}
func rewriteValueRISCV64_OpConvert(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Convert x mem)
	// result: (MOVconvert x mem)
	for {
		x := v_0
		mem := v_1
		v.reset(OpRISCV64MOVconvert)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueRISCV64_OpCvt32Fto32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt32Fto32 x)
	// result: (FCVTWS x)
	for {
		x := v_0
		v.reset(OpRISCV64FCVTWS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt32Fto64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt32Fto64 x)
	// result: (FCVTLS x)
	for {
		x := v_0
		v.reset(OpRISCV64FCVTLS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt32Fto64F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt32Fto64F x)
	// result: (FCVTDS x)
	for {
		x := v_0
		v.reset(OpRISCV64FCVTDS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt32to32F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt32to32F x)
	// result: (FCVTSW x)
	for {
		x := v_0
		v.reset(OpRISCV64FCVTSW)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt32to64F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt32to64F x)
	// result: (FCVTDW x)
	for {
		x := v_0
		v.reset(OpRISCV64FCVTDW)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt64Fto32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64Fto32 x)
	// result: (FCVTWD x)
	for {
		x := v_0
		v.reset(OpRISCV64FCVTWD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt64Fto32F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64Fto32F x)
	// result: (FCVTSD x)
	for {
		x := v_0
		v.reset(OpRISCV64FCVTSD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt64Fto64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64Fto64 x)
	// result: (FCVTLD x)
	for {
		x := v_0
		v.reset(OpRISCV64FCVTLD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt64to32F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64to32F x)
	// result: (FCVTSL x)
	for {
		x := v_0
		v.reset(OpRISCV64FCVTSL)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt64to64F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64to64F x)
	// result: (FCVTDL x)
	for {
		x := v_0
		v.reset(OpRISCV64FCVTDL)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpDiv16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 x y)
	// result: (DIVW (SignExt16to32 x) (SignExt16to32 y))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpDiv16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// result: (DIVUW (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpDiv32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div32 x y)
	// result: (DIVW x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64DIVW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpDiv32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div32F x y)
	// result: (FDIVS x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FDIVS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpDiv32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div32u x y)
	// result: (DIVUW x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64DIVUW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpDiv64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div64 x y)
	// result: (DIV x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64DIV)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpDiv64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div64F x y)
	// result: (FDIVD x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FDIVD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpDiv64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div64u x y)
	// result: (DIVU x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64DIVU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpDiv8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (DIVW (SignExt8to32 x) (SignExt8to32 y))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpDiv8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// result: (DIVUW (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpEq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq16 x y)
	// result: (SEQZ (ZeroExt16to64 (SUB <x.Type> x y)))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpEq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq32 x y)
	// result: (SEQZ (ZeroExt32to64 (SUB <x.Type> x y)))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpEq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Eq32F x y)
	// result: (FEQS x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FEQS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpEq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64 x y)
	// result: (SEQZ (SUB <x.Type> x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpEq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Eq64F x y)
	// result: (FEQD x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FEQD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpEq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq8 x y)
	// result: (SEQZ (ZeroExt8to64 (SUB <x.Type> x y)))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpEqB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqB x y)
	// result: (XORI [1] (XOR <typ.Bool> x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64XORI)
		v.AuxInt = 1
		v0 := b.NewValue0(v.Pos, OpRISCV64XOR, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpEqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (EqPtr x y)
	// result: (SEQZ (SUB <x.Type> x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq16 x y)
	// result: (Not (Less16 x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess16, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGeq16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq16U x y)
	// result: (Not (Less16U x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess16U, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq32 x y)
	// result: (Not (Less32 x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess32, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Geq32F x y)
	// result: (FLES y x)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FLES)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGeq32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq32U x y)
	// result: (Not (Less32U x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess32U, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq64 x y)
	// result: (Not (Less64 x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess64, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Geq64F x y)
	// result: (FLED y x)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FLED)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGeq64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq64U x y)
	// result: (Not (Less64U x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess64U, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq8 x y)
	// result: (Not (Less8 x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess8, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGeq8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq8U x y)
	// result: (Not (Less8U x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess8U, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGetCallerPC(v *Value) bool {
	// match: (GetCallerPC)
	// result: (LoweredGetCallerPC)
	for {
		v.reset(OpRISCV64LoweredGetCallerPC)
		return true
	}
}
func rewriteValueRISCV64_OpGetCallerSP(v *Value) bool {
	// match: (GetCallerSP)
	// result: (LoweredGetCallerSP)
	for {
		v.reset(OpRISCV64LoweredGetCallerSP)
		return true
	}
}
func rewriteValueRISCV64_OpGetClosurePtr(v *Value) bool {
	// match: (GetClosurePtr)
	// result: (LoweredGetClosurePtr)
	for {
		v.reset(OpRISCV64LoweredGetClosurePtr)
		return true
	}
}
func rewriteValueRISCV64_OpGreater16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Greater16 x y)
	// result: (Less16 y x)
	for {
		x := v_0
		y := v_1
		v.reset(OpLess16)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Greater16U x y)
	// result: (Less16U y x)
	for {
		x := v_0
		y := v_1
		v.reset(OpLess16U)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Greater32 x y)
	// result: (Less32 y x)
	for {
		x := v_0
		y := v_1
		v.reset(OpLess32)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Greater32F x y)
	// result: (FLTS y x)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FLTS)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Greater32U x y)
	// result: (Less32U y x)
	for {
		x := v_0
		y := v_1
		v.reset(OpLess32U)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Greater64 x y)
	// result: (Less64 y x)
	for {
		x := v_0
		y := v_1
		v.reset(OpLess64)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Greater64F x y)
	// result: (FLTD y x)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FLTD)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Greater64U x y)
	// result: (Less64U y x)
	for {
		x := v_0
		y := v_1
		v.reset(OpLess64U)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Greater8 x y)
	// result: (Less8 y x)
	for {
		x := v_0
		y := v_1
		v.reset(OpLess8)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Greater8U x y)
	// result: (Less8U y x)
	for {
		x := v_0
		y := v_1
		v.reset(OpLess8U)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpHmul32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32 x y)
	// result: (SRAI [32] (MUL (SignExt32to64 x) (SignExt32to64 y)))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpHmul32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32u x y)
	// result: (SRLI [32] (MUL (ZeroExt32to64 x) (ZeroExt32to64 y)))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpHmul64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Hmul64 x y)
	// result: (MULH x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64MULH)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpHmul64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Hmul64u x y)
	// result: (MULHU x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64MULHU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpInterCall(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (InterCall [argwid] entry mem)
	// result: (CALLinter [argwid] entry mem)
	for {
		argwid := v.AuxInt
		entry := v_0
		mem := v_1
		v.reset(OpRISCV64CALLinter)
		v.AuxInt = argwid
		v.AddArg(entry)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueRISCV64_OpIsInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IsInBounds idx len)
	// result: (Less64U idx len)
	for {
		idx := v_0
		len := v_1
		v.reset(OpLess64U)
		v.AddArg(idx)
		v.AddArg(len)
		return true
	}
}
func rewriteValueRISCV64_OpIsNonNil(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (IsNonNil p)
	// result: (NeqPtr (MOVDconst) p)
	for {
		p := v_0
		v.reset(OpNeqPtr)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v.AddArg(v0)
		v.AddArg(p)
		return true
	}
}
func rewriteValueRISCV64_OpIsSliceInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IsSliceInBounds idx len)
	// result: (Leq64U idx len)
	for {
		idx := v_0
		len := v_1
		v.reset(OpLeq64U)
		v.AddArg(idx)
		v.AddArg(len)
		return true
	}
}
func rewriteValueRISCV64_OpLeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16 x y)
	// result: (Not (Less16 y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess16, typ.Bool)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16U x y)
	// result: (Not (Less16U y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess16U, typ.Bool)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32 x y)
	// result: (Not (Less32 y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess32, typ.Bool)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Leq32F x y)
	// result: (FLES x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FLES)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpLeq32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32U x y)
	// result: (Not (Less32U y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess32U, typ.Bool)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq64 x y)
	// result: (Not (Less64 y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess64, typ.Bool)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Leq64F x y)
	// result: (FLED x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FLED)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpLeq64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq64U x y)
	// result: (Not (Less64U y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess64U, typ.Bool)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8 x y)
	// result: (Not (Less8 y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess8, typ.Bool)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8U x y)
	// result: (Not (Less8U y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess8U, typ.Bool)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLess16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16 x y)
	// result: (SLT (SignExt16to64 x) (SignExt16to64 y))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLess16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16U x y)
	// result: (SLTU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLess32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32 x y)
	// result: (SLT (SignExt32to64 x) (SignExt32to64 y))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLess32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less32F x y)
	// result: (FLTS x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FLTS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpLess32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32U x y)
	// result: (SLTU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLess64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less64 x y)
	// result: (SLT x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SLT)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpLess64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less64F x y)
	// result: (FLTD x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FLTD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpLess64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less64U x y)
	// result: (SLTU x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SLTU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpLess8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8 x y)
	// result: (SLT (SignExt8to64 x) (SignExt8to64 y))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLess8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8U x y)
	// result: (SLTU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLoad(v *Value) bool {
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
		ptr := v_0
		mem := v_1
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
		ptr := v_0
		mem := v_1
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
		ptr := v_0
		mem := v_1
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
		ptr := v_0
		mem := v_1
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
		ptr := v_0
		mem := v_1
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
		ptr := v_0
		mem := v_1
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
		ptr := v_0
		mem := v_1
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
		ptr := v_0
		mem := v_1
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
		ptr := v_0
		mem := v_1
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
func rewriteValueRISCV64_OpLocalAddr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (LocalAddr {sym} base _)
	// result: (MOVaddr {sym} base)
	for {
		sym := v.Aux
		base := v_0
		v.reset(OpRISCV64MOVaddr)
		v.Aux = sym
		v.AddArg(base)
		return true
	}
}
func rewriteValueRISCV64_OpLsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x16 <t> x y)
	// result: (AND (SLL <t> x y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x32 <t> x y)
	// result: (AND (SLL <t> x y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh16x64 <t> x y)
	// result: (AND (SLL <t> x y) (Neg16 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x8 <t> x y)
	// result: (AND (SLL <t> x y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x16 <t> x y)
	// result: (AND (SLL <t> x y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x32 <t> x y)
	// result: (AND (SLL <t> x y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh32x64 <t> x y)
	// result: (AND (SLL <t> x y) (Neg32 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x8 <t> x y)
	// result: (AND (SLL <t> x y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x16 <t> x y)
	// result: (AND (SLL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x32 <t> x y)
	// result: (AND (SLL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh64x64 <t> x y)
	// result: (AND (SLL <t> x y) (Neg64 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x8 <t> x y)
	// result: (AND (SLL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x16 <t> x y)
	// result: (AND (SLL <t> x y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x32 <t> x y)
	// result: (AND (SLL <t> x y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh8x64 <t> x y)
	// result: (AND (SLL <t> x y) (Neg8 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpLsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x8 <t> x y)
	// result: (AND (SLL <t> x y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpMod16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16 x y)
	// result: (REMW (SignExt16to32 x) (SignExt16to32 y))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpMod16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16u x y)
	// result: (REMUW (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpMod32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mod32 x y)
	// result: (REMW x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64REMW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpMod32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mod32u x y)
	// result: (REMUW x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64REMUW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpMod64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mod64 x y)
	// result: (REM x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64REM)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpMod64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mod64u x y)
	// result: (REMU x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64REMU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpMod8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// result: (REMW (SignExt8to32 x) (SignExt8to32 y))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpMod8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// result: (REMUW (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpMove(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Move [0] _ _ mem)
	// result: mem
	for {
		if v.AuxInt != 0 {
			break
		}
		mem := v_2
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
		dst := v_0
		src := v_1
		mem := v_2
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
		dst := v_0
		src := v_1
		mem := v_2
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
		dst := v_0
		src := v_1
		mem := v_2
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
		dst := v_0
		src := v_1
		mem := v_2
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
		dst := v_0
		src := v_1
		mem := v_2
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
func rewriteValueRISCV64_OpMul16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul16 x y)
	// result: (MULW (SignExt16to32 x) (SignExt16to32 y))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpMul32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mul32 x y)
	// result: (MULW x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64MULW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpMul32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mul32F x y)
	// result: (FMULS x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FMULS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpMul64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mul64 x y)
	// result: (MUL x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64MUL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpMul64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mul64F x y)
	// result: (FMULD x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FMULD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpMul8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul8 x y)
	// result: (MULW (SignExt8to32 x) (SignExt8to32 y))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpNeg16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg16 x)
	// result: (SUB (MOVHconst) x)
	for {
		x := v_0
		v.reset(OpRISCV64SUB)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVHconst, typ.UInt16)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpNeg32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg32 x)
	// result: (SUB (MOVWconst) x)
	for {
		x := v_0
		v.reset(OpRISCV64SUB)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVWconst, typ.UInt32)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpNeg32F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Neg32F x)
	// result: (FNEGS x)
	for {
		x := v_0
		v.reset(OpRISCV64FNEGS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpNeg64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg64 x)
	// result: (SUB (MOVDconst) x)
	for {
		x := v_0
		v.reset(OpRISCV64SUB)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpNeg64F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Neg64F x)
	// result: (FNEGD x)
	for {
		x := v_0
		v.reset(OpRISCV64FNEGD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpNeg8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg8 x)
	// result: (SUB (MOVBconst) x)
	for {
		x := v_0
		v.reset(OpRISCV64SUB)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVBconst, typ.UInt8)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpNeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq16 x y)
	// result: (SNEZ (ZeroExt16to64 (SUB <x.Type> x y)))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpNeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq32 x y)
	// result: (SNEZ (ZeroExt32to64 (SUB <x.Type> x y)))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpNeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Neq32F x y)
	// result: (FNES x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FNES)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpNeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64 x y)
	// result: (SNEZ (SUB <x.Type> x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpNeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Neq64F x y)
	// result: (FNED x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FNED)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpNeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq8 x y)
	// result: (SNEZ (ZeroExt8to64 (SUB <x.Type> x y)))
	for {
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpNeqB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (NeqB x y)
	// result: (XOR x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64XOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpNeqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (NeqPtr x y)
	// result: (SNEZ (SUB <x.Type> x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpNilCheck(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (NilCheck ptr mem)
	// result: (LoweredNilCheck ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.reset(OpRISCV64LoweredNilCheck)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueRISCV64_OpNot(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Not x)
	// result: (XORI [1] x)
	for {
		x := v_0
		v.reset(OpRISCV64XORI)
		v.AuxInt = 1
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpOffPtr(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (OffPtr [off] ptr:(SP))
	// result: (MOVaddr [off] ptr)
	for {
		off := v.AuxInt
		ptr := v_0
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
		ptr := v_0
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
		ptr := v_0
		v.reset(OpRISCV64ADD)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = off
		v.AddArg(v0)
		v.AddArg(ptr)
		return true
	}
}
func rewriteValueRISCV64_OpOr16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Or16 x y)
	// result: (OR x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64OR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpOr32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Or32 x y)
	// result: (OR x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64OR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpOr64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Or64 x y)
	// result: (OR x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64OR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpOr8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Or8 x y)
	// result: (OR x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64OR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpOrB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (OrB x y)
	// result: (OR x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64OR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpPanicBounds(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (PanicBounds [kind] x y mem)
	// cond: boundsABI(kind) == 0
	// result: (LoweredPanicBoundsA [kind] x y mem)
	for {
		kind := v.AuxInt
		x := v_0
		y := v_1
		mem := v_2
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
		x := v_0
		y := v_1
		mem := v_2
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
		x := v_0
		y := v_1
		mem := v_2
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
func rewriteValueRISCV64_OpRISCV64ADD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADD (MOVDconst [off]) ptr)
	// cond: is32Bit(off)
	// result: (ADDI [off] ptr)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64MOVDconst {
				continue
			}
			off := v_0.AuxInt
			ptr := v_1
			if !(is32Bit(off)) {
				continue
			}
			v.reset(OpRISCV64ADDI)
			v.AuxInt = off
			v.AddArg(ptr)
			return true
		}
		break
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64ADDI(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ADDI [c] (MOVaddr [d] {s} x))
	// cond: is32Bit(c+d)
	// result: (MOVaddr [c+d] {s} x)
	for {
		c := v.AuxInt
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
		x := v_0
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVBUload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBUload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVBUload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v_1
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
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		mem := v_1
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
func rewriteValueRISCV64_OpRISCV64MOVBload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVBload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v_1
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
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		mem := v_1
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
func rewriteValueRISCV64_OpRISCV64MOVBstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVBstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v_1
		mem := v_2
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
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		val := v_1
		mem := v_2
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
func rewriteValueRISCV64_OpRISCV64MOVDconst(v *Value) bool {
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
func rewriteValueRISCV64_OpRISCV64MOVDload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVDload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v_1
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
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		mem := v_1
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
func rewriteValueRISCV64_OpRISCV64MOVDstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVDstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v_1
		mem := v_2
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
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		val := v_1
		mem := v_2
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
func rewriteValueRISCV64_OpRISCV64MOVHUload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHUload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVHUload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v_1
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
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		mem := v_1
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
func rewriteValueRISCV64_OpRISCV64MOVHload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVHload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v_1
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
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		mem := v_1
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
func rewriteValueRISCV64_OpRISCV64MOVHstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVHstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v_1
		mem := v_2
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
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		val := v_1
		mem := v_2
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
func rewriteValueRISCV64_OpRISCV64MOVWUload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWUload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVWUload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v_1
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
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		mem := v_1
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
func rewriteValueRISCV64_OpRISCV64MOVWload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVWload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v_1
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
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		mem := v_1
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
func rewriteValueRISCV64_OpRISCV64MOVWstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVWstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v_1
		mem := v_2
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
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		val := v_1
		mem := v_2
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
func rewriteValueRISCV64_OpRotateLeft16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft16 <t> x (MOVHconst [c]))
	// result: (Or16 (Lsh16x64 <t> x (MOVHconst [c&15])) (Rsh16Ux64 <t> x (MOVHconst [-c&15])))
	for {
		t := v.Type
		x := v_0
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
func rewriteValueRISCV64_OpRotateLeft32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft32 <t> x (MOVWconst [c]))
	// result: (Or32 (Lsh32x64 <t> x (MOVWconst [c&31])) (Rsh32Ux64 <t> x (MOVWconst [-c&31])))
	for {
		t := v.Type
		x := v_0
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
func rewriteValueRISCV64_OpRotateLeft64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft64 <t> x (MOVDconst [c]))
	// result: (Or64 (Lsh64x64 <t> x (MOVDconst [c&63])) (Rsh64Ux64 <t> x (MOVDconst [-c&63])))
	for {
		t := v.Type
		x := v_0
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
func rewriteValueRISCV64_OpRotateLeft8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft8 <t> x (MOVBconst [c]))
	// result: (Or8 (Lsh8x64 <t> x (MOVBconst [c&7])) (Rsh8Ux64 <t> x (MOVBconst [-c&7])))
	for {
		t := v.Type
		x := v_0
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
func rewriteValueRISCV64_OpRound32F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Round32F x)
	// result: x
	for {
		x := v_0
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpRound64F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Round64F x)
	// result: x
	for {
		x := v_0
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpRsh16Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux16 <t> x y)
	// result: (AND (SRL <t> (ZeroExt16to64 x) y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh16Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 <t> x y)
	// result: (AND (SRL <t> (ZeroExt16to64 x) y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh16Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 <t> x y)
	// result: (AND (SRL <t> (ZeroExt16to64 x) y) (Neg16 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh16Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux8 <t> x y)
	// result: (AND (SRL <t> (ZeroExt16to64 x) y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x16 <t> x y)
	// result: (SRA <t> (SignExt16to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt16to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x32 <t> x y)
	// result: (SRA <t> (SignExt16to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt32to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 <t> x y)
	// result: (SRA <t> (SignExt16to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x8 <t> x y)
	// result: (SRA <t> (SignExt16to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt8to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh32Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 <t> x y)
	// result: (AND (SRL <t> (ZeroExt32to64 x) y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh32Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux32 <t> x y)
	// result: (AND (SRL <t> (ZeroExt32to64 x) y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh32Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux64 <t> x y)
	// result: (AND (SRL <t> (ZeroExt32to64 x) y) (Neg32 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh32Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux8 <t> x y)
	// result: (AND (SRL <t> (ZeroExt32to64 x) y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x16 <t> x y)
	// result: (SRA <t> (SignExt32to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt16to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x32 <t> x y)
	// result: (SRA <t> (SignExt32to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt32to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x64 <t> x y)
	// result: (SRA <t> (SignExt32to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x8 <t> x y)
	// result: (SRA <t> (SignExt32to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt8to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh64Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux16 <t> x y)
	// result: (AND (SRL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh64Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux32 <t> x y)
	// result: (AND (SRL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh64Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64Ux64 <t> x y)
	// result: (AND (SRL <t> x y) (Neg64 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh64Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux8 <t> x y)
	// result: (AND (SRL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x16 <t> x y)
	// result: (SRA <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt16to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x32 <t> x y)
	// result: (SRA <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt32to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x64 <t> x y)
	// result: (SRA <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x8 <t> x y)
	// result: (SRA <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt8to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh8Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux16 <t> x y)
	// result: (AND (SRL <t> (ZeroExt8to64 x) y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh8Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 <t> x y)
	// result: (AND (SRL <t> (ZeroExt8to64 x) y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh8Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 <t> x y)
	// result: (AND (SRL <t> (ZeroExt8to64 x) y) (Neg8 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh8Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux8 <t> x y)
	// result: (AND (SRL <t> (ZeroExt8to64 x) y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x16 <t> x y)
	// result: (SRA <t> (SignExt8to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt16to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x32 <t> x y)
	// result: (SRA <t> (SignExt8to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt32to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x64 <t> x y)
	// result: (SRA <t> (SignExt8to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpRsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x8 <t> x y)
	// result: (SRA <t> (SignExt8to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt8to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
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
func rewriteValueRISCV64_OpSignExt16to32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (SignExt16to32 <t> x)
	// result: (SRAI [48] (SLLI <t> [48] x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpRISCV64SRAI)
		v.AuxInt = 48
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 48
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpSignExt16to64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (SignExt16to64 <t> x)
	// result: (SRAI [48] (SLLI <t> [48] x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpRISCV64SRAI)
		v.AuxInt = 48
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 48
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpSignExt32to64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (SignExt32to64 <t> x)
	// result: (SRAI [32] (SLLI <t> [32] x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpRISCV64SRAI)
		v.AuxInt = 32
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 32
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpSignExt8to16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (SignExt8to16 <t> x)
	// result: (SRAI [56] (SLLI <t> [56] x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpRISCV64SRAI)
		v.AuxInt = 56
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 56
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpSignExt8to32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (SignExt8to32 <t> x)
	// result: (SRAI [56] (SLLI <t> [56] x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpRISCV64SRAI)
		v.AuxInt = 56
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 56
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpSignExt8to64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (SignExt8to64 <t> x)
	// result: (SRAI [56] (SLLI <t> [56] x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpRISCV64SRAI)
		v.AuxInt = 56
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 56
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpSlicemask(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Slicemask <t> x)
	// result: (XOR (MOVDconst [-1]) (SRA <t> (SUB <t> x (MOVDconst [1])) (MOVDconst [63])))
	for {
		t := v.Type
		x := v_0
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
func rewriteValueRISCV64_OpSqrt(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Sqrt x)
	// result: (FSQRTD x)
	for {
		x := v_0
		v.reset(OpRISCV64FSQRTD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpStaticCall(v *Value) bool {
	v_0 := v.Args[0]
	// match: (StaticCall [argwid] {target} mem)
	// result: (CALLstatic [argwid] {target} mem)
	for {
		argwid := v.AuxInt
		target := v.Aux
		mem := v_0
		v.reset(OpRISCV64CALLstatic)
		v.AuxInt = argwid
		v.Aux = target
		v.AddArg(mem)
		return true
	}
}
func rewriteValueRISCV64_OpStore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 1
	// result: (MOVBstore ptr val mem)
	for {
		t := v.Aux
		ptr := v_0
		val := v_1
		mem := v_2
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
		ptr := v_0
		val := v_1
		mem := v_2
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
		ptr := v_0
		val := v_1
		mem := v_2
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
		ptr := v_0
		val := v_1
		mem := v_2
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
		ptr := v_0
		val := v_1
		mem := v_2
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
		ptr := v_0
		val := v_1
		mem := v_2
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
func rewriteValueRISCV64_OpSub16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Sub16 x y)
	// result: (SUB x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpSub32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Sub32 x y)
	// result: (SUB x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpSub32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Sub32F x y)
	// result: (FSUBS x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FSUBS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpSub64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Sub64 x y)
	// result: (SUB x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpSub64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Sub64F x y)
	// result: (FSUBD x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64FSUBD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpSub8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Sub8 x y)
	// result: (SUB x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpSubPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SubPtr x y)
	// result: (SUB x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpTrunc16to8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc16to8 x)
	// result: x
	for {
		x := v_0
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpTrunc32to16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc32to16 x)
	// result: x
	for {
		x := v_0
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpTrunc32to8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc32to8 x)
	// result: x
	for {
		x := v_0
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpTrunc64to16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc64to16 x)
	// result: x
	for {
		x := v_0
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpTrunc64to32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc64to32 x)
	// result: x
	for {
		x := v_0
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpTrunc64to8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc64to8 x)
	// result: x
	for {
		x := v_0
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpWB(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (WB {fn} destptr srcptr mem)
	// result: (LoweredWB {fn} destptr srcptr mem)
	for {
		fn := v.Aux
		destptr := v_0
		srcptr := v_1
		mem := v_2
		v.reset(OpRISCV64LoweredWB)
		v.Aux = fn
		v.AddArg(destptr)
		v.AddArg(srcptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueRISCV64_OpXor16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Xor16 x y)
	// result: (XOR x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64XOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpXor32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Xor32 x y)
	// result: (XOR x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64XOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpXor64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Xor64 x y)
	// result: (XOR x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64XOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpXor8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Xor8 x y)
	// result: (XOR x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64XOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpZero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Zero [0] _ mem)
	// result: mem
	for {
		if v.AuxInt != 0 {
			break
		}
		mem := v_1
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
		ptr := v_0
		mem := v_1
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
		ptr := v_0
		mem := v_1
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
		ptr := v_0
		mem := v_1
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
		ptr := v_0
		mem := v_1
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
		ptr := v_0
		mem := v_1
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
func rewriteValueRISCV64_OpZeroExt16to32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (ZeroExt16to32 <t> x)
	// result: (SRLI [48] (SLLI <t> [48] x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpRISCV64SRLI)
		v.AuxInt = 48
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 48
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpZeroExt16to64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (ZeroExt16to64 <t> x)
	// result: (SRLI [48] (SLLI <t> [48] x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpRISCV64SRLI)
		v.AuxInt = 48
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 48
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpZeroExt32to64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (ZeroExt32to64 <t> x)
	// result: (SRLI [32] (SLLI <t> [32] x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpRISCV64SRLI)
		v.AuxInt = 32
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 32
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpZeroExt8to16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (ZeroExt8to16 <t> x)
	// result: (SRLI [56] (SLLI <t> [56] x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpRISCV64SRLI)
		v.AuxInt = 56
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 56
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpZeroExt8to32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (ZeroExt8to32 <t> x)
	// result: (SRLI [56] (SLLI <t> [56] x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpRISCV64SRLI)
		v.AuxInt = 56
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 56
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpZeroExt8to64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (ZeroExt8to64 <t> x)
	// result: (SRLI [56] (SLLI <t> [56] x))
	for {
		t := v.Type
		x := v_0
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
