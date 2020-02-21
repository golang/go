// Code generated from gen/Wasm.rules; DO NOT EDIT.
// generated with: cd gen; go run *.go

package ssa

import "cmd/internal/objabi"
import "cmd/compile/internal/types"

func rewriteValueWasm(v *Value) bool {
	switch v.Op {
	case OpAbs:
		return rewriteValueWasm_OpAbs(v)
	case OpAdd16:
		return rewriteValueWasm_OpAdd16(v)
	case OpAdd32:
		return rewriteValueWasm_OpAdd32(v)
	case OpAdd32F:
		return rewriteValueWasm_OpAdd32F(v)
	case OpAdd64:
		return rewriteValueWasm_OpAdd64(v)
	case OpAdd64F:
		return rewriteValueWasm_OpAdd64F(v)
	case OpAdd8:
		return rewriteValueWasm_OpAdd8(v)
	case OpAddPtr:
		return rewriteValueWasm_OpAddPtr(v)
	case OpAddr:
		return rewriteValueWasm_OpAddr(v)
	case OpAnd16:
		return rewriteValueWasm_OpAnd16(v)
	case OpAnd32:
		return rewriteValueWasm_OpAnd32(v)
	case OpAnd64:
		return rewriteValueWasm_OpAnd64(v)
	case OpAnd8:
		return rewriteValueWasm_OpAnd8(v)
	case OpAndB:
		return rewriteValueWasm_OpAndB(v)
	case OpBitLen64:
		return rewriteValueWasm_OpBitLen64(v)
	case OpCeil:
		return rewriteValueWasm_OpCeil(v)
	case OpClosureCall:
		return rewriteValueWasm_OpClosureCall(v)
	case OpCom16:
		return rewriteValueWasm_OpCom16(v)
	case OpCom32:
		return rewriteValueWasm_OpCom32(v)
	case OpCom64:
		return rewriteValueWasm_OpCom64(v)
	case OpCom8:
		return rewriteValueWasm_OpCom8(v)
	case OpCondSelect:
		return rewriteValueWasm_OpCondSelect(v)
	case OpConst16:
		return rewriteValueWasm_OpConst16(v)
	case OpConst32:
		return rewriteValueWasm_OpConst32(v)
	case OpConst32F:
		return rewriteValueWasm_OpConst32F(v)
	case OpConst64:
		return rewriteValueWasm_OpConst64(v)
	case OpConst64F:
		return rewriteValueWasm_OpConst64F(v)
	case OpConst8:
		return rewriteValueWasm_OpConst8(v)
	case OpConstBool:
		return rewriteValueWasm_OpConstBool(v)
	case OpConstNil:
		return rewriteValueWasm_OpConstNil(v)
	case OpConvert:
		return rewriteValueWasm_OpConvert(v)
	case OpCopysign:
		return rewriteValueWasm_OpCopysign(v)
	case OpCtz16:
		return rewriteValueWasm_OpCtz16(v)
	case OpCtz16NonZero:
		return rewriteValueWasm_OpCtz16NonZero(v)
	case OpCtz32:
		return rewriteValueWasm_OpCtz32(v)
	case OpCtz32NonZero:
		return rewriteValueWasm_OpCtz32NonZero(v)
	case OpCtz64:
		return rewriteValueWasm_OpCtz64(v)
	case OpCtz64NonZero:
		return rewriteValueWasm_OpCtz64NonZero(v)
	case OpCtz8:
		return rewriteValueWasm_OpCtz8(v)
	case OpCtz8NonZero:
		return rewriteValueWasm_OpCtz8NonZero(v)
	case OpCvt32Fto32:
		return rewriteValueWasm_OpCvt32Fto32(v)
	case OpCvt32Fto32U:
		return rewriteValueWasm_OpCvt32Fto32U(v)
	case OpCvt32Fto64:
		return rewriteValueWasm_OpCvt32Fto64(v)
	case OpCvt32Fto64F:
		return rewriteValueWasm_OpCvt32Fto64F(v)
	case OpCvt32Fto64U:
		return rewriteValueWasm_OpCvt32Fto64U(v)
	case OpCvt32Uto32F:
		return rewriteValueWasm_OpCvt32Uto32F(v)
	case OpCvt32Uto64F:
		return rewriteValueWasm_OpCvt32Uto64F(v)
	case OpCvt32to32F:
		return rewriteValueWasm_OpCvt32to32F(v)
	case OpCvt32to64F:
		return rewriteValueWasm_OpCvt32to64F(v)
	case OpCvt64Fto32:
		return rewriteValueWasm_OpCvt64Fto32(v)
	case OpCvt64Fto32F:
		return rewriteValueWasm_OpCvt64Fto32F(v)
	case OpCvt64Fto32U:
		return rewriteValueWasm_OpCvt64Fto32U(v)
	case OpCvt64Fto64:
		return rewriteValueWasm_OpCvt64Fto64(v)
	case OpCvt64Fto64U:
		return rewriteValueWasm_OpCvt64Fto64U(v)
	case OpCvt64Uto32F:
		return rewriteValueWasm_OpCvt64Uto32F(v)
	case OpCvt64Uto64F:
		return rewriteValueWasm_OpCvt64Uto64F(v)
	case OpCvt64to32F:
		return rewriteValueWasm_OpCvt64to32F(v)
	case OpCvt64to64F:
		return rewriteValueWasm_OpCvt64to64F(v)
	case OpDiv16:
		return rewriteValueWasm_OpDiv16(v)
	case OpDiv16u:
		return rewriteValueWasm_OpDiv16u(v)
	case OpDiv32:
		return rewriteValueWasm_OpDiv32(v)
	case OpDiv32F:
		return rewriteValueWasm_OpDiv32F(v)
	case OpDiv32u:
		return rewriteValueWasm_OpDiv32u(v)
	case OpDiv64:
		return rewriteValueWasm_OpDiv64(v)
	case OpDiv64F:
		return rewriteValueWasm_OpDiv64F(v)
	case OpDiv64u:
		return rewriteValueWasm_OpDiv64u(v)
	case OpDiv8:
		return rewriteValueWasm_OpDiv8(v)
	case OpDiv8u:
		return rewriteValueWasm_OpDiv8u(v)
	case OpEq16:
		return rewriteValueWasm_OpEq16(v)
	case OpEq32:
		return rewriteValueWasm_OpEq32(v)
	case OpEq32F:
		return rewriteValueWasm_OpEq32F(v)
	case OpEq64:
		return rewriteValueWasm_OpEq64(v)
	case OpEq64F:
		return rewriteValueWasm_OpEq64F(v)
	case OpEq8:
		return rewriteValueWasm_OpEq8(v)
	case OpEqB:
		return rewriteValueWasm_OpEqB(v)
	case OpEqPtr:
		return rewriteValueWasm_OpEqPtr(v)
	case OpFloor:
		return rewriteValueWasm_OpFloor(v)
	case OpGeq16:
		return rewriteValueWasm_OpGeq16(v)
	case OpGeq16U:
		return rewriteValueWasm_OpGeq16U(v)
	case OpGeq32:
		return rewriteValueWasm_OpGeq32(v)
	case OpGeq32F:
		return rewriteValueWasm_OpGeq32F(v)
	case OpGeq32U:
		return rewriteValueWasm_OpGeq32U(v)
	case OpGeq64:
		return rewriteValueWasm_OpGeq64(v)
	case OpGeq64F:
		return rewriteValueWasm_OpGeq64F(v)
	case OpGeq64U:
		return rewriteValueWasm_OpGeq64U(v)
	case OpGeq8:
		return rewriteValueWasm_OpGeq8(v)
	case OpGeq8U:
		return rewriteValueWasm_OpGeq8U(v)
	case OpGetCallerPC:
		return rewriteValueWasm_OpGetCallerPC(v)
	case OpGetCallerSP:
		return rewriteValueWasm_OpGetCallerSP(v)
	case OpGetClosurePtr:
		return rewriteValueWasm_OpGetClosurePtr(v)
	case OpGreater16:
		return rewriteValueWasm_OpGreater16(v)
	case OpGreater16U:
		return rewriteValueWasm_OpGreater16U(v)
	case OpGreater32:
		return rewriteValueWasm_OpGreater32(v)
	case OpGreater32F:
		return rewriteValueWasm_OpGreater32F(v)
	case OpGreater32U:
		return rewriteValueWasm_OpGreater32U(v)
	case OpGreater64:
		return rewriteValueWasm_OpGreater64(v)
	case OpGreater64F:
		return rewriteValueWasm_OpGreater64F(v)
	case OpGreater64U:
		return rewriteValueWasm_OpGreater64U(v)
	case OpGreater8:
		return rewriteValueWasm_OpGreater8(v)
	case OpGreater8U:
		return rewriteValueWasm_OpGreater8U(v)
	case OpInterCall:
		return rewriteValueWasm_OpInterCall(v)
	case OpIsInBounds:
		return rewriteValueWasm_OpIsInBounds(v)
	case OpIsNonNil:
		return rewriteValueWasm_OpIsNonNil(v)
	case OpIsSliceInBounds:
		return rewriteValueWasm_OpIsSliceInBounds(v)
	case OpLeq16:
		return rewriteValueWasm_OpLeq16(v)
	case OpLeq16U:
		return rewriteValueWasm_OpLeq16U(v)
	case OpLeq32:
		return rewriteValueWasm_OpLeq32(v)
	case OpLeq32F:
		return rewriteValueWasm_OpLeq32F(v)
	case OpLeq32U:
		return rewriteValueWasm_OpLeq32U(v)
	case OpLeq64:
		return rewriteValueWasm_OpLeq64(v)
	case OpLeq64F:
		return rewriteValueWasm_OpLeq64F(v)
	case OpLeq64U:
		return rewriteValueWasm_OpLeq64U(v)
	case OpLeq8:
		return rewriteValueWasm_OpLeq8(v)
	case OpLeq8U:
		return rewriteValueWasm_OpLeq8U(v)
	case OpLess16:
		return rewriteValueWasm_OpLess16(v)
	case OpLess16U:
		return rewriteValueWasm_OpLess16U(v)
	case OpLess32:
		return rewriteValueWasm_OpLess32(v)
	case OpLess32F:
		return rewriteValueWasm_OpLess32F(v)
	case OpLess32U:
		return rewriteValueWasm_OpLess32U(v)
	case OpLess64:
		return rewriteValueWasm_OpLess64(v)
	case OpLess64F:
		return rewriteValueWasm_OpLess64F(v)
	case OpLess64U:
		return rewriteValueWasm_OpLess64U(v)
	case OpLess8:
		return rewriteValueWasm_OpLess8(v)
	case OpLess8U:
		return rewriteValueWasm_OpLess8U(v)
	case OpLoad:
		return rewriteValueWasm_OpLoad(v)
	case OpLocalAddr:
		return rewriteValueWasm_OpLocalAddr(v)
	case OpLsh16x16:
		return rewriteValueWasm_OpLsh16x16(v)
	case OpLsh16x32:
		return rewriteValueWasm_OpLsh16x32(v)
	case OpLsh16x64:
		return rewriteValueWasm_OpLsh16x64(v)
	case OpLsh16x8:
		return rewriteValueWasm_OpLsh16x8(v)
	case OpLsh32x16:
		return rewriteValueWasm_OpLsh32x16(v)
	case OpLsh32x32:
		return rewriteValueWasm_OpLsh32x32(v)
	case OpLsh32x64:
		return rewriteValueWasm_OpLsh32x64(v)
	case OpLsh32x8:
		return rewriteValueWasm_OpLsh32x8(v)
	case OpLsh64x16:
		return rewriteValueWasm_OpLsh64x16(v)
	case OpLsh64x32:
		return rewriteValueWasm_OpLsh64x32(v)
	case OpLsh64x64:
		return rewriteValueWasm_OpLsh64x64(v)
	case OpLsh64x8:
		return rewriteValueWasm_OpLsh64x8(v)
	case OpLsh8x16:
		return rewriteValueWasm_OpLsh8x16(v)
	case OpLsh8x32:
		return rewriteValueWasm_OpLsh8x32(v)
	case OpLsh8x64:
		return rewriteValueWasm_OpLsh8x64(v)
	case OpLsh8x8:
		return rewriteValueWasm_OpLsh8x8(v)
	case OpMod16:
		return rewriteValueWasm_OpMod16(v)
	case OpMod16u:
		return rewriteValueWasm_OpMod16u(v)
	case OpMod32:
		return rewriteValueWasm_OpMod32(v)
	case OpMod32u:
		return rewriteValueWasm_OpMod32u(v)
	case OpMod64:
		return rewriteValueWasm_OpMod64(v)
	case OpMod64u:
		return rewriteValueWasm_OpMod64u(v)
	case OpMod8:
		return rewriteValueWasm_OpMod8(v)
	case OpMod8u:
		return rewriteValueWasm_OpMod8u(v)
	case OpMove:
		return rewriteValueWasm_OpMove(v)
	case OpMul16:
		return rewriteValueWasm_OpMul16(v)
	case OpMul32:
		return rewriteValueWasm_OpMul32(v)
	case OpMul32F:
		return rewriteValueWasm_OpMul32F(v)
	case OpMul64:
		return rewriteValueWasm_OpMul64(v)
	case OpMul64F:
		return rewriteValueWasm_OpMul64F(v)
	case OpMul8:
		return rewriteValueWasm_OpMul8(v)
	case OpNeg16:
		return rewriteValueWasm_OpNeg16(v)
	case OpNeg32:
		return rewriteValueWasm_OpNeg32(v)
	case OpNeg32F:
		return rewriteValueWasm_OpNeg32F(v)
	case OpNeg64:
		return rewriteValueWasm_OpNeg64(v)
	case OpNeg64F:
		return rewriteValueWasm_OpNeg64F(v)
	case OpNeg8:
		return rewriteValueWasm_OpNeg8(v)
	case OpNeq16:
		return rewriteValueWasm_OpNeq16(v)
	case OpNeq32:
		return rewriteValueWasm_OpNeq32(v)
	case OpNeq32F:
		return rewriteValueWasm_OpNeq32F(v)
	case OpNeq64:
		return rewriteValueWasm_OpNeq64(v)
	case OpNeq64F:
		return rewriteValueWasm_OpNeq64F(v)
	case OpNeq8:
		return rewriteValueWasm_OpNeq8(v)
	case OpNeqB:
		return rewriteValueWasm_OpNeqB(v)
	case OpNeqPtr:
		return rewriteValueWasm_OpNeqPtr(v)
	case OpNilCheck:
		return rewriteValueWasm_OpNilCheck(v)
	case OpNot:
		return rewriteValueWasm_OpNot(v)
	case OpOffPtr:
		return rewriteValueWasm_OpOffPtr(v)
	case OpOr16:
		return rewriteValueWasm_OpOr16(v)
	case OpOr32:
		return rewriteValueWasm_OpOr32(v)
	case OpOr64:
		return rewriteValueWasm_OpOr64(v)
	case OpOr8:
		return rewriteValueWasm_OpOr8(v)
	case OpOrB:
		return rewriteValueWasm_OpOrB(v)
	case OpPopCount16:
		return rewriteValueWasm_OpPopCount16(v)
	case OpPopCount32:
		return rewriteValueWasm_OpPopCount32(v)
	case OpPopCount64:
		return rewriteValueWasm_OpPopCount64(v)
	case OpPopCount8:
		return rewriteValueWasm_OpPopCount8(v)
	case OpRotateLeft16:
		return rewriteValueWasm_OpRotateLeft16(v)
	case OpRotateLeft32:
		return rewriteValueWasm_OpRotateLeft32(v)
	case OpRotateLeft64:
		return rewriteValueWasm_OpRotateLeft64(v)
	case OpRotateLeft8:
		return rewriteValueWasm_OpRotateLeft8(v)
	case OpRound32F:
		return rewriteValueWasm_OpRound32F(v)
	case OpRound64F:
		return rewriteValueWasm_OpRound64F(v)
	case OpRoundToEven:
		return rewriteValueWasm_OpRoundToEven(v)
	case OpRsh16Ux16:
		return rewriteValueWasm_OpRsh16Ux16(v)
	case OpRsh16Ux32:
		return rewriteValueWasm_OpRsh16Ux32(v)
	case OpRsh16Ux64:
		return rewriteValueWasm_OpRsh16Ux64(v)
	case OpRsh16Ux8:
		return rewriteValueWasm_OpRsh16Ux8(v)
	case OpRsh16x16:
		return rewriteValueWasm_OpRsh16x16(v)
	case OpRsh16x32:
		return rewriteValueWasm_OpRsh16x32(v)
	case OpRsh16x64:
		return rewriteValueWasm_OpRsh16x64(v)
	case OpRsh16x8:
		return rewriteValueWasm_OpRsh16x8(v)
	case OpRsh32Ux16:
		return rewriteValueWasm_OpRsh32Ux16(v)
	case OpRsh32Ux32:
		return rewriteValueWasm_OpRsh32Ux32(v)
	case OpRsh32Ux64:
		return rewriteValueWasm_OpRsh32Ux64(v)
	case OpRsh32Ux8:
		return rewriteValueWasm_OpRsh32Ux8(v)
	case OpRsh32x16:
		return rewriteValueWasm_OpRsh32x16(v)
	case OpRsh32x32:
		return rewriteValueWasm_OpRsh32x32(v)
	case OpRsh32x64:
		return rewriteValueWasm_OpRsh32x64(v)
	case OpRsh32x8:
		return rewriteValueWasm_OpRsh32x8(v)
	case OpRsh64Ux16:
		return rewriteValueWasm_OpRsh64Ux16(v)
	case OpRsh64Ux32:
		return rewriteValueWasm_OpRsh64Ux32(v)
	case OpRsh64Ux64:
		return rewriteValueWasm_OpRsh64Ux64(v)
	case OpRsh64Ux8:
		return rewriteValueWasm_OpRsh64Ux8(v)
	case OpRsh64x16:
		return rewriteValueWasm_OpRsh64x16(v)
	case OpRsh64x32:
		return rewriteValueWasm_OpRsh64x32(v)
	case OpRsh64x64:
		return rewriteValueWasm_OpRsh64x64(v)
	case OpRsh64x8:
		return rewriteValueWasm_OpRsh64x8(v)
	case OpRsh8Ux16:
		return rewriteValueWasm_OpRsh8Ux16(v)
	case OpRsh8Ux32:
		return rewriteValueWasm_OpRsh8Ux32(v)
	case OpRsh8Ux64:
		return rewriteValueWasm_OpRsh8Ux64(v)
	case OpRsh8Ux8:
		return rewriteValueWasm_OpRsh8Ux8(v)
	case OpRsh8x16:
		return rewriteValueWasm_OpRsh8x16(v)
	case OpRsh8x32:
		return rewriteValueWasm_OpRsh8x32(v)
	case OpRsh8x64:
		return rewriteValueWasm_OpRsh8x64(v)
	case OpRsh8x8:
		return rewriteValueWasm_OpRsh8x8(v)
	case OpSignExt16to32:
		return rewriteValueWasm_OpSignExt16to32(v)
	case OpSignExt16to64:
		return rewriteValueWasm_OpSignExt16to64(v)
	case OpSignExt32to64:
		return rewriteValueWasm_OpSignExt32to64(v)
	case OpSignExt8to16:
		return rewriteValueWasm_OpSignExt8to16(v)
	case OpSignExt8to32:
		return rewriteValueWasm_OpSignExt8to32(v)
	case OpSignExt8to64:
		return rewriteValueWasm_OpSignExt8to64(v)
	case OpSlicemask:
		return rewriteValueWasm_OpSlicemask(v)
	case OpSqrt:
		return rewriteValueWasm_OpSqrt(v)
	case OpStaticCall:
		return rewriteValueWasm_OpStaticCall(v)
	case OpStore:
		return rewriteValueWasm_OpStore(v)
	case OpSub16:
		return rewriteValueWasm_OpSub16(v)
	case OpSub32:
		return rewriteValueWasm_OpSub32(v)
	case OpSub32F:
		return rewriteValueWasm_OpSub32F(v)
	case OpSub64:
		return rewriteValueWasm_OpSub64(v)
	case OpSub64F:
		return rewriteValueWasm_OpSub64F(v)
	case OpSub8:
		return rewriteValueWasm_OpSub8(v)
	case OpSubPtr:
		return rewriteValueWasm_OpSubPtr(v)
	case OpTrunc:
		return rewriteValueWasm_OpTrunc(v)
	case OpTrunc16to8:
		return rewriteValueWasm_OpTrunc16to8(v)
	case OpTrunc32to16:
		return rewriteValueWasm_OpTrunc32to16(v)
	case OpTrunc32to8:
		return rewriteValueWasm_OpTrunc32to8(v)
	case OpTrunc64to16:
		return rewriteValueWasm_OpTrunc64to16(v)
	case OpTrunc64to32:
		return rewriteValueWasm_OpTrunc64to32(v)
	case OpTrunc64to8:
		return rewriteValueWasm_OpTrunc64to8(v)
	case OpWB:
		return rewriteValueWasm_OpWB(v)
	case OpWasmF64Add:
		return rewriteValueWasm_OpWasmF64Add(v)
	case OpWasmF64Mul:
		return rewriteValueWasm_OpWasmF64Mul(v)
	case OpWasmI64Add:
		return rewriteValueWasm_OpWasmI64Add(v)
	case OpWasmI64AddConst:
		return rewriteValueWasm_OpWasmI64AddConst(v)
	case OpWasmI64And:
		return rewriteValueWasm_OpWasmI64And(v)
	case OpWasmI64Eq:
		return rewriteValueWasm_OpWasmI64Eq(v)
	case OpWasmI64Eqz:
		return rewriteValueWasm_OpWasmI64Eqz(v)
	case OpWasmI64Load:
		return rewriteValueWasm_OpWasmI64Load(v)
	case OpWasmI64Load16S:
		return rewriteValueWasm_OpWasmI64Load16S(v)
	case OpWasmI64Load16U:
		return rewriteValueWasm_OpWasmI64Load16U(v)
	case OpWasmI64Load32S:
		return rewriteValueWasm_OpWasmI64Load32S(v)
	case OpWasmI64Load32U:
		return rewriteValueWasm_OpWasmI64Load32U(v)
	case OpWasmI64Load8S:
		return rewriteValueWasm_OpWasmI64Load8S(v)
	case OpWasmI64Load8U:
		return rewriteValueWasm_OpWasmI64Load8U(v)
	case OpWasmI64Mul:
		return rewriteValueWasm_OpWasmI64Mul(v)
	case OpWasmI64Ne:
		return rewriteValueWasm_OpWasmI64Ne(v)
	case OpWasmI64Or:
		return rewriteValueWasm_OpWasmI64Or(v)
	case OpWasmI64Shl:
		return rewriteValueWasm_OpWasmI64Shl(v)
	case OpWasmI64ShrS:
		return rewriteValueWasm_OpWasmI64ShrS(v)
	case OpWasmI64ShrU:
		return rewriteValueWasm_OpWasmI64ShrU(v)
	case OpWasmI64Store:
		return rewriteValueWasm_OpWasmI64Store(v)
	case OpWasmI64Store16:
		return rewriteValueWasm_OpWasmI64Store16(v)
	case OpWasmI64Store32:
		return rewriteValueWasm_OpWasmI64Store32(v)
	case OpWasmI64Store8:
		return rewriteValueWasm_OpWasmI64Store8(v)
	case OpWasmI64Xor:
		return rewriteValueWasm_OpWasmI64Xor(v)
	case OpXor16:
		return rewriteValueWasm_OpXor16(v)
	case OpXor32:
		return rewriteValueWasm_OpXor32(v)
	case OpXor64:
		return rewriteValueWasm_OpXor64(v)
	case OpXor8:
		return rewriteValueWasm_OpXor8(v)
	case OpZero:
		return rewriteValueWasm_OpZero(v)
	case OpZeroExt16to32:
		return rewriteValueWasm_OpZeroExt16to32(v)
	case OpZeroExt16to64:
		return rewriteValueWasm_OpZeroExt16to64(v)
	case OpZeroExt32to64:
		return rewriteValueWasm_OpZeroExt32to64(v)
	case OpZeroExt8to16:
		return rewriteValueWasm_OpZeroExt8to16(v)
	case OpZeroExt8to32:
		return rewriteValueWasm_OpZeroExt8to32(v)
	case OpZeroExt8to64:
		return rewriteValueWasm_OpZeroExt8to64(v)
	}
	return false
}
func rewriteValueWasm_OpAbs(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Abs x)
	// result: (F64Abs x)
	for {
		x := v_0
		v.reset(OpWasmF64Abs)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpAdd16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Add16 x y)
	// result: (I64Add x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Add)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpAdd32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Add32 x y)
	// result: (I64Add x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Add)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpAdd32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Add32F x y)
	// result: (F32Add x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF32Add)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpAdd64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Add64 x y)
	// result: (I64Add x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Add)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpAdd64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Add64F x y)
	// result: (F64Add x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF64Add)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpAdd8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Add8 x y)
	// result: (I64Add x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Add)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpAddPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AddPtr x y)
	// result: (I64Add x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Add)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpAddr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Addr {sym} base)
	// result: (LoweredAddr {sym} base)
	for {
		sym := v.Aux
		base := v_0
		v.reset(OpWasmLoweredAddr)
		v.Aux = sym
		v.AddArg(base)
		return true
	}
}
func rewriteValueWasm_OpAnd16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (And16 x y)
	// result: (I64And x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64And)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpAnd32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (And32 x y)
	// result: (I64And x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64And)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpAnd64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (And64 x y)
	// result: (I64And x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64And)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpAnd8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (And8 x y)
	// result: (I64And x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64And)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpAndB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AndB x y)
	// result: (I64And x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64And)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpBitLen64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen64 x)
	// result: (I64Sub (I64Const [64]) (I64Clz x))
	for {
		x := v_0
		v.reset(OpWasmI64Sub)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 64
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpWasmI64Clz, typ.Int64)
		v1.AddArg(x)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpCeil(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Ceil x)
	// result: (F64Ceil x)
	for {
		x := v_0
		v.reset(OpWasmF64Ceil)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpClosureCall(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ClosureCall [argwid] entry closure mem)
	// result: (LoweredClosureCall [argwid] entry closure mem)
	for {
		argwid := v.AuxInt
		entry := v_0
		closure := v_1
		mem := v_2
		v.reset(OpWasmLoweredClosureCall)
		v.AuxInt = argwid
		v.AddArg(entry)
		v.AddArg(closure)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueWasm_OpCom16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com16 x)
	// result: (I64Xor x (I64Const [-1]))
	for {
		x := v_0
		v.reset(OpWasmI64Xor)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = -1
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCom32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com32 x)
	// result: (I64Xor x (I64Const [-1]))
	for {
		x := v_0
		v.reset(OpWasmI64Xor)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = -1
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCom64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com64 x)
	// result: (I64Xor x (I64Const [-1]))
	for {
		x := v_0
		v.reset(OpWasmI64Xor)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = -1
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCom8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com8 x)
	// result: (I64Xor x (I64Const [-1]))
	for {
		x := v_0
		v.reset(OpWasmI64Xor)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = -1
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCondSelect(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CondSelect <t> x y cond)
	// result: (Select <t> x y cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		cond := v_2
		v.reset(OpWasmSelect)
		v.Type = t
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(cond)
		return true
	}
}
func rewriteValueWasm_OpConst16(v *Value) bool {
	// match: (Const16 [val])
	// result: (I64Const [val])
	for {
		val := v.AuxInt
		v.reset(OpWasmI64Const)
		v.AuxInt = val
		return true
	}
}
func rewriteValueWasm_OpConst32(v *Value) bool {
	// match: (Const32 [val])
	// result: (I64Const [val])
	for {
		val := v.AuxInt
		v.reset(OpWasmI64Const)
		v.AuxInt = val
		return true
	}
}
func rewriteValueWasm_OpConst32F(v *Value) bool {
	// match: (Const32F [val])
	// result: (F32Const [val])
	for {
		val := v.AuxInt
		v.reset(OpWasmF32Const)
		v.AuxInt = val
		return true
	}
}
func rewriteValueWasm_OpConst64(v *Value) bool {
	// match: (Const64 [val])
	// result: (I64Const [val])
	for {
		val := v.AuxInt
		v.reset(OpWasmI64Const)
		v.AuxInt = val
		return true
	}
}
func rewriteValueWasm_OpConst64F(v *Value) bool {
	// match: (Const64F [val])
	// result: (F64Const [val])
	for {
		val := v.AuxInt
		v.reset(OpWasmF64Const)
		v.AuxInt = val
		return true
	}
}
func rewriteValueWasm_OpConst8(v *Value) bool {
	// match: (Const8 [val])
	// result: (I64Const [val])
	for {
		val := v.AuxInt
		v.reset(OpWasmI64Const)
		v.AuxInt = val
		return true
	}
}
func rewriteValueWasm_OpConstBool(v *Value) bool {
	// match: (ConstBool [b])
	// result: (I64Const [b])
	for {
		b := v.AuxInt
		v.reset(OpWasmI64Const)
		v.AuxInt = b
		return true
	}
}
func rewriteValueWasm_OpConstNil(v *Value) bool {
	// match: (ConstNil)
	// result: (I64Const [0])
	for {
		v.reset(OpWasmI64Const)
		v.AuxInt = 0
		return true
	}
}
func rewriteValueWasm_OpConvert(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Convert <t> x mem)
	// result: (LoweredConvert <t> x mem)
	for {
		t := v.Type
		x := v_0
		mem := v_1
		v.reset(OpWasmLoweredConvert)
		v.Type = t
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueWasm_OpCopysign(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Copysign x y)
	// result: (F64Copysign x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF64Copysign)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpCtz16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz16 x)
	// result: (I64Ctz (I64Or x (I64Const [0x10000])))
	for {
		x := v_0
		v.reset(OpWasmI64Ctz)
		v0 := b.NewValue0(v.Pos, OpWasmI64Or, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v1.AuxInt = 0x10000
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCtz16NonZero(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Ctz16NonZero x)
	// result: (I64Ctz x)
	for {
		x := v_0
		v.reset(OpWasmI64Ctz)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCtz32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz32 x)
	// result: (I64Ctz (I64Or x (I64Const [0x100000000])))
	for {
		x := v_0
		v.reset(OpWasmI64Ctz)
		v0 := b.NewValue0(v.Pos, OpWasmI64Or, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v1.AuxInt = 0x100000000
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCtz32NonZero(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Ctz32NonZero x)
	// result: (I64Ctz x)
	for {
		x := v_0
		v.reset(OpWasmI64Ctz)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCtz64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Ctz64 x)
	// result: (I64Ctz x)
	for {
		x := v_0
		v.reset(OpWasmI64Ctz)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCtz64NonZero(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Ctz64NonZero x)
	// result: (I64Ctz x)
	for {
		x := v_0
		v.reset(OpWasmI64Ctz)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCtz8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz8 x)
	// result: (I64Ctz (I64Or x (I64Const [0x100])))
	for {
		x := v_0
		v.reset(OpWasmI64Ctz)
		v0 := b.NewValue0(v.Pos, OpWasmI64Or, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v1.AuxInt = 0x100
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCtz8NonZero(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Ctz8NonZero x)
	// result: (I64Ctz x)
	for {
		x := v_0
		v.reset(OpWasmI64Ctz)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCvt32Fto32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt32Fto32 x)
	// result: (I64TruncSatF32S x)
	for {
		x := v_0
		v.reset(OpWasmI64TruncSatF32S)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCvt32Fto32U(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt32Fto32U x)
	// result: (I64TruncSatF32U x)
	for {
		x := v_0
		v.reset(OpWasmI64TruncSatF32U)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCvt32Fto64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt32Fto64 x)
	// result: (I64TruncSatF32S x)
	for {
		x := v_0
		v.reset(OpWasmI64TruncSatF32S)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCvt32Fto64F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt32Fto64F x)
	// result: (F64PromoteF32 x)
	for {
		x := v_0
		v.reset(OpWasmF64PromoteF32)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCvt32Fto64U(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt32Fto64U x)
	// result: (I64TruncSatF32U x)
	for {
		x := v_0
		v.reset(OpWasmI64TruncSatF32U)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCvt32Uto32F(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt32Uto32F x)
	// result: (F32ConvertI64U (ZeroExt32to64 x))
	for {
		x := v_0
		v.reset(OpWasmF32ConvertI64U)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCvt32Uto64F(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt32Uto64F x)
	// result: (F64ConvertI64U (ZeroExt32to64 x))
	for {
		x := v_0
		v.reset(OpWasmF64ConvertI64U)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCvt32to32F(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt32to32F x)
	// result: (F32ConvertI64S (SignExt32to64 x))
	for {
		x := v_0
		v.reset(OpWasmF32ConvertI64S)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCvt32to64F(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt32to64F x)
	// result: (F64ConvertI64S (SignExt32to64 x))
	for {
		x := v_0
		v.reset(OpWasmF64ConvertI64S)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCvt64Fto32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64Fto32 x)
	// result: (I64TruncSatF64S x)
	for {
		x := v_0
		v.reset(OpWasmI64TruncSatF64S)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCvt64Fto32F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64Fto32F x)
	// result: (F32DemoteF64 x)
	for {
		x := v_0
		v.reset(OpWasmF32DemoteF64)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCvt64Fto32U(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64Fto32U x)
	// result: (I64TruncSatF64U x)
	for {
		x := v_0
		v.reset(OpWasmI64TruncSatF64U)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCvt64Fto64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64Fto64 x)
	// result: (I64TruncSatF64S x)
	for {
		x := v_0
		v.reset(OpWasmI64TruncSatF64S)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCvt64Fto64U(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64Fto64U x)
	// result: (I64TruncSatF64U x)
	for {
		x := v_0
		v.reset(OpWasmI64TruncSatF64U)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCvt64Uto32F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64Uto32F x)
	// result: (F32ConvertI64U x)
	for {
		x := v_0
		v.reset(OpWasmF32ConvertI64U)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCvt64Uto64F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64Uto64F x)
	// result: (F64ConvertI64U x)
	for {
		x := v_0
		v.reset(OpWasmF64ConvertI64U)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCvt64to32F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64to32F x)
	// result: (F32ConvertI64S x)
	for {
		x := v_0
		v.reset(OpWasmF32ConvertI64S)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpCvt64to64F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Cvt64to64F x)
	// result: (F64ConvertI64S x)
	for {
		x := v_0
		v.reset(OpWasmF64ConvertI64S)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpDiv16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 x y)
	// result: (I64DivS (SignExt16to64 x) (SignExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64DivS)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpDiv16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// result: (I64DivU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64DivU)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpDiv32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32 x y)
	// result: (I64DivS (SignExt32to64 x) (SignExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64DivS)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpDiv32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div32F x y)
	// result: (F32Div x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF32Div)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpDiv32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32u x y)
	// result: (I64DivU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64DivU)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpDiv64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div64 x y)
	// result: (I64DivS x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64DivS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpDiv64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div64F x y)
	// result: (F64Div x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF64Div)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpDiv64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div64u x y)
	// result: (I64DivU x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64DivU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpDiv8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (I64DivS (SignExt8to64 x) (SignExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64DivS)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpDiv8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// result: (I64DivU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64DivU)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpEq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq16 x y)
	// result: (I64Eq (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Eq)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpEq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq32 x y)
	// result: (I64Eq (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Eq)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpEq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Eq32F x y)
	// result: (F32Eq x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF32Eq)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpEq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Eq64 x y)
	// result: (I64Eq x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Eq)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpEq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Eq64F x y)
	// result: (F64Eq x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF64Eq)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpEq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq8 x y)
	// result: (I64Eq (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Eq)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpEqB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (EqB x y)
	// result: (I64Eq x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Eq)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpEqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (EqPtr x y)
	// result: (I64Eq x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Eq)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpFloor(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Floor x)
	// result: (F64Floor x)
	for {
		x := v_0
		v.reset(OpWasmF64Floor)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpGeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq16 x y)
	// result: (I64GeS (SignExt16to64 x) (SignExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64GeS)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpGeq16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq16U x y)
	// result: (I64GeU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64GeU)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpGeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq32 x y)
	// result: (I64GeS (SignExt32to64 x) (SignExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64GeS)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpGeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Geq32F x y)
	// result: (F32Ge x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF32Ge)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpGeq32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq32U x y)
	// result: (I64GeU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64GeU)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpGeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Geq64 x y)
	// result: (I64GeS x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64GeS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpGeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Geq64F x y)
	// result: (F64Ge x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF64Ge)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpGeq64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Geq64U x y)
	// result: (I64GeU x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64GeU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpGeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq8 x y)
	// result: (I64GeS (SignExt8to64 x) (SignExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64GeS)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpGeq8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq8U x y)
	// result: (I64GeU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64GeU)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpGetCallerPC(v *Value) bool {
	// match: (GetCallerPC)
	// result: (LoweredGetCallerPC)
	for {
		v.reset(OpWasmLoweredGetCallerPC)
		return true
	}
}
func rewriteValueWasm_OpGetCallerSP(v *Value) bool {
	// match: (GetCallerSP)
	// result: (LoweredGetCallerSP)
	for {
		v.reset(OpWasmLoweredGetCallerSP)
		return true
	}
}
func rewriteValueWasm_OpGetClosurePtr(v *Value) bool {
	// match: (GetClosurePtr)
	// result: (LoweredGetClosurePtr)
	for {
		v.reset(OpWasmLoweredGetClosurePtr)
		return true
	}
}
func rewriteValueWasm_OpGreater16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Greater16 x y)
	// result: (I64GtS (SignExt16to64 x) (SignExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64GtS)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpGreater16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Greater16U x y)
	// result: (I64GtU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64GtU)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpGreater32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Greater32 x y)
	// result: (I64GtS (SignExt32to64 x) (SignExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64GtS)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpGreater32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Greater32F x y)
	// result: (F32Gt x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF32Gt)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpGreater32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Greater32U x y)
	// result: (I64GtU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64GtU)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpGreater64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Greater64 x y)
	// result: (I64GtS x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64GtS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpGreater64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Greater64F x y)
	// result: (F64Gt x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF64Gt)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpGreater64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Greater64U x y)
	// result: (I64GtU x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64GtU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpGreater8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Greater8 x y)
	// result: (I64GtS (SignExt8to64 x) (SignExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64GtS)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpGreater8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Greater8U x y)
	// result: (I64GtU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64GtU)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpInterCall(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (InterCall [argwid] entry mem)
	// result: (LoweredInterCall [argwid] entry mem)
	for {
		argwid := v.AuxInt
		entry := v_0
		mem := v_1
		v.reset(OpWasmLoweredInterCall)
		v.AuxInt = argwid
		v.AddArg(entry)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueWasm_OpIsInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IsInBounds idx len)
	// result: (I64LtU idx len)
	for {
		idx := v_0
		len := v_1
		v.reset(OpWasmI64LtU)
		v.AddArg(idx)
		v.AddArg(len)
		return true
	}
}
func rewriteValueWasm_OpIsNonNil(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (IsNonNil p)
	// result: (I64Eqz (I64Eqz p))
	for {
		p := v_0
		v.reset(OpWasmI64Eqz)
		v0 := b.NewValue0(v.Pos, OpWasmI64Eqz, typ.Bool)
		v0.AddArg(p)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpIsSliceInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IsSliceInBounds idx len)
	// result: (I64LeU idx len)
	for {
		idx := v_0
		len := v_1
		v.reset(OpWasmI64LeU)
		v.AddArg(idx)
		v.AddArg(len)
		return true
	}
}
func rewriteValueWasm_OpLeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16 x y)
	// result: (I64LeS (SignExt16to64 x) (SignExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64LeS)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpLeq16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16U x y)
	// result: (I64LeU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64LeU)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpLeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32 x y)
	// result: (I64LeS (SignExt32to64 x) (SignExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64LeS)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpLeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Leq32F x y)
	// result: (F32Le x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF32Le)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpLeq32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32U x y)
	// result: (I64LeU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64LeU)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpLeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Leq64 x y)
	// result: (I64LeS x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64LeS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpLeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Leq64F x y)
	// result: (F64Le x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF64Le)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpLeq64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Leq64U x y)
	// result: (I64LeU x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64LeU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpLeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8 x y)
	// result: (I64LeS (SignExt8to64 x) (SignExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64LeS)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpLeq8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8U x y)
	// result: (I64LeU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64LeU)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpLess16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16 x y)
	// result: (I64LtS (SignExt16to64 x) (SignExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64LtS)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpLess16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16U x y)
	// result: (I64LtU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64LtU)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpLess32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32 x y)
	// result: (I64LtS (SignExt32to64 x) (SignExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64LtS)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpLess32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less32F x y)
	// result: (F32Lt x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF32Lt)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpLess32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32U x y)
	// result: (I64LtU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64LtU)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpLess64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less64 x y)
	// result: (I64LtS x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64LtS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpLess64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less64F x y)
	// result: (F64Lt x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF64Lt)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpLess64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less64U x y)
	// result: (I64LtU x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64LtU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpLess8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8 x y)
	// result: (I64LtS (SignExt8to64 x) (SignExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64LtS)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpLess8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8U x y)
	// result: (I64LtU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64LtU)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpLoad(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Load <t> ptr mem)
	// cond: is32BitFloat(t)
	// result: (F32Load ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is32BitFloat(t)) {
			break
		}
		v.reset(OpWasmF32Load)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is64BitFloat(t)
	// result: (F64Load ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is64BitFloat(t)) {
			break
		}
		v.reset(OpWasmF64Load)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 8
	// result: (I64Load ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 8) {
			break
		}
		v.reset(OpWasmI64Load)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 4 && !t.IsSigned()
	// result: (I64Load32U ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 4 && !t.IsSigned()) {
			break
		}
		v.reset(OpWasmI64Load32U)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 4 && t.IsSigned()
	// result: (I64Load32S ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 4 && t.IsSigned()) {
			break
		}
		v.reset(OpWasmI64Load32S)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 2 && !t.IsSigned()
	// result: (I64Load16U ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 2 && !t.IsSigned()) {
			break
		}
		v.reset(OpWasmI64Load16U)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 2 && t.IsSigned()
	// result: (I64Load16S ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 2 && t.IsSigned()) {
			break
		}
		v.reset(OpWasmI64Load16S)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 1 && !t.IsSigned()
	// result: (I64Load8U ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 1 && !t.IsSigned()) {
			break
		}
		v.reset(OpWasmI64Load8U)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 1 && t.IsSigned()
	// result: (I64Load8S ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 1 && t.IsSigned()) {
			break
		}
		v.reset(OpWasmI64Load8S)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpLocalAddr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (LocalAddr {sym} base _)
	// result: (LoweredAddr {sym} base)
	for {
		sym := v.Aux
		base := v_0
		v.reset(OpWasmLoweredAddr)
		v.Aux = sym
		v.AddArg(base)
		return true
	}
}
func rewriteValueWasm_OpLsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x16 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt16to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpLsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpLsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x32 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt32to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpLsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpLsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Lsh16x64 [c] x y)
	// result: (Lsh64x64 [c] x y)
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpLsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpLsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x8 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt8to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpLsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpLsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x16 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt16to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpLsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpLsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x32 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt32to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpLsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpLsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Lsh32x64 [c] x y)
	// result: (Lsh64x64 [c] x y)
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpLsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpLsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x8 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt8to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpLsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpLsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x16 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt16to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpLsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpLsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x32 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt32to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpLsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpLsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x64 x y)
	// cond: shiftIsBounded(v)
	// result: (I64Shl x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpWasmI64Shl)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Lsh64x64 x (I64Const [c]))
	// cond: uint64(c) < 64
	// result: (I64Shl x (I64Const [c]))
	for {
		x := v_0
		if v_1.Op != OpWasmI64Const {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 64) {
			break
		}
		v.reset(OpWasmI64Shl)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = c
		v.AddArg(v0)
		return true
	}
	// match: (Lsh64x64 x (I64Const [c]))
	// cond: uint64(c) >= 64
	// result: (I64Const [0])
	for {
		if v_1.Op != OpWasmI64Const {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 64) {
			break
		}
		v.reset(OpWasmI64Const)
		v.AuxInt = 0
		return true
	}
	// match: (Lsh64x64 x y)
	// result: (Select (I64Shl x y) (I64Const [0]) (I64LtU y (I64Const [64])))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmSelect)
		v0 := b.NewValue0(v.Pos, OpWasmI64Shl, typ.Int64)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v1.AuxInt = 0
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpWasmI64LtU, typ.Bool)
		v2.AddArg(y)
		v3 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v3.AuxInt = 64
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueWasm_OpLsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x8 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt8to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpLsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpLsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x16 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt16to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpLsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpLsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x32 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt32to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpLsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpLsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Lsh8x64 [c] x y)
	// result: (Lsh64x64 [c] x y)
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpLsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpLsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x8 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt8to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpLsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpMod16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16 x y)
	// result: (I64RemS (SignExt16to64 x) (SignExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64RemS)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpMod16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16u x y)
	// result: (I64RemU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64RemU)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpMod32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32 x y)
	// result: (I64RemS (SignExt32to64 x) (SignExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64RemS)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpMod32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32u x y)
	// result: (I64RemU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64RemU)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpMod64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mod64 x y)
	// result: (I64RemS x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64RemS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpMod64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mod64u x y)
	// result: (I64RemU x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64RemU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpMod8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// result: (I64RemS (SignExt8to64 x) (SignExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64RemS)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpMod8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// result: (I64RemU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64RemU)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpMove(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
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
	// result: (I64Store8 dst (I64Load8U src mem) mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpWasmI64Store8)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpWasmI64Load8U, typ.UInt8)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [2] dst src mem)
	// result: (I64Store16 dst (I64Load16U src mem) mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpWasmI64Store16)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpWasmI64Load16U, typ.UInt16)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [4] dst src mem)
	// result: (I64Store32 dst (I64Load32U src mem) mem)
	for {
		if v.AuxInt != 4 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpWasmI64Store32)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpWasmI64Load32U, typ.UInt32)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [8] dst src mem)
	// result: (I64Store dst (I64Load src mem) mem)
	for {
		if v.AuxInt != 8 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpWasmI64Store)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpWasmI64Load, typ.UInt64)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [16] dst src mem)
	// result: (I64Store [8] dst (I64Load [8] src mem) (I64Store dst (I64Load src mem) mem))
	for {
		if v.AuxInt != 16 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpWasmI64Store)
		v.AuxInt = 8
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpWasmI64Load, typ.UInt64)
		v0.AuxInt = 8
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpWasmI64Store, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpWasmI64Load, typ.UInt64)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Move [3] dst src mem)
	// result: (I64Store8 [2] dst (I64Load8U [2] src mem) (I64Store16 dst (I64Load16U src mem) mem))
	for {
		if v.AuxInt != 3 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpWasmI64Store8)
		v.AuxInt = 2
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpWasmI64Load8U, typ.UInt8)
		v0.AuxInt = 2
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpWasmI64Store16, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpWasmI64Load16U, typ.UInt16)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Move [5] dst src mem)
	// result: (I64Store8 [4] dst (I64Load8U [4] src mem) (I64Store32 dst (I64Load32U src mem) mem))
	for {
		if v.AuxInt != 5 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpWasmI64Store8)
		v.AuxInt = 4
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpWasmI64Load8U, typ.UInt8)
		v0.AuxInt = 4
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpWasmI64Store32, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpWasmI64Load32U, typ.UInt32)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Move [6] dst src mem)
	// result: (I64Store16 [4] dst (I64Load16U [4] src mem) (I64Store32 dst (I64Load32U src mem) mem))
	for {
		if v.AuxInt != 6 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpWasmI64Store16)
		v.AuxInt = 4
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpWasmI64Load16U, typ.UInt16)
		v0.AuxInt = 4
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpWasmI64Store32, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpWasmI64Load32U, typ.UInt32)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Move [7] dst src mem)
	// result: (I64Store32 [3] dst (I64Load32U [3] src mem) (I64Store32 dst (I64Load32U src mem) mem))
	for {
		if v.AuxInt != 7 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpWasmI64Store32)
		v.AuxInt = 3
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpWasmI64Load32U, typ.UInt32)
		v0.AuxInt = 3
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpWasmI64Store32, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpWasmI64Load32U, typ.UInt32)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 8 && s < 16
	// result: (I64Store [s-8] dst (I64Load [s-8] src mem) (I64Store dst (I64Load src mem) mem))
	for {
		s := v.AuxInt
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 8 && s < 16) {
			break
		}
		v.reset(OpWasmI64Store)
		v.AuxInt = s - 8
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpWasmI64Load, typ.UInt64)
		v0.AuxInt = s - 8
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpWasmI64Store, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpWasmI64Load, typ.UInt64)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 16 && s%16 != 0 && s%16 <= 8
	// result: (Move [s-s%16] (OffPtr <dst.Type> dst [s%16]) (OffPtr <src.Type> src [s%16]) (I64Store dst (I64Load src mem) mem))
	for {
		s := v.AuxInt
		dst := v_0
		src := v_1
		mem := v_2
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
		v2 := b.NewValue0(v.Pos, OpWasmI64Store, types.TypeMem)
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, OpWasmI64Load, typ.UInt64)
		v3.AddArg(src)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v2.AddArg(mem)
		v.AddArg(v2)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 16 && s%16 != 0 && s%16 > 8
	// result: (Move [s-s%16] (OffPtr <dst.Type> dst [s%16]) (OffPtr <src.Type> src [s%16]) (I64Store [8] dst (I64Load [8] src mem) (I64Store dst (I64Load src mem) mem)))
	for {
		s := v.AuxInt
		dst := v_0
		src := v_1
		mem := v_2
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
		v2 := b.NewValue0(v.Pos, OpWasmI64Store, types.TypeMem)
		v2.AuxInt = 8
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, OpWasmI64Load, typ.UInt64)
		v3.AuxInt = 8
		v3.AddArg(src)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpWasmI64Store, types.TypeMem)
		v4.AddArg(dst)
		v5 := b.NewValue0(v.Pos, OpWasmI64Load, typ.UInt64)
		v5.AddArg(src)
		v5.AddArg(mem)
		v4.AddArg(v5)
		v4.AddArg(mem)
		v2.AddArg(v4)
		v.AddArg(v2)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s%8 == 0
	// result: (LoweredMove [s/8] dst src mem)
	for {
		s := v.AuxInt
		dst := v_0
		src := v_1
		mem := v_2
		if !(s%8 == 0) {
			break
		}
		v.reset(OpWasmLoweredMove)
		v.AuxInt = s / 8
		v.AddArg(dst)
		v.AddArg(src)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpMul16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mul16 x y)
	// result: (I64Mul x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Mul)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpMul32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mul32 x y)
	// result: (I64Mul x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Mul)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpMul32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mul32F x y)
	// result: (F32Mul x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF32Mul)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpMul64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mul64 x y)
	// result: (I64Mul x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Mul)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpMul64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mul64F x y)
	// result: (F64Mul x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF64Mul)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpMul8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mul8 x y)
	// result: (I64Mul x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Mul)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpNeg16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg16 x)
	// result: (I64Sub (I64Const [0]) x)
	for {
		x := v_0
		v.reset(OpWasmI64Sub)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpNeg32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg32 x)
	// result: (I64Sub (I64Const [0]) x)
	for {
		x := v_0
		v.reset(OpWasmI64Sub)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpNeg32F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Neg32F x)
	// result: (F32Neg x)
	for {
		x := v_0
		v.reset(OpWasmF32Neg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpNeg64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg64 x)
	// result: (I64Sub (I64Const [0]) x)
	for {
		x := v_0
		v.reset(OpWasmI64Sub)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpNeg64F(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Neg64F x)
	// result: (F64Neg x)
	for {
		x := v_0
		v.reset(OpWasmF64Neg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpNeg8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg8 x)
	// result: (I64Sub (I64Const [0]) x)
	for {
		x := v_0
		v.reset(OpWasmI64Sub)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpNeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq16 x y)
	// result: (I64Ne (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Ne)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpNeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq32 x y)
	// result: (I64Ne (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Ne)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpNeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Neq32F x y)
	// result: (F32Ne x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF32Ne)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpNeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Neq64 x y)
	// result: (I64Ne x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Ne)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpNeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Neq64F x y)
	// result: (F64Ne x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF64Ne)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpNeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq8 x y)
	// result: (I64Ne (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Ne)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpNeqB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (NeqB x y)
	// result: (I64Ne x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Ne)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpNeqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (NeqPtr x y)
	// result: (I64Ne x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Ne)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpNilCheck(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (NilCheck ptr mem)
	// result: (LoweredNilCheck ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.reset(OpWasmLoweredNilCheck)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueWasm_OpNot(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Not x)
	// result: (I64Eqz x)
	for {
		x := v_0
		v.reset(OpWasmI64Eqz)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpOffPtr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (OffPtr [off] ptr)
	// result: (I64AddConst [off] ptr)
	for {
		off := v.AuxInt
		ptr := v_0
		v.reset(OpWasmI64AddConst)
		v.AuxInt = off
		v.AddArg(ptr)
		return true
	}
}
func rewriteValueWasm_OpOr16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Or16 x y)
	// result: (I64Or x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Or)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpOr32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Or32 x y)
	// result: (I64Or x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Or)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpOr64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Or64 x y)
	// result: (I64Or x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Or)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpOr8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Or8 x y)
	// result: (I64Or x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Or)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpOrB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (OrB x y)
	// result: (I64Or x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Or)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpPopCount16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount16 x)
	// result: (I64Popcnt (ZeroExt16to64 x))
	for {
		x := v_0
		v.reset(OpWasmI64Popcnt)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpPopCount32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount32 x)
	// result: (I64Popcnt (ZeroExt32to64 x))
	for {
		x := v_0
		v.reset(OpWasmI64Popcnt)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpPopCount64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (PopCount64 x)
	// result: (I64Popcnt x)
	for {
		x := v_0
		v.reset(OpWasmI64Popcnt)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpPopCount8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount8 x)
	// result: (I64Popcnt (ZeroExt8to64 x))
	for {
		x := v_0
		v.reset(OpWasmI64Popcnt)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpRotateLeft16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft16 <t> x (I64Const [c]))
	// result: (Or16 (Lsh16x64 <t> x (I64Const [c&15])) (Rsh16Ux64 <t> x (I64Const [-c&15])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpWasmI64Const {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr16)
		v0 := b.NewValue0(v.Pos, OpLsh16x64, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v1.AuxInt = c & 15
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpRsh16Ux64, t)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v3.AuxInt = -c & 15
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
	return false
}
func rewriteValueWasm_OpRotateLeft32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (RotateLeft32 x y)
	// result: (I32Rotl x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI32Rotl)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpRotateLeft64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (RotateLeft64 x y)
	// result: (I64Rotl x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Rotl)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpRotateLeft8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft8 <t> x (I64Const [c]))
	// result: (Or8 (Lsh8x64 <t> x (I64Const [c&7])) (Rsh8Ux64 <t> x (I64Const [-c&7])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpWasmI64Const {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr8)
		v0 := b.NewValue0(v.Pos, OpLsh8x64, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v1.AuxInt = c & 7
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpRsh8Ux64, t)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v3.AuxInt = -c & 7
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
	return false
}
func rewriteValueWasm_OpRound32F(v *Value) bool {
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
func rewriteValueWasm_OpRound64F(v *Value) bool {
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
func rewriteValueWasm_OpRoundToEven(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEven x)
	// result: (F64Nearest x)
	for {
		x := v_0
		v.reset(OpWasmF64Nearest)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpRsh16Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux16 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64Ux64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpRsh16Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt16to64 x) (ZeroExt32to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64Ux64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpRsh16Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt16to64 x) y)
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64Ux64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpRsh16Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux8 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt16to64 x) (ZeroExt8to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64Ux64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpRsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x16 [c] x y)
	// result: (Rsh64x64 [c] (SignExt16to64 x) (ZeroExt16to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64x64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpRsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x32 [c] x y)
	// result: (Rsh64x64 [c] (SignExt16to64 x) (ZeroExt32to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64x64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpRsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 [c] x y)
	// result: (Rsh64x64 [c] (SignExt16to64 x) y)
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64x64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpRsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x8 [c] x y)
	// result: (Rsh64x64 [c] (SignExt16to64 x) (ZeroExt8to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64x64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpRsh32Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt32to64 x) (ZeroExt16to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64Ux64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpRsh32Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux32 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64Ux64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpRsh32Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux64 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt32to64 x) y)
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64Ux64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpRsh32Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux8 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt32to64 x) (ZeroExt8to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64Ux64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpRsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x16 [c] x y)
	// result: (Rsh64x64 [c] (SignExt32to64 x) (ZeroExt16to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64x64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpRsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x32 [c] x y)
	// result: (Rsh64x64 [c] (SignExt32to64 x) (ZeroExt32to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64x64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpRsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x64 [c] x y)
	// result: (Rsh64x64 [c] (SignExt32to64 x) y)
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64x64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpRsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x8 [c] x y)
	// result: (Rsh64x64 [c] (SignExt32to64 x) (ZeroExt8to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64x64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpRsh64Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux16 [c] x y)
	// result: (Rsh64Ux64 [c] x (ZeroExt16to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64Ux64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpRsh64Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux32 [c] x y)
	// result: (Rsh64Ux64 [c] x (ZeroExt32to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64Ux64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpRsh64Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux64 x y)
	// cond: shiftIsBounded(v)
	// result: (I64ShrU x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpWasmI64ShrU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Rsh64Ux64 x (I64Const [c]))
	// cond: uint64(c) < 64
	// result: (I64ShrU x (I64Const [c]))
	for {
		x := v_0
		if v_1.Op != OpWasmI64Const {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 64) {
			break
		}
		v.reset(OpWasmI64ShrU)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = c
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64Ux64 x (I64Const [c]))
	// cond: uint64(c) >= 64
	// result: (I64Const [0])
	for {
		if v_1.Op != OpWasmI64Const {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 64) {
			break
		}
		v.reset(OpWasmI64Const)
		v.AuxInt = 0
		return true
	}
	// match: (Rsh64Ux64 x y)
	// result: (Select (I64ShrU x y) (I64Const [0]) (I64LtU y (I64Const [64])))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmSelect)
		v0 := b.NewValue0(v.Pos, OpWasmI64ShrU, typ.Int64)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v1.AuxInt = 0
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpWasmI64LtU, typ.Bool)
		v2.AddArg(y)
		v3 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v3.AuxInt = 64
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueWasm_OpRsh64Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux8 [c] x y)
	// result: (Rsh64Ux64 [c] x (ZeroExt8to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64Ux64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpRsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x16 [c] x y)
	// result: (Rsh64x64 [c] x (ZeroExt16to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpRsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x32 [c] x y)
	// result: (Rsh64x64 [c] x (ZeroExt32to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpRsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x64 x y)
	// cond: shiftIsBounded(v)
	// result: (I64ShrS x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpWasmI64ShrS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Rsh64x64 x (I64Const [c]))
	// cond: uint64(c) < 64
	// result: (I64ShrS x (I64Const [c]))
	for {
		x := v_0
		if v_1.Op != OpWasmI64Const {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 64) {
			break
		}
		v.reset(OpWasmI64ShrS)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = c
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64x64 x (I64Const [c]))
	// cond: uint64(c) >= 64
	// result: (I64ShrS x (I64Const [63]))
	for {
		x := v_0
		if v_1.Op != OpWasmI64Const {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 64) {
			break
		}
		v.reset(OpWasmI64ShrS)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 63
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64x64 x y)
	// result: (I64ShrS x (Select <typ.Int64> y (I64Const [63]) (I64LtU y (I64Const [64]))))
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64ShrS)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpWasmSelect, typ.Int64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v1.AuxInt = 63
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpWasmI64LtU, typ.Bool)
		v2.AddArg(y)
		v3 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v3.AuxInt = 64
		v2.AddArg(v3)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpRsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x8 [c] x y)
	// result: (Rsh64x64 [c] x (ZeroExt8to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64x64)
		v.AuxInt = c
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpRsh8Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux16 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt8to64 x) (ZeroExt16to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64Ux64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpRsh8Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt8to64 x) (ZeroExt32to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64Ux64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpRsh8Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt8to64 x) y)
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64Ux64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpRsh8Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux8 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64Ux64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpRsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x16 [c] x y)
	// result: (Rsh64x64 [c] (SignExt8to64 x) (ZeroExt16to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64x64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpRsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x32 [c] x y)
	// result: (Rsh64x64 [c] (SignExt8to64 x) (ZeroExt32to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64x64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpRsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x64 [c] x y)
	// result: (Rsh64x64 [c] (SignExt8to64 x) y)
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64x64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpRsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x8 [c] x y)
	// result: (Rsh64x64 [c] (SignExt8to64 x) (ZeroExt8to64 y))
	for {
		c := v.AuxInt
		x := v_0
		y := v_1
		v.reset(OpRsh64x64)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueWasm_OpSignExt16to32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SignExt16to32 x:(I64Load16S _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpWasmI64Load16S {
			break
		}
		_ = x.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (SignExt16to32 x)
	// cond: objabi.GOWASM.SignExt
	// result: (I64Extend16S x)
	for {
		x := v_0
		if !(objabi.GOWASM.SignExt) {
			break
		}
		v.reset(OpWasmI64Extend16S)
		v.AddArg(x)
		return true
	}
	// match: (SignExt16to32 x)
	// result: (I64ShrS (I64Shl x (I64Const [48])) (I64Const [48]))
	for {
		x := v_0
		v.reset(OpWasmI64ShrS)
		v0 := b.NewValue0(v.Pos, OpWasmI64Shl, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v1.AuxInt = 48
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v2.AuxInt = 48
		v.AddArg(v2)
		return true
	}
}
func rewriteValueWasm_OpSignExt16to64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SignExt16to64 x:(I64Load16S _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpWasmI64Load16S {
			break
		}
		_ = x.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (SignExt16to64 x)
	// cond: objabi.GOWASM.SignExt
	// result: (I64Extend16S x)
	for {
		x := v_0
		if !(objabi.GOWASM.SignExt) {
			break
		}
		v.reset(OpWasmI64Extend16S)
		v.AddArg(x)
		return true
	}
	// match: (SignExt16to64 x)
	// result: (I64ShrS (I64Shl x (I64Const [48])) (I64Const [48]))
	for {
		x := v_0
		v.reset(OpWasmI64ShrS)
		v0 := b.NewValue0(v.Pos, OpWasmI64Shl, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v1.AuxInt = 48
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v2.AuxInt = 48
		v.AddArg(v2)
		return true
	}
}
func rewriteValueWasm_OpSignExt32to64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SignExt32to64 x:(I64Load32S _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpWasmI64Load32S {
			break
		}
		_ = x.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (SignExt32to64 x)
	// cond: objabi.GOWASM.SignExt
	// result: (I64Extend32S x)
	for {
		x := v_0
		if !(objabi.GOWASM.SignExt) {
			break
		}
		v.reset(OpWasmI64Extend32S)
		v.AddArg(x)
		return true
	}
	// match: (SignExt32to64 x)
	// result: (I64ShrS (I64Shl x (I64Const [32])) (I64Const [32]))
	for {
		x := v_0
		v.reset(OpWasmI64ShrS)
		v0 := b.NewValue0(v.Pos, OpWasmI64Shl, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v1.AuxInt = 32
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v2.AuxInt = 32
		v.AddArg(v2)
		return true
	}
}
func rewriteValueWasm_OpSignExt8to16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SignExt8to16 x:(I64Load8S _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpWasmI64Load8S {
			break
		}
		_ = x.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (SignExt8to16 x)
	// cond: objabi.GOWASM.SignExt
	// result: (I64Extend8S x)
	for {
		x := v_0
		if !(objabi.GOWASM.SignExt) {
			break
		}
		v.reset(OpWasmI64Extend8S)
		v.AddArg(x)
		return true
	}
	// match: (SignExt8to16 x)
	// result: (I64ShrS (I64Shl x (I64Const [56])) (I64Const [56]))
	for {
		x := v_0
		v.reset(OpWasmI64ShrS)
		v0 := b.NewValue0(v.Pos, OpWasmI64Shl, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v1.AuxInt = 56
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v2.AuxInt = 56
		v.AddArg(v2)
		return true
	}
}
func rewriteValueWasm_OpSignExt8to32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SignExt8to32 x:(I64Load8S _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpWasmI64Load8S {
			break
		}
		_ = x.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (SignExt8to32 x)
	// cond: objabi.GOWASM.SignExt
	// result: (I64Extend8S x)
	for {
		x := v_0
		if !(objabi.GOWASM.SignExt) {
			break
		}
		v.reset(OpWasmI64Extend8S)
		v.AddArg(x)
		return true
	}
	// match: (SignExt8to32 x)
	// result: (I64ShrS (I64Shl x (I64Const [56])) (I64Const [56]))
	for {
		x := v_0
		v.reset(OpWasmI64ShrS)
		v0 := b.NewValue0(v.Pos, OpWasmI64Shl, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v1.AuxInt = 56
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v2.AuxInt = 56
		v.AddArg(v2)
		return true
	}
}
func rewriteValueWasm_OpSignExt8to64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SignExt8to64 x:(I64Load8S _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpWasmI64Load8S {
			break
		}
		_ = x.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (SignExt8to64 x)
	// cond: objabi.GOWASM.SignExt
	// result: (I64Extend8S x)
	for {
		x := v_0
		if !(objabi.GOWASM.SignExt) {
			break
		}
		v.reset(OpWasmI64Extend8S)
		v.AddArg(x)
		return true
	}
	// match: (SignExt8to64 x)
	// result: (I64ShrS (I64Shl x (I64Const [56])) (I64Const [56]))
	for {
		x := v_0
		v.reset(OpWasmI64ShrS)
		v0 := b.NewValue0(v.Pos, OpWasmI64Shl, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v1.AuxInt = 56
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v2.AuxInt = 56
		v.AddArg(v2)
		return true
	}
}
func rewriteValueWasm_OpSlicemask(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Slicemask x)
	// result: (I64ShrS (I64Sub (I64Const [0]) x) (I64Const [63]))
	for {
		x := v_0
		v.reset(OpWasmI64ShrS)
		v0 := b.NewValue0(v.Pos, OpWasmI64Sub, typ.Int64)
		v1 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v1.AuxInt = 0
		v0.AddArg(v1)
		v0.AddArg(x)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v2.AuxInt = 63
		v.AddArg(v2)
		return true
	}
}
func rewriteValueWasm_OpSqrt(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Sqrt x)
	// result: (F64Sqrt x)
	for {
		x := v_0
		v.reset(OpWasmF64Sqrt)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpStaticCall(v *Value) bool {
	v_0 := v.Args[0]
	// match: (StaticCall [argwid] {target} mem)
	// result: (LoweredStaticCall [argwid] {target} mem)
	for {
		argwid := v.AuxInt
		target := v.Aux
		mem := v_0
		v.reset(OpWasmLoweredStaticCall)
		v.AuxInt = argwid
		v.Aux = target
		v.AddArg(mem)
		return true
	}
}
func rewriteValueWasm_OpStore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Store {t} ptr val mem)
	// cond: is64BitFloat(t.(*types.Type))
	// result: (F64Store ptr val mem)
	for {
		t := v.Aux
		ptr := v_0
		val := v_1
		mem := v_2
		if !(is64BitFloat(t.(*types.Type))) {
			break
		}
		v.reset(OpWasmF64Store)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: is32BitFloat(t.(*types.Type))
	// result: (F32Store ptr val mem)
	for {
		t := v.Aux
		ptr := v_0
		val := v_1
		mem := v_2
		if !(is32BitFloat(t.(*types.Type))) {
			break
		}
		v.reset(OpWasmF32Store)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 8
	// result: (I64Store ptr val mem)
	for {
		t := v.Aux
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.(*types.Type).Size() == 8) {
			break
		}
		v.reset(OpWasmI64Store)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 4
	// result: (I64Store32 ptr val mem)
	for {
		t := v.Aux
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.(*types.Type).Size() == 4) {
			break
		}
		v.reset(OpWasmI64Store32)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 2
	// result: (I64Store16 ptr val mem)
	for {
		t := v.Aux
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.(*types.Type).Size() == 2) {
			break
		}
		v.reset(OpWasmI64Store16)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 1
	// result: (I64Store8 ptr val mem)
	for {
		t := v.Aux
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.(*types.Type).Size() == 1) {
			break
		}
		v.reset(OpWasmI64Store8)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpSub16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Sub16 x y)
	// result: (I64Sub x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Sub)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpSub32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Sub32 x y)
	// result: (I64Sub x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Sub)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpSub32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Sub32F x y)
	// result: (F32Sub x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF32Sub)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpSub64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Sub64 x y)
	// result: (I64Sub x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Sub)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpSub64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Sub64F x y)
	// result: (F64Sub x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmF64Sub)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpSub8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Sub8 x y)
	// result: (I64Sub x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Sub)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpSubPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SubPtr x y)
	// result: (I64Sub x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Sub)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpTrunc(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc x)
	// result: (F64Trunc x)
	for {
		x := v_0
		v.reset(OpWasmF64Trunc)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpTrunc16to8(v *Value) bool {
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
func rewriteValueWasm_OpTrunc32to16(v *Value) bool {
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
func rewriteValueWasm_OpTrunc32to8(v *Value) bool {
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
func rewriteValueWasm_OpTrunc64to16(v *Value) bool {
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
func rewriteValueWasm_OpTrunc64to32(v *Value) bool {
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
func rewriteValueWasm_OpTrunc64to8(v *Value) bool {
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
func rewriteValueWasm_OpWB(v *Value) bool {
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
		v.reset(OpWasmLoweredWB)
		v.Aux = fn
		v.AddArg(destptr)
		v.AddArg(srcptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueWasm_OpWasmF64Add(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (F64Add (F64Const [x]) (F64Const [y]))
	// result: (F64Const [auxFrom64F(auxTo64F(x) + auxTo64F(y))])
	for {
		if v_0.Op != OpWasmF64Const {
			break
		}
		x := v_0.AuxInt
		if v_1.Op != OpWasmF64Const {
			break
		}
		y := v_1.AuxInt
		v.reset(OpWasmF64Const)
		v.AuxInt = auxFrom64F(auxTo64F(x) + auxTo64F(y))
		return true
	}
	// match: (F64Add (F64Const [x]) y)
	// result: (F64Add y (F64Const [x]))
	for {
		if v_0.Op != OpWasmF64Const {
			break
		}
		x := v_0.AuxInt
		y := v_1
		v.reset(OpWasmF64Add)
		v.AddArg(y)
		v0 := b.NewValue0(v.Pos, OpWasmF64Const, typ.Float64)
		v0.AuxInt = x
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmF64Mul(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (F64Mul (F64Const [x]) (F64Const [y]))
	// result: (F64Const [auxFrom64F(auxTo64F(x) * auxTo64F(y))])
	for {
		if v_0.Op != OpWasmF64Const {
			break
		}
		x := v_0.AuxInt
		if v_1.Op != OpWasmF64Const {
			break
		}
		y := v_1.AuxInt
		v.reset(OpWasmF64Const)
		v.AuxInt = auxFrom64F(auxTo64F(x) * auxTo64F(y))
		return true
	}
	// match: (F64Mul (F64Const [x]) y)
	// result: (F64Mul y (F64Const [x]))
	for {
		if v_0.Op != OpWasmF64Const {
			break
		}
		x := v_0.AuxInt
		y := v_1
		v.reset(OpWasmF64Mul)
		v.AddArg(y)
		v0 := b.NewValue0(v.Pos, OpWasmF64Const, typ.Float64)
		v0.AuxInt = x
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Add(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (I64Add (I64Const [x]) (I64Const [y]))
	// result: (I64Const [x + y])
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		if v_1.Op != OpWasmI64Const {
			break
		}
		y := v_1.AuxInt
		v.reset(OpWasmI64Const)
		v.AuxInt = x + y
		return true
	}
	// match: (I64Add (I64Const [x]) y)
	// result: (I64Add y (I64Const [x]))
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		y := v_1
		v.reset(OpWasmI64Add)
		v.AddArg(y)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = x
		v.AddArg(v0)
		return true
	}
	// match: (I64Add x (I64Const [y]))
	// result: (I64AddConst [y] x)
	for {
		x := v_0
		if v_1.Op != OpWasmI64Const {
			break
		}
		y := v_1.AuxInt
		v.reset(OpWasmI64AddConst)
		v.AuxInt = y
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64AddConst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (I64AddConst [0] x)
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
	// match: (I64AddConst [off] (LoweredAddr {sym} [off2] base))
	// cond: isU32Bit(off+off2)
	// result: (LoweredAddr {sym} [off+off2] base)
	for {
		off := v.AuxInt
		if v_0.Op != OpWasmLoweredAddr {
			break
		}
		off2 := v_0.AuxInt
		sym := v_0.Aux
		base := v_0.Args[0]
		if !(isU32Bit(off + off2)) {
			break
		}
		v.reset(OpWasmLoweredAddr)
		v.AuxInt = off + off2
		v.Aux = sym
		v.AddArg(base)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64And(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (I64And (I64Const [x]) (I64Const [y]))
	// result: (I64Const [x & y])
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		if v_1.Op != OpWasmI64Const {
			break
		}
		y := v_1.AuxInt
		v.reset(OpWasmI64Const)
		v.AuxInt = x & y
		return true
	}
	// match: (I64And (I64Const [x]) y)
	// result: (I64And y (I64Const [x]))
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		y := v_1
		v.reset(OpWasmI64And)
		v.AddArg(y)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = x
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Eq(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (I64Eq (I64Const [x]) (I64Const [y]))
	// cond: x == y
	// result: (I64Const [1])
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		if v_1.Op != OpWasmI64Const {
			break
		}
		y := v_1.AuxInt
		if !(x == y) {
			break
		}
		v.reset(OpWasmI64Const)
		v.AuxInt = 1
		return true
	}
	// match: (I64Eq (I64Const [x]) (I64Const [y]))
	// cond: x != y
	// result: (I64Const [0])
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		if v_1.Op != OpWasmI64Const {
			break
		}
		y := v_1.AuxInt
		if !(x != y) {
			break
		}
		v.reset(OpWasmI64Const)
		v.AuxInt = 0
		return true
	}
	// match: (I64Eq (I64Const [x]) y)
	// result: (I64Eq y (I64Const [x]))
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		y := v_1
		v.reset(OpWasmI64Eq)
		v.AddArg(y)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = x
		v.AddArg(v0)
		return true
	}
	// match: (I64Eq x (I64Const [0]))
	// result: (I64Eqz x)
	for {
		x := v_0
		if v_1.Op != OpWasmI64Const || v_1.AuxInt != 0 {
			break
		}
		v.reset(OpWasmI64Eqz)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Eqz(v *Value) bool {
	v_0 := v.Args[0]
	// match: (I64Eqz (I64Eqz (I64Eqz x)))
	// result: (I64Eqz x)
	for {
		if v_0.Op != OpWasmI64Eqz {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpWasmI64Eqz {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpWasmI64Eqz)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Load(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (I64Load [off] (I64AddConst [off2] ptr) mem)
	// cond: isU32Bit(off+off2)
	// result: (I64Load [off+off2] ptr mem)
	for {
		off := v.AuxInt
		if v_0.Op != OpWasmI64AddConst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v_1
		if !(isU32Bit(off + off2)) {
			break
		}
		v.reset(OpWasmI64Load)
		v.AuxInt = off + off2
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (I64Load [off] (LoweredAddr {sym} [off2] (SB)) _)
	// cond: symIsRO(sym) && isU32Bit(off+off2)
	// result: (I64Const [int64(read64(sym, off+off2, config.BigEndian))])
	for {
		off := v.AuxInt
		if v_0.Op != OpWasmLoweredAddr {
			break
		}
		off2 := v_0.AuxInt
		sym := v_0.Aux
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpSB || !(symIsRO(sym) && isU32Bit(off+off2)) {
			break
		}
		v.reset(OpWasmI64Const)
		v.AuxInt = int64(read64(sym, off+off2, config.BigEndian))
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Load16S(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64Load16S [off] (I64AddConst [off2] ptr) mem)
	// cond: isU32Bit(off+off2)
	// result: (I64Load16S [off+off2] ptr mem)
	for {
		off := v.AuxInt
		if v_0.Op != OpWasmI64AddConst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v_1
		if !(isU32Bit(off + off2)) {
			break
		}
		v.reset(OpWasmI64Load16S)
		v.AuxInt = off + off2
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Load16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (I64Load16U [off] (I64AddConst [off2] ptr) mem)
	// cond: isU32Bit(off+off2)
	// result: (I64Load16U [off+off2] ptr mem)
	for {
		off := v.AuxInt
		if v_0.Op != OpWasmI64AddConst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v_1
		if !(isU32Bit(off + off2)) {
			break
		}
		v.reset(OpWasmI64Load16U)
		v.AuxInt = off + off2
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (I64Load16U [off] (LoweredAddr {sym} [off2] (SB)) _)
	// cond: symIsRO(sym) && isU32Bit(off+off2)
	// result: (I64Const [int64(read16(sym, off+off2, config.BigEndian))])
	for {
		off := v.AuxInt
		if v_0.Op != OpWasmLoweredAddr {
			break
		}
		off2 := v_0.AuxInt
		sym := v_0.Aux
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpSB || !(symIsRO(sym) && isU32Bit(off+off2)) {
			break
		}
		v.reset(OpWasmI64Const)
		v.AuxInt = int64(read16(sym, off+off2, config.BigEndian))
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Load32S(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64Load32S [off] (I64AddConst [off2] ptr) mem)
	// cond: isU32Bit(off+off2)
	// result: (I64Load32S [off+off2] ptr mem)
	for {
		off := v.AuxInt
		if v_0.Op != OpWasmI64AddConst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v_1
		if !(isU32Bit(off + off2)) {
			break
		}
		v.reset(OpWasmI64Load32S)
		v.AuxInt = off + off2
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Load32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (I64Load32U [off] (I64AddConst [off2] ptr) mem)
	// cond: isU32Bit(off+off2)
	// result: (I64Load32U [off+off2] ptr mem)
	for {
		off := v.AuxInt
		if v_0.Op != OpWasmI64AddConst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v_1
		if !(isU32Bit(off + off2)) {
			break
		}
		v.reset(OpWasmI64Load32U)
		v.AuxInt = off + off2
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (I64Load32U [off] (LoweredAddr {sym} [off2] (SB)) _)
	// cond: symIsRO(sym) && isU32Bit(off+off2)
	// result: (I64Const [int64(read32(sym, off+off2, config.BigEndian))])
	for {
		off := v.AuxInt
		if v_0.Op != OpWasmLoweredAddr {
			break
		}
		off2 := v_0.AuxInt
		sym := v_0.Aux
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpSB || !(symIsRO(sym) && isU32Bit(off+off2)) {
			break
		}
		v.reset(OpWasmI64Const)
		v.AuxInt = int64(read32(sym, off+off2, config.BigEndian))
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Load8S(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64Load8S [off] (I64AddConst [off2] ptr) mem)
	// cond: isU32Bit(off+off2)
	// result: (I64Load8S [off+off2] ptr mem)
	for {
		off := v.AuxInt
		if v_0.Op != OpWasmI64AddConst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v_1
		if !(isU32Bit(off + off2)) {
			break
		}
		v.reset(OpWasmI64Load8S)
		v.AuxInt = off + off2
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Load8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64Load8U [off] (I64AddConst [off2] ptr) mem)
	// cond: isU32Bit(off+off2)
	// result: (I64Load8U [off+off2] ptr mem)
	for {
		off := v.AuxInt
		if v_0.Op != OpWasmI64AddConst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v_1
		if !(isU32Bit(off + off2)) {
			break
		}
		v.reset(OpWasmI64Load8U)
		v.AuxInt = off + off2
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (I64Load8U [off] (LoweredAddr {sym} [off2] (SB)) _)
	// cond: symIsRO(sym) && isU32Bit(off+off2)
	// result: (I64Const [int64(read8(sym, off+off2))])
	for {
		off := v.AuxInt
		if v_0.Op != OpWasmLoweredAddr {
			break
		}
		off2 := v_0.AuxInt
		sym := v_0.Aux
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpSB || !(symIsRO(sym) && isU32Bit(off+off2)) {
			break
		}
		v.reset(OpWasmI64Const)
		v.AuxInt = int64(read8(sym, off+off2))
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Mul(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (I64Mul (I64Const [x]) (I64Const [y]))
	// result: (I64Const [x * y])
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		if v_1.Op != OpWasmI64Const {
			break
		}
		y := v_1.AuxInt
		v.reset(OpWasmI64Const)
		v.AuxInt = x * y
		return true
	}
	// match: (I64Mul (I64Const [x]) y)
	// result: (I64Mul y (I64Const [x]))
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		y := v_1
		v.reset(OpWasmI64Mul)
		v.AddArg(y)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = x
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Ne(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (I64Ne (I64Const [x]) (I64Const [y]))
	// cond: x == y
	// result: (I64Const [0])
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		if v_1.Op != OpWasmI64Const {
			break
		}
		y := v_1.AuxInt
		if !(x == y) {
			break
		}
		v.reset(OpWasmI64Const)
		v.AuxInt = 0
		return true
	}
	// match: (I64Ne (I64Const [x]) (I64Const [y]))
	// cond: x != y
	// result: (I64Const [1])
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		if v_1.Op != OpWasmI64Const {
			break
		}
		y := v_1.AuxInt
		if !(x != y) {
			break
		}
		v.reset(OpWasmI64Const)
		v.AuxInt = 1
		return true
	}
	// match: (I64Ne (I64Const [x]) y)
	// result: (I64Ne y (I64Const [x]))
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		y := v_1
		v.reset(OpWasmI64Ne)
		v.AddArg(y)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = x
		v.AddArg(v0)
		return true
	}
	// match: (I64Ne x (I64Const [0]))
	// result: (I64Eqz (I64Eqz x))
	for {
		x := v_0
		if v_1.Op != OpWasmI64Const || v_1.AuxInt != 0 {
			break
		}
		v.reset(OpWasmI64Eqz)
		v0 := b.NewValue0(v.Pos, OpWasmI64Eqz, typ.Bool)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Or(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (I64Or (I64Const [x]) (I64Const [y]))
	// result: (I64Const [x | y])
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		if v_1.Op != OpWasmI64Const {
			break
		}
		y := v_1.AuxInt
		v.reset(OpWasmI64Const)
		v.AuxInt = x | y
		return true
	}
	// match: (I64Or (I64Const [x]) y)
	// result: (I64Or y (I64Const [x]))
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		y := v_1
		v.reset(OpWasmI64Or)
		v.AddArg(y)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = x
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Shl(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64Shl (I64Const [x]) (I64Const [y]))
	// result: (I64Const [x << uint64(y)])
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		if v_1.Op != OpWasmI64Const {
			break
		}
		y := v_1.AuxInt
		v.reset(OpWasmI64Const)
		v.AuxInt = x << uint64(y)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64ShrS(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64ShrS (I64Const [x]) (I64Const [y]))
	// result: (I64Const [x >> uint64(y)])
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		if v_1.Op != OpWasmI64Const {
			break
		}
		y := v_1.AuxInt
		v.reset(OpWasmI64Const)
		v.AuxInt = x >> uint64(y)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64ShrU(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64ShrU (I64Const [x]) (I64Const [y]))
	// result: (I64Const [int64(uint64(x) >> uint64(y))])
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		if v_1.Op != OpWasmI64Const {
			break
		}
		y := v_1.AuxInt
		v.reset(OpWasmI64Const)
		v.AuxInt = int64(uint64(x) >> uint64(y))
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Store(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64Store [off] (I64AddConst [off2] ptr) val mem)
	// cond: isU32Bit(off+off2)
	// result: (I64Store [off+off2] ptr val mem)
	for {
		off := v.AuxInt
		if v_0.Op != OpWasmI64AddConst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(isU32Bit(off + off2)) {
			break
		}
		v.reset(OpWasmI64Store)
		v.AuxInt = off + off2
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Store16(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64Store16 [off] (I64AddConst [off2] ptr) val mem)
	// cond: isU32Bit(off+off2)
	// result: (I64Store16 [off+off2] ptr val mem)
	for {
		off := v.AuxInt
		if v_0.Op != OpWasmI64AddConst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(isU32Bit(off + off2)) {
			break
		}
		v.reset(OpWasmI64Store16)
		v.AuxInt = off + off2
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Store32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64Store32 [off] (I64AddConst [off2] ptr) val mem)
	// cond: isU32Bit(off+off2)
	// result: (I64Store32 [off+off2] ptr val mem)
	for {
		off := v.AuxInt
		if v_0.Op != OpWasmI64AddConst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(isU32Bit(off + off2)) {
			break
		}
		v.reset(OpWasmI64Store32)
		v.AuxInt = off + off2
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Store8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64Store8 [off] (I64AddConst [off2] ptr) val mem)
	// cond: isU32Bit(off+off2)
	// result: (I64Store8 [off+off2] ptr val mem)
	for {
		off := v.AuxInt
		if v_0.Op != OpWasmI64AddConst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(isU32Bit(off + off2)) {
			break
		}
		v.reset(OpWasmI64Store8)
		v.AuxInt = off + off2
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Xor(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (I64Xor (I64Const [x]) (I64Const [y]))
	// result: (I64Const [x ^ y])
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		if v_1.Op != OpWasmI64Const {
			break
		}
		y := v_1.AuxInt
		v.reset(OpWasmI64Const)
		v.AuxInt = x ^ y
		return true
	}
	// match: (I64Xor (I64Const [x]) y)
	// result: (I64Xor y (I64Const [x]))
	for {
		if v_0.Op != OpWasmI64Const {
			break
		}
		x := v_0.AuxInt
		y := v_1
		v.reset(OpWasmI64Xor)
		v.AddArg(y)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = x
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueWasm_OpXor16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Xor16 x y)
	// result: (I64Xor x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Xor)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpXor32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Xor32 x y)
	// result: (I64Xor x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Xor)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpXor64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Xor64 x y)
	// result: (I64Xor x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Xor)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpXor8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Xor8 x y)
	// result: (I64Xor x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpWasmI64Xor)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueWasm_OpZero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
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
	// match: (Zero [1] destptr mem)
	// result: (I64Store8 destptr (I64Const [0]) mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpWasmI64Store8)
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [2] destptr mem)
	// result: (I64Store16 destptr (I64Const [0]) mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpWasmI64Store16)
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [4] destptr mem)
	// result: (I64Store32 destptr (I64Const [0]) mem)
	for {
		if v.AuxInt != 4 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpWasmI64Store32)
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [8] destptr mem)
	// result: (I64Store destptr (I64Const [0]) mem)
	for {
		if v.AuxInt != 8 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpWasmI64Store)
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [3] destptr mem)
	// result: (I64Store8 [2] destptr (I64Const [0]) (I64Store16 destptr (I64Const [0]) mem))
	for {
		if v.AuxInt != 3 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpWasmI64Store8)
		v.AuxInt = 2
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpWasmI64Store16, types.TypeMem)
		v1.AddArg(destptr)
		v2 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [5] destptr mem)
	// result: (I64Store8 [4] destptr (I64Const [0]) (I64Store32 destptr (I64Const [0]) mem))
	for {
		if v.AuxInt != 5 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpWasmI64Store8)
		v.AuxInt = 4
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpWasmI64Store32, types.TypeMem)
		v1.AddArg(destptr)
		v2 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [6] destptr mem)
	// result: (I64Store16 [4] destptr (I64Const [0]) (I64Store32 destptr (I64Const [0]) mem))
	for {
		if v.AuxInt != 6 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpWasmI64Store16)
		v.AuxInt = 4
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpWasmI64Store32, types.TypeMem)
		v1.AddArg(destptr)
		v2 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [7] destptr mem)
	// result: (I64Store32 [3] destptr (I64Const [0]) (I64Store32 destptr (I64Const [0]) mem))
	for {
		if v.AuxInt != 7 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpWasmI64Store32)
		v.AuxInt = 3
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpWasmI64Store32, types.TypeMem)
		v1.AddArg(destptr)
		v2 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s%8 != 0 && s > 8
	// result: (Zero [s-s%8] (OffPtr <destptr.Type> destptr [s%8]) (I64Store destptr (I64Const [0]) mem))
	for {
		s := v.AuxInt
		destptr := v_0
		mem := v_1
		if !(s%8 != 0 && s > 8) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = s - s%8
		v0 := b.NewValue0(v.Pos, OpOffPtr, destptr.Type)
		v0.AuxInt = s % 8
		v0.AddArg(destptr)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpWasmI64Store, types.TypeMem)
		v1.AddArg(destptr)
		v2 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [16] destptr mem)
	// result: (I64Store [8] destptr (I64Const [0]) (I64Store destptr (I64Const [0]) mem))
	for {
		if v.AuxInt != 16 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpWasmI64Store)
		v.AuxInt = 8
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpWasmI64Store, types.TypeMem)
		v1.AddArg(destptr)
		v2 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [24] destptr mem)
	// result: (I64Store [16] destptr (I64Const [0]) (I64Store [8] destptr (I64Const [0]) (I64Store destptr (I64Const [0]) mem)))
	for {
		if v.AuxInt != 24 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpWasmI64Store)
		v.AuxInt = 16
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpWasmI64Store, types.TypeMem)
		v1.AuxInt = 8
		v1.AddArg(destptr)
		v2 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpWasmI64Store, types.TypeMem)
		v3.AddArg(destptr)
		v4 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v4.AuxInt = 0
		v3.AddArg(v4)
		v3.AddArg(mem)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [32] destptr mem)
	// result: (I64Store [24] destptr (I64Const [0]) (I64Store [16] destptr (I64Const [0]) (I64Store [8] destptr (I64Const [0]) (I64Store destptr (I64Const [0]) mem))))
	for {
		if v.AuxInt != 32 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpWasmI64Store)
		v.AuxInt = 24
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpWasmI64Store, types.TypeMem)
		v1.AuxInt = 16
		v1.AddArg(destptr)
		v2 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpWasmI64Store, types.TypeMem)
		v3.AuxInt = 8
		v3.AddArg(destptr)
		v4 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v4.AuxInt = 0
		v3.AddArg(v4)
		v5 := b.NewValue0(v.Pos, OpWasmI64Store, types.TypeMem)
		v5.AddArg(destptr)
		v6 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v6.AuxInt = 0
		v5.AddArg(v6)
		v5.AddArg(mem)
		v3.AddArg(v5)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s%8 == 0 && s > 32
	// result: (LoweredZero [s/8] destptr mem)
	for {
		s := v.AuxInt
		destptr := v_0
		mem := v_1
		if !(s%8 == 0 && s > 32) {
			break
		}
		v.reset(OpWasmLoweredZero)
		v.AuxInt = s / 8
		v.AddArg(destptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpZeroExt16to32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ZeroExt16to32 x:(I64Load16U _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpWasmI64Load16U {
			break
		}
		_ = x.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (ZeroExt16to32 x)
	// result: (I64And x (I64Const [0xffff]))
	for {
		x := v_0
		v.reset(OpWasmI64And)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0xffff
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpZeroExt16to64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ZeroExt16to64 x:(I64Load16U _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpWasmI64Load16U {
			break
		}
		_ = x.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (ZeroExt16to64 x)
	// result: (I64And x (I64Const [0xffff]))
	for {
		x := v_0
		v.reset(OpWasmI64And)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0xffff
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpZeroExt32to64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ZeroExt32to64 x:(I64Load32U _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpWasmI64Load32U {
			break
		}
		_ = x.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (ZeroExt32to64 x)
	// result: (I64And x (I64Const [0xffffffff]))
	for {
		x := v_0
		v.reset(OpWasmI64And)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0xffffffff
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpZeroExt8to16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ZeroExt8to16 x:(I64Load8U _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpWasmI64Load8U {
			break
		}
		_ = x.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (ZeroExt8to16 x)
	// result: (I64And x (I64Const [0xff]))
	for {
		x := v_0
		v.reset(OpWasmI64And)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0xff
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpZeroExt8to32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ZeroExt8to32 x:(I64Load8U _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpWasmI64Load8U {
			break
		}
		_ = x.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (ZeroExt8to32 x)
	// result: (I64And x (I64Const [0xff]))
	for {
		x := v_0
		v.reset(OpWasmI64And)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0xff
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpZeroExt8to64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ZeroExt8to64 x:(I64Load8U _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpWasmI64Load8U {
			break
		}
		_ = x.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (ZeroExt8to64 x)
	// result: (I64And x (I64Const [0xff]))
	for {
		x := v_0
		v.reset(OpWasmI64And)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpWasmI64Const, typ.Int64)
		v0.AuxInt = 0xff
		v.AddArg(v0)
		return true
	}
}
func rewriteBlockWasm(b *Block) bool {
	switch b.Kind {
	}
	return false
}
