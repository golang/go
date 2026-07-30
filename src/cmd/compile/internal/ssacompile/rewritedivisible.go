// Code generated from _gen/divisible.rules using 'go generate'; DO NOT EDIT.

package ssacompile

import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa/ssacore"

func rewriteValuedivisible(v *ssacore.Value) bool {
	switch v.Op {
	case ssaop.OpEq16:
		return rewriteValuedivisible_OpEq16(v)
	case ssaop.OpEq32:
		return rewriteValuedivisible_OpEq32(v)
	case ssaop.OpEq64:
		return rewriteValuedivisible_OpEq64(v)
	case ssaop.OpEq8:
		return rewriteValuedivisible_OpEq8(v)
	case ssaop.OpNeq16:
		return rewriteValuedivisible_OpNeq16(v)
	case ssaop.OpNeq32:
		return rewriteValuedivisible_OpNeq32(v)
	case ssaop.OpNeq64:
		return rewriteValuedivisible_OpNeq64(v)
	case ssaop.OpNeq8:
		return rewriteValuedivisible_OpNeq8(v)
	}
	return false
}
func rewriteValuedivisible_OpEq16(v *ssacore.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq16 x (Mul16 <t> (Div16u x (Const16 [c])) (Const16 [c])))
	// cond: x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)
	// result: (Eq16 (And16 <t> x (Const16 <t> [c-1])) (Const16 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul16 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != ssaop.OpDiv16u {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != ssaop.OpConst16 {
					continue
				}
				c := AuxIntToInt16(v_1_0_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst16 || AuxIntToInt16(v_1_1.AuxInt) != c || !(x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)) {
					continue
				}
				v.Reset(ssaop.OpEq16)
				v0 := b.NewValue0(v.Pos, ssaop.OpAnd16, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v1.AuxInt = Int16ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v2.AuxInt = Int16ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq16 x (Mul16 <t> (Div16 x (Const16 [c])) (Const16 [c])))
	// cond: x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)
	// result: (Eq16 (And16 <t> x (Const16 <t> [c-1])) (Const16 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul16 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != ssaop.OpDiv16 {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != ssaop.OpConst16 {
					continue
				}
				c := AuxIntToInt16(v_1_0_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst16 || AuxIntToInt16(v_1_1.AuxInt) != c || !(x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)) {
					continue
				}
				v.Reset(ssaop.OpEq16)
				v0 := b.NewValue0(v.Pos, ssaop.OpAnd16, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v1.AuxInt = Int16ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v2.AuxInt = Int16ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq16 x (Mul16 <t> div:(Div16u x (Const16 [c])) (Const16 [c])))
	// cond: div.Uses == 1 && x.Op != ssaop.OpConst16 && UdivisibleOK16(c)
	// result: (Leq16U (RotateLeft16 <t> (Mul16 <t> x (Const16 <t> [int16(udivisible16(c).M)])) (Const16 <t> [int16(16 - udivisible16(c).K)])) (Const16 <t> [int16(udivisible16(c).Max)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul16 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != ssaop.OpDiv16u {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != ssaop.OpConst16 {
					continue
				}
				c := AuxIntToInt16(div_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst16 || AuxIntToInt16(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != ssaop.OpConst16 && UdivisibleOK16(c)) {
					continue
				}
				v.Reset(ssaop.OpLeq16U)
				v0 := b.NewValue0(v.Pos, ssaop.OpRotateLeft16, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpMul16, t)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v2.AuxInt = Int16ToAuxInt(int16(udivisible16(c).M))
				v1.AddArg2(x, v2)
				v3 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v3.AuxInt = Int16ToAuxInt(int16(16 - udivisible16(c).K))
				v0.AddArg2(v1, v3)
				v4 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v4.AuxInt = Int16ToAuxInt(int16(udivisible16(c).Max))
				v.AddArg2(v0, v4)
				return true
			}
		}
		break
	}
	// match: (Eq16 x (Mul16 <t> div:(Div16 x (Const16 [c])) (Const16 [c])))
	// cond: div.Uses == 1 && x.Op != ssaop.OpConst16 && SdivisibleOK16(c)
	// result: (Leq16U (RotateLeft16 <t> (Add16 <t> (Mul16 <t> x (Const16 <t> [int16(sdivisible16(c).M)])) (Const16 <t> [int16(sdivisible16(c).A)])) (Const16 <t> [int16(16 - sdivisible16(c).K)])) (Const16 <t> [int16(sdivisible16(c).Max)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul16 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != ssaop.OpDiv16 {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != ssaop.OpConst16 {
					continue
				}
				c := AuxIntToInt16(div_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst16 || AuxIntToInt16(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != ssaop.OpConst16 && SdivisibleOK16(c)) {
					continue
				}
				v.Reset(ssaop.OpLeq16U)
				v0 := b.NewValue0(v.Pos, ssaop.OpRotateLeft16, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpAdd16, t)
				v2 := b.NewValue0(v.Pos, ssaop.OpMul16, t)
				v3 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v3.AuxInt = Int16ToAuxInt(int16(sdivisible16(c).M))
				v2.AddArg2(x, v3)
				v4 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v4.AuxInt = Int16ToAuxInt(int16(sdivisible16(c).A))
				v1.AddArg2(v2, v4)
				v5 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v5.AuxInt = Int16ToAuxInt(int16(16 - sdivisible16(c).K))
				v0.AddArg2(v1, v5)
				v6 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v6.AuxInt = Int16ToAuxInt(int16(sdivisible16(c).Max))
				v.AddArg2(v0, v6)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuedivisible_OpEq32(v *ssacore.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32 x (Mul32 <t> (Div32u x (Const32 [c])) (Const32 [c])))
	// cond: x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)
	// result: (Eq32 (And32 <t> x (Const32 <t> [c-1])) (Const32 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul32 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != ssaop.OpDiv32u {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != ssaop.OpConst32 {
					continue
				}
				c := AuxIntToInt32(v_1_0_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst32 || AuxIntToInt32(v_1_1.AuxInt) != c || !(x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)) {
					continue
				}
				v.Reset(ssaop.OpEq32)
				v0 := b.NewValue0(v.Pos, ssaop.OpAnd32, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v1.AuxInt = Int32ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v2.AuxInt = Int32ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq32 x (Mul32 <t> (Div32 x (Const32 [c])) (Const32 [c])))
	// cond: x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)
	// result: (Eq32 (And32 <t> x (Const32 <t> [c-1])) (Const32 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul32 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != ssaop.OpDiv32 {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != ssaop.OpConst32 {
					continue
				}
				c := AuxIntToInt32(v_1_0_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst32 || AuxIntToInt32(v_1_1.AuxInt) != c || !(x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)) {
					continue
				}
				v.Reset(ssaop.OpEq32)
				v0 := b.NewValue0(v.Pos, ssaop.OpAnd32, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v1.AuxInt = Int32ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v2.AuxInt = Int32ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq32 x (Mul32 <t> div:(Div32u x (Const32 [c])) (Const32 [c])))
	// cond: div.Uses == 1 && x.Op != ssaop.OpConst32 && udivisibleOK32(c)
	// result: (Leq32U (RotateLeft32 <t> (Mul32 <t> x (Const32 <t> [int32(udivisible32(c).M)])) (Const32 <t> [int32(32 - udivisible32(c).K)])) (Const32 <t> [int32(udivisible32(c).Max)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul32 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != ssaop.OpDiv32u {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != ssaop.OpConst32 {
					continue
				}
				c := AuxIntToInt32(div_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst32 || AuxIntToInt32(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != ssaop.OpConst32 && udivisibleOK32(c)) {
					continue
				}
				v.Reset(ssaop.OpLeq32U)
				v0 := b.NewValue0(v.Pos, ssaop.OpRotateLeft32, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpMul32, t)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v2.AuxInt = Int32ToAuxInt(int32(udivisible32(c).M))
				v1.AddArg2(x, v2)
				v3 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v3.AuxInt = Int32ToAuxInt(int32(32 - udivisible32(c).K))
				v0.AddArg2(v1, v3)
				v4 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v4.AuxInt = Int32ToAuxInt(int32(udivisible32(c).Max))
				v.AddArg2(v0, v4)
				return true
			}
		}
		break
	}
	// match: (Eq32 x (Mul32 <t> div:(Div32 x (Const32 [c])) (Const32 [c])))
	// cond: div.Uses == 1 && x.Op != ssaop.OpConst32 && sdivisibleOK32(c)
	// result: (Leq32U (RotateLeft32 <t> (Add32 <t> (Mul32 <t> x (Const32 <t> [int32(sdivisible32(c).M)])) (Const32 <t> [int32(sdivisible32(c).A)])) (Const32 <t> [int32(32 - sdivisible32(c).K)])) (Const32 <t> [int32(sdivisible32(c).Max)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul32 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != ssaop.OpDiv32 {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != ssaop.OpConst32 {
					continue
				}
				c := AuxIntToInt32(div_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst32 || AuxIntToInt32(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != ssaop.OpConst32 && sdivisibleOK32(c)) {
					continue
				}
				v.Reset(ssaop.OpLeq32U)
				v0 := b.NewValue0(v.Pos, ssaop.OpRotateLeft32, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpAdd32, t)
				v2 := b.NewValue0(v.Pos, ssaop.OpMul32, t)
				v3 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v3.AuxInt = Int32ToAuxInt(int32(sdivisible32(c).M))
				v2.AddArg2(x, v3)
				v4 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v4.AuxInt = Int32ToAuxInt(int32(sdivisible32(c).A))
				v1.AddArg2(v2, v4)
				v5 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v5.AuxInt = Int32ToAuxInt(int32(32 - sdivisible32(c).K))
				v0.AddArg2(v1, v5)
				v6 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v6.AuxInt = Int32ToAuxInt(int32(sdivisible32(c).Max))
				v.AddArg2(v0, v6)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuedivisible_OpEq64(v *ssacore.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64 x (Mul64 <t> (Div64u x (Const64 [c])) (Const64 [c])))
	// cond: x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)
	// result: (Eq64 (And64 <t> x (Const64 <t> [c-1])) (Const64 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul64 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != ssaop.OpDiv64u {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != ssaop.OpConst64 {
					continue
				}
				c := AuxIntToInt64(v_1_0_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst64 || AuxIntToInt64(v_1_1.AuxInt) != c || !(x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)) {
					continue
				}
				v.Reset(ssaop.OpEq64)
				v0 := b.NewValue0(v.Pos, ssaop.OpAnd64, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v1.AuxInt = Int64ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v2.AuxInt = Int64ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq64 x (Mul64 <t> (Div64 x (Const64 [c])) (Const64 [c])))
	// cond: x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)
	// result: (Eq64 (And64 <t> x (Const64 <t> [c-1])) (Const64 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul64 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != ssaop.OpDiv64 {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != ssaop.OpConst64 {
					continue
				}
				c := AuxIntToInt64(v_1_0_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst64 || AuxIntToInt64(v_1_1.AuxInt) != c || !(x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)) {
					continue
				}
				v.Reset(ssaop.OpEq64)
				v0 := b.NewValue0(v.Pos, ssaop.OpAnd64, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v1.AuxInt = Int64ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v2.AuxInt = Int64ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq64 x (Mul64 <t> div:(Div64u x (Const64 [c])) (Const64 [c])))
	// cond: div.Uses == 1 && x.Op != ssaop.OpConst64 && udivisibleOK64(c)
	// result: (Leq64U (RotateLeft64 <t> (Mul64 <t> x (Const64 <t> [int64(udivisible64(c).M)])) (Const64 <t> [int64(64 - udivisible64(c).K)])) (Const64 <t> [int64(udivisible64(c).Max)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul64 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != ssaop.OpDiv64u {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != ssaop.OpConst64 {
					continue
				}
				c := AuxIntToInt64(div_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst64 || AuxIntToInt64(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != ssaop.OpConst64 && udivisibleOK64(c)) {
					continue
				}
				v.Reset(ssaop.OpLeq64U)
				v0 := b.NewValue0(v.Pos, ssaop.OpRotateLeft64, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpMul64, t)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v2.AuxInt = Int64ToAuxInt(int64(udivisible64(c).M))
				v1.AddArg2(x, v2)
				v3 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v3.AuxInt = Int64ToAuxInt(int64(64 - udivisible64(c).K))
				v0.AddArg2(v1, v3)
				v4 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v4.AuxInt = Int64ToAuxInt(int64(udivisible64(c).Max))
				v.AddArg2(v0, v4)
				return true
			}
		}
		break
	}
	// match: (Eq64 x (Mul64 <t> div:(Div64 x (Const64 [c])) (Const64 [c])))
	// cond: div.Uses == 1 && x.Op != ssaop.OpConst64 && sdivisibleOK64(c)
	// result: (Leq64U (RotateLeft64 <t> (Add64 <t> (Mul64 <t> x (Const64 <t> [int64(sdivisible64(c).M)])) (Const64 <t> [int64(sdivisible64(c).A)])) (Const64 <t> [int64(64 - sdivisible64(c).K)])) (Const64 <t> [int64(sdivisible64(c).Max)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul64 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != ssaop.OpDiv64 {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != ssaop.OpConst64 {
					continue
				}
				c := AuxIntToInt64(div_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst64 || AuxIntToInt64(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != ssaop.OpConst64 && sdivisibleOK64(c)) {
					continue
				}
				v.Reset(ssaop.OpLeq64U)
				v0 := b.NewValue0(v.Pos, ssaop.OpRotateLeft64, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpAdd64, t)
				v2 := b.NewValue0(v.Pos, ssaop.OpMul64, t)
				v3 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v3.AuxInt = Int64ToAuxInt(int64(sdivisible64(c).M))
				v2.AddArg2(x, v3)
				v4 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v4.AuxInt = Int64ToAuxInt(int64(sdivisible64(c).A))
				v1.AddArg2(v2, v4)
				v5 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v5.AuxInt = Int64ToAuxInt(int64(64 - sdivisible64(c).K))
				v0.AddArg2(v1, v5)
				v6 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v6.AuxInt = Int64ToAuxInt(int64(sdivisible64(c).Max))
				v.AddArg2(v0, v6)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuedivisible_OpEq8(v *ssacore.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq8 x (Mul8 <t> (Div8u x (Const8 [c])) (Const8 [c])))
	// cond: x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)
	// result: (Eq8 (And8 <t> x (Const8 <t> [c-1])) (Const8 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul8 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != ssaop.OpDiv8u {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != ssaop.OpConst8 {
					continue
				}
				c := AuxIntToInt8(v_1_0_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst8 || AuxIntToInt8(v_1_1.AuxInt) != c || !(x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)) {
					continue
				}
				v.Reset(ssaop.OpEq8)
				v0 := b.NewValue0(v.Pos, ssaop.OpAnd8, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v1.AuxInt = Int8ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v2.AuxInt = Int8ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq8 x (Mul8 <t> (Div8 x (Const8 [c])) (Const8 [c])))
	// cond: x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)
	// result: (Eq8 (And8 <t> x (Const8 <t> [c-1])) (Const8 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul8 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != ssaop.OpDiv8 {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != ssaop.OpConst8 {
					continue
				}
				c := AuxIntToInt8(v_1_0_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst8 || AuxIntToInt8(v_1_1.AuxInt) != c || !(x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)) {
					continue
				}
				v.Reset(ssaop.OpEq8)
				v0 := b.NewValue0(v.Pos, ssaop.OpAnd8, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v1.AuxInt = Int8ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v2.AuxInt = Int8ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq8 x (Mul8 <t> div:(Div8u x (Const8 [c])) (Const8 [c])))
	// cond: div.Uses == 1 && x.Op != ssaop.OpConst8 && UdivisibleOK8(c)
	// result: (Leq8U (RotateLeft8 <t> (Mul8 <t> x (Const8 <t> [int8(udivisible8(c).M)])) (Const8 <t> [int8(8 - udivisible8(c).K)])) (Const8 <t> [int8(udivisible8(c).Max)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul8 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != ssaop.OpDiv8u {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != ssaop.OpConst8 {
					continue
				}
				c := AuxIntToInt8(div_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst8 || AuxIntToInt8(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != ssaop.OpConst8 && UdivisibleOK8(c)) {
					continue
				}
				v.Reset(ssaop.OpLeq8U)
				v0 := b.NewValue0(v.Pos, ssaop.OpRotateLeft8, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpMul8, t)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v2.AuxInt = Int8ToAuxInt(int8(udivisible8(c).M))
				v1.AddArg2(x, v2)
				v3 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v3.AuxInt = Int8ToAuxInt(int8(8 - udivisible8(c).K))
				v0.AddArg2(v1, v3)
				v4 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v4.AuxInt = Int8ToAuxInt(int8(udivisible8(c).Max))
				v.AddArg2(v0, v4)
				return true
			}
		}
		break
	}
	// match: (Eq8 x (Mul8 <t> div:(Div8 x (Const8 [c])) (Const8 [c])))
	// cond: div.Uses == 1 && x.Op != ssaop.OpConst8 && SdivisibleOK8(c)
	// result: (Leq8U (RotateLeft8 <t> (Add8 <t> (Mul8 <t> x (Const8 <t> [int8(sdivisible8(c).M)])) (Const8 <t> [int8(sdivisible8(c).A)])) (Const8 <t> [int8(8 - sdivisible8(c).K)])) (Const8 <t> [int8(sdivisible8(c).Max)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul8 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != ssaop.OpDiv8 {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != ssaop.OpConst8 {
					continue
				}
				c := AuxIntToInt8(div_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst8 || AuxIntToInt8(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != ssaop.OpConst8 && SdivisibleOK8(c)) {
					continue
				}
				v.Reset(ssaop.OpLeq8U)
				v0 := b.NewValue0(v.Pos, ssaop.OpRotateLeft8, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpAdd8, t)
				v2 := b.NewValue0(v.Pos, ssaop.OpMul8, t)
				v3 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v3.AuxInt = Int8ToAuxInt(int8(sdivisible8(c).M))
				v2.AddArg2(x, v3)
				v4 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v4.AuxInt = Int8ToAuxInt(int8(sdivisible8(c).A))
				v1.AddArg2(v2, v4)
				v5 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v5.AuxInt = Int8ToAuxInt(int8(8 - sdivisible8(c).K))
				v0.AddArg2(v1, v5)
				v6 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v6.AuxInt = Int8ToAuxInt(int8(sdivisible8(c).Max))
				v.AddArg2(v0, v6)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuedivisible_OpNeq16(v *ssacore.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq16 x (Mul16 <t> (Div16u x (Const16 [c])) (Const16 [c])))
	// cond: x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)
	// result: (Neq16 (And16 <t> x (Const16 <t> [c-1])) (Const16 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul16 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != ssaop.OpDiv16u {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != ssaop.OpConst16 {
					continue
				}
				c := AuxIntToInt16(v_1_0_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst16 || AuxIntToInt16(v_1_1.AuxInt) != c || !(x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)) {
					continue
				}
				v.Reset(ssaop.OpNeq16)
				v0 := b.NewValue0(v.Pos, ssaop.OpAnd16, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v1.AuxInt = Int16ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v2.AuxInt = Int16ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq16 x (Mul16 <t> (Div16 x (Const16 [c])) (Const16 [c])))
	// cond: x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)
	// result: (Neq16 (And16 <t> x (Const16 <t> [c-1])) (Const16 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul16 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != ssaop.OpDiv16 {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != ssaop.OpConst16 {
					continue
				}
				c := AuxIntToInt16(v_1_0_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst16 || AuxIntToInt16(v_1_1.AuxInt) != c || !(x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)) {
					continue
				}
				v.Reset(ssaop.OpNeq16)
				v0 := b.NewValue0(v.Pos, ssaop.OpAnd16, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v1.AuxInt = Int16ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v2.AuxInt = Int16ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq16 x (Mul16 <t> div:(Div16u x (Const16 [c])) (Const16 [c])))
	// cond: div.Uses == 1 && x.Op != ssaop.OpConst16 && UdivisibleOK16(c)
	// result: (Less16U (Const16 <t> [int16(udivisible16(c).Max)]) (RotateLeft16 <t> (Mul16 <t> x (Const16 <t> [int16(udivisible16(c).M)])) (Const16 <t> [int16(16 - udivisible16(c).K)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul16 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != ssaop.OpDiv16u {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != ssaop.OpConst16 {
					continue
				}
				c := AuxIntToInt16(div_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst16 || AuxIntToInt16(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != ssaop.OpConst16 && UdivisibleOK16(c)) {
					continue
				}
				v.Reset(ssaop.OpLess16U)
				v0 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v0.AuxInt = Int16ToAuxInt(int16(udivisible16(c).Max))
				v1 := b.NewValue0(v.Pos, ssaop.OpRotateLeft16, t)
				v2 := b.NewValue0(v.Pos, ssaop.OpMul16, t)
				v3 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v3.AuxInt = Int16ToAuxInt(int16(udivisible16(c).M))
				v2.AddArg2(x, v3)
				v4 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v4.AuxInt = Int16ToAuxInt(int16(16 - udivisible16(c).K))
				v1.AddArg2(v2, v4)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	// match: (Neq16 x (Mul16 <t> div:(Div16 x (Const16 [c])) (Const16 [c])))
	// cond: div.Uses == 1 && x.Op != ssaop.OpConst16 && SdivisibleOK16(c)
	// result: (Less16U (Const16 <t> [int16(sdivisible16(c).Max)]) (RotateLeft16 <t> (Add16 <t> (Mul16 <t> x (Const16 <t> [int16(sdivisible16(c).M)])) (Const16 <t> [int16(sdivisible16(c).A)])) (Const16 <t> [int16(16 - sdivisible16(c).K)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul16 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != ssaop.OpDiv16 {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != ssaop.OpConst16 {
					continue
				}
				c := AuxIntToInt16(div_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst16 || AuxIntToInt16(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != ssaop.OpConst16 && SdivisibleOK16(c)) {
					continue
				}
				v.Reset(ssaop.OpLess16U)
				v0 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v0.AuxInt = Int16ToAuxInt(int16(sdivisible16(c).Max))
				v1 := b.NewValue0(v.Pos, ssaop.OpRotateLeft16, t)
				v2 := b.NewValue0(v.Pos, ssaop.OpAdd16, t)
				v3 := b.NewValue0(v.Pos, ssaop.OpMul16, t)
				v4 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v4.AuxInt = Int16ToAuxInt(int16(sdivisible16(c).M))
				v3.AddArg2(x, v4)
				v5 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v5.AuxInt = Int16ToAuxInt(int16(sdivisible16(c).A))
				v2.AddArg2(v3, v5)
				v6 := b.NewValue0(v.Pos, ssaop.OpConst16, t)
				v6.AuxInt = Int16ToAuxInt(int16(16 - sdivisible16(c).K))
				v1.AddArg2(v2, v6)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuedivisible_OpNeq32(v *ssacore.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32 x (Mul32 <t> (Div32u x (Const32 [c])) (Const32 [c])))
	// cond: x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)
	// result: (Neq32 (And32 <t> x (Const32 <t> [c-1])) (Const32 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul32 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != ssaop.OpDiv32u {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != ssaop.OpConst32 {
					continue
				}
				c := AuxIntToInt32(v_1_0_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst32 || AuxIntToInt32(v_1_1.AuxInt) != c || !(x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)) {
					continue
				}
				v.Reset(ssaop.OpNeq32)
				v0 := b.NewValue0(v.Pos, ssaop.OpAnd32, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v1.AuxInt = Int32ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v2.AuxInt = Int32ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq32 x (Mul32 <t> (Div32 x (Const32 [c])) (Const32 [c])))
	// cond: x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)
	// result: (Neq32 (And32 <t> x (Const32 <t> [c-1])) (Const32 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul32 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != ssaop.OpDiv32 {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != ssaop.OpConst32 {
					continue
				}
				c := AuxIntToInt32(v_1_0_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst32 || AuxIntToInt32(v_1_1.AuxInt) != c || !(x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)) {
					continue
				}
				v.Reset(ssaop.OpNeq32)
				v0 := b.NewValue0(v.Pos, ssaop.OpAnd32, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v1.AuxInt = Int32ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v2.AuxInt = Int32ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq32 x (Mul32 <t> div:(Div32u x (Const32 [c])) (Const32 [c])))
	// cond: div.Uses == 1 && x.Op != ssaop.OpConst32 && udivisibleOK32(c)
	// result: (Less32U (Const32 <t> [int32(udivisible32(c).Max)]) (RotateLeft32 <t> (Mul32 <t> x (Const32 <t> [int32(udivisible32(c).M)])) (Const32 <t> [int32(32 - udivisible32(c).K)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul32 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != ssaop.OpDiv32u {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != ssaop.OpConst32 {
					continue
				}
				c := AuxIntToInt32(div_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst32 || AuxIntToInt32(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != ssaop.OpConst32 && udivisibleOK32(c)) {
					continue
				}
				v.Reset(ssaop.OpLess32U)
				v0 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v0.AuxInt = Int32ToAuxInt(int32(udivisible32(c).Max))
				v1 := b.NewValue0(v.Pos, ssaop.OpRotateLeft32, t)
				v2 := b.NewValue0(v.Pos, ssaop.OpMul32, t)
				v3 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v3.AuxInt = Int32ToAuxInt(int32(udivisible32(c).M))
				v2.AddArg2(x, v3)
				v4 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v4.AuxInt = Int32ToAuxInt(int32(32 - udivisible32(c).K))
				v1.AddArg2(v2, v4)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	// match: (Neq32 x (Mul32 <t> div:(Div32 x (Const32 [c])) (Const32 [c])))
	// cond: div.Uses == 1 && x.Op != ssaop.OpConst32 && sdivisibleOK32(c)
	// result: (Less32U (Const32 <t> [int32(sdivisible32(c).Max)]) (RotateLeft32 <t> (Add32 <t> (Mul32 <t> x (Const32 <t> [int32(sdivisible32(c).M)])) (Const32 <t> [int32(sdivisible32(c).A)])) (Const32 <t> [int32(32 - sdivisible32(c).K)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul32 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != ssaop.OpDiv32 {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != ssaop.OpConst32 {
					continue
				}
				c := AuxIntToInt32(div_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst32 || AuxIntToInt32(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != ssaop.OpConst32 && sdivisibleOK32(c)) {
					continue
				}
				v.Reset(ssaop.OpLess32U)
				v0 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v0.AuxInt = Int32ToAuxInt(int32(sdivisible32(c).Max))
				v1 := b.NewValue0(v.Pos, ssaop.OpRotateLeft32, t)
				v2 := b.NewValue0(v.Pos, ssaop.OpAdd32, t)
				v3 := b.NewValue0(v.Pos, ssaop.OpMul32, t)
				v4 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v4.AuxInt = Int32ToAuxInt(int32(sdivisible32(c).M))
				v3.AddArg2(x, v4)
				v5 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v5.AuxInt = Int32ToAuxInt(int32(sdivisible32(c).A))
				v2.AddArg2(v3, v5)
				v6 := b.NewValue0(v.Pos, ssaop.OpConst32, t)
				v6.AuxInt = Int32ToAuxInt(int32(32 - sdivisible32(c).K))
				v1.AddArg2(v2, v6)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuedivisible_OpNeq64(v *ssacore.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64 x (Mul64 <t> (Div64u x (Const64 [c])) (Const64 [c])))
	// cond: x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)
	// result: (Neq64 (And64 <t> x (Const64 <t> [c-1])) (Const64 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul64 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != ssaop.OpDiv64u {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != ssaop.OpConst64 {
					continue
				}
				c := AuxIntToInt64(v_1_0_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst64 || AuxIntToInt64(v_1_1.AuxInt) != c || !(x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)) {
					continue
				}
				v.Reset(ssaop.OpNeq64)
				v0 := b.NewValue0(v.Pos, ssaop.OpAnd64, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v1.AuxInt = Int64ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v2.AuxInt = Int64ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq64 x (Mul64 <t> (Div64 x (Const64 [c])) (Const64 [c])))
	// cond: x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)
	// result: (Neq64 (And64 <t> x (Const64 <t> [c-1])) (Const64 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul64 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != ssaop.OpDiv64 {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != ssaop.OpConst64 {
					continue
				}
				c := AuxIntToInt64(v_1_0_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst64 || AuxIntToInt64(v_1_1.AuxInt) != c || !(x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)) {
					continue
				}
				v.Reset(ssaop.OpNeq64)
				v0 := b.NewValue0(v.Pos, ssaop.OpAnd64, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v1.AuxInt = Int64ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v2.AuxInt = Int64ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq64 x (Mul64 <t> div:(Div64u x (Const64 [c])) (Const64 [c])))
	// cond: div.Uses == 1 && x.Op != ssaop.OpConst64 && udivisibleOK64(c)
	// result: (Less64U (Const64 <t> [int64(udivisible64(c).Max)]) (RotateLeft64 <t> (Mul64 <t> x (Const64 <t> [int64(udivisible64(c).M)])) (Const64 <t> [int64(64 - udivisible64(c).K)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul64 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != ssaop.OpDiv64u {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != ssaop.OpConst64 {
					continue
				}
				c := AuxIntToInt64(div_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst64 || AuxIntToInt64(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != ssaop.OpConst64 && udivisibleOK64(c)) {
					continue
				}
				v.Reset(ssaop.OpLess64U)
				v0 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v0.AuxInt = Int64ToAuxInt(int64(udivisible64(c).Max))
				v1 := b.NewValue0(v.Pos, ssaop.OpRotateLeft64, t)
				v2 := b.NewValue0(v.Pos, ssaop.OpMul64, t)
				v3 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v3.AuxInt = Int64ToAuxInt(int64(udivisible64(c).M))
				v2.AddArg2(x, v3)
				v4 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v4.AuxInt = Int64ToAuxInt(int64(64 - udivisible64(c).K))
				v1.AddArg2(v2, v4)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	// match: (Neq64 x (Mul64 <t> div:(Div64 x (Const64 [c])) (Const64 [c])))
	// cond: div.Uses == 1 && x.Op != ssaop.OpConst64 && sdivisibleOK64(c)
	// result: (Less64U (Const64 <t> [int64(sdivisible64(c).Max)]) (RotateLeft64 <t> (Add64 <t> (Mul64 <t> x (Const64 <t> [int64(sdivisible64(c).M)])) (Const64 <t> [int64(sdivisible64(c).A)])) (Const64 <t> [int64(64 - sdivisible64(c).K)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul64 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != ssaop.OpDiv64 {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != ssaop.OpConst64 {
					continue
				}
				c := AuxIntToInt64(div_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst64 || AuxIntToInt64(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != ssaop.OpConst64 && sdivisibleOK64(c)) {
					continue
				}
				v.Reset(ssaop.OpLess64U)
				v0 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v0.AuxInt = Int64ToAuxInt(int64(sdivisible64(c).Max))
				v1 := b.NewValue0(v.Pos, ssaop.OpRotateLeft64, t)
				v2 := b.NewValue0(v.Pos, ssaop.OpAdd64, t)
				v3 := b.NewValue0(v.Pos, ssaop.OpMul64, t)
				v4 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v4.AuxInt = Int64ToAuxInt(int64(sdivisible64(c).M))
				v3.AddArg2(x, v4)
				v5 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v5.AuxInt = Int64ToAuxInt(int64(sdivisible64(c).A))
				v2.AddArg2(v3, v5)
				v6 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
				v6.AuxInt = Int64ToAuxInt(int64(64 - sdivisible64(c).K))
				v1.AddArg2(v2, v6)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuedivisible_OpNeq8(v *ssacore.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq8 x (Mul8 <t> (Div8u x (Const8 [c])) (Const8 [c])))
	// cond: x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)
	// result: (Neq8 (And8 <t> x (Const8 <t> [c-1])) (Const8 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul8 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != ssaop.OpDiv8u {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != ssaop.OpConst8 {
					continue
				}
				c := AuxIntToInt8(v_1_0_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst8 || AuxIntToInt8(v_1_1.AuxInt) != c || !(x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)) {
					continue
				}
				v.Reset(ssaop.OpNeq8)
				v0 := b.NewValue0(v.Pos, ssaop.OpAnd8, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v1.AuxInt = Int8ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v2.AuxInt = Int8ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq8 x (Mul8 <t> (Div8 x (Const8 [c])) (Const8 [c])))
	// cond: x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)
	// result: (Neq8 (And8 <t> x (Const8 <t> [c-1])) (Const8 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul8 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != ssaop.OpDiv8 {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != ssaop.OpConst8 {
					continue
				}
				c := AuxIntToInt8(v_1_0_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst8 || AuxIntToInt8(v_1_1.AuxInt) != c || !(x.Op != ssaop.OpConst64 && IsPowerOfTwo(c)) {
					continue
				}
				v.Reset(ssaop.OpNeq8)
				v0 := b.NewValue0(v.Pos, ssaop.OpAnd8, t)
				v1 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v1.AuxInt = Int8ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v2.AuxInt = Int8ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq8 x (Mul8 <t> div:(Div8u x (Const8 [c])) (Const8 [c])))
	// cond: div.Uses == 1 && x.Op != ssaop.OpConst8 && UdivisibleOK8(c)
	// result: (Less8U (Const8 <t> [int8(udivisible8(c).Max)]) (RotateLeft8 <t> (Mul8 <t> x (Const8 <t> [int8(udivisible8(c).M)])) (Const8 <t> [int8(8 - udivisible8(c).K)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul8 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != ssaop.OpDiv8u {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != ssaop.OpConst8 {
					continue
				}
				c := AuxIntToInt8(div_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst8 || AuxIntToInt8(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != ssaop.OpConst8 && UdivisibleOK8(c)) {
					continue
				}
				v.Reset(ssaop.OpLess8U)
				v0 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v0.AuxInt = Int8ToAuxInt(int8(udivisible8(c).Max))
				v1 := b.NewValue0(v.Pos, ssaop.OpRotateLeft8, t)
				v2 := b.NewValue0(v.Pos, ssaop.OpMul8, t)
				v3 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v3.AuxInt = Int8ToAuxInt(int8(udivisible8(c).M))
				v2.AddArg2(x, v3)
				v4 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v4.AuxInt = Int8ToAuxInt(int8(8 - udivisible8(c).K))
				v1.AddArg2(v2, v4)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	// match: (Neq8 x (Mul8 <t> div:(Div8 x (Const8 [c])) (Const8 [c])))
	// cond: div.Uses == 1 && x.Op != ssaop.OpConst8 && SdivisibleOK8(c)
	// result: (Less8U (Const8 <t> [int8(sdivisible8(c).Max)]) (RotateLeft8 <t> (Add8 <t> (Mul8 <t> x (Const8 <t> [int8(sdivisible8(c).M)])) (Const8 <t> [int8(sdivisible8(c).A)])) (Const8 <t> [int8(8 - sdivisible8(c).K)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMul8 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != ssaop.OpDiv8 {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != ssaop.OpConst8 {
					continue
				}
				c := AuxIntToInt8(div_1.AuxInt)
				if v_1_1.Op != ssaop.OpConst8 || AuxIntToInt8(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != ssaop.OpConst8 && SdivisibleOK8(c)) {
					continue
				}
				v.Reset(ssaop.OpLess8U)
				v0 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v0.AuxInt = Int8ToAuxInt(int8(sdivisible8(c).Max))
				v1 := b.NewValue0(v.Pos, ssaop.OpRotateLeft8, t)
				v2 := b.NewValue0(v.Pos, ssaop.OpAdd8, t)
				v3 := b.NewValue0(v.Pos, ssaop.OpMul8, t)
				v4 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v4.AuxInt = Int8ToAuxInt(int8(sdivisible8(c).M))
				v3.AddArg2(x, v4)
				v5 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v5.AuxInt = Int8ToAuxInt(int8(sdivisible8(c).A))
				v2.AddArg2(v3, v5)
				v6 := b.NewValue0(v.Pos, ssaop.OpConst8, t)
				v6.AuxInt = Int8ToAuxInt(int8(8 - sdivisible8(c).K))
				v1.AddArg2(v2, v6)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	return false
}
func rewriteBlockdivisible(b *ssacore.Block) bool {
	return false
}
