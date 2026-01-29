// Code generated from _gen/divisible.rules using 'go generate'; DO NOT EDIT.

package ssa

func rewriteValuedivisible(v *Value) bool {
	switch v.Op {
	case OpEq16:
		return rewriteValuedivisible_OpEq16(v)
	case OpEq32:
		return rewriteValuedivisible_OpEq32(v)
	case OpEq64:
		return rewriteValuedivisible_OpEq64(v)
	case OpEq8:
		return rewriteValuedivisible_OpEq8(v)
	case OpNeq16:
		return rewriteValuedivisible_OpNeq16(v)
	case OpNeq32:
		return rewriteValuedivisible_OpNeq32(v)
	case OpNeq64:
		return rewriteValuedivisible_OpNeq64(v)
	case OpNeq8:
		return rewriteValuedivisible_OpNeq8(v)
	}
	return false
}
func rewriteValuedivisible_OpEq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq16 x (Mul16 <t> (Div16u x (Const16 [c])) (Const16 [c])))
	// cond: x.Op != OpConst64 && isPowerOfTwo(c)
	// result: (Eq16 (And16 <t> x (Const16 <t> [c-1])) (Const16 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul16 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpDiv16u {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst16 {
					continue
				}
				c := auxIntToInt16(v_1_0_1.AuxInt)
				if v_1_1.Op != OpConst16 || auxIntToInt16(v_1_1.AuxInt) != c || !(x.Op != OpConst64 && isPowerOfTwo(c)) {
					continue
				}
				v.reset(OpEq16)
				v0 := b.NewValue0(v.Pos, OpAnd16, t)
				v1 := b.NewValue0(v.Pos, OpConst16, t)
				v1.AuxInt = int16ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst16, t)
				v2.AuxInt = int16ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq16 x (Mul16 <t> (Div16 x (Const16 [c])) (Const16 [c])))
	// cond: x.Op != OpConst64 && isPowerOfTwo(c)
	// result: (Eq16 (And16 <t> x (Const16 <t> [c-1])) (Const16 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul16 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpDiv16 {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst16 {
					continue
				}
				c := auxIntToInt16(v_1_0_1.AuxInt)
				if v_1_1.Op != OpConst16 || auxIntToInt16(v_1_1.AuxInt) != c || !(x.Op != OpConst64 && isPowerOfTwo(c)) {
					continue
				}
				v.reset(OpEq16)
				v0 := b.NewValue0(v.Pos, OpAnd16, t)
				v1 := b.NewValue0(v.Pos, OpConst16, t)
				v1.AuxInt = int16ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst16, t)
				v2.AuxInt = int16ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq16 x (Mul16 <t> div:(Div16u x (Const16 [c])) (Const16 [c])))
	// cond: div.Uses == 1 && x.Op != OpConst16 && udivisibleOK16(c)
	// result: (Leq16U (RotateLeft16 <t> (Mul16 <t> x (Const16 <t> [int16(udivisible16(c).m)])) (Const16 <t> [int16(16 - udivisible16(c).k)])) (Const16 <t> [int16(udivisible16(c).max)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul16 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != OpDiv16u {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != OpConst16 {
					continue
				}
				c := auxIntToInt16(div_1.AuxInt)
				if v_1_1.Op != OpConst16 || auxIntToInt16(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != OpConst16 && udivisibleOK16(c)) {
					continue
				}
				v.reset(OpLeq16U)
				v0 := b.NewValue0(v.Pos, OpRotateLeft16, t)
				v1 := b.NewValue0(v.Pos, OpMul16, t)
				v2 := b.NewValue0(v.Pos, OpConst16, t)
				v2.AuxInt = int16ToAuxInt(int16(udivisible16(c).m))
				v1.AddArg2(x, v2)
				v3 := b.NewValue0(v.Pos, OpConst16, t)
				v3.AuxInt = int16ToAuxInt(int16(16 - udivisible16(c).k))
				v0.AddArg2(v1, v3)
				v4 := b.NewValue0(v.Pos, OpConst16, t)
				v4.AuxInt = int16ToAuxInt(int16(udivisible16(c).max))
				v.AddArg2(v0, v4)
				return true
			}
		}
		break
	}
	// match: (Eq16 x (Mul16 <t> div:(Div16 x (Const16 [c])) (Const16 [c])))
	// cond: div.Uses == 1 && x.Op != OpConst16 && sdivisibleOK16(c)
	// result: (Leq16U (RotateLeft16 <t> (Add16 <t> (Mul16 <t> x (Const16 <t> [int16(sdivisible16(c).m)])) (Const16 <t> [int16(sdivisible16(c).a)])) (Const16 <t> [int16(16 - sdivisible16(c).k)])) (Const16 <t> [int16(sdivisible16(c).max)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul16 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != OpDiv16 {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != OpConst16 {
					continue
				}
				c := auxIntToInt16(div_1.AuxInt)
				if v_1_1.Op != OpConst16 || auxIntToInt16(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != OpConst16 && sdivisibleOK16(c)) {
					continue
				}
				v.reset(OpLeq16U)
				v0 := b.NewValue0(v.Pos, OpRotateLeft16, t)
				v1 := b.NewValue0(v.Pos, OpAdd16, t)
				v2 := b.NewValue0(v.Pos, OpMul16, t)
				v3 := b.NewValue0(v.Pos, OpConst16, t)
				v3.AuxInt = int16ToAuxInt(int16(sdivisible16(c).m))
				v2.AddArg2(x, v3)
				v4 := b.NewValue0(v.Pos, OpConst16, t)
				v4.AuxInt = int16ToAuxInt(int16(sdivisible16(c).a))
				v1.AddArg2(v2, v4)
				v5 := b.NewValue0(v.Pos, OpConst16, t)
				v5.AuxInt = int16ToAuxInt(int16(16 - sdivisible16(c).k))
				v0.AddArg2(v1, v5)
				v6 := b.NewValue0(v.Pos, OpConst16, t)
				v6.AuxInt = int16ToAuxInt(int16(sdivisible16(c).max))
				v.AddArg2(v0, v6)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuedivisible_OpEq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32 x (Mul32 <t> (Div32u x (Const32 [c])) (Const32 [c])))
	// cond: x.Op != OpConst64 && isPowerOfTwo(c)
	// result: (Eq32 (And32 <t> x (Const32 <t> [c-1])) (Const32 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul32 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpDiv32u {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst32 {
					continue
				}
				c := auxIntToInt32(v_1_0_1.AuxInt)
				if v_1_1.Op != OpConst32 || auxIntToInt32(v_1_1.AuxInt) != c || !(x.Op != OpConst64 && isPowerOfTwo(c)) {
					continue
				}
				v.reset(OpEq32)
				v0 := b.NewValue0(v.Pos, OpAnd32, t)
				v1 := b.NewValue0(v.Pos, OpConst32, t)
				v1.AuxInt = int32ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst32, t)
				v2.AuxInt = int32ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq32 x (Mul32 <t> (Div32 x (Const32 [c])) (Const32 [c])))
	// cond: x.Op != OpConst64 && isPowerOfTwo(c)
	// result: (Eq32 (And32 <t> x (Const32 <t> [c-1])) (Const32 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul32 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpDiv32 {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst32 {
					continue
				}
				c := auxIntToInt32(v_1_0_1.AuxInt)
				if v_1_1.Op != OpConst32 || auxIntToInt32(v_1_1.AuxInt) != c || !(x.Op != OpConst64 && isPowerOfTwo(c)) {
					continue
				}
				v.reset(OpEq32)
				v0 := b.NewValue0(v.Pos, OpAnd32, t)
				v1 := b.NewValue0(v.Pos, OpConst32, t)
				v1.AuxInt = int32ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst32, t)
				v2.AuxInt = int32ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq32 x (Mul32 <t> div:(Div32u x (Const32 [c])) (Const32 [c])))
	// cond: div.Uses == 1 && x.Op != OpConst32 && udivisibleOK32(c)
	// result: (Leq32U (RotateLeft32 <t> (Mul32 <t> x (Const32 <t> [int32(udivisible32(c).m)])) (Const32 <t> [int32(32 - udivisible32(c).k)])) (Const32 <t> [int32(udivisible32(c).max)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul32 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != OpDiv32u {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != OpConst32 {
					continue
				}
				c := auxIntToInt32(div_1.AuxInt)
				if v_1_1.Op != OpConst32 || auxIntToInt32(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != OpConst32 && udivisibleOK32(c)) {
					continue
				}
				v.reset(OpLeq32U)
				v0 := b.NewValue0(v.Pos, OpRotateLeft32, t)
				v1 := b.NewValue0(v.Pos, OpMul32, t)
				v2 := b.NewValue0(v.Pos, OpConst32, t)
				v2.AuxInt = int32ToAuxInt(int32(udivisible32(c).m))
				v1.AddArg2(x, v2)
				v3 := b.NewValue0(v.Pos, OpConst32, t)
				v3.AuxInt = int32ToAuxInt(int32(32 - udivisible32(c).k))
				v0.AddArg2(v1, v3)
				v4 := b.NewValue0(v.Pos, OpConst32, t)
				v4.AuxInt = int32ToAuxInt(int32(udivisible32(c).max))
				v.AddArg2(v0, v4)
				return true
			}
		}
		break
	}
	// match: (Eq32 x (Mul32 <t> div:(Div32 x (Const32 [c])) (Const32 [c])))
	// cond: div.Uses == 1 && x.Op != OpConst32 && sdivisibleOK32(c)
	// result: (Leq32U (RotateLeft32 <t> (Add32 <t> (Mul32 <t> x (Const32 <t> [int32(sdivisible32(c).m)])) (Const32 <t> [int32(sdivisible32(c).a)])) (Const32 <t> [int32(32 - sdivisible32(c).k)])) (Const32 <t> [int32(sdivisible32(c).max)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul32 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != OpDiv32 {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != OpConst32 {
					continue
				}
				c := auxIntToInt32(div_1.AuxInt)
				if v_1_1.Op != OpConst32 || auxIntToInt32(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != OpConst32 && sdivisibleOK32(c)) {
					continue
				}
				v.reset(OpLeq32U)
				v0 := b.NewValue0(v.Pos, OpRotateLeft32, t)
				v1 := b.NewValue0(v.Pos, OpAdd32, t)
				v2 := b.NewValue0(v.Pos, OpMul32, t)
				v3 := b.NewValue0(v.Pos, OpConst32, t)
				v3.AuxInt = int32ToAuxInt(int32(sdivisible32(c).m))
				v2.AddArg2(x, v3)
				v4 := b.NewValue0(v.Pos, OpConst32, t)
				v4.AuxInt = int32ToAuxInt(int32(sdivisible32(c).a))
				v1.AddArg2(v2, v4)
				v5 := b.NewValue0(v.Pos, OpConst32, t)
				v5.AuxInt = int32ToAuxInt(int32(32 - sdivisible32(c).k))
				v0.AddArg2(v1, v5)
				v6 := b.NewValue0(v.Pos, OpConst32, t)
				v6.AuxInt = int32ToAuxInt(int32(sdivisible32(c).max))
				v.AddArg2(v0, v6)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuedivisible_OpEq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64 x (Mul64 <t> (Div64u x (Const64 [c])) (Const64 [c])))
	// cond: x.Op != OpConst64 && isPowerOfTwo(c)
	// result: (Eq64 (And64 <t> x (Const64 <t> [c-1])) (Const64 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul64 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpDiv64u {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst64 {
					continue
				}
				c := auxIntToInt64(v_1_0_1.AuxInt)
				if v_1_1.Op != OpConst64 || auxIntToInt64(v_1_1.AuxInt) != c || !(x.Op != OpConst64 && isPowerOfTwo(c)) {
					continue
				}
				v.reset(OpEq64)
				v0 := b.NewValue0(v.Pos, OpAnd64, t)
				v1 := b.NewValue0(v.Pos, OpConst64, t)
				v1.AuxInt = int64ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst64, t)
				v2.AuxInt = int64ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq64 x (Mul64 <t> (Div64 x (Const64 [c])) (Const64 [c])))
	// cond: x.Op != OpConst64 && isPowerOfTwo(c)
	// result: (Eq64 (And64 <t> x (Const64 <t> [c-1])) (Const64 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul64 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpDiv64 {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst64 {
					continue
				}
				c := auxIntToInt64(v_1_0_1.AuxInt)
				if v_1_1.Op != OpConst64 || auxIntToInt64(v_1_1.AuxInt) != c || !(x.Op != OpConst64 && isPowerOfTwo(c)) {
					continue
				}
				v.reset(OpEq64)
				v0 := b.NewValue0(v.Pos, OpAnd64, t)
				v1 := b.NewValue0(v.Pos, OpConst64, t)
				v1.AuxInt = int64ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst64, t)
				v2.AuxInt = int64ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq64 x (Mul64 <t> div:(Div64u x (Const64 [c])) (Const64 [c])))
	// cond: div.Uses == 1 && x.Op != OpConst64 && udivisibleOK64(c)
	// result: (Leq64U (RotateLeft64 <t> (Mul64 <t> x (Const64 <t> [int64(udivisible64(c).m)])) (Const64 <t> [int64(64 - udivisible64(c).k)])) (Const64 <t> [int64(udivisible64(c).max)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul64 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != OpDiv64u {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != OpConst64 {
					continue
				}
				c := auxIntToInt64(div_1.AuxInt)
				if v_1_1.Op != OpConst64 || auxIntToInt64(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != OpConst64 && udivisibleOK64(c)) {
					continue
				}
				v.reset(OpLeq64U)
				v0 := b.NewValue0(v.Pos, OpRotateLeft64, t)
				v1 := b.NewValue0(v.Pos, OpMul64, t)
				v2 := b.NewValue0(v.Pos, OpConst64, t)
				v2.AuxInt = int64ToAuxInt(int64(udivisible64(c).m))
				v1.AddArg2(x, v2)
				v3 := b.NewValue0(v.Pos, OpConst64, t)
				v3.AuxInt = int64ToAuxInt(int64(64 - udivisible64(c).k))
				v0.AddArg2(v1, v3)
				v4 := b.NewValue0(v.Pos, OpConst64, t)
				v4.AuxInt = int64ToAuxInt(int64(udivisible64(c).max))
				v.AddArg2(v0, v4)
				return true
			}
		}
		break
	}
	// match: (Eq64 x (Mul64 <t> div:(Div64 x (Const64 [c])) (Const64 [c])))
	// cond: div.Uses == 1 && x.Op != OpConst64 && sdivisibleOK64(c)
	// result: (Leq64U (RotateLeft64 <t> (Add64 <t> (Mul64 <t> x (Const64 <t> [int64(sdivisible64(c).m)])) (Const64 <t> [int64(sdivisible64(c).a)])) (Const64 <t> [int64(64 - sdivisible64(c).k)])) (Const64 <t> [int64(sdivisible64(c).max)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul64 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != OpDiv64 {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != OpConst64 {
					continue
				}
				c := auxIntToInt64(div_1.AuxInt)
				if v_1_1.Op != OpConst64 || auxIntToInt64(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != OpConst64 && sdivisibleOK64(c)) {
					continue
				}
				v.reset(OpLeq64U)
				v0 := b.NewValue0(v.Pos, OpRotateLeft64, t)
				v1 := b.NewValue0(v.Pos, OpAdd64, t)
				v2 := b.NewValue0(v.Pos, OpMul64, t)
				v3 := b.NewValue0(v.Pos, OpConst64, t)
				v3.AuxInt = int64ToAuxInt(int64(sdivisible64(c).m))
				v2.AddArg2(x, v3)
				v4 := b.NewValue0(v.Pos, OpConst64, t)
				v4.AuxInt = int64ToAuxInt(int64(sdivisible64(c).a))
				v1.AddArg2(v2, v4)
				v5 := b.NewValue0(v.Pos, OpConst64, t)
				v5.AuxInt = int64ToAuxInt(int64(64 - sdivisible64(c).k))
				v0.AddArg2(v1, v5)
				v6 := b.NewValue0(v.Pos, OpConst64, t)
				v6.AuxInt = int64ToAuxInt(int64(sdivisible64(c).max))
				v.AddArg2(v0, v6)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuedivisible_OpEq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq8 x (Mul8 <t> (Div8u x (Const8 [c])) (Const8 [c])))
	// cond: x.Op != OpConst64 && isPowerOfTwo(c)
	// result: (Eq8 (And8 <t> x (Const8 <t> [c-1])) (Const8 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul8 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpDiv8u {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst8 {
					continue
				}
				c := auxIntToInt8(v_1_0_1.AuxInt)
				if v_1_1.Op != OpConst8 || auxIntToInt8(v_1_1.AuxInt) != c || !(x.Op != OpConst64 && isPowerOfTwo(c)) {
					continue
				}
				v.reset(OpEq8)
				v0 := b.NewValue0(v.Pos, OpAnd8, t)
				v1 := b.NewValue0(v.Pos, OpConst8, t)
				v1.AuxInt = int8ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst8, t)
				v2.AuxInt = int8ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq8 x (Mul8 <t> (Div8 x (Const8 [c])) (Const8 [c])))
	// cond: x.Op != OpConst64 && isPowerOfTwo(c)
	// result: (Eq8 (And8 <t> x (Const8 <t> [c-1])) (Const8 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul8 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpDiv8 {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst8 {
					continue
				}
				c := auxIntToInt8(v_1_0_1.AuxInt)
				if v_1_1.Op != OpConst8 || auxIntToInt8(v_1_1.AuxInt) != c || !(x.Op != OpConst64 && isPowerOfTwo(c)) {
					continue
				}
				v.reset(OpEq8)
				v0 := b.NewValue0(v.Pos, OpAnd8, t)
				v1 := b.NewValue0(v.Pos, OpConst8, t)
				v1.AuxInt = int8ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst8, t)
				v2.AuxInt = int8ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Eq8 x (Mul8 <t> div:(Div8u x (Const8 [c])) (Const8 [c])))
	// cond: div.Uses == 1 && x.Op != OpConst8 && udivisibleOK8(c)
	// result: (Leq8U (RotateLeft8 <t> (Mul8 <t> x (Const8 <t> [int8(udivisible8(c).m)])) (Const8 <t> [int8(8 - udivisible8(c).k)])) (Const8 <t> [int8(udivisible8(c).max)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul8 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != OpDiv8u {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != OpConst8 {
					continue
				}
				c := auxIntToInt8(div_1.AuxInt)
				if v_1_1.Op != OpConst8 || auxIntToInt8(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != OpConst8 && udivisibleOK8(c)) {
					continue
				}
				v.reset(OpLeq8U)
				v0 := b.NewValue0(v.Pos, OpRotateLeft8, t)
				v1 := b.NewValue0(v.Pos, OpMul8, t)
				v2 := b.NewValue0(v.Pos, OpConst8, t)
				v2.AuxInt = int8ToAuxInt(int8(udivisible8(c).m))
				v1.AddArg2(x, v2)
				v3 := b.NewValue0(v.Pos, OpConst8, t)
				v3.AuxInt = int8ToAuxInt(int8(8 - udivisible8(c).k))
				v0.AddArg2(v1, v3)
				v4 := b.NewValue0(v.Pos, OpConst8, t)
				v4.AuxInt = int8ToAuxInt(int8(udivisible8(c).max))
				v.AddArg2(v0, v4)
				return true
			}
		}
		break
	}
	// match: (Eq8 x (Mul8 <t> div:(Div8 x (Const8 [c])) (Const8 [c])))
	// cond: div.Uses == 1 && x.Op != OpConst8 && sdivisibleOK8(c)
	// result: (Leq8U (RotateLeft8 <t> (Add8 <t> (Mul8 <t> x (Const8 <t> [int8(sdivisible8(c).m)])) (Const8 <t> [int8(sdivisible8(c).a)])) (Const8 <t> [int8(8 - sdivisible8(c).k)])) (Const8 <t> [int8(sdivisible8(c).max)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul8 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != OpDiv8 {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != OpConst8 {
					continue
				}
				c := auxIntToInt8(div_1.AuxInt)
				if v_1_1.Op != OpConst8 || auxIntToInt8(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != OpConst8 && sdivisibleOK8(c)) {
					continue
				}
				v.reset(OpLeq8U)
				v0 := b.NewValue0(v.Pos, OpRotateLeft8, t)
				v1 := b.NewValue0(v.Pos, OpAdd8, t)
				v2 := b.NewValue0(v.Pos, OpMul8, t)
				v3 := b.NewValue0(v.Pos, OpConst8, t)
				v3.AuxInt = int8ToAuxInt(int8(sdivisible8(c).m))
				v2.AddArg2(x, v3)
				v4 := b.NewValue0(v.Pos, OpConst8, t)
				v4.AuxInt = int8ToAuxInt(int8(sdivisible8(c).a))
				v1.AddArg2(v2, v4)
				v5 := b.NewValue0(v.Pos, OpConst8, t)
				v5.AuxInt = int8ToAuxInt(int8(8 - sdivisible8(c).k))
				v0.AddArg2(v1, v5)
				v6 := b.NewValue0(v.Pos, OpConst8, t)
				v6.AuxInt = int8ToAuxInt(int8(sdivisible8(c).max))
				v.AddArg2(v0, v6)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuedivisible_OpNeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq16 x (Mul16 <t> (Div16u x (Const16 [c])) (Const16 [c])))
	// cond: x.Op != OpConst64 && isPowerOfTwo(c)
	// result: (Neq16 (And16 <t> x (Const16 <t> [c-1])) (Const16 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul16 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpDiv16u {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst16 {
					continue
				}
				c := auxIntToInt16(v_1_0_1.AuxInt)
				if v_1_1.Op != OpConst16 || auxIntToInt16(v_1_1.AuxInt) != c || !(x.Op != OpConst64 && isPowerOfTwo(c)) {
					continue
				}
				v.reset(OpNeq16)
				v0 := b.NewValue0(v.Pos, OpAnd16, t)
				v1 := b.NewValue0(v.Pos, OpConst16, t)
				v1.AuxInt = int16ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst16, t)
				v2.AuxInt = int16ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq16 x (Mul16 <t> (Div16 x (Const16 [c])) (Const16 [c])))
	// cond: x.Op != OpConst64 && isPowerOfTwo(c)
	// result: (Neq16 (And16 <t> x (Const16 <t> [c-1])) (Const16 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul16 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpDiv16 {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst16 {
					continue
				}
				c := auxIntToInt16(v_1_0_1.AuxInt)
				if v_1_1.Op != OpConst16 || auxIntToInt16(v_1_1.AuxInt) != c || !(x.Op != OpConst64 && isPowerOfTwo(c)) {
					continue
				}
				v.reset(OpNeq16)
				v0 := b.NewValue0(v.Pos, OpAnd16, t)
				v1 := b.NewValue0(v.Pos, OpConst16, t)
				v1.AuxInt = int16ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst16, t)
				v2.AuxInt = int16ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq16 x (Mul16 <t> div:(Div16u x (Const16 [c])) (Const16 [c])))
	// cond: div.Uses == 1 && x.Op != OpConst16 && udivisibleOK16(c)
	// result: (Less16U (Const16 <t> [int16(udivisible16(c).max)]) (RotateLeft16 <t> (Mul16 <t> x (Const16 <t> [int16(udivisible16(c).m)])) (Const16 <t> [int16(16 - udivisible16(c).k)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul16 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != OpDiv16u {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != OpConst16 {
					continue
				}
				c := auxIntToInt16(div_1.AuxInt)
				if v_1_1.Op != OpConst16 || auxIntToInt16(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != OpConst16 && udivisibleOK16(c)) {
					continue
				}
				v.reset(OpLess16U)
				v0 := b.NewValue0(v.Pos, OpConst16, t)
				v0.AuxInt = int16ToAuxInt(int16(udivisible16(c).max))
				v1 := b.NewValue0(v.Pos, OpRotateLeft16, t)
				v2 := b.NewValue0(v.Pos, OpMul16, t)
				v3 := b.NewValue0(v.Pos, OpConst16, t)
				v3.AuxInt = int16ToAuxInt(int16(udivisible16(c).m))
				v2.AddArg2(x, v3)
				v4 := b.NewValue0(v.Pos, OpConst16, t)
				v4.AuxInt = int16ToAuxInt(int16(16 - udivisible16(c).k))
				v1.AddArg2(v2, v4)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	// match: (Neq16 x (Mul16 <t> div:(Div16 x (Const16 [c])) (Const16 [c])))
	// cond: div.Uses == 1 && x.Op != OpConst16 && sdivisibleOK16(c)
	// result: (Less16U (Const16 <t> [int16(sdivisible16(c).max)]) (RotateLeft16 <t> (Add16 <t> (Mul16 <t> x (Const16 <t> [int16(sdivisible16(c).m)])) (Const16 <t> [int16(sdivisible16(c).a)])) (Const16 <t> [int16(16 - sdivisible16(c).k)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul16 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != OpDiv16 {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != OpConst16 {
					continue
				}
				c := auxIntToInt16(div_1.AuxInt)
				if v_1_1.Op != OpConst16 || auxIntToInt16(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != OpConst16 && sdivisibleOK16(c)) {
					continue
				}
				v.reset(OpLess16U)
				v0 := b.NewValue0(v.Pos, OpConst16, t)
				v0.AuxInt = int16ToAuxInt(int16(sdivisible16(c).max))
				v1 := b.NewValue0(v.Pos, OpRotateLeft16, t)
				v2 := b.NewValue0(v.Pos, OpAdd16, t)
				v3 := b.NewValue0(v.Pos, OpMul16, t)
				v4 := b.NewValue0(v.Pos, OpConst16, t)
				v4.AuxInt = int16ToAuxInt(int16(sdivisible16(c).m))
				v3.AddArg2(x, v4)
				v5 := b.NewValue0(v.Pos, OpConst16, t)
				v5.AuxInt = int16ToAuxInt(int16(sdivisible16(c).a))
				v2.AddArg2(v3, v5)
				v6 := b.NewValue0(v.Pos, OpConst16, t)
				v6.AuxInt = int16ToAuxInt(int16(16 - sdivisible16(c).k))
				v1.AddArg2(v2, v6)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuedivisible_OpNeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32 x (Mul32 <t> (Div32u x (Const32 [c])) (Const32 [c])))
	// cond: x.Op != OpConst64 && isPowerOfTwo(c)
	// result: (Neq32 (And32 <t> x (Const32 <t> [c-1])) (Const32 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul32 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpDiv32u {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst32 {
					continue
				}
				c := auxIntToInt32(v_1_0_1.AuxInt)
				if v_1_1.Op != OpConst32 || auxIntToInt32(v_1_1.AuxInt) != c || !(x.Op != OpConst64 && isPowerOfTwo(c)) {
					continue
				}
				v.reset(OpNeq32)
				v0 := b.NewValue0(v.Pos, OpAnd32, t)
				v1 := b.NewValue0(v.Pos, OpConst32, t)
				v1.AuxInt = int32ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst32, t)
				v2.AuxInt = int32ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq32 x (Mul32 <t> (Div32 x (Const32 [c])) (Const32 [c])))
	// cond: x.Op != OpConst64 && isPowerOfTwo(c)
	// result: (Neq32 (And32 <t> x (Const32 <t> [c-1])) (Const32 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul32 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpDiv32 {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst32 {
					continue
				}
				c := auxIntToInt32(v_1_0_1.AuxInt)
				if v_1_1.Op != OpConst32 || auxIntToInt32(v_1_1.AuxInt) != c || !(x.Op != OpConst64 && isPowerOfTwo(c)) {
					continue
				}
				v.reset(OpNeq32)
				v0 := b.NewValue0(v.Pos, OpAnd32, t)
				v1 := b.NewValue0(v.Pos, OpConst32, t)
				v1.AuxInt = int32ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst32, t)
				v2.AuxInt = int32ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq32 x (Mul32 <t> div:(Div32u x (Const32 [c])) (Const32 [c])))
	// cond: div.Uses == 1 && x.Op != OpConst32 && udivisibleOK32(c)
	// result: (Less32U (Const32 <t> [int32(udivisible32(c).max)]) (RotateLeft32 <t> (Mul32 <t> x (Const32 <t> [int32(udivisible32(c).m)])) (Const32 <t> [int32(32 - udivisible32(c).k)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul32 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != OpDiv32u {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != OpConst32 {
					continue
				}
				c := auxIntToInt32(div_1.AuxInt)
				if v_1_1.Op != OpConst32 || auxIntToInt32(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != OpConst32 && udivisibleOK32(c)) {
					continue
				}
				v.reset(OpLess32U)
				v0 := b.NewValue0(v.Pos, OpConst32, t)
				v0.AuxInt = int32ToAuxInt(int32(udivisible32(c).max))
				v1 := b.NewValue0(v.Pos, OpRotateLeft32, t)
				v2 := b.NewValue0(v.Pos, OpMul32, t)
				v3 := b.NewValue0(v.Pos, OpConst32, t)
				v3.AuxInt = int32ToAuxInt(int32(udivisible32(c).m))
				v2.AddArg2(x, v3)
				v4 := b.NewValue0(v.Pos, OpConst32, t)
				v4.AuxInt = int32ToAuxInt(int32(32 - udivisible32(c).k))
				v1.AddArg2(v2, v4)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	// match: (Neq32 x (Mul32 <t> div:(Div32 x (Const32 [c])) (Const32 [c])))
	// cond: div.Uses == 1 && x.Op != OpConst32 && sdivisibleOK32(c)
	// result: (Less32U (Const32 <t> [int32(sdivisible32(c).max)]) (RotateLeft32 <t> (Add32 <t> (Mul32 <t> x (Const32 <t> [int32(sdivisible32(c).m)])) (Const32 <t> [int32(sdivisible32(c).a)])) (Const32 <t> [int32(32 - sdivisible32(c).k)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul32 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != OpDiv32 {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != OpConst32 {
					continue
				}
				c := auxIntToInt32(div_1.AuxInt)
				if v_1_1.Op != OpConst32 || auxIntToInt32(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != OpConst32 && sdivisibleOK32(c)) {
					continue
				}
				v.reset(OpLess32U)
				v0 := b.NewValue0(v.Pos, OpConst32, t)
				v0.AuxInt = int32ToAuxInt(int32(sdivisible32(c).max))
				v1 := b.NewValue0(v.Pos, OpRotateLeft32, t)
				v2 := b.NewValue0(v.Pos, OpAdd32, t)
				v3 := b.NewValue0(v.Pos, OpMul32, t)
				v4 := b.NewValue0(v.Pos, OpConst32, t)
				v4.AuxInt = int32ToAuxInt(int32(sdivisible32(c).m))
				v3.AddArg2(x, v4)
				v5 := b.NewValue0(v.Pos, OpConst32, t)
				v5.AuxInt = int32ToAuxInt(int32(sdivisible32(c).a))
				v2.AddArg2(v3, v5)
				v6 := b.NewValue0(v.Pos, OpConst32, t)
				v6.AuxInt = int32ToAuxInt(int32(32 - sdivisible32(c).k))
				v1.AddArg2(v2, v6)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuedivisible_OpNeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64 x (Mul64 <t> (Div64u x (Const64 [c])) (Const64 [c])))
	// cond: x.Op != OpConst64 && isPowerOfTwo(c)
	// result: (Neq64 (And64 <t> x (Const64 <t> [c-1])) (Const64 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul64 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpDiv64u {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst64 {
					continue
				}
				c := auxIntToInt64(v_1_0_1.AuxInt)
				if v_1_1.Op != OpConst64 || auxIntToInt64(v_1_1.AuxInt) != c || !(x.Op != OpConst64 && isPowerOfTwo(c)) {
					continue
				}
				v.reset(OpNeq64)
				v0 := b.NewValue0(v.Pos, OpAnd64, t)
				v1 := b.NewValue0(v.Pos, OpConst64, t)
				v1.AuxInt = int64ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst64, t)
				v2.AuxInt = int64ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq64 x (Mul64 <t> (Div64 x (Const64 [c])) (Const64 [c])))
	// cond: x.Op != OpConst64 && isPowerOfTwo(c)
	// result: (Neq64 (And64 <t> x (Const64 <t> [c-1])) (Const64 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul64 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpDiv64 {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst64 {
					continue
				}
				c := auxIntToInt64(v_1_0_1.AuxInt)
				if v_1_1.Op != OpConst64 || auxIntToInt64(v_1_1.AuxInt) != c || !(x.Op != OpConst64 && isPowerOfTwo(c)) {
					continue
				}
				v.reset(OpNeq64)
				v0 := b.NewValue0(v.Pos, OpAnd64, t)
				v1 := b.NewValue0(v.Pos, OpConst64, t)
				v1.AuxInt = int64ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst64, t)
				v2.AuxInt = int64ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq64 x (Mul64 <t> div:(Div64u x (Const64 [c])) (Const64 [c])))
	// cond: div.Uses == 1 && x.Op != OpConst64 && udivisibleOK64(c)
	// result: (Less64U (Const64 <t> [int64(udivisible64(c).max)]) (RotateLeft64 <t> (Mul64 <t> x (Const64 <t> [int64(udivisible64(c).m)])) (Const64 <t> [int64(64 - udivisible64(c).k)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul64 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != OpDiv64u {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != OpConst64 {
					continue
				}
				c := auxIntToInt64(div_1.AuxInt)
				if v_1_1.Op != OpConst64 || auxIntToInt64(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != OpConst64 && udivisibleOK64(c)) {
					continue
				}
				v.reset(OpLess64U)
				v0 := b.NewValue0(v.Pos, OpConst64, t)
				v0.AuxInt = int64ToAuxInt(int64(udivisible64(c).max))
				v1 := b.NewValue0(v.Pos, OpRotateLeft64, t)
				v2 := b.NewValue0(v.Pos, OpMul64, t)
				v3 := b.NewValue0(v.Pos, OpConst64, t)
				v3.AuxInt = int64ToAuxInt(int64(udivisible64(c).m))
				v2.AddArg2(x, v3)
				v4 := b.NewValue0(v.Pos, OpConst64, t)
				v4.AuxInt = int64ToAuxInt(int64(64 - udivisible64(c).k))
				v1.AddArg2(v2, v4)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	// match: (Neq64 x (Mul64 <t> div:(Div64 x (Const64 [c])) (Const64 [c])))
	// cond: div.Uses == 1 && x.Op != OpConst64 && sdivisibleOK64(c)
	// result: (Less64U (Const64 <t> [int64(sdivisible64(c).max)]) (RotateLeft64 <t> (Add64 <t> (Mul64 <t> x (Const64 <t> [int64(sdivisible64(c).m)])) (Const64 <t> [int64(sdivisible64(c).a)])) (Const64 <t> [int64(64 - sdivisible64(c).k)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul64 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != OpDiv64 {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != OpConst64 {
					continue
				}
				c := auxIntToInt64(div_1.AuxInt)
				if v_1_1.Op != OpConst64 || auxIntToInt64(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != OpConst64 && sdivisibleOK64(c)) {
					continue
				}
				v.reset(OpLess64U)
				v0 := b.NewValue0(v.Pos, OpConst64, t)
				v0.AuxInt = int64ToAuxInt(int64(sdivisible64(c).max))
				v1 := b.NewValue0(v.Pos, OpRotateLeft64, t)
				v2 := b.NewValue0(v.Pos, OpAdd64, t)
				v3 := b.NewValue0(v.Pos, OpMul64, t)
				v4 := b.NewValue0(v.Pos, OpConst64, t)
				v4.AuxInt = int64ToAuxInt(int64(sdivisible64(c).m))
				v3.AddArg2(x, v4)
				v5 := b.NewValue0(v.Pos, OpConst64, t)
				v5.AuxInt = int64ToAuxInt(int64(sdivisible64(c).a))
				v2.AddArg2(v3, v5)
				v6 := b.NewValue0(v.Pos, OpConst64, t)
				v6.AuxInt = int64ToAuxInt(int64(64 - sdivisible64(c).k))
				v1.AddArg2(v2, v6)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuedivisible_OpNeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq8 x (Mul8 <t> (Div8u x (Const8 [c])) (Const8 [c])))
	// cond: x.Op != OpConst64 && isPowerOfTwo(c)
	// result: (Neq8 (And8 <t> x (Const8 <t> [c-1])) (Const8 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul8 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpDiv8u {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst8 {
					continue
				}
				c := auxIntToInt8(v_1_0_1.AuxInt)
				if v_1_1.Op != OpConst8 || auxIntToInt8(v_1_1.AuxInt) != c || !(x.Op != OpConst64 && isPowerOfTwo(c)) {
					continue
				}
				v.reset(OpNeq8)
				v0 := b.NewValue0(v.Pos, OpAnd8, t)
				v1 := b.NewValue0(v.Pos, OpConst8, t)
				v1.AuxInt = int8ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst8, t)
				v2.AuxInt = int8ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq8 x (Mul8 <t> (Div8 x (Const8 [c])) (Const8 [c])))
	// cond: x.Op != OpConst64 && isPowerOfTwo(c)
	// result: (Neq8 (And8 <t> x (Const8 <t> [c-1])) (Const8 <t> [0]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul8 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if v_1_0.Op != OpDiv8 {
					continue
				}
				_ = v_1_0.Args[1]
				if x != v_1_0.Args[0] {
					continue
				}
				v_1_0_1 := v_1_0.Args[1]
				if v_1_0_1.Op != OpConst8 {
					continue
				}
				c := auxIntToInt8(v_1_0_1.AuxInt)
				if v_1_1.Op != OpConst8 || auxIntToInt8(v_1_1.AuxInt) != c || !(x.Op != OpConst64 && isPowerOfTwo(c)) {
					continue
				}
				v.reset(OpNeq8)
				v0 := b.NewValue0(v.Pos, OpAnd8, t)
				v1 := b.NewValue0(v.Pos, OpConst8, t)
				v1.AuxInt = int8ToAuxInt(c - 1)
				v0.AddArg2(x, v1)
				v2 := b.NewValue0(v.Pos, OpConst8, t)
				v2.AuxInt = int8ToAuxInt(0)
				v.AddArg2(v0, v2)
				return true
			}
		}
		break
	}
	// match: (Neq8 x (Mul8 <t> div:(Div8u x (Const8 [c])) (Const8 [c])))
	// cond: div.Uses == 1 && x.Op != OpConst8 && udivisibleOK8(c)
	// result: (Less8U (Const8 <t> [int8(udivisible8(c).max)]) (RotateLeft8 <t> (Mul8 <t> x (Const8 <t> [int8(udivisible8(c).m)])) (Const8 <t> [int8(8 - udivisible8(c).k)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul8 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != OpDiv8u {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != OpConst8 {
					continue
				}
				c := auxIntToInt8(div_1.AuxInt)
				if v_1_1.Op != OpConst8 || auxIntToInt8(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != OpConst8 && udivisibleOK8(c)) {
					continue
				}
				v.reset(OpLess8U)
				v0 := b.NewValue0(v.Pos, OpConst8, t)
				v0.AuxInt = int8ToAuxInt(int8(udivisible8(c).max))
				v1 := b.NewValue0(v.Pos, OpRotateLeft8, t)
				v2 := b.NewValue0(v.Pos, OpMul8, t)
				v3 := b.NewValue0(v.Pos, OpConst8, t)
				v3.AuxInt = int8ToAuxInt(int8(udivisible8(c).m))
				v2.AddArg2(x, v3)
				v4 := b.NewValue0(v.Pos, OpConst8, t)
				v4.AuxInt = int8ToAuxInt(int8(8 - udivisible8(c).k))
				v1.AddArg2(v2, v4)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	// match: (Neq8 x (Mul8 <t> div:(Div8 x (Const8 [c])) (Const8 [c])))
	// cond: div.Uses == 1 && x.Op != OpConst8 && sdivisibleOK8(c)
	// result: (Less8U (Const8 <t> [int8(sdivisible8(c).max)]) (RotateLeft8 <t> (Add8 <t> (Mul8 <t> x (Const8 <t> [int8(sdivisible8(c).m)])) (Const8 <t> [int8(sdivisible8(c).a)])) (Const8 <t> [int8(8 - sdivisible8(c).k)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpMul8 {
				continue
			}
			t := v_1.Type
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				div := v_1_0
				if div.Op != OpDiv8 {
					continue
				}
				_ = div.Args[1]
				if x != div.Args[0] {
					continue
				}
				div_1 := div.Args[1]
				if div_1.Op != OpConst8 {
					continue
				}
				c := auxIntToInt8(div_1.AuxInt)
				if v_1_1.Op != OpConst8 || auxIntToInt8(v_1_1.AuxInt) != c || !(div.Uses == 1 && x.Op != OpConst8 && sdivisibleOK8(c)) {
					continue
				}
				v.reset(OpLess8U)
				v0 := b.NewValue0(v.Pos, OpConst8, t)
				v0.AuxInt = int8ToAuxInt(int8(sdivisible8(c).max))
				v1 := b.NewValue0(v.Pos, OpRotateLeft8, t)
				v2 := b.NewValue0(v.Pos, OpAdd8, t)
				v3 := b.NewValue0(v.Pos, OpMul8, t)
				v4 := b.NewValue0(v.Pos, OpConst8, t)
				v4.AuxInt = int8ToAuxInt(int8(sdivisible8(c).m))
				v3.AddArg2(x, v4)
				v5 := b.NewValue0(v.Pos, OpConst8, t)
				v5.AuxInt = int8ToAuxInt(int8(sdivisible8(c).a))
				v2.AddArg2(v3, v5)
				v6 := b.NewValue0(v.Pos, OpConst8, t)
				v6.AuxInt = int8ToAuxInt(int8(8 - sdivisible8(c).k))
				v1.AddArg2(v2, v6)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	return false
}
func rewriteBlockdivisible(b *Block) bool {
	return false
}
