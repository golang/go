// Code generated from _gen/RISCV64latelower.rules using 'go generate'; DO NOT EDIT.

package ssa

func rewriteValueRISCV64latelower(v *Value) bool {
	switch v.Op {
	case OpRISCV64AND:
		return rewriteValueRISCV64latelower_OpRISCV64AND(v)
	case OpRISCV64NOT:
		return rewriteValueRISCV64latelower_OpRISCV64NOT(v)
	case OpRISCV64OR:
		return rewriteValueRISCV64latelower_OpRISCV64OR(v)
	case OpRISCV64SLLI:
		return rewriteValueRISCV64latelower_OpRISCV64SLLI(v)
	case OpRISCV64SRAI:
		return rewriteValueRISCV64latelower_OpRISCV64SRAI(v)
	case OpRISCV64SRLI:
		return rewriteValueRISCV64latelower_OpRISCV64SRLI(v)
	case OpRISCV64XOR:
		return rewriteValueRISCV64latelower_OpRISCV64XOR(v)
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64AND(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AND x (NOT y))
	// result: (ANDN x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpRISCV64NOT {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpRISCV64ANDN)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64NOT(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NOT (XOR x y))
	// result: (XNOR x y)
	for {
		if v_0.Op != OpRISCV64XOR {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpRISCV64XNOR)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64OR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (OR x (NOT y))
	// result: (ORN x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpRISCV64NOT {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpRISCV64ORN)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (OR (CZERONEZ <t> x x) (CZEROEQZ <t> y x))
	// result: (CZEROEQZ <t> y x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZERONEZ {
				continue
			}
			t := v_0.Type
			x := v_0.Args[1]
			if x != v_0.Args[0] || v_1.Op != OpRISCV64CZEROEQZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			y := v_1.Args[0]
			if x != v_1.Args[1] {
				continue
			}
			v.reset(OpRISCV64CZEROEQZ)
			v.Type = t
			v.AddArg2(y, x)
			return true
		}
		break
	}
	// match: (OR (CZERONEZ <t> (ADD x y) z) (CZEROEQZ <t> x z))
	// result: (ADD x (CZERONEZ <t> y z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZERONEZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64ADD {
				continue
			}
			_ = v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0_0, v_0_0_1 = _i1+1, v_0_0_1, v_0_0_0 {
				x := v_0_0_0
				y := v_0_0_1
				if v_1.Op != OpRISCV64CZEROEQZ || v_1.Type != t {
					continue
				}
				_ = v_1.Args[1]
				if x != v_1.Args[0] || z != v_1.Args[1] {
					continue
				}
				v.reset(OpRISCV64ADD)
				v0 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
				v0.AddArg2(y, z)
				v.AddArg2(x, v0)
				return true
			}
		}
		break
	}
	// match: (OR (CZERONEZ <t> (SUB x y) z) (CZEROEQZ <t> x z))
	// result: (SUB x (CZERONEZ <t> y z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZERONEZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64SUB {
				continue
			}
			y := v_0_0.Args[1]
			x := v_0_0.Args[0]
			if v_1.Op != OpRISCV64CZEROEQZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || z != v_1.Args[1] {
				continue
			}
			v.reset(OpRISCV64SUB)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
			v0.AddArg2(y, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZERONEZ <t> (OR x y) z) (CZEROEQZ <t> x z))
	// result: (OR x (CZERONEZ <t> y z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZERONEZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64OR {
				continue
			}
			_ = v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0_0, v_0_0_1 = _i1+1, v_0_0_1, v_0_0_0 {
				x := v_0_0_0
				y := v_0_0_1
				if v_1.Op != OpRISCV64CZEROEQZ || v_1.Type != t {
					continue
				}
				_ = v_1.Args[1]
				if x != v_1.Args[0] || z != v_1.Args[1] {
					continue
				}
				v.reset(OpRISCV64OR)
				v0 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
				v0.AddArg2(y, z)
				v.AddArg2(x, v0)
				return true
			}
		}
		break
	}
	// match: (OR (CZERONEZ <t> (XOR x y) z) (CZEROEQZ <t> x z))
	// result: (XOR x (CZERONEZ <t> y z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZERONEZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64XOR {
				continue
			}
			_ = v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0_0, v_0_0_1 = _i1+1, v_0_0_1, v_0_0_0 {
				x := v_0_0_0
				y := v_0_0_1
				if v_1.Op != OpRISCV64CZEROEQZ || v_1.Type != t {
					continue
				}
				_ = v_1.Args[1]
				if x != v_1.Args[0] || z != v_1.Args[1] {
					continue
				}
				v.reset(OpRISCV64XOR)
				v0 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
				v0.AddArg2(y, z)
				v.AddArg2(x, v0)
				return true
			}
		}
		break
	}
	// match: (OR (CZERONEZ <t> (SUBW x y) z) (CZEROEQZ <t> x z))
	// result: (SUBW x (CZERONEZ <t> y z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZERONEZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64SUBW {
				continue
			}
			y := v_0_0.Args[1]
			x := v_0_0.Args[0]
			if v_1.Op != OpRISCV64CZEROEQZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || z != v_1.Args[1] {
				continue
			}
			v.reset(OpRISCV64SUBW)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
			v0.AddArg2(y, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (ADD x y) z) (CZERONEZ <t> x z))
	// result: (ADD x (CZEROEQZ <t> y z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64ADD {
				continue
			}
			_ = v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0_0, v_0_0_1 = _i1+1, v_0_0_1, v_0_0_0 {
				x := v_0_0_0
				y := v_0_0_1
				if v_1.Op != OpRISCV64CZERONEZ || v_1.Type != t {
					continue
				}
				_ = v_1.Args[1]
				if x != v_1.Args[0] || z != v_1.Args[1] {
					continue
				}
				v.reset(OpRISCV64ADD)
				v0 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
				v0.AddArg2(y, z)
				v.AddArg2(x, v0)
				return true
			}
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (SUB x y) z) (CZERONEZ <t> x z))
	// result: (SUB x (CZEROEQZ <t> y z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64SUB {
				continue
			}
			y := v_0_0.Args[1]
			x := v_0_0.Args[0]
			if v_1.Op != OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || z != v_1.Args[1] {
				continue
			}
			v.reset(OpRISCV64SUB)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
			v0.AddArg2(y, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (OR x y) z) (CZERONEZ <t> x z))
	// result: (OR x (CZEROEQZ <t> y z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64OR {
				continue
			}
			_ = v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0_0, v_0_0_1 = _i1+1, v_0_0_1, v_0_0_0 {
				x := v_0_0_0
				y := v_0_0_1
				if v_1.Op != OpRISCV64CZERONEZ || v_1.Type != t {
					continue
				}
				_ = v_1.Args[1]
				if x != v_1.Args[0] || z != v_1.Args[1] {
					continue
				}
				v.reset(OpRISCV64OR)
				v0 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
				v0.AddArg2(y, z)
				v.AddArg2(x, v0)
				return true
			}
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (XOR x y) z) (CZERONEZ <t> x z))
	// result: (XOR x (CZEROEQZ <t> y z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64XOR {
				continue
			}
			_ = v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0_0, v_0_0_1 = _i1+1, v_0_0_1, v_0_0_0 {
				x := v_0_0_0
				y := v_0_0_1
				if v_1.Op != OpRISCV64CZERONEZ || v_1.Type != t {
					continue
				}
				_ = v_1.Args[1]
				if x != v_1.Args[0] || z != v_1.Args[1] {
					continue
				}
				v.reset(OpRISCV64XOR)
				v0 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
				v0.AddArg2(y, z)
				v.AddArg2(x, v0)
				return true
			}
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (SUBW x y) z) (CZERONEZ <t> x z))
	// result: (SUBW x (CZEROEQZ <t> y z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64SUBW {
				continue
			}
			y := v_0_0.Args[1]
			x := v_0_0.Args[0]
			if v_1.Op != OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || z != v_1.Args[1] {
				continue
			}
			v.reset(OpRISCV64SUBW)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
			v0.AddArg2(y, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZERONEZ <t> (AND x y) z) (CZEROEQZ <t> x z))
	// result: (OR (AND <t> x y) (CZEROEQZ <t> x z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZERONEZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64AND {
				continue
			}
			_ = v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0_0, v_0_0_1 = _i1+1, v_0_0_1, v_0_0_0 {
				x := v_0_0_0
				y := v_0_0_1
				if v_1.Op != OpRISCV64CZEROEQZ || v_1.Type != t {
					continue
				}
				_ = v_1.Args[1]
				if x != v_1.Args[0] || z != v_1.Args[1] {
					continue
				}
				v.reset(OpRISCV64OR)
				v0 := b.NewValue0(v.Pos, OpRISCV64AND, t)
				v0.AddArg2(x, y)
				v1 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
				v1.AddArg2(x, z)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (AND x y) z) (CZERONEZ <t> x z))
	// result: (OR (AND <t> x y) (CZERONEZ <t> x z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64AND {
				continue
			}
			_ = v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0_0, v_0_0_1 = _i1+1, v_0_0_1, v_0_0_0 {
				x := v_0_0_0
				y := v_0_0_1
				if v_1.Op != OpRISCV64CZERONEZ || v_1.Type != t {
					continue
				}
				_ = v_1.Args[1]
				if x != v_1.Args[0] || z != v_1.Args[1] {
					continue
				}
				v.reset(OpRISCV64OR)
				v0 := b.NewValue0(v.Pos, OpRISCV64AND, t)
				v0.AddArg2(x, y)
				v1 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
				v1.AddArg2(x, z)
				v.AddArg2(v0, v1)
				return true
			}
		}
		break
	}
	// match: (OR (CZERONEZ <t> (ANDI [c] x) z) (CZEROEQZ <t> x z))
	// result: (OR (AND <t> x (MOVDconst [c])) (CZEROEQZ <t> x z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZERONEZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64ANDI {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			x := v_0_0.Args[0]
			if v_1.Op != OpRISCV64CZEROEQZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || z != v_1.Args[1] {
				continue
			}
			v.reset(OpRISCV64OR)
			v0 := b.NewValue0(v.Pos, OpRISCV64AND, t)
			v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
			v1.AuxInt = int64ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
			v2.AddArg2(x, z)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (ANDI [c] x) z) (CZERONEZ <t> x z))
	// result: (OR (AND <t> x (MOVDconst [c])) (CZERONEZ <t> x z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64ANDI {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			x := v_0_0.Args[0]
			if v_1.Op != OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || z != v_1.Args[1] {
				continue
			}
			v.reset(OpRISCV64OR)
			v0 := b.NewValue0(v.Pos, OpRISCV64AND, t)
			v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
			v1.AuxInt = int64ToAuxInt(c)
			v0.AddArg2(x, v1)
			v2 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
			v2.AddArg2(x, z)
			v.AddArg2(v0, v2)
			return true
		}
		break
	}
	// match: (OR (CZERONEZ <t> (ADDI [c] x) z) (CZEROEQZ <t> x z))
	// result: (ADD x (CZERONEZ <t> (MOVDconst [c]) z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZERONEZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64ADDI {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			x := v_0_0.Args[0]
			if v_1.Op != OpRISCV64CZEROEQZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || z != v_1.Args[1] {
				continue
			}
			v.reset(OpRISCV64ADD)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
			v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
			v1.AuxInt = int64ToAuxInt(c)
			v0.AddArg2(v1, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZERONEZ <t> (ORI [c] x) z) (CZEROEQZ <t> x z))
	// result: (OR x (CZERONEZ <t> (MOVDconst [c]) z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZERONEZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64ORI {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			x := v_0_0.Args[0]
			if v_1.Op != OpRISCV64CZEROEQZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || z != v_1.Args[1] {
				continue
			}
			v.reset(OpRISCV64OR)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
			v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
			v1.AuxInt = int64ToAuxInt(c)
			v0.AddArg2(v1, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZERONEZ <t> (XORI [c] x) z) (CZEROEQZ <t> x z))
	// result: (XOR x (CZERONEZ <t> (MOVDconst [c]) z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZERONEZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64XORI {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			x := v_0_0.Args[0]
			if v_1.Op != OpRISCV64CZEROEQZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || z != v_1.Args[1] {
				continue
			}
			v.reset(OpRISCV64XOR)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
			v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
			v1.AuxInt = int64ToAuxInt(c)
			v0.AddArg2(v1, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (ADDI [c] x) z) (CZERONEZ <t> x z))
	// result: (ADD x (CZEROEQZ <t> (MOVDconst [c]) z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64ADDI {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			x := v_0_0.Args[0]
			if v_1.Op != OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || z != v_1.Args[1] {
				continue
			}
			v.reset(OpRISCV64ADD)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
			v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
			v1.AuxInt = int64ToAuxInt(c)
			v0.AddArg2(v1, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (ORI [c] x) z) (CZERONEZ <t> x z))
	// result: (OR x (CZEROEQZ <t> (MOVDconst [c]) z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64ORI {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			x := v_0_0.Args[0]
			if v_1.Op != OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || z != v_1.Args[1] {
				continue
			}
			v.reset(OpRISCV64OR)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
			v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
			v1.AuxInt = int64ToAuxInt(c)
			v0.AddArg2(v1, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (XORI [c] x) z) (CZERONEZ <t> x z))
	// result: (XOR x (CZEROEQZ <t> (MOVDconst [c]) z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64XORI {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			x := v_0_0.Args[0]
			if v_1.Op != OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || z != v_1.Args[1] {
				continue
			}
			v.reset(OpRISCV64XOR)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
			v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
			v1.AuxInt = int64ToAuxInt(c)
			v0.AddArg2(v1, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZERONEZ <t> (ADDIW [c] x) z) (CZEROEQZ <t> x z))
	// result: (ADD x (CZERONEZ <t> (MOVDconst [int64(int32(c))]) z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZERONEZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64ADDIW {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			x := v_0_0.Args[0]
			if v_1.Op != OpRISCV64CZEROEQZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || z != v_1.Args[1] {
				continue
			}
			v.reset(OpRISCV64ADD)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
			v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
			v1.AuxInt = int64ToAuxInt(int64(int32(c)))
			v0.AddArg2(v1, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (ADDIW [c] x) z) (CZERONEZ <t> x z))
	// result: (ADD x (CZEROEQZ <t> (MOVDconst [int64(int32(c))]) z))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			z := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64ADDIW {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			x := v_0_0.Args[0]
			if v_1.Op != OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || z != v_1.Args[1] {
				continue
			}
			v.reset(OpRISCV64ADD)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
			v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
			v1.AuxInt = int64ToAuxInt(int64(int32(c)))
			v0.AddArg2(v1, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZERONEZ <t> (MOVDconst [0]) cond) (CZEROEQZ <t> (MOVDconst [c]) cond))
	// result: (CZERONEZ <t> (MOVDconst [c]) cond)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZERONEZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_0.AuxInt) != 0 || v_1.Op != OpRISCV64CZEROEQZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRISCV64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1_0.AuxInt)
			if cond != v_1.Args[1] {
				continue
			}
			v.reset(OpRISCV64CZERONEZ)
			v.Type = t
			v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
			v0.AuxInt = int64ToAuxInt(c)
			v.AddArg2(v0, cond)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (MOVDconst [0]) cond) (CZERONEZ <t> (MOVDconst [c]) cond))
	// result: (CZEROEQZ <t> (MOVDconst [c]) cond)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_0.AuxInt) != 0 || v_1.Op != OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRISCV64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1_0.AuxInt)
			if cond != v_1.Args[1] {
				continue
			}
			v.reset(OpRISCV64CZEROEQZ)
			v.Type = t
			v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
			v0.AuxInt = int64ToAuxInt(c)
			v.AddArg2(v0, cond)
			return true
		}
		break
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64SLLI(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SLLI [c] (MOVBUreg x))
	// cond: c <= 56
	// result: (SRLI [56-c] (SLLI <typ.UInt64> [56] x))
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		if !(c <= 56) {
			break
		}
		v.reset(OpRISCV64SRLI)
		v.AuxInt = int64ToAuxInt(56 - c)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(56)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SLLI [c] (MOVHUreg x))
	// cond: c <= 48
	// result: (SRLI [48-c] (SLLI <typ.UInt64> [48] x))
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		if !(c <= 48) {
			break
		}
		v.reset(OpRISCV64SRLI)
		v.AuxInt = int64ToAuxInt(48 - c)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(48)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SLLI [c] (MOVWUreg x))
	// cond: c <= 32
	// result: (SRLI [32-c] (SLLI <typ.UInt64> [32] x))
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		if !(c <= 32) {
			break
		}
		v.reset(OpRISCV64SRLI)
		v.AuxInt = int64ToAuxInt(32 - c)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SLLI [0] x)
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64SRAI(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SRAI [c] (MOVBreg x))
	// cond: c < 8
	// result: (SRAI [56+c] (SLLI <typ.Int64> [56] x))
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVBreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 8) {
			break
		}
		v.reset(OpRISCV64SRAI)
		v.AuxInt = int64ToAuxInt(56 + c)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, typ.Int64)
		v0.AuxInt = int64ToAuxInt(56)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRAI [c] (MOVHreg x))
	// cond: c < 16
	// result: (SRAI [48+c] (SLLI <typ.Int64> [48] x))
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVHreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 16) {
			break
		}
		v.reset(OpRISCV64SRAI)
		v.AuxInt = int64ToAuxInt(48 + c)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, typ.Int64)
		v0.AuxInt = int64ToAuxInt(48)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRAI [c] (MOVWreg x))
	// cond: c < 32
	// result: (SRAI [32+c] (SLLI <typ.Int64> [32] x))
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVWreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 32) {
			break
		}
		v.reset(OpRISCV64SRAI)
		v.AuxInt = int64ToAuxInt(32 + c)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, typ.Int64)
		v0.AuxInt = int64ToAuxInt(32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRAI [0] x)
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64SRLI(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SRLI [c] (MOVBUreg x))
	// cond: c < 8
	// result: (SRLI [56+c] (SLLI <typ.UInt64> [56] x))
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 8) {
			break
		}
		v.reset(OpRISCV64SRLI)
		v.AuxInt = int64ToAuxInt(56 + c)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(56)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRLI [c] (MOVHUreg x))
	// cond: c < 16
	// result: (SRLI [48+c] (SLLI <typ.UInt64> [48] x))
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 16) {
			break
		}
		v.reset(OpRISCV64SRLI)
		v.AuxInt = int64ToAuxInt(48 + c)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(48)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRLI [c] (MOVWUreg x))
	// cond: c < 32
	// result: (SRLI [32+c] (SLLI <typ.UInt64> [32] x))
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 32) {
			break
		}
		v.reset(OpRISCV64SRLI)
		v.AuxInt = int64ToAuxInt(32 + c)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRLI [0] x)
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64XOR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XOR x (NOT y))
	// result: (XNOR x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpRISCV64NOT {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpRISCV64XNOR)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteBlockRISCV64latelower(b *Block) bool {
	return false
}
