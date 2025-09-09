// Code generated from _gen/RISCV64latelower.rules using 'go generate'; DO NOT EDIT.

package ssa

import "internal/buildcfg"

func rewriteValueRISCV64latelower(v *Value) bool {
	switch v.Op {
	case OpCondSelect:
		return rewriteValueRISCV64latelower_OpCondSelect(v)
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
func rewriteValueRISCV64latelower_OpCondSelect(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CondSelect <t> x y (SEQZ x))
	// cond: buildcfg.GORISCV64 >= 23
	// result: (CZEROEQZ <t> y x)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpRISCV64SEQZ || x != v_2.Args[0] || !(buildcfg.GORISCV64 >= 23) {
			break
		}
		v.reset(OpRISCV64CZEROEQZ)
		v.Type = t
		v.AddArg2(y, x)
		return true
	}
	// match: (CondSelect <t> (ADD x y) x (SEQZ z))
	// cond: buildcfg.GORISCV64 >= 23
	// result: (ADD x (CZERONEZ <t> y z))
	for {
		t := v.Type
		if v_0.Op != OpRISCV64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 || v_2.Op != OpRISCV64SEQZ {
				continue
			}
			z := v_2.Args[0]
			if !(buildcfg.GORISCV64 >= 23) {
				continue
			}
			v.reset(OpRISCV64ADD)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
			v0.AddArg2(y, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (CondSelect <t> (ADD x y) x (SNEZ z))
	// cond: buildcfg.GORISCV64 >= 23
	// result: (ADD x (CZEROEQZ <t> y z))
	for {
		t := v.Type
		if v_0.Op != OpRISCV64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 || v_2.Op != OpRISCV64SNEZ {
				continue
			}
			z := v_2.Args[0]
			if !(buildcfg.GORISCV64 >= 23) {
				continue
			}
			v.reset(OpRISCV64ADD)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
			v0.AddArg2(y, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (CondSelect <t> (SUB x y) x (SEQZ z))
	// cond: buildcfg.GORISCV64 >= 23
	// result: (SUB x (CZERONEZ <t> y z))
	for {
		t := v.Type
		if v_0.Op != OpRISCV64SUB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 || v_2.Op != OpRISCV64SEQZ {
			break
		}
		z := v_2.Args[0]
		if !(buildcfg.GORISCV64 >= 23) {
			break
		}
		v.reset(OpRISCV64SUB)
		v0 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
		v0.AddArg2(y, z)
		v.AddArg2(x, v0)
		return true
	}
	// match: (CondSelect <t> (SUB x y) x (SNEZ z))
	// cond: buildcfg.GORISCV64 >= 23
	// result: (SUB x (CZEROEQZ <t> y z))
	for {
		t := v.Type
		if v_0.Op != OpRISCV64SUB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 || v_2.Op != OpRISCV64SNEZ {
			break
		}
		z := v_2.Args[0]
		if !(buildcfg.GORISCV64 >= 23) {
			break
		}
		v.reset(OpRISCV64SUB)
		v0 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
		v0.AddArg2(y, z)
		v.AddArg2(x, v0)
		return true
	}
	// match: (CondSelect <t> (OR x y) x (SEQZ z))
	// cond: buildcfg.GORISCV64 >= 23
	// result: (OR x (CZERONEZ <t> y z))
	for {
		t := v.Type
		if v_0.Op != OpRISCV64OR {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 || v_2.Op != OpRISCV64SEQZ {
				continue
			}
			z := v_2.Args[0]
			if !(buildcfg.GORISCV64 >= 23) {
				continue
			}
			v.reset(OpRISCV64OR)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
			v0.AddArg2(y, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (CondSelect <t> (OR x y) x (SNEZ z))
	// cond: buildcfg.GORISCV64 >= 23
	// result: (OR x (CZEROEQZ <t> y z))
	for {
		t := v.Type
		if v_0.Op != OpRISCV64OR {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 || v_2.Op != OpRISCV64SNEZ {
				continue
			}
			z := v_2.Args[0]
			if !(buildcfg.GORISCV64 >= 23) {
				continue
			}
			v.reset(OpRISCV64OR)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
			v0.AddArg2(y, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (CondSelect <t> (XOR x y) x (SEQZ z))
	// cond: buildcfg.GORISCV64 >= 23
	// result: (XOR x (CZERONEZ <t> y z))
	for {
		t := v.Type
		if v_0.Op != OpRISCV64XOR {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 || v_2.Op != OpRISCV64SEQZ {
				continue
			}
			z := v_2.Args[0]
			if !(buildcfg.GORISCV64 >= 23) {
				continue
			}
			v.reset(OpRISCV64XOR)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
			v0.AddArg2(y, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (CondSelect <t> (XOR x y) x (SNEZ z))
	// cond: buildcfg.GORISCV64 >= 23
	// result: (XOR x (CZEROEQZ <t> y z))
	for {
		t := v.Type
		if v_0.Op != OpRISCV64XOR {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 || v_2.Op != OpRISCV64SNEZ {
				continue
			}
			z := v_2.Args[0]
			if !(buildcfg.GORISCV64 >= 23) {
				continue
			}
			v.reset(OpRISCV64XOR)
			v0 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
			v0.AddArg2(y, z)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (CondSelect <t> (AND x y) x (SEQZ z))
	// cond: buildcfg.GORISCV64 >= 23
	// result: (OR (AND <t> x y) (CZEROEQZ <t> x z))
	for {
		t := v.Type
		if v_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 || v_2.Op != OpRISCV64SEQZ {
				continue
			}
			z := v_2.Args[0]
			if !(buildcfg.GORISCV64 >= 23) {
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
		break
	}
	// match: (CondSelect <t> (AND x y) x (SNEZ z))
	// cond: buildcfg.GORISCV64 >= 23
	// result: (OR (AND <t> x y) (CZERONEZ <t> x z))
	for {
		t := v.Type
		if v_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 || v_2.Op != OpRISCV64SNEZ {
				continue
			}
			z := v_2.Args[0]
			if !(buildcfg.GORISCV64 >= 23) {
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
		break
	}
	// match: (CondSelect <t> x y (SEQZ z))
	// cond: buildcfg.GORISCV64 >= 23
	// result: (OR (CZERONEZ <t> x z) (CZEROEQZ <t> y z))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpRISCV64SEQZ {
			break
		}
		z := v_2.Args[0]
		if !(buildcfg.GORISCV64 >= 23) {
			break
		}
		v.reset(OpRISCV64OR)
		v0 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
		v0.AddArg2(x, z)
		v1 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
		v1.AddArg2(y, z)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (CondSelect <t> x y (SNEZ z))
	// cond: buildcfg.GORISCV64 >= 23
	// result: (OR (CZEROEQZ <t> x z) (CZERONEZ <t> y z))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpRISCV64SNEZ {
			break
		}
		z := v_2.Args[0]
		if !(buildcfg.GORISCV64 >= 23) {
			break
		}
		v.reset(OpRISCV64OR)
		v0 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
		v0.AddArg2(x, z)
		v1 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
		v1.AddArg2(y, z)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (CondSelect <t> x y cond)
	// cond: buildcfg.GORISCV64 >= 23
	// result: (OR (CZEROEQZ <t> x cond) (CZERONEZ <t> y cond))
	for {
		t := v.Type
		x := v_0
		y := v_1
		cond := v_2
		if !(buildcfg.GORISCV64 >= 23) {
			break
		}
		v.reset(OpRISCV64OR)
		v0 := b.NewValue0(v.Pos, OpRISCV64CZEROEQZ, t)
		v0.AddArg2(x, cond)
		v1 := b.NewValue0(v.Pos, OpRISCV64CZERONEZ, t)
		v1.AddArg2(y, cond)
		v.AddArg2(v0, v1)
		return true
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
