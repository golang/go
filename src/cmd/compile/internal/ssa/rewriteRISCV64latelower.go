// Code generated from _gen/RISCV64latelower.rules using 'go generate'; DO NOT EDIT.

package ssa

import "internal/buildcfg"

func rewriteValueRISCV64latelower(v *Value) bool {
	switch v.Op {
	case OpRISCV64ADD:
		return rewriteValueRISCV64latelower_OpRISCV64ADD(v)
	case OpRISCV64AND:
		return rewriteValueRISCV64latelower_OpRISCV64AND(v)
	case OpRISCV64ANDI:
		return rewriteValueRISCV64latelower_OpRISCV64ANDI(v)
	case OpRISCV64ANDN:
		return rewriteValueRISCV64latelower_OpRISCV64ANDN(v)
	case OpRISCV64BEXTI:
		return rewriteValueRISCV64latelower_OpRISCV64BEXTI(v)
	case OpRISCV64NOT:
		return rewriteValueRISCV64latelower_OpRISCV64NOT(v)
	case OpRISCV64OR:
		return rewriteValueRISCV64latelower_OpRISCV64OR(v)
	case OpRISCV64ORI:
		return rewriteValueRISCV64latelower_OpRISCV64ORI(v)
	case OpRISCV64SEQZ:
		return rewriteValueRISCV64latelower_OpRISCV64SEQZ(v)
	case OpRISCV64SLLI:
		return rewriteValueRISCV64latelower_OpRISCV64SLLI(v)
	case OpRISCV64SNEZ:
		return rewriteValueRISCV64latelower_OpRISCV64SNEZ(v)
	case OpRISCV64SRAI:
		return rewriteValueRISCV64latelower_OpRISCV64SRAI(v)
	case OpRISCV64SRLI:
		return rewriteValueRISCV64latelower_OpRISCV64SRLI(v)
	case OpRISCV64XOR:
		return rewriteValueRISCV64latelower_OpRISCV64XOR(v)
	case OpRISCV64XORI:
		return rewriteValueRISCV64latelower_OpRISCV64XORI(v)
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64ADD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADD (SLLI [1] x) y)
	// cond: buildcfg.GORISCV64 >= 22
	// result: (SH1ADD x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64SLLI || auxIntToInt64(v_0.AuxInt) != 1 {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			if !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64SH1ADD)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADD (SLLI [2] x) y)
	// cond: buildcfg.GORISCV64 >= 22
	// result: (SH2ADD x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64SLLI || auxIntToInt64(v_0.AuxInt) != 2 {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			if !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64SH2ADD)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADD (SLLI [3] x) y)
	// cond: buildcfg.GORISCV64 >= 22
	// result: (SH3ADD x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64SLLI || auxIntToInt64(v_0.AuxInt) != 3 {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			if !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64SH3ADD)
			v.AddArg2(x, y)
			return true
		}
		break
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
	// match: (AND x (MOVDconst [y]))
	// cond: isUnsignedPowerOfTwo(uint64(^y)) && log64(^y) >= 10 && buildcfg.GORISCV64 >= 22
	// result: (BCLRI [log64(^y)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpRISCV64MOVDconst {
				continue
			}
			y := auxIntToInt64(v_1.AuxInt)
			if !(isUnsignedPowerOfTwo(uint64(^y)) && log64(^y) >= 10 && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BCLRI)
			v.AuxInt = int64ToAuxInt(log64(^y))
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64ANDI(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ANDI x [c])
	// cond: isUnsignedPowerOfTwo(uint64(^c)) && log64(^c) >= 10 && buildcfg.GORISCV64 >= 22
	// result: (BCLRI [log64(^c)] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		x := v_0
		if !(isUnsignedPowerOfTwo(uint64(^c)) && log64(^c) >= 10 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BCLRI)
		v.AuxInt = int64ToAuxInt(log64(^c))
		v.AddArg(x)
		return true
	}
	// match: (ANDI (AND (SRL x y) mask) [1])
	// cond: buildcfg.GORISCV64 >= 22
	// result: (AND (BEXT <typ.UInt64> x y) mask)
	for {
		if auxIntToInt64(v.AuxInt) != 1 || v_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRISCV64SRL {
				continue
			}
			y := v_0_0.Args[1]
			x := v_0_0.Args[0]
			mask := v_0_1
			if !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64AND)
			v0 := b.NewValue0(v.Pos, OpRISCV64BEXT, typ.UInt64)
			v0.AddArg2(x, y)
			v.AddArg2(v0, mask)
			return true
		}
		break
	}
	// match: (ANDI (AND (SRLW x y) mask) [1])
	// cond: buildcfg.GORISCV64 >= 22
	// result: (AND (BEXT <typ.UInt64> x y) mask)
	for {
		if auxIntToInt64(v.AuxInt) != 1 || v_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpRISCV64SRLW {
				continue
			}
			y := v_0_0.Args[1]
			x := v_0_0.Args[0]
			mask := v_0_1
			if !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64AND)
			v0 := b.NewValue0(v.Pos, OpRISCV64BEXT, typ.UInt64)
			v0.AddArg2(x, y)
			v.AddArg2(v0, mask)
			return true
		}
		break
	}
	// match: (ANDI (SRL x y) [1])
	// cond: buildcfg.GORISCV64 >= 22
	// result: (BEXT x y)
	for {
		if auxIntToInt64(v.AuxInt) != 1 || v_0.Op != OpRISCV64SRL {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BEXT)
		v.AddArg2(x, y)
		return true
	}
	// match: (ANDI (SRLI x [c]) [1])
	// cond: c < 64 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI [c] x)
	for {
		if auxIntToInt64(v.AuxInt) != 1 || v_0.Op != OpRISCV64SRLI {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c < 64 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BEXTI)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (ANDI (SRLW x y) [1])
	// cond: buildcfg.GORISCV64 >= 22
	// result: (BEXT x y)
	for {
		if auxIntToInt64(v.AuxInt) != 1 || v_0.Op != OpRISCV64SRLW {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BEXT)
		v.AddArg2(x, y)
		return true
	}
	// match: (ANDI (SRLIW x [c]) [1])
	// cond: c < 32 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI [c] x)
	for {
		if auxIntToInt64(v.AuxInt) != 1 || v_0.Op != OpRISCV64SRLIW {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c < 32 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BEXTI)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (ANDI (SRLI x [c]) [mask])
	// cond: isUnsignedPowerOfTwo(uint64(mask)) && c+log64(mask) < 64 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI [c+log64(mask)] x)
	for {
		mask := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64SRLI {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isUnsignedPowerOfTwo(uint64(mask)) && c+log64(mask) < 64 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BEXTI)
		v.AuxInt = int64ToAuxInt(c + log64(mask))
		v.AddArg(x)
		return true
	}
	// match: (ANDI (SRLIW x [c]) [mask])
	// cond: isUnsignedPowerOfTwo(uint64(mask)) && c+log64(mask) < 32 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI [c+log64(mask)] x)
	for {
		mask := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64SRLIW {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isUnsignedPowerOfTwo(uint64(mask)) && c+log64(mask) < 32 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BEXTI)
		v.AuxInt = int64ToAuxInt(c + log64(mask))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64ANDN(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ANDN x (SLL (MOVDconst [1]) y))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (BCLR x y)
	for {
		x := v_0
		if v_1.Op != OpRISCV64SLL {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1_0.AuxInt) != 1 || !(buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BCLR)
		v.AddArg2(x, y)
		return true
	}
	// match: (ANDN x (SLLW (MOVDconst [1]) y))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (BCLR x y)
	for {
		x := v_0
		if v_1.Op != OpRISCV64SLLW {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1_0.AuxInt) != 1 || !(buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BCLR)
		v.AddArg2(x, y)
		return true
	}
	// match: (ANDN x (SLLI [c] (MOVDconst [1])))
	// cond: c < 64 && buildcfg.GORISCV64 >= 22
	// result: (BCLRI [c] x)
	for {
		x := v_0
		if v_1.Op != OpRISCV64SLLI {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1_0.AuxInt) != 1 || !(c < 64 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BCLRI)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (ANDN x (SLLIW [c] (MOVDconst [1])))
	// cond: c < 32 && buildcfg.GORISCV64 >= 22
	// result: (BCLRI [c] x)
	for {
		x := v_0
		if v_1.Op != OpRISCV64SLLIW {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1_0.AuxInt) != 1 || !(c < 32 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BCLRI)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64BEXTI(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BEXTI [c] (SRLI [d] x))
	// cond: c+d < 64 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI <typ.UInt64> [c+d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64SRLI {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c+d < 64 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BEXTI)
		v.Type = typ.UInt64
		v.AuxInt = int64ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	// match: (BEXTI [c] (SRLIW [d] x))
	// cond: c+d < 32 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI <typ.UInt64> [c+d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64SRLIW {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c+d < 32 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BEXTI)
		v.Type = typ.UInt64
		v.AuxInt = int64ToAuxInt(c + d)
		v.AddArg(x)
		return true
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
	// match: (OR x (SLL (MOVDconst [1]) y))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (BSET x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpRISCV64SLL {
				continue
			}
			y := v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1_0.AuxInt) != 1 || !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BSET)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (OR x (SLLW (MOVDconst [1]) y))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (BSET x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpRISCV64SLLW {
				continue
			}
			y := v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1_0.AuxInt) != 1 || !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BSET)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (OR x (SLLI [c] (MOVDconst [1])))
	// cond: c < 64 && buildcfg.GORISCV64 >= 22
	// result: (BSETI [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpRISCV64SLLI {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1_0.AuxInt) != 1 || !(c < 64 && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BSETI)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (OR x (SLLIW [c] (MOVDconst [1])))
	// cond: c < 32 && buildcfg.GORISCV64 >= 22
	// result: (BSETI [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpRISCV64SLLIW {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1_0.AuxInt) != 1 || !(c < 32 && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BSETI)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (OR x (MOVDconst [y]))
	// cond: oneBit(y) && log64(y) >= 10 && buildcfg.GORISCV64 >= 22
	// result: (BSETI [log64(y)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpRISCV64MOVDconst {
				continue
			}
			y := auxIntToInt64(v_1.AuxInt)
			if !(oneBit(y) && log64(y) >= 10 && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BSETI)
			v.AuxInt = int64ToAuxInt(log64(y))
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64ORI(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ORI x [c])
	// cond: oneBit(c) && log64(c) >= 10 && buildcfg.GORISCV64 >= 22
	// result: (BSETI [log64(c)] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		x := v_0
		if !(oneBit(c) && log64(c) >= 10 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BSETI)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64SEQZ(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SEQZ (ANDI x [c]))
	// cond: isUnsignedPowerOfTwo(uint64(c)) && log64(c) >= 10 && buildcfg.GORISCV64 >= 22
	// result: (SEQZ (BEXTI <typ.UInt64> [log64(c)] x))
	for {
		if v_0.Op != OpRISCV64ANDI {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isUnsignedPowerOfTwo(uint64(c)) && log64(c) >= 10 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64BEXTI, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(log64(c))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SEQZ (AND x (SLL (MOVDconst [1]) y)))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (SEQZ (BEXT <typ.UInt64> x y))
	for {
		if v_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpRISCV64SLL {
				continue
			}
			y := v_0_1.Args[1]
			v_0_1_0 := v_0_1.Args[0]
			if v_0_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_1_0.AuxInt) != 1 || !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64SEQZ)
			v0 := b.NewValue0(v.Pos, OpRISCV64BEXT, typ.UInt64)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SEQZ (AND x (SLLW (MOVDconst [1]) y)))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (SEQZ (BEXT <typ.UInt64> x y))
	for {
		if v_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpRISCV64SLLW {
				continue
			}
			y := v_0_1.Args[1]
			v_0_1_0 := v_0_1.Args[0]
			if v_0_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_1_0.AuxInt) != 1 || !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64SEQZ)
			v0 := b.NewValue0(v.Pos, OpRISCV64BEXT, typ.UInt64)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SEQZ (AND x (SLLI [c] (MOVDconst [1]))))
	// cond: c < 64 && buildcfg.GORISCV64 >= 22
	// result: (SEQZ (BEXTI <typ.UInt64> [c] x))
	for {
		if v_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpRISCV64SLLI {
				continue
			}
			c := auxIntToInt64(v_0_1.AuxInt)
			v_0_1_0 := v_0_1.Args[0]
			if v_0_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_1_0.AuxInt) != 1 || !(c < 64 && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64SEQZ)
			v0 := b.NewValue0(v.Pos, OpRISCV64BEXTI, typ.UInt64)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SEQZ (AND x (SLLIW [c] (MOVDconst [1]))))
	// cond: c < 32 && buildcfg.GORISCV64 >= 22
	// result: (SEQZ (BEXTI <typ.UInt64> [c] x))
	for {
		if v_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpRISCV64SLLIW {
				continue
			}
			c := auxIntToInt64(v_0_1.AuxInt)
			v_0_1_0 := v_0_1.Args[0]
			if v_0_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_1_0.AuxInt) != 1 || !(c < 32 && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64SEQZ)
			v0 := b.NewValue0(v.Pos, OpRISCV64BEXTI, typ.UInt64)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SEQZ (AND x (MOVDconst [c])))
	// cond: oneBit(c) && log64(c) >= 10 && buildcfg.GORISCV64 >= 22
	// result: (SEQZ (BEXTI <typ.UInt64> [log64(c)] x))
	for {
		if v_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpRISCV64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_0_1.AuxInt)
			if !(oneBit(c) && log64(c) >= 10 && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64SEQZ)
			v0 := b.NewValue0(v.Pos, OpRISCV64BEXTI, typ.UInt64)
			v0.AuxInt = int64ToAuxInt(log64(c))
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SEQZ (ANDI (SRL x y) [1]))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (SEQZ (BEXT <typ.UInt64> x y))
	for {
		if v_0.Op != OpRISCV64ANDI || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64SRL {
			break
		}
		y := v_0_0.Args[1]
		x := v_0_0.Args[0]
		if !(buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64BEXT, typ.UInt64)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (SEQZ (ANDI (SRLW x y) [1]))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (SEQZ (BEXT <typ.UInt64> x y))
	for {
		if v_0.Op != OpRISCV64ANDI || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64SRLW {
			break
		}
		y := v_0_0.Args[1]
		x := v_0_0.Args[0]
		if !(buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64BEXT, typ.UInt64)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (SEQZ (ANDI (SRLI x [c]) [1]))
	// cond: c < 64 && buildcfg.GORISCV64 >= 22
	// result: (SEQZ (BEXTI <typ.UInt64> [c] x))
	for {
		if v_0.Op != OpRISCV64ANDI || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64SRLI {
			break
		}
		c := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if !(c < 64 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64BEXTI, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SEQZ (ANDI (SRLIW x [c]) [1]))
	// cond: c < 32 && buildcfg.GORISCV64 >= 22
	// result: (SEQZ (BEXTI <typ.UInt64> [c] x))
	for {
		if v_0.Op != OpRISCV64ANDI || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64SRLIW {
			break
		}
		c := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if !(c < 32 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64BEXTI, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SEQZ (ANDI (SLLI x [c]) [1]))
	// cond: c > 0 && c < 64 && buildcfg.GORISCV64 >= 22
	// result: (SEQZ (BEXTI <typ.UInt64> [64-c] x))
	for {
		if v_0.Op != OpRISCV64ANDI || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64SLLI {
			break
		}
		c := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if !(c > 0 && c < 64 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64BEXTI, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(64 - c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SEQZ (ANDI (SLLIW x [c]) [1]))
	// cond: c > 0 && c < 32 && buildcfg.GORISCV64 >= 22
	// result: (SEQZ (BEXTI <typ.UInt64> [32-c] x))
	for {
		if v_0.Op != OpRISCV64ANDI || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64SLLIW {
			break
		}
		c := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if !(c > 0 && c < 32 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64BEXTI, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(32 - c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
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
func rewriteValueRISCV64latelower_OpRISCV64SNEZ(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SNEZ (ANDI x [c]))
	// cond: isUnsignedPowerOfTwo(uint64(c)) && log64(c) >= 10 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI [log64(c)] x)
	for {
		if v_0.Op != OpRISCV64ANDI {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isUnsignedPowerOfTwo(uint64(c)) && log64(c) >= 10 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BEXTI)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg(x)
		return true
	}
	// match: (SNEZ (AND x (SLL (MOVDconst [1]) y)))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (BEXT x y)
	for {
		if v_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpRISCV64SLL {
				continue
			}
			y := v_0_1.Args[1]
			v_0_1_0 := v_0_1.Args[0]
			if v_0_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_1_0.AuxInt) != 1 || !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BEXT)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (SNEZ (AND x (SLLW (MOVDconst [1]) y)))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (BEXT x y)
	for {
		if v_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpRISCV64SLLW {
				continue
			}
			y := v_0_1.Args[1]
			v_0_1_0 := v_0_1.Args[0]
			if v_0_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_1_0.AuxInt) != 1 || !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BEXT)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (SNEZ (AND x (SLLI [c] (MOVDconst [1]))))
	// cond: c < 64 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI [c] x)
	for {
		if v_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpRISCV64SLLI {
				continue
			}
			c := auxIntToInt64(v_0_1.AuxInt)
			v_0_1_0 := v_0_1.Args[0]
			if v_0_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_1_0.AuxInt) != 1 || !(c < 64 && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BEXTI)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (SNEZ (AND x (SLLIW [c] (MOVDconst [1]))))
	// cond: c < 32 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI [c] x)
	for {
		if v_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpRISCV64SLLIW {
				continue
			}
			c := auxIntToInt64(v_0_1.AuxInt)
			v_0_1_0 := v_0_1.Args[0]
			if v_0_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_1_0.AuxInt) != 1 || !(c < 32 && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BEXTI)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (SNEZ (AND x (MOVDconst [c])))
	// cond: oneBit(c) && log64(c) >= 10 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI [log64(c)] x)
	for {
		if v_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpRISCV64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_0_1.AuxInt)
			if !(oneBit(c) && log64(c) >= 10 && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BEXTI)
			v.AuxInt = int64ToAuxInt(log64(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (SNEZ (MOVWUreg (AND x (SLL (MOVDconst [1]) y))))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (BEXT x y)
	for {
		if v_0.Op != OpRISCV64MOVWUreg {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0_0, v_0_0_1 = _i0+1, v_0_0_1, v_0_0_0 {
			x := v_0_0_0
			if v_0_0_1.Op != OpRISCV64SLL {
				continue
			}
			y := v_0_0_1.Args[1]
			v_0_0_1_0 := v_0_0_1.Args[0]
			if v_0_0_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_0_1_0.AuxInt) != 1 || !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BEXT)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (SNEZ (MOVWUreg (AND x (SLLW (MOVDconst [1]) y))))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (BEXT x y)
	for {
		if v_0.Op != OpRISCV64MOVWUreg {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0_0, v_0_0_1 = _i0+1, v_0_0_1, v_0_0_0 {
			x := v_0_0_0
			if v_0_0_1.Op != OpRISCV64SLLW {
				continue
			}
			y := v_0_0_1.Args[1]
			v_0_0_1_0 := v_0_0_1.Args[0]
			if v_0_0_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_0_1_0.AuxInt) != 1 || !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BEXT)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (SNEZ (MOVWUreg (AND x (SLLI [c] (MOVDconst [1])))))
	// cond: c < 64 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI [c] x)
	for {
		if v_0.Op != OpRISCV64MOVWUreg {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0_0, v_0_0_1 = _i0+1, v_0_0_1, v_0_0_0 {
			x := v_0_0_0
			if v_0_0_1.Op != OpRISCV64SLLI {
				continue
			}
			c := auxIntToInt64(v_0_0_1.AuxInt)
			v_0_0_1_0 := v_0_0_1.Args[0]
			if v_0_0_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_0_1_0.AuxInt) != 1 || !(c < 64 && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BEXTI)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (SNEZ (MOVWUreg (AND x (SLLIW [c] (MOVDconst [1])))))
	// cond: c < 32 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI [c] x)
	for {
		if v_0.Op != OpRISCV64MOVWUreg {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0_0, v_0_0_1 = _i0+1, v_0_0_1, v_0_0_0 {
			x := v_0_0_0
			if v_0_0_1.Op != OpRISCV64SLLIW {
				continue
			}
			c := auxIntToInt64(v_0_0_1.AuxInt)
			v_0_0_1_0 := v_0_0_1.Args[0]
			if v_0_0_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_0_1_0.AuxInt) != 1 || !(c < 32 && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BEXTI)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (SNEZ (MOVWreg (AND x (SLL (MOVDconst [1]) y))))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (BEXT x y)
	for {
		if v_0.Op != OpRISCV64MOVWreg {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0_0, v_0_0_1 = _i0+1, v_0_0_1, v_0_0_0 {
			x := v_0_0_0
			if v_0_0_1.Op != OpRISCV64SLL {
				continue
			}
			y := v_0_0_1.Args[1]
			v_0_0_1_0 := v_0_0_1.Args[0]
			if v_0_0_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_0_1_0.AuxInt) != 1 || !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BEXT)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (SNEZ (MOVWreg (AND x (SLLW (MOVDconst [1]) y))))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (BEXT x y)
	for {
		if v_0.Op != OpRISCV64MOVWreg {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0_0, v_0_0_1 = _i0+1, v_0_0_1, v_0_0_0 {
			x := v_0_0_0
			if v_0_0_1.Op != OpRISCV64SLLW {
				continue
			}
			y := v_0_0_1.Args[1]
			v_0_0_1_0 := v_0_0_1.Args[0]
			if v_0_0_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_0_1_0.AuxInt) != 1 || !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BEXT)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (SNEZ (MOVWreg (AND x (SLLI [c] (MOVDconst [1])))))
	// cond: c < 64 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI [c] x)
	for {
		if v_0.Op != OpRISCV64MOVWreg {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0_0, v_0_0_1 = _i0+1, v_0_0_1, v_0_0_0 {
			x := v_0_0_0
			if v_0_0_1.Op != OpRISCV64SLLI {
				continue
			}
			c := auxIntToInt64(v_0_0_1.AuxInt)
			v_0_0_1_0 := v_0_0_1.Args[0]
			if v_0_0_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_0_1_0.AuxInt) != 1 || !(c < 64 && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BEXTI)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (SNEZ (MOVWreg (AND x (SLLIW [c] (MOVDconst [1])))))
	// cond: c < 32 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI [c] x)
	for {
		if v_0.Op != OpRISCV64MOVWreg {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64AND {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0_0, v_0_0_1 = _i0+1, v_0_0_1, v_0_0_0 {
			x := v_0_0_0
			if v_0_0_1.Op != OpRISCV64SLLIW {
				continue
			}
			c := auxIntToInt64(v_0_0_1.AuxInt)
			v_0_0_1_0 := v_0_0_1.Args[0]
			if v_0_0_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0_0_1_0.AuxInt) != 1 || !(c < 32 && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BEXTI)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (SNEZ (ANDI (SRL x y) [1]))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (BEXT x y)
	for {
		if v_0.Op != OpRISCV64ANDI || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64SRL {
			break
		}
		y := v_0_0.Args[1]
		x := v_0_0.Args[0]
		if !(buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BEXT)
		v.AddArg2(x, y)
		return true
	}
	// match: (SNEZ (ANDI (SRLW x y) [1]))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (BEXT x y)
	for {
		if v_0.Op != OpRISCV64ANDI || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64SRLW {
			break
		}
		y := v_0_0.Args[1]
		x := v_0_0.Args[0]
		if !(buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BEXT)
		v.AddArg2(x, y)
		return true
	}
	// match: (SNEZ (ANDI (SRLI x [c]) [1]))
	// cond: c < 64 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI [c] x)
	for {
		if v_0.Op != OpRISCV64ANDI || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64SRLI {
			break
		}
		c := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if !(c < 64 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BEXTI)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SNEZ (ANDI (SRLIW x [c]) [1]))
	// cond: c < 32 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI [c] x)
	for {
		if v_0.Op != OpRISCV64ANDI || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64SRLIW {
			break
		}
		c := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if !(c < 32 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BEXTI)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SNEZ (ANDI (SLLI x [c]) [1]))
	// cond: c > 0 && c < 64 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI [64-c] x)
	for {
		if v_0.Op != OpRISCV64ANDI || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64SLLI {
			break
		}
		c := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if !(c > 0 && c < 64 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BEXTI)
		v.AuxInt = int64ToAuxInt(64 - c)
		v.AddArg(x)
		return true
	}
	// match: (SNEZ (ANDI (SLLIW x [c]) [1]))
	// cond: c > 0 && c < 32 && buildcfg.GORISCV64 >= 22
	// result: (BEXTI [32-c] x)
	for {
		if v_0.Op != OpRISCV64ANDI || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRISCV64SLLIW {
			break
		}
		c := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if !(c > 0 && c < 32 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BEXTI)
		v.AuxInt = int64ToAuxInt(32 - c)
		v.AddArg(x)
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
	// match: (XOR x (SLL (MOVDconst [1]) y))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (BINV x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpRISCV64SLL {
				continue
			}
			y := v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1_0.AuxInt) != 1 || !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BINV)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (XOR x (SLLW (MOVDconst [1]) y))
	// cond: buildcfg.GORISCV64 >= 22
	// result: (BINV x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpRISCV64SLLW {
				continue
			}
			y := v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1_0.AuxInt) != 1 || !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BINV)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (XOR x (SLLI [c] (MOVDconst [1])))
	// cond: c < 64 && buildcfg.GORISCV64 >= 22
	// result: (BINVI [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpRISCV64SLLI {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1_0.AuxInt) != 1 || !(c < 64 && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BINVI)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XOR x (SLLIW [c] (MOVDconst [1])))
	// cond: c < 32 && buildcfg.GORISCV64 >= 22
	// result: (BINVI [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpRISCV64SLLIW {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1_0.AuxInt) != 1 || !(c < 32 && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BINVI)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XOR x (MOVDconst [y]))
	// cond: oneBit(y) && log64(y) >= 10 && buildcfg.GORISCV64 >= 22
	// result: (BINVI [log64(y)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpRISCV64MOVDconst {
				continue
			}
			y := auxIntToInt64(v_1.AuxInt)
			if !(oneBit(y) && log64(y) >= 10 && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.reset(OpRISCV64BINVI)
			v.AuxInt = int64ToAuxInt(log64(y))
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64XORI(v *Value) bool {
	v_0 := v.Args[0]
	// match: (XORI x [c])
	// cond: oneBit(c) && log64(c) >= 10 && buildcfg.GORISCV64 >= 22
	// result: (BINVI [log64(c)] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		x := v_0
		if !(oneBit(c) && log64(c) >= 10 && buildcfg.GORISCV64 >= 22) {
			break
		}
		v.reset(OpRISCV64BINVI)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteBlockRISCV64latelower(b *Block) bool {
	return false
}
