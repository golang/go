// Code generated from _gen/RISCV64latelower.rules using 'go generate'; DO NOT EDIT.

package riscv64latelower

import "internal/buildcfg"
import "cmd/compile/internal/ssa/block"
import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa"

func RewriteValue(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpRISCV64AND:
		return rewriteValue_OpRISCV64AND(v)
	case ssaop.OpRISCV64ANDI:
		return rewriteValue_OpRISCV64ANDI(v)
	case ssaop.OpRISCV64NOT:
		return rewriteValue_OpRISCV64NOT(v)
	case ssaop.OpRISCV64OR:
		return rewriteValue_OpRISCV64OR(v)
	case ssaop.OpRISCV64SLLI:
		return rewriteValue_OpRISCV64SLLI(v)
	case ssaop.OpRISCV64SNEZ:
		return rewriteValue_OpRISCV64SNEZ(v)
	case ssaop.OpRISCV64SRAI:
		return rewriteValue_OpRISCV64SRAI(v)
	case ssaop.OpRISCV64SRLI:
		return rewriteValue_OpRISCV64SRLI(v)
	case ssaop.OpRISCV64XOR:
		return rewriteValue_OpRISCV64XOR(v)
	case ssaop.OpRISCV64XORI:
		return rewriteValue_OpRISCV64XORI(v)
	}
	return false
}
func rewriteValue_OpRISCV64AND(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AND x (NOT y))
	// result: (ANDN x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpRISCV64NOT {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpRISCV64ANDN)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpRISCV64ANDI(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ANDI [1] (SRLI [c] x))
	// cond: buildcfg.GORISCV64 >= 23
	// result: (BEXTI [c] x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 1 || v_0.Op != ssaop.OpRISCV64SRLI {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(buildcfg.GORISCV64 >= 23) {
			break
		}
		v.Reset(ssaop.OpRISCV64BEXTI)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (ANDI [1] (SRLIW [c] x))
	// cond: buildcfg.GORISCV64 >= 23
	// result: (BEXTI [c] x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 1 || v_0.Op != ssaop.OpRISCV64SRLIW {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(buildcfg.GORISCV64 >= 23) {
			break
		}
		v.Reset(ssaop.OpRISCV64BEXTI)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64NOT(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (NOT (XOR x y))
	// result: (XNOR x y)
	for {
		if v_0.Op != ssaop.OpRISCV64XOR {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpRISCV64XNOR)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64OR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (OR x (NOT y))
	// result: (ORN x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpRISCV64NOT {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpRISCV64ORN)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpRISCV64SLLI(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SLLI [c] (MOVBUreg x))
	// cond: c <= 56
	// result: (SRLI [56-c] (SLLI <typ.UInt64> [56] x))
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		if !(c <= 56) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRLI)
		v.AuxInt = ssa.Int64ToAuxInt(56 - c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(56)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SLLI [c] (MOVHUreg x))
	// cond: c <= 48
	// result: (SRLI [48-c] (SLLI <typ.UInt64> [48] x))
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		if !(c <= 48) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRLI)
		v.AuxInt = ssa.Int64ToAuxInt(48 - c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(48)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SLLI [c] (MOVWUreg x))
	// cond: c <= 32
	// result: (SRLI [32-c] (SLLI <typ.UInt64> [32] x))
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		if !(c <= 32) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRLI)
		v.AuxInt = ssa.Int64ToAuxInt(32 - c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SLLI [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SNEZ(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SNEZ (ANDI [c] y))
	// cond: ssa.IsPowerOfTwo(uint64(c)) && buildcfg.GORISCV64 >= 23
	// result: (BEXTI [ssa.Log64u(uint64(c))] y)
	for {
		if v_0.Op != ssaop.OpRISCV64ANDI {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		y := v_0.Args[0]
		if !(ssa.IsPowerOfTwo(uint64(c)) && buildcfg.GORISCV64 >= 23) {
			break
		}
		v.Reset(ssaop.OpRISCV64BEXTI)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64u(uint64(c)))
		v.AddArg(y)
		return true
	}
	// match: (SNEZ x:(BEXTI [_] _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64BEXTI {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SRAI(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SRAI [c] (MOVBreg x))
	// cond: c < 8
	// result: (SRAI [56+c] (SLLI <typ.Int64> [56] x))
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVBreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 8) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAI)
		v.AuxInt = ssa.Int64ToAuxInt(56 + c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(56)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRAI [c] (MOVHreg x))
	// cond: c < 16
	// result: (SRAI [48+c] (SLLI <typ.Int64> [48] x))
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVHreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 16) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAI)
		v.AuxInt = ssa.Int64ToAuxInt(48 + c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(48)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRAI [c] (MOVWreg x))
	// cond: c < 32
	// result: (SRAI [32+c] (SLLI <typ.Int64> [32] x))
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVWreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 32) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAI)
		v.AuxInt = ssa.Int64ToAuxInt(32 + c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRAI [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SRLI(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SRLI [c] (MOVBUreg x))
	// cond: c < 8
	// result: (SRLI [56+c] (SLLI <typ.UInt64> [56] x))
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 8) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRLI)
		v.AuxInt = ssa.Int64ToAuxInt(56 + c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(56)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRLI [c] (MOVHUreg x))
	// cond: c < 16
	// result: (SRLI [48+c] (SLLI <typ.UInt64> [48] x))
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 16) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRLI)
		v.AuxInt = ssa.Int64ToAuxInt(48 + c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(48)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRLI [c] (MOVWUreg x))
	// cond: c < 32
	// result: (SRLI [32+c] (SLLI <typ.UInt64> [32] x))
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 32) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRLI)
		v.AuxInt = ssa.Int64ToAuxInt(32 + c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRLI [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64XOR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (XOR x (NOT y))
	// result: (XNOR x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpRISCV64NOT {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpRISCV64XNOR)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (XOR (SLL (MOVDconst [1]) x) y)
	// cond: buildcfg.GORISCV64 >= 23
	// result: (BINV y x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64SLL {
				continue
			}
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpRISCV64MOVDconst || ssa.AuxIntToInt64(v_0_0.AuxInt) != 1 {
				continue
			}
			y := v_1
			if !(buildcfg.GORISCV64 >= 23) {
				continue
			}
			v.Reset(ssaop.OpRISCV64BINV)
			v.AddArg2(y, x)
			return true
		}
		break
	}
	// match: (XOR (SLLW <t> (MOVDconst [1]) x) y)
	// cond: buildcfg.GORISCV64 >= 23
	// result: (BINV y (ANDI <t> [31] x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64SLLW {
				continue
			}
			t := v_0.Type
			x := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpRISCV64MOVDconst || ssa.AuxIntToInt64(v_0_0.AuxInt) != 1 {
				continue
			}
			y := v_1
			if !(buildcfg.GORISCV64 >= 23) {
				continue
			}
			v.Reset(ssaop.OpRISCV64BINV)
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, t)
			v0.AuxInt = ssa.Int64ToAuxInt(31)
			v0.AddArg(x)
			v.AddArg2(y, v0)
			return true
		}
		break
	}
	// match: (XOR (MOVDconst [c]) x)
	// cond: ssa.IsPowerOfTwo(uint64(c)) && buildcfg.GORISCV64 >= 23
	// result: (BINVI [ssa.Log64u(uint64(c))] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			x := v_1
			if !(ssa.IsPowerOfTwo(uint64(c)) && buildcfg.GORISCV64 >= 23) {
				continue
			}
			v.Reset(ssaop.OpRISCV64BINVI)
			v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64u(uint64(c)))
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpRISCV64XORI(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (XORI <t> [c] x)
	// cond: t.Size() == 2 && !ssa.Is12Bit(c) && ssa.IsPowerOfTwo(uint16(c)) && buildcfg.GORISCV64 >= 23
	// result: (BINVI [ssa.Log16u(uint16(c))] x)
	for {
		t := v.Type
		c := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if !(t.Size() == 2 && !ssa.Is12Bit(c) && ssa.IsPowerOfTwo(uint16(c)) && buildcfg.GORISCV64 >= 23) {
			break
		}
		v.Reset(ssaop.OpRISCV64BINVI)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log16u(uint16(c)))
		v.AddArg(x)
		return true
	}
	// match: (XORI <t> [c] x)
	// cond: t.Size() == 4 && !ssa.Is12Bit(c) && ssa.IsPowerOfTwo(uint32(c)) && buildcfg.GORISCV64 >= 23
	// result: (BINVI [ssa.Log32u(uint32(c))] x)
	for {
		t := v.Type
		c := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if !(t.Size() == 4 && !ssa.Is12Bit(c) && ssa.IsPowerOfTwo(uint32(c)) && buildcfg.GORISCV64 >= 23) {
			break
		}
		v.Reset(ssaop.OpRISCV64BINVI)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log32u(uint32(c)))
		v.AddArg(x)
		return true
	}
	// match: (XORI <t> [c] x)
	// cond: t.Size() == 8 && !ssa.Is12Bit(c) && ssa.IsPowerOfTwo(uint64(c)) && buildcfg.GORISCV64 >= 23
	// result: (BINVI [ssa.Log64u(uint64(c))] x)
	for {
		t := v.Type
		c := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if !(t.Size() == 8 && !ssa.Is12Bit(c) && ssa.IsPowerOfTwo(uint64(c)) && buildcfg.GORISCV64 >= 23) {
			break
		}
		v.Reset(ssaop.OpRISCV64BINVI)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64u(uint64(c)))
		v.AddArg(x)
		return true
	}
	return false
}
func RewriteBlock(b *ssa.Block) bool {
	typ := &b.Func.Config.Types
	switch b.Kind {
	case block.BlockRISCV64BEQZ:
		// match: (BEQZ (ANDI [1] (SRL x y)) yes no)
		// cond: buildcfg.GORISCV64 >= 23
		// result: (BEQZ (BEXT <typ.UInt64> x y) yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64ANDI {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpRISCV64SRL {
				break
			}
			y := v_0_0.Args[1]
			x := v_0_0.Args[0]
			if !(buildcfg.GORISCV64 >= 23) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpRISCV64BEXT, typ.UInt64)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockRISCV64BEQZ, v0)
			return true
		}
	case block.BlockRISCV64BNEZ:
		// match: (BNEZ (ANDI [1] (SRL x y)) yes no)
		// cond: buildcfg.GORISCV64 >= 23
		// result: (BNEZ (BEXT <typ.UInt64> x y) yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64ANDI {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpRISCV64SRL {
				break
			}
			y := v_0_0.Args[1]
			x := v_0_0.Args[0]
			if !(buildcfg.GORISCV64 >= 23) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpRISCV64BEXT, typ.UInt64)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockRISCV64BNEZ, v0)
			return true
		}
	}
	return false
}
