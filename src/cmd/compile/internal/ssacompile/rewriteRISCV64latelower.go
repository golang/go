// Code generated from _gen/RISCV64latelower.rules using 'go generate'; DO NOT EDIT.

package ssacompile

import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa/ssacore"

func rewriteValueRISCV64latelower(v *ssacore.Value) bool {
	switch v.Op {
	case ssaop.OpRISCV64AND:
		return rewriteValueRISCV64latelower_OpRISCV64AND(v)
	case ssaop.OpRISCV64NOT:
		return rewriteValueRISCV64latelower_OpRISCV64NOT(v)
	case ssaop.OpRISCV64OR:
		return rewriteValueRISCV64latelower_OpRISCV64OR(v)
	case ssaop.OpRISCV64SLLI:
		return rewriteValueRISCV64latelower_OpRISCV64SLLI(v)
	case ssaop.OpRISCV64SRAI:
		return rewriteValueRISCV64latelower_OpRISCV64SRAI(v)
	case ssaop.OpRISCV64SRLI:
		return rewriteValueRISCV64latelower_OpRISCV64SRLI(v)
	case ssaop.OpRISCV64XOR:
		return rewriteValueRISCV64latelower_OpRISCV64XOR(v)
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64AND(v *ssacore.Value) bool {
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
func rewriteValueRISCV64latelower_OpRISCV64NOT(v *ssacore.Value) bool {
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
func rewriteValueRISCV64latelower_OpRISCV64OR(v *ssacore.Value) bool {
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
func rewriteValueRISCV64latelower_OpRISCV64SLLI(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SLLI [c] (MOVBUreg x))
	// cond: c <= 56
	// result: (SRLI [56-c] (SLLI <typ.UInt64> [56] x))
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		if !(c <= 56) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRLI)
		v.AuxInt = Int64ToAuxInt(56 - c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(56)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SLLI [c] (MOVHUreg x))
	// cond: c <= 48
	// result: (SRLI [48-c] (SLLI <typ.UInt64> [48] x))
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		if !(c <= 48) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRLI)
		v.AuxInt = Int64ToAuxInt(48 - c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(48)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SLLI [c] (MOVWUreg x))
	// cond: c <= 32
	// result: (SRLI [32-c] (SLLI <typ.UInt64> [32] x))
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		if !(c <= 32) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRLI)
		v.AuxInt = Int64ToAuxInt(32 - c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SLLI [0] x)
	// result: x
	for {
		if AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64SRAI(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SRAI [c] (MOVBreg x))
	// cond: c < 8
	// result: (SRAI [56+c] (SLLI <typ.Int64> [56] x))
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVBreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 8) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAI)
		v.AuxInt = Int64ToAuxInt(56 + c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(56)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRAI [c] (MOVHreg x))
	// cond: c < 16
	// result: (SRAI [48+c] (SLLI <typ.Int64> [48] x))
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVHreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 16) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAI)
		v.AuxInt = Int64ToAuxInt(48 + c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(48)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRAI [c] (MOVWreg x))
	// cond: c < 32
	// result: (SRAI [32+c] (SLLI <typ.Int64> [32] x))
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVWreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 32) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAI)
		v.AuxInt = Int64ToAuxInt(32 + c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRAI [0] x)
	// result: x
	for {
		if AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64SRLI(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SRLI [c] (MOVBUreg x))
	// cond: c < 8
	// result: (SRLI [56+c] (SLLI <typ.UInt64> [56] x))
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 8) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRLI)
		v.AuxInt = Int64ToAuxInt(56 + c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(56)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRLI [c] (MOVHUreg x))
	// cond: c < 16
	// result: (SRLI [48+c] (SLLI <typ.UInt64> [48] x))
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 16) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRLI)
		v.AuxInt = Int64ToAuxInt(48 + c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(48)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRLI [c] (MOVWUreg x))
	// cond: c < 32
	// result: (SRLI [32+c] (SLLI <typ.UInt64> [32] x))
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		if !(c < 32) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRLI)
		v.AuxInt = Int64ToAuxInt(32 + c)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SRLI [0] x)
	// result: x
	for {
		if AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValueRISCV64latelower_OpRISCV64XOR(v *ssacore.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
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
	return false
}
func rewriteBlockRISCV64latelower(b *ssacore.Block) bool {
	return false
}
