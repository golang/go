// Code generated from _gen/AMD64latelower.rules using 'go generate'; DO NOT EDIT.

package amd64latelower

import "internal/buildcfg"
import "cmd/compile/internal/types"
import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa"

func RewriteValue(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpAMD64MOVBQZX:
		return rewriteValue_OpAMD64MOVBQZX(v)
	case ssaop.OpAMD64MOVLQZX:
		return rewriteValue_OpAMD64MOVLQZX(v)
	case ssaop.OpAMD64MOVWQZX:
		return rewriteValue_OpAMD64MOVWQZX(v)
	case ssaop.OpAMD64SARL:
		return rewriteValue_OpAMD64SARL(v)
	case ssaop.OpAMD64SARQ:
		return rewriteValue_OpAMD64SARQ(v)
	case ssaop.OpAMD64SHLL:
		return rewriteValue_OpAMD64SHLL(v)
	case ssaop.OpAMD64SHLQ:
		return rewriteValue_OpAMD64SHLQ(v)
	case ssaop.OpAMD64SHRL:
		return rewriteValue_OpAMD64SHRL(v)
	case ssaop.OpAMD64SHRQ:
		return rewriteValue_OpAMD64SHRQ(v)
	case ssaop.OpAMD64VBROADCASTSS128:
		return rewriteValue_OpAMD64VBROADCASTSS128(v)
	case ssaop.OpAMD64VPBROADCASTB128:
		return rewriteValue_OpAMD64VPBROADCASTB128(v)
	case ssaop.OpAMD64VPBROADCASTB128load:
		return rewriteValue_OpAMD64VPBROADCASTB128load(v)
	case ssaop.OpAMD64VPBROADCASTD128:
		return rewriteValue_OpAMD64VPBROADCASTD128(v)
	case ssaop.OpAMD64VPBROADCASTD128load:
		return rewriteValue_OpAMD64VPBROADCASTD128load(v)
	case ssaop.OpAMD64VPBROADCASTQ128:
		return rewriteValue_OpAMD64VPBROADCASTQ128(v)
	case ssaop.OpAMD64VPBROADCASTQ128load:
		return rewriteValue_OpAMD64VPBROADCASTQ128load(v)
	case ssaop.OpAMD64VPBROADCASTW128:
		return rewriteValue_OpAMD64VPBROADCASTW128(v)
	case ssaop.OpAMD64VPBROADCASTW128load:
		return rewriteValue_OpAMD64VPBROADCASTW128load(v)
	}
	return false
}
func rewriteValue_OpAMD64MOVBQZX(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBQZX x)
	// cond: ssa.ZeroUpper56Bits(x)
	// result: x
	for {
		x := v_0
		if !(ssa.ZeroUpper56Bits(x)) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpAMD64MOVLQZX(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVLQZX x)
	// cond: ssa.ZeroUpper32Bits(x)
	// result: x
	for {
		x := v_0
		if !(ssa.ZeroUpper32Bits(x)) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpAMD64MOVWQZX(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWQZX x)
	// cond: ssa.ZeroUpper48Bits(x)
	// result: x
	for {
		x := v_0
		if !(ssa.ZeroUpper48Bits(x)) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpAMD64SARL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SARL x y)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (SARXL x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.Reset(ssaop.OpAMD64SARXL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpAMD64SARQ(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SARQ x y)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (SARXQ x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.Reset(ssaop.OpAMD64SARXQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpAMD64SHLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SHLL x y)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (SHLXL x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.Reset(ssaop.OpAMD64SHLXL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpAMD64SHLQ(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SHLQ x y)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (SHLXQ x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.Reset(ssaop.OpAMD64SHLXQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpAMD64SHRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SHRL x y)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (SHRXL x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.Reset(ssaop.OpAMD64SHRXL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpAMD64SHRQ(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SHRQ x y)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (SHRXQ x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.Reset(ssaop.OpAMD64SHRXQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpAMD64VBROADCASTSS128(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VBROADCASTSS128 x)
	// cond: !hasAVX2(v)
	// result: (VPSHUFD128 [0] x)
	for {
		x := v_0
		if !(!hasAVX2(v)) {
			break
		}
		v.Reset(ssaop.OpAMD64VPSHUFD128)
		v.AuxInt = ssa.Uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpAMD64VPBROADCASTB128(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (VPBROADCASTB128 x)
	// cond: !hasAVX2(v)
	// result: (VPSHUFB128 x (Zero128 <x.Type>))
	for {
		x := v_0
		if !(!hasAVX2(v)) {
			break
		}
		v.Reset(ssaop.OpAMD64VPSHUFB128)
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64Zero128, x.Type)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValue_OpAMD64VPBROADCASTB128load(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (VPBROADCASTB128load [off] {sym} ptr mem)
	// cond: !hasAVX2(v)
	// result: (VPBROADCASTB128 (VMOVDload <types.TypeVec128> [off] {sym} ptr mem))
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(!hasAVX2(v)) {
			break
		}
		v.Reset(ssaop.OpAMD64VPBROADCASTB128)
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64VMOVDload, types.TypeVec128)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpAMD64VPBROADCASTD128(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VPBROADCASTD128 x)
	// cond: !hasAVX2(v)
	// result: (VPSHUFD128 [0] x)
	for {
		x := v_0
		if !(!hasAVX2(v)) {
			break
		}
		v.Reset(ssaop.OpAMD64VPSHUFD128)
		v.AuxInt = ssa.Uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpAMD64VPBROADCASTD128load(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPBROADCASTD128load [off] {sym} ptr mem)
	// cond: !hasAVX2(v)
	// result: (VBROADCASTSS128load [off] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(!hasAVX2(v)) {
			break
		}
		v.Reset(ssaop.OpAMD64VBROADCASTSS128load)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpAMD64VPBROADCASTQ128(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VPBROADCASTQ128 x)
	// cond: !hasAVX2(v)
	// result: (VPSHUFD128 [0x44] x)
	for {
		x := v_0
		if !(!hasAVX2(v)) {
			break
		}
		v.Reset(ssaop.OpAMD64VPSHUFD128)
		v.AuxInt = ssa.Uint8ToAuxInt(0x44)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpAMD64VPBROADCASTQ128load(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (VPBROADCASTQ128load [off] {sym} ptr mem)
	// cond: !hasAVX2(v)
	// result: (VPBROADCASTQ128 (VMOVQload <types.TypeVec128> [off] {sym} ptr mem))
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(!hasAVX2(v)) {
			break
		}
		v.Reset(ssaop.OpAMD64VPBROADCASTQ128)
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64VMOVQload, types.TypeVec128)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpAMD64VPBROADCASTW128(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (VPBROADCASTW128 x)
	// cond: !hasAVX2(v)
	// result: (VPSHUFD128 [0] (VPUNPCKLWD128 <x.Type> x x))
	for {
		x := v_0
		if !(!hasAVX2(v)) {
			break
		}
		v.Reset(ssaop.OpAMD64VPSHUFD128)
		v.AuxInt = ssa.Uint8ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64VPUNPCKLWD128, x.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpAMD64VPBROADCASTW128load(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (VPBROADCASTW128load [off] {sym} ptr mem)
	// cond: !hasAVX2(v)
	// result: (VPBROADCASTW128 (VMOVDload <types.TypeVec128> [off] {sym} ptr mem))
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		mem := v_1
		if !(!hasAVX2(v)) {
			break
		}
		v.Reset(ssaop.OpAMD64VPBROADCASTW128)
		v0 := b.NewValue0(v.Pos, ssaop.OpAMD64VMOVDload, types.TypeVec128)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func RewriteBlock(b *ssa.Block) bool {
	return false
}
