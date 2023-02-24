// Code generated from _gen/PPC64latelower.rules using 'go generate'; DO NOT EDIT.

package ssa

import "internal/buildcfg"

func rewriteValuePPC64latelower(v *Value) bool {
	switch v.Op {
	case OpPPC64ISEL:
		return rewriteValuePPC64latelower_OpPPC64ISEL(v)
	case OpPPC64SETBC:
		return rewriteValuePPC64latelower_OpPPC64SETBC(v)
	case OpPPC64SETBCR:
		return rewriteValuePPC64latelower_OpPPC64SETBCR(v)
	}
	return false
}
func rewriteValuePPC64latelower_OpPPC64ISEL(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ISEL [a] x (MOVDconst [0]) z)
	// result: (ISELZ [a] x z)
	for {
		a := auxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != OpPPC64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		z := v_2
		v.reset(OpPPC64ISELZ)
		v.AuxInt = int32ToAuxInt(a)
		v.AddArg2(x, z)
		return true
	}
	// match: (ISEL [a] (MOVDconst [0]) y z)
	// result: (ISELZ [a^0x4] y z)
	for {
		a := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpPPC64MOVDconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		y := v_1
		z := v_2
		v.reset(OpPPC64ISELZ)
		v.AuxInt = int32ToAuxInt(a ^ 0x4)
		v.AddArg2(y, z)
		return true
	}
	return false
}
func rewriteValuePPC64latelower_OpPPC64SETBC(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETBC [2] cmp)
	// cond: buildcfg.GOPPC64 <= 9
	// result: (ISELZ [2] (MOVDconst [1]) cmp)
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		cmp := v_0
		if !(buildcfg.GOPPC64 <= 9) {
			break
		}
		v.reset(OpPPC64ISELZ)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v0.AuxInt = int64ToAuxInt(1)
		v.AddArg2(v0, cmp)
		return true
	}
	// match: (SETBC [0] cmp)
	// cond: buildcfg.GOPPC64 <= 9
	// result: (ISELZ [0] (MOVDconst [1]) cmp)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		cmp := v_0
		if !(buildcfg.GOPPC64 <= 9) {
			break
		}
		v.reset(OpPPC64ISELZ)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v0.AuxInt = int64ToAuxInt(1)
		v.AddArg2(v0, cmp)
		return true
	}
	// match: (SETBC [1] cmp)
	// cond: buildcfg.GOPPC64 <= 9
	// result: (ISELZ [1] (MOVDconst [1]) cmp)
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		cmp := v_0
		if !(buildcfg.GOPPC64 <= 9) {
			break
		}
		v.reset(OpPPC64ISELZ)
		v.AuxInt = int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v0.AuxInt = int64ToAuxInt(1)
		v.AddArg2(v0, cmp)
		return true
	}
	return false
}
func rewriteValuePPC64latelower_OpPPC64SETBCR(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETBCR [2] cmp)
	// cond: buildcfg.GOPPC64 <= 9
	// result: (ISELZ [6] (MOVDconst [1]) cmp)
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		cmp := v_0
		if !(buildcfg.GOPPC64 <= 9) {
			break
		}
		v.reset(OpPPC64ISELZ)
		v.AuxInt = int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v0.AuxInt = int64ToAuxInt(1)
		v.AddArg2(v0, cmp)
		return true
	}
	// match: (SETBCR [0] cmp)
	// cond: buildcfg.GOPPC64 <= 9
	// result: (ISELZ [4] (MOVDconst [1]) cmp)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		cmp := v_0
		if !(buildcfg.GOPPC64 <= 9) {
			break
		}
		v.reset(OpPPC64ISELZ)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v0.AuxInt = int64ToAuxInt(1)
		v.AddArg2(v0, cmp)
		return true
	}
	// match: (SETBCR [1] cmp)
	// cond: buildcfg.GOPPC64 <= 9
	// result: (ISELZ [5] (MOVDconst [1]) cmp)
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		cmp := v_0
		if !(buildcfg.GOPPC64 <= 9) {
			break
		}
		v.reset(OpPPC64ISELZ)
		v.AuxInt = int32ToAuxInt(5)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v0.AuxInt = int64ToAuxInt(1)
		v.AddArg2(v0, cmp)
		return true
	}
	return false
}
func rewriteBlockPPC64latelower(b *Block) bool {
	return false
}
