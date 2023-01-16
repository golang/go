// Code generated from _gen/AMD64latelower.rules; DO NOT EDIT.
// generated with: cd _gen; go run .

package ssa

import "internal/buildcfg"

func rewriteValueAMD64latelower(v *Value) bool {
	switch v.Op {
	case OpAMD64LEAL1:
		return rewriteValueAMD64latelower_OpAMD64LEAL1(v)
	case OpAMD64LEAL2:
		return rewriteValueAMD64latelower_OpAMD64LEAL2(v)
	case OpAMD64LEAL4:
		return rewriteValueAMD64latelower_OpAMD64LEAL4(v)
	case OpAMD64LEAL8:
		return rewriteValueAMD64latelower_OpAMD64LEAL8(v)
	case OpAMD64LEAQ1:
		return rewriteValueAMD64latelower_OpAMD64LEAQ1(v)
	case OpAMD64LEAQ2:
		return rewriteValueAMD64latelower_OpAMD64LEAQ2(v)
	case OpAMD64LEAQ4:
		return rewriteValueAMD64latelower_OpAMD64LEAQ4(v)
	case OpAMD64LEAQ8:
		return rewriteValueAMD64latelower_OpAMD64LEAQ8(v)
	case OpAMD64LEAW1:
		return rewriteValueAMD64latelower_OpAMD64LEAW1(v)
	case OpAMD64LEAW2:
		return rewriteValueAMD64latelower_OpAMD64LEAW2(v)
	case OpAMD64LEAW4:
		return rewriteValueAMD64latelower_OpAMD64LEAW4(v)
	case OpAMD64LEAW8:
		return rewriteValueAMD64latelower_OpAMD64LEAW8(v)
	case OpAMD64SARL:
		return rewriteValueAMD64latelower_OpAMD64SARL(v)
	case OpAMD64SARQ:
		return rewriteValueAMD64latelower_OpAMD64SARQ(v)
	case OpAMD64SHLL:
		return rewriteValueAMD64latelower_OpAMD64SHLL(v)
	case OpAMD64SHLQ:
		return rewriteValueAMD64latelower_OpAMD64SHLQ(v)
	case OpAMD64SHRL:
		return rewriteValueAMD64latelower_OpAMD64SHRL(v)
	case OpAMD64SHRQ:
		return rewriteValueAMD64latelower_OpAMD64SHRQ(v)
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64LEAL1(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LEAL1 <t> [c] {s} x y)
	// cond: isPtr(x.Type) && c != 0 && s == nil
	// result: (ADDL x (ADDLconst <y.Type> [c] y))
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			y := v_1
			if !(isPtr(x.Type) && c != 0 && s == nil) {
				continue
			}
			v.reset(OpAMD64ADDL)
			v0 := b.NewValue0(v.Pos, OpAMD64ADDLconst, y.Type)
			v0.AuxInt = int32ToAuxInt(c)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (LEAL1 <t> [c] {s} x y)
	// cond: !isPtr(x.Type) && c != 0 && s == nil
	// result: (ADDL y (ADDLconst <x.Type> [c] x))
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			y := v_1
			if !(!isPtr(x.Type) && c != 0 && s == nil) {
				continue
			}
			v.reset(OpAMD64ADDL)
			v0 := b.NewValue0(v.Pos, OpAMD64ADDLconst, x.Type)
			v0.AuxInt = int32ToAuxInt(c)
			v0.AddArg(x)
			v.AddArg2(y, v0)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64LEAL2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LEAL2 <t> [c] {s} x y)
	// cond: !isPtr(t) && c != 0 && s == nil
	// result: (ADDLconst [c] (LEAL2 <x.Type> x y))
	for {
		t := v.Type
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		y := v_1
		if !(!isPtr(t) && c != 0 && s == nil) {
			break
		}
		v.reset(OpAMD64ADDLconst)
		v.AuxInt = int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL2, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64LEAL4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LEAL4 <t> [c] {s} x y)
	// cond: !isPtr(t) && c != 0 && s == nil
	// result: (ADDLconst [c] (LEAL4 <x.Type> x y))
	for {
		t := v.Type
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		y := v_1
		if !(!isPtr(t) && c != 0 && s == nil) {
			break
		}
		v.reset(OpAMD64ADDLconst)
		v.AuxInt = int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL4, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64LEAL8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LEAL8 <t> [c] {s} x y)
	// cond: !isPtr(t) && c != 0 && s == nil
	// result: (ADDLconst [c] (LEAL8 <x.Type> x y))
	for {
		t := v.Type
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		y := v_1
		if !(!isPtr(t) && c != 0 && s == nil) {
			break
		}
		v.reset(OpAMD64ADDLconst)
		v.AuxInt = int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL8, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64LEAQ1(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LEAQ1 <t> [c] {s} x y)
	// cond: isPtr(x.Type) && c != 0 && s == nil
	// result: (ADDQ x (ADDQconst <y.Type> [c] y))
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			y := v_1
			if !(isPtr(x.Type) && c != 0 && s == nil) {
				continue
			}
			v.reset(OpAMD64ADDQ)
			v0 := b.NewValue0(v.Pos, OpAMD64ADDQconst, y.Type)
			v0.AuxInt = int32ToAuxInt(c)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (LEAQ1 <t> [c] {s} x y)
	// cond: !isPtr(x.Type) && c != 0 && s == nil
	// result: (ADDQ y (ADDQconst <x.Type> [c] x))
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			y := v_1
			if !(!isPtr(x.Type) && c != 0 && s == nil) {
				continue
			}
			v.reset(OpAMD64ADDQ)
			v0 := b.NewValue0(v.Pos, OpAMD64ADDQconst, x.Type)
			v0.AuxInt = int32ToAuxInt(c)
			v0.AddArg(x)
			v.AddArg2(y, v0)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64LEAQ2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LEAQ2 <t> [c] {s} x y)
	// cond: !isPtr(t) && c != 0 && s == nil
	// result: (ADDQconst [c] (LEAQ2 <x.Type> x y))
	for {
		t := v.Type
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		y := v_1
		if !(!isPtr(t) && c != 0 && s == nil) {
			break
		}
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ2, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64LEAQ4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LEAQ4 <t> [c] {s} x y)
	// cond: !isPtr(t) && c != 0 && s == nil
	// result: (ADDQconst [c] (LEAQ4 <x.Type> x y))
	for {
		t := v.Type
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		y := v_1
		if !(!isPtr(t) && c != 0 && s == nil) {
			break
		}
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ4, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64LEAQ8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LEAQ8 <t> [c] {s} x y)
	// cond: !isPtr(t) && c != 0 && s == nil
	// result: (ADDQconst [c] (LEAQ8 <x.Type> x y))
	for {
		t := v.Type
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		y := v_1
		if !(!isPtr(t) && c != 0 && s == nil) {
			break
		}
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ8, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64LEAW1(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LEAW1 <t> [c] {s} x y)
	// cond: isPtr(x.Type) && c != 0 && s == nil
	// result: (ADDL x (ADDLconst <y.Type> [c] y))
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			y := v_1
			if !(isPtr(x.Type) && c != 0 && s == nil) {
				continue
			}
			v.reset(OpAMD64ADDL)
			v0 := b.NewValue0(v.Pos, OpAMD64ADDLconst, y.Type)
			v0.AuxInt = int32ToAuxInt(c)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (LEAW1 <t> [c] {s} x y)
	// cond: !isPtr(x.Type) && c != 0 && s == nil
	// result: (ADDL y (ADDLconst <x.Type> [c] x))
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			y := v_1
			if !(!isPtr(x.Type) && c != 0 && s == nil) {
				continue
			}
			v.reset(OpAMD64ADDL)
			v0 := b.NewValue0(v.Pos, OpAMD64ADDLconst, x.Type)
			v0.AuxInt = int32ToAuxInt(c)
			v0.AddArg(x)
			v.AddArg2(y, v0)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64LEAW2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LEAW2 <t> [c] {s} x y)
	// cond: !isPtr(t) && c != 0 && s == nil
	// result: (ADDLconst [c] (LEAW2 <x.Type> x y))
	for {
		t := v.Type
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		y := v_1
		if !(!isPtr(t) && c != 0 && s == nil) {
			break
		}
		v.reset(OpAMD64ADDLconst)
		v.AuxInt = int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAW2, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64LEAW4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LEAW4 <t> [c] {s} x y)
	// cond: !isPtr(t) && c != 0 && s == nil
	// result: (ADDLconst [c] (LEAW4 <x.Type> x y))
	for {
		t := v.Type
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		y := v_1
		if !(!isPtr(t) && c != 0 && s == nil) {
			break
		}
		v.reset(OpAMD64ADDLconst)
		v.AuxInt = int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAW4, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64LEAW8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LEAW8 <t> [c] {s} x y)
	// cond: !isPtr(t) && c != 0 && s == nil
	// result: (ADDLconst [c] (LEAW8 <x.Type> x y))
	for {
		t := v.Type
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		y := v_1
		if !(!isPtr(t) && c != 0 && s == nil) {
			break
		}
		v.reset(OpAMD64ADDLconst)
		v.AuxInt = int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAW8, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64SARL(v *Value) bool {
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
		v.reset(OpAMD64SARXL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64SARQ(v *Value) bool {
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
		v.reset(OpAMD64SARXQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64SHLL(v *Value) bool {
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
		v.reset(OpAMD64SHLXL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64SHLQ(v *Value) bool {
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
		v.reset(OpAMD64SHLXQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64SHRL(v *Value) bool {
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
		v.reset(OpAMD64SHRXL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64SHRQ(v *Value) bool {
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
		v.reset(OpAMD64SHRXQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteBlockAMD64latelower(b *Block) bool {
	return false
}
