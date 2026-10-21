// Code generated from _gen/AMD64latelower.rules using 'go generate'; DO NOT EDIT.

package ssa

import "internal/buildcfg"

func rewriteValueAMD64latelower(v *Value) bool {
	switch v.Op {
	case OpAMD64MOVBQZX:
		return rewriteValueAMD64latelower_OpAMD64MOVBQZX(v)
	case OpAMD64MOVLQZX:
		return rewriteValueAMD64latelower_OpAMD64MOVLQZX(v)
	case OpAMD64MOVWQZX:
		return rewriteValueAMD64latelower_OpAMD64MOVWQZX(v)
	case OpAMD64ROLLconst:
		return rewriteValueAMD64latelower_OpAMD64ROLLconst(v)
	case OpAMD64ROLQconst:
		return rewriteValueAMD64latelower_OpAMD64ROLQconst(v)
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
func rewriteValueAMD64latelower_OpAMD64MOVBQZX(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBQZX x)
	// cond: ZeroUpper56Bits(x,3)
	// result: x
	for {
		x := v_0
		if !(ZeroUpper56Bits(x, 3)) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64MOVLQZX(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVLQZX x)
	// cond: ZeroUpper32Bits(x,3)
	// result: x
	for {
		x := v_0
		if !(ZeroUpper32Bits(x, 3)) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64MOVWQZX(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWQZX x)
	// cond: ZeroUpper48Bits(x,3)
	// result: x
	for {
		x := v_0
		if !(ZeroUpper48Bits(x, 3)) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64ROLLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ROLLconst [c] x)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (RORXL [(-c)&31] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		x := v_0
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64RORXL)
		v.AuxInt = int8ToAuxInt((-c) & 31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64ROLQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ROLQconst [c] x)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (RORXQ [(-c)&63] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		x := v_0
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64RORXQ)
		v.AuxInt = int8ToAuxInt((-c) & 63)
		v.AddArg(x)
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
