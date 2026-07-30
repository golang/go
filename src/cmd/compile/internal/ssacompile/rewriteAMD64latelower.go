// Code generated from _gen/AMD64latelower.rules using 'go generate'; DO NOT EDIT.

package ssacompile

import "internal/buildcfg"
import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa/ssacore"

func rewriteValueAMD64latelower(v *ssacore.Value) bool {
	switch v.Op {
	case ssaop.OpAMD64MOVBQZX:
		return rewriteValueAMD64latelower_OpAMD64MOVBQZX(v)
	case ssaop.OpAMD64MOVLQZX:
		return rewriteValueAMD64latelower_OpAMD64MOVLQZX(v)
	case ssaop.OpAMD64MOVWQZX:
		return rewriteValueAMD64latelower_OpAMD64MOVWQZX(v)
	case ssaop.OpAMD64SARL:
		return rewriteValueAMD64latelower_OpAMD64SARL(v)
	case ssaop.OpAMD64SARQ:
		return rewriteValueAMD64latelower_OpAMD64SARQ(v)
	case ssaop.OpAMD64SHLL:
		return rewriteValueAMD64latelower_OpAMD64SHLL(v)
	case ssaop.OpAMD64SHLQ:
		return rewriteValueAMD64latelower_OpAMD64SHLQ(v)
	case ssaop.OpAMD64SHRL:
		return rewriteValueAMD64latelower_OpAMD64SHRL(v)
	case ssaop.OpAMD64SHRQ:
		return rewriteValueAMD64latelower_OpAMD64SHRQ(v)
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64MOVBQZX(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBQZX x)
	// cond: ssacore.ZeroUpper56Bits(x)
	// result: x
	for {
		x := v_0
		if !(ssacore.ZeroUpper56Bits(x)) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64MOVLQZX(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVLQZX x)
	// cond: ssacore.ZeroUpper32Bits(x)
	// result: x
	for {
		x := v_0
		if !(ssacore.ZeroUpper32Bits(x)) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64MOVWQZX(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWQZX x)
	// cond: ssacore.ZeroUpper48Bits(x)
	// result: x
	for {
		x := v_0
		if !(ssacore.ZeroUpper48Bits(x)) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64latelower_OpAMD64SARL(v *ssacore.Value) bool {
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
func rewriteValueAMD64latelower_OpAMD64SARQ(v *ssacore.Value) bool {
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
func rewriteValueAMD64latelower_OpAMD64SHLL(v *ssacore.Value) bool {
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
func rewriteValueAMD64latelower_OpAMD64SHLQ(v *ssacore.Value) bool {
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
func rewriteValueAMD64latelower_OpAMD64SHRL(v *ssacore.Value) bool {
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
func rewriteValueAMD64latelower_OpAMD64SHRQ(v *ssacore.Value) bool {
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
func rewriteBlockAMD64latelower(b *ssacore.Block) bool {
	return false
}
