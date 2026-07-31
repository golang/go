// Code generated from _gen/AMD64latelower.rules using 'go generate'; DO NOT EDIT.

package rewriteamd64latelower

import "internal/buildcfg"
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
func RewriteBlock(b *ssa.Block) bool {
	return false
}
