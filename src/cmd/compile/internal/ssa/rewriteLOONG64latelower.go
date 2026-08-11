// Code generated from _gen/LOONG64latelower.rules using 'go generate'; DO NOT EDIT.

package ssa

import "cmd/compile/internal/ssa/block"

func rewriteValueLOONG64latelower(v *Value) bool {
	switch v.Op {
	case OpLOONG64MOVBUreg:
		return rewriteValueLOONG64latelower_OpLOONG64MOVBUreg(v)
	case OpLOONG64MOVHUreg:
		return rewriteValueLOONG64latelower_OpLOONG64MOVHUreg(v)
	case OpLOONG64MOVVconst:
		return rewriteValueLOONG64latelower_OpLOONG64MOVVconst(v)
	case OpLOONG64MOVWUreg:
		return rewriteValueLOONG64latelower_OpLOONG64MOVWUreg(v)
	case OpLOONG64SLLVconst:
		return rewriteValueLOONG64latelower_OpLOONG64SLLVconst(v)
	}
	return false
}
func rewriteValueLOONG64latelower_OpLOONG64MOVBUreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBUreg x)
	// cond: ZeroUpper56Bits(x)
	// result: x
	for {
		x := v_0
		if !(ZeroUpper56Bits(x)) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueLOONG64latelower_OpLOONG64MOVHUreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHUreg x)
	// cond: ZeroUpper48Bits(x)
	// result: x
	for {
		x := v_0
		if !(ZeroUpper48Bits(x)) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueLOONG64latelower_OpLOONG64MOVVconst(v *Value) bool {
	// match: (MOVVconst [0])
	// result: (ZERO)
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		v.reset(OpLOONG64ZERO)
		return true
	}
	return false
}
func rewriteValueLOONG64latelower_OpLOONG64MOVWUreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWUreg x)
	// cond: ZeroUpper32Bits(x)
	// result: x
	for {
		x := v_0
		if !(ZeroUpper32Bits(x)) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueLOONG64latelower_OpLOONG64SLLVconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SLLVconst [1] x)
	// result: (ADDV x x)
	for {
		if auxIntToInt64(v.AuxInt) != 1 {
			break
		}
		x := v_0
		v.reset(OpLOONG64ADDV)
		v.AddArg2(x, x)
		return true
	}
	return false
}
func rewriteBlockLOONG64latelower(b *Block) bool {
	switch b.Kind {
	case block.BlockLOONG64EQZ:
		// match: (EQZ (XOR x y) yes no)
		// result: (BEQ x y yes no)
		for b.Controls[0].Op == OpLOONG64XOR {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				b.resetWithControl2(block.BlockLOONG64BEQ, x, y)
				return true
			}
		}
	case block.BlockLOONG64NEZ:
		// match: (NEZ (XOR x y) yes no)
		// result: (BNE x y yes no)
		for b.Controls[0].Op == OpLOONG64XOR {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				b.resetWithControl2(block.BlockLOONG64BNE, x, y)
				return true
			}
		}
	}
	return false
}
