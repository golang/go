// Code generated from _gen/LOONG64latelower.rules using 'go generate'; DO NOT EDIT.

package rewriteloong64latelower

import "cmd/compile/internal/ssa/block"
import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa"

func RewriteValue(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpLOONG64MOVBUreg:
		return rewriteValue_OpLOONG64MOVBUreg(v)
	case ssaop.OpLOONG64MOVBreg:
		return rewriteValue_OpLOONG64MOVBreg(v)
	case ssaop.OpLOONG64MOVHUreg:
		return rewriteValue_OpLOONG64MOVHUreg(v)
	case ssaop.OpLOONG64MOVHreg:
		return rewriteValue_OpLOONG64MOVHreg(v)
	case ssaop.OpLOONG64MOVVconst:
		return rewriteValue_OpLOONG64MOVVconst(v)
	case ssaop.OpLOONG64MOVVnop:
		return rewriteValue_OpLOONG64MOVVnop(v)
	case ssaop.OpLOONG64MOVVreg:
		return rewriteValue_OpLOONG64MOVVreg(v)
	case ssaop.OpLOONG64MOVWUreg:
		return rewriteValue_OpLOONG64MOVWUreg(v)
	case ssaop.OpLOONG64MOVWreg:
		return rewriteValue_OpLOONG64MOVWreg(v)
	case ssaop.OpLOONG64SLLVconst:
		return rewriteValue_OpLOONG64SLLVconst(v)
	}
	return false
}
func rewriteValue_OpLOONG64MOVBUreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBUreg x)
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
	// match: (MOVBUreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVBreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBreg x:(MOVBload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVHUreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHUreg x)
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
	// match: (MOVHUreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHUreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVHreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHreg x:(MOVBload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVVconst(v *ssa.Value) bool {
	// match: (MOVVconst [0])
	// result: (ZERO)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpLOONG64ZERO)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVVnop(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVVnop (MOVVconst [c]))
	// result: (MOVVconst [c])
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVVreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVVreg x)
	// cond: x.Uses == 1
	// result: (MOVVnop x)
	for {
		x := v_0
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVnop)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVWUreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWUreg x)
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
	// match: (MOVWUreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVWUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVWUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHUreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVWUreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVWreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWreg x:(MOVBload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVWload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVWloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVWreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SLLVconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SLLVconst [1] x)
	// result: (ADDV x x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		x := v_0
		v.Reset(ssaop.OpLOONG64ADDV)
		v.AddArg2(x, x)
		return true
	}
	return false
}
func RewriteBlock(b *ssa.Block) bool {
	switch b.Kind {
	case block.BlockLOONG64EQZ:
		// match: (EQZ (XOR x y) yes no)
		// result: (BEQ x y yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64XOR {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				b.ResetWithControl2(block.BlockLOONG64BEQ, x, y)
				return true
			}
		}
	case block.BlockLOONG64NEZ:
		// match: (NEZ (XOR x y) yes no)
		// result: (BNE x y yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64XOR {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				b.ResetWithControl2(block.BlockLOONG64BNE, x, y)
				return true
			}
		}
	}
	return false
}
