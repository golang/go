// Code generated from _gen/MIPS64latelower.rules using 'go generate'; DO NOT EDIT.

package rewritemips64latelower

import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa"

func RewriteValue(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpMIPS64MOVVconst:
		return rewriteValue_OpMIPS64MOVVconst(v)
	}
	return false
}
func rewriteValue_OpMIPS64MOVVconst(v *ssa.Value) bool {
	// match: (MOVVconst [0])
	// result: (ZERO)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpMIPS64ZERO)
		return true
	}
	return false
}
func RewriteBlock(b *ssa.Block) bool {
	return false
}
