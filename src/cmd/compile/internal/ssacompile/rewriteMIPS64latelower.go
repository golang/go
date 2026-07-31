// Code generated from _gen/MIPS64latelower.rules using 'go generate'; DO NOT EDIT.

package ssacompile

import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa"

func rewriteValueMIPS64latelower(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpMIPS64MOVVconst:
		return rewriteValueMIPS64latelower_OpMIPS64MOVVconst(v)
	}
	return false
}
func rewriteValueMIPS64latelower_OpMIPS64MOVVconst(v *ssa.Value) bool {
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
func rewriteBlockMIPS64latelower(b *ssa.Block) bool {
	return false
}
