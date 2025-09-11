// Code generated from _gen/MIPS64latelower.rules using 'go generate'; DO NOT EDIT.

package ssa

func rewriteValueMIPS64latelower(v *Value) bool {
	switch v.Op {
	case OpMIPS64MOVVconst:
		return rewriteValueMIPS64latelower_OpMIPS64MOVVconst(v)
	}
	return false
}
func rewriteValueMIPS64latelower_OpMIPS64MOVVconst(v *Value) bool {
	// match: (MOVVconst [0])
	// result: (ZERO)
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		v.reset(OpMIPS64ZERO)
		return true
	}
	return false
}
func rewriteBlockMIPS64latelower(b *Block) bool {
	return false
}
