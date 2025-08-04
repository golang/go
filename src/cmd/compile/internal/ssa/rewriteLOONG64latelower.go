// Code generated from _gen/LOONG64latelower.rules using 'go generate'; DO NOT EDIT.

package ssa

func rewriteValueLOONG64latelower(v *Value) bool {
	switch v.Op {
	case OpLOONG64SLLVconst:
		return rewriteValueLOONG64latelower_OpLOONG64SLLVconst(v)
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
	return false
}
