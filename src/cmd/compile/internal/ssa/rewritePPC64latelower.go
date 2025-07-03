// Code generated from _gen/PPC64latelower.rules using 'go generate'; DO NOT EDIT.

package ssa

import "internal/buildcfg"

func rewriteValuePPC64latelower(v *Value) bool {
	switch v.Op {
	case OpPPC64ADD:
		return rewriteValuePPC64latelower_OpPPC64ADD(v)
	case OpPPC64AND:
		return rewriteValuePPC64latelower_OpPPC64AND(v)
	case OpPPC64ANDconst:
		return rewriteValuePPC64latelower_OpPPC64ANDconst(v)
	case OpPPC64CMPconst:
		return rewriteValuePPC64latelower_OpPPC64CMPconst(v)
	case OpPPC64ISEL:
		return rewriteValuePPC64latelower_OpPPC64ISEL(v)
	case OpPPC64RLDICL:
		return rewriteValuePPC64latelower_OpPPC64RLDICL(v)
	case OpPPC64RLDICLCC:
		return rewriteValuePPC64latelower_OpPPC64RLDICLCC(v)
	case OpPPC64SETBC:
		return rewriteValuePPC64latelower_OpPPC64SETBC(v)
	case OpPPC64SETBCR:
		return rewriteValuePPC64latelower_OpPPC64SETBCR(v)
	}
	return false
}
func rewriteValuePPC64latelower_OpPPC64ADD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADD (MOVDconst [m]) x)
	// cond: supportsPPC64PCRel() && (m<<30)>>30 == m
	// result: (ADDconst [m] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpPPC64MOVDconst {
				continue
			}
			m := auxIntToInt64(v_0.AuxInt)
			x := v_1
			if !(supportsPPC64PCRel() && (m<<30)>>30 == m) {
				continue
			}
			v.reset(OpPPC64ADDconst)
			v.AuxInt = int64ToAuxInt(m)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64latelower_OpPPC64AND(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AND <t> x:(MOVDconst [m]) n)
	// cond: t.Size() <= 2
	// result: (ANDconst [int64(int16(m))] n)
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if x.Op != OpPPC64MOVDconst {
				continue
			}
			m := auxIntToInt64(x.AuxInt)
			n := v_1
			if !(t.Size() <= 2) {
				continue
			}
			v.reset(OpPPC64ANDconst)
			v.AuxInt = int64ToAuxInt(int64(int16(m)))
			v.AddArg(n)
			return true
		}
		break
	}
	// match: (AND x:(MOVDconst [m]) n)
	// cond: isPPC64ValidShiftMask(m)
	// result: (RLDICL [encodePPC64RotateMask(0,m,64)] n)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if x.Op != OpPPC64MOVDconst {
				continue
			}
			m := auxIntToInt64(x.AuxInt)
			n := v_1
			if !(isPPC64ValidShiftMask(m)) {
				continue
			}
			v.reset(OpPPC64RLDICL)
			v.AuxInt = int64ToAuxInt(encodePPC64RotateMask(0, m, 64))
			v.AddArg(n)
			return true
		}
		break
	}
	// match: (AND x:(MOVDconst [m]) n)
	// cond: m != 0 && isPPC64ValidShiftMask(^m)
	// result: (RLDICR [encodePPC64RotateMask(0,m,64)] n)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if x.Op != OpPPC64MOVDconst {
				continue
			}
			m := auxIntToInt64(x.AuxInt)
			n := v_1
			if !(m != 0 && isPPC64ValidShiftMask(^m)) {
				continue
			}
			v.reset(OpPPC64RLDICR)
			v.AuxInt = int64ToAuxInt(encodePPC64RotateMask(0, m, 64))
			v.AddArg(n)
			return true
		}
		break
	}
	// match: (AND <t> x:(MOVDconst [m]) n)
	// cond: t.Size() == 4 && isPPC64WordRotateMask(m)
	// result: (RLWINM [encodePPC64RotateMask(0,m,32)] n)
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if x.Op != OpPPC64MOVDconst {
				continue
			}
			m := auxIntToInt64(x.AuxInt)
			n := v_1
			if !(t.Size() == 4 && isPPC64WordRotateMask(m)) {
				continue
			}
			v.reset(OpPPC64RLWINM)
			v.AuxInt = int64ToAuxInt(encodePPC64RotateMask(0, m, 32))
			v.AddArg(n)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64latelower_OpPPC64ANDconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ANDconst [m] x)
	// cond: isPPC64ValidShiftMask(m)
	// result: (RLDICL [encodePPC64RotateMask(0,m,64)] x)
	for {
		m := auxIntToInt64(v.AuxInt)
		x := v_0
		if !(isPPC64ValidShiftMask(m)) {
			break
		}
		v.reset(OpPPC64RLDICL)
		v.AuxInt = int64ToAuxInt(encodePPC64RotateMask(0, m, 64))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64latelower_OpPPC64CMPconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CMPconst [0] z:(ADD x y))
	// cond: v.Block == z.Block
	// result: (CMPconst [0] convertPPC64OpToOpCC(z))
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		z := v_0
		if z.Op != OpPPC64ADD {
			break
		}
		if !(v.Block == z.Block) {
			break
		}
		v.reset(OpPPC64CMPconst)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg(convertPPC64OpToOpCC(z))
		return true
	}
	// match: (CMPconst [0] z:(AND x y))
	// cond: v.Block == z.Block
	// result: (CMPconst [0] convertPPC64OpToOpCC(z))
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		z := v_0
		if z.Op != OpPPC64AND {
			break
		}
		if !(v.Block == z.Block) {
			break
		}
		v.reset(OpPPC64CMPconst)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg(convertPPC64OpToOpCC(z))
		return true
	}
	// match: (CMPconst [0] z:(ANDN x y))
	// cond: v.Block == z.Block
	// result: (CMPconst [0] convertPPC64OpToOpCC(z))
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		z := v_0
		if z.Op != OpPPC64ANDN {
			break
		}
		if !(v.Block == z.Block) {
			break
		}
		v.reset(OpPPC64CMPconst)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg(convertPPC64OpToOpCC(z))
		return true
	}
	// match: (CMPconst [0] z:(OR x y))
	// cond: v.Block == z.Block
	// result: (CMPconst [0] convertPPC64OpToOpCC(z))
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		z := v_0
		if z.Op != OpPPC64OR {
			break
		}
		if !(v.Block == z.Block) {
			break
		}
		v.reset(OpPPC64CMPconst)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg(convertPPC64OpToOpCC(z))
		return true
	}
	// match: (CMPconst [0] z:(SUB x y))
	// cond: v.Block == z.Block
	// result: (CMPconst [0] convertPPC64OpToOpCC(z))
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		z := v_0
		if z.Op != OpPPC64SUB {
			break
		}
		if !(v.Block == z.Block) {
			break
		}
		v.reset(OpPPC64CMPconst)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg(convertPPC64OpToOpCC(z))
		return true
	}
	// match: (CMPconst [0] z:(NOR x y))
	// cond: v.Block == z.Block
	// result: (CMPconst [0] convertPPC64OpToOpCC(z))
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		z := v_0
		if z.Op != OpPPC64NOR {
			break
		}
		if !(v.Block == z.Block) {
			break
		}
		v.reset(OpPPC64CMPconst)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg(convertPPC64OpToOpCC(z))
		return true
	}
	// match: (CMPconst [0] z:(XOR x y))
	// cond: v.Block == z.Block
	// result: (CMPconst [0] convertPPC64OpToOpCC(z))
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		z := v_0
		if z.Op != OpPPC64XOR {
			break
		}
		if !(v.Block == z.Block) {
			break
		}
		v.reset(OpPPC64CMPconst)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg(convertPPC64OpToOpCC(z))
		return true
	}
	// match: (CMPconst [0] z:(MULHDU x y))
	// cond: v.Block == z.Block
	// result: (CMPconst [0] convertPPC64OpToOpCC(z))
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		z := v_0
		if z.Op != OpPPC64MULHDU {
			break
		}
		if !(v.Block == z.Block) {
			break
		}
		v.reset(OpPPC64CMPconst)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg(convertPPC64OpToOpCC(z))
		return true
	}
	// match: (CMPconst [0] z:(NEG x))
	// cond: v.Block == z.Block
	// result: (CMPconst [0] convertPPC64OpToOpCC(z))
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		z := v_0
		if z.Op != OpPPC64NEG {
			break
		}
		if !(v.Block == z.Block) {
			break
		}
		v.reset(OpPPC64CMPconst)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg(convertPPC64OpToOpCC(z))
		return true
	}
	// match: (CMPconst [0] z:(CNTLZD x))
	// cond: v.Block == z.Block
	// result: (CMPconst [0] convertPPC64OpToOpCC(z))
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		z := v_0
		if z.Op != OpPPC64CNTLZD {
			break
		}
		if !(v.Block == z.Block) {
			break
		}
		v.reset(OpPPC64CMPconst)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg(convertPPC64OpToOpCC(z))
		return true
	}
	// match: (CMPconst [0] z:(RLDICL x))
	// cond: v.Block == z.Block
	// result: (CMPconst [0] convertPPC64OpToOpCC(z))
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		z := v_0
		if z.Op != OpPPC64RLDICL {
			break
		}
		if !(v.Block == z.Block) {
			break
		}
		v.reset(OpPPC64CMPconst)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg(convertPPC64OpToOpCC(z))
		return true
	}
	// match: (CMPconst [0] z:(ADDconst [c] x))
	// cond: int64(int16(c)) == c && v.Block == z.Block
	// result: (CMPconst [0] convertPPC64OpToOpCC(z))
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		z := v_0
		if z.Op != OpPPC64ADDconst {
			break
		}
		c := auxIntToInt64(z.AuxInt)
		if !(int64(int16(c)) == c && v.Block == z.Block) {
			break
		}
		v.reset(OpPPC64CMPconst)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg(convertPPC64OpToOpCC(z))
		return true
	}
	// match: (CMPconst [0] z:(ANDconst [c] x))
	// cond: int64(uint16(c)) == c && v.Block == z.Block
	// result: (CMPconst [0] convertPPC64OpToOpCC(z))
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		z := v_0
		if z.Op != OpPPC64ANDconst {
			break
		}
		c := auxIntToInt64(z.AuxInt)
		if !(int64(uint16(c)) == c && v.Block == z.Block) {
			break
		}
		v.reset(OpPPC64CMPconst)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg(convertPPC64OpToOpCC(z))
		return true
	}
	// match: (CMPconst <t> [0] (Select0 z:(ADDCC x y)))
	// result: (Select1 <t> z)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpSelect0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpPPC64ADDCC {
			break
		}
		v.reset(OpSelect1)
		v.Type = t
		v.AddArg(z)
		return true
	}
	// match: (CMPconst <t> [0] (Select0 z:(ANDCC x y)))
	// result: (Select1 <t> z)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpSelect0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpPPC64ANDCC {
			break
		}
		v.reset(OpSelect1)
		v.Type = t
		v.AddArg(z)
		return true
	}
	// match: (CMPconst <t> [0] (Select0 z:(ANDNCC x y)))
	// result: (Select1 <t> z)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpSelect0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpPPC64ANDNCC {
			break
		}
		v.reset(OpSelect1)
		v.Type = t
		v.AddArg(z)
		return true
	}
	// match: (CMPconst <t> [0] (Select0 z:(ORCC x y)))
	// result: (Select1 <t> z)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpSelect0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpPPC64ORCC {
			break
		}
		v.reset(OpSelect1)
		v.Type = t
		v.AddArg(z)
		return true
	}
	// match: (CMPconst <t> [0] (Select0 z:(SUBCC x y)))
	// result: (Select1 <t> z)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpSelect0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpPPC64SUBCC {
			break
		}
		v.reset(OpSelect1)
		v.Type = t
		v.AddArg(z)
		return true
	}
	// match: (CMPconst <t> [0] (Select0 z:(NORCC x y)))
	// result: (Select1 <t> z)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpSelect0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpPPC64NORCC {
			break
		}
		v.reset(OpSelect1)
		v.Type = t
		v.AddArg(z)
		return true
	}
	// match: (CMPconst <t> [0] (Select0 z:(XORCC x y)))
	// result: (Select1 <t> z)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpSelect0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpPPC64XORCC {
			break
		}
		v.reset(OpSelect1)
		v.Type = t
		v.AddArg(z)
		return true
	}
	// match: (CMPconst <t> [0] (Select0 z:(MULHDUCC x y)))
	// result: (Select1 <t> z)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpSelect0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpPPC64MULHDUCC {
			break
		}
		v.reset(OpSelect1)
		v.Type = t
		v.AddArg(z)
		return true
	}
	// match: (CMPconst <t> [0] (Select0 z:(ADDCCconst y)))
	// result: (Select1 <t> z)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpSelect0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpPPC64ADDCCconst {
			break
		}
		v.reset(OpSelect1)
		v.Type = t
		v.AddArg(z)
		return true
	}
	// match: (CMPconst <t> [0] (Select0 z:(ANDCCconst y)))
	// result: (Select1 <t> z)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpSelect0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpPPC64ANDCCconst {
			break
		}
		v.reset(OpSelect1)
		v.Type = t
		v.AddArg(z)
		return true
	}
	// match: (CMPconst <t> [0] (Select0 z:(NEGCC y)))
	// result: (Select1 <t> z)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpSelect0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpPPC64NEGCC {
			break
		}
		v.reset(OpSelect1)
		v.Type = t
		v.AddArg(z)
		return true
	}
	// match: (CMPconst <t> [0] (Select0 z:(CNTLZDCC y)))
	// result: (Select1 <t> z)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpSelect0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpPPC64CNTLZDCC {
			break
		}
		v.reset(OpSelect1)
		v.Type = t
		v.AddArg(z)
		return true
	}
	// match: (CMPconst <t> [0] (Select0 z:(RLDICLCC y)))
	// result: (Select1 <t> z)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 0 || v_0.Op != OpSelect0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpPPC64RLDICLCC {
			break
		}
		v.reset(OpSelect1)
		v.Type = t
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValuePPC64latelower_OpPPC64ISEL(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ISEL [a] x (MOVDconst [0]) z)
	// result: (ISELZ [a] x z)
	for {
		a := auxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != OpPPC64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		z := v_2
		v.reset(OpPPC64ISELZ)
		v.AuxInt = int32ToAuxInt(a)
		v.AddArg2(x, z)
		return true
	}
	// match: (ISEL [a] (MOVDconst [0]) y z)
	// result: (ISELZ [a^0x4] y z)
	for {
		a := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpPPC64MOVDconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		y := v_1
		z := v_2
		v.reset(OpPPC64ISELZ)
		v.AuxInt = int32ToAuxInt(a ^ 0x4)
		v.AddArg2(y, z)
		return true
	}
	return false
}
func rewriteValuePPC64latelower_OpPPC64RLDICL(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RLDICL [em] x:(SRDconst [s] a))
	// cond: (em&0xFF0000) == 0
	// result: (RLDICL [mergePPC64RLDICLandSRDconst(em, s)] a)
	for {
		em := auxIntToInt64(v.AuxInt)
		x := v_0
		if x.Op != OpPPC64SRDconst {
			break
		}
		s := auxIntToInt64(x.AuxInt)
		a := x.Args[0]
		if !((em & 0xFF0000) == 0) {
			break
		}
		v.reset(OpPPC64RLDICL)
		v.AuxInt = int64ToAuxInt(mergePPC64RLDICLandSRDconst(em, s))
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValuePPC64latelower_OpPPC64RLDICLCC(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RLDICLCC [a] x)
	// cond: convertPPC64RldiclAndccconst(a) != 0
	// result: (ANDCCconst [convertPPC64RldiclAndccconst(a)] x)
	for {
		a := auxIntToInt64(v.AuxInt)
		x := v_0
		if !(convertPPC64RldiclAndccconst(a) != 0) {
			break
		}
		v.reset(OpPPC64ANDCCconst)
		v.AuxInt = int64ToAuxInt(convertPPC64RldiclAndccconst(a))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64latelower_OpPPC64SETBC(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETBC [2] cmp)
	// cond: buildcfg.GOPPC64 <= 9
	// result: (ISELZ [2] (MOVDconst [1]) cmp)
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		cmp := v_0
		if !(buildcfg.GOPPC64 <= 9) {
			break
		}
		v.reset(OpPPC64ISELZ)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v0.AuxInt = int64ToAuxInt(1)
		v.AddArg2(v0, cmp)
		return true
	}
	// match: (SETBC [0] cmp)
	// cond: buildcfg.GOPPC64 <= 9
	// result: (ISELZ [0] (MOVDconst [1]) cmp)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		cmp := v_0
		if !(buildcfg.GOPPC64 <= 9) {
			break
		}
		v.reset(OpPPC64ISELZ)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v0.AuxInt = int64ToAuxInt(1)
		v.AddArg2(v0, cmp)
		return true
	}
	// match: (SETBC [1] cmp)
	// cond: buildcfg.GOPPC64 <= 9
	// result: (ISELZ [1] (MOVDconst [1]) cmp)
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		cmp := v_0
		if !(buildcfg.GOPPC64 <= 9) {
			break
		}
		v.reset(OpPPC64ISELZ)
		v.AuxInt = int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v0.AuxInt = int64ToAuxInt(1)
		v.AddArg2(v0, cmp)
		return true
	}
	return false
}
func rewriteValuePPC64latelower_OpPPC64SETBCR(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETBCR [2] cmp)
	// cond: buildcfg.GOPPC64 <= 9
	// result: (ISELZ [6] (MOVDconst [1]) cmp)
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		cmp := v_0
		if !(buildcfg.GOPPC64 <= 9) {
			break
		}
		v.reset(OpPPC64ISELZ)
		v.AuxInt = int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v0.AuxInt = int64ToAuxInt(1)
		v.AddArg2(v0, cmp)
		return true
	}
	// match: (SETBCR [0] cmp)
	// cond: buildcfg.GOPPC64 <= 9
	// result: (ISELZ [4] (MOVDconst [1]) cmp)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		cmp := v_0
		if !(buildcfg.GOPPC64 <= 9) {
			break
		}
		v.reset(OpPPC64ISELZ)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v0.AuxInt = int64ToAuxInt(1)
		v.AddArg2(v0, cmp)
		return true
	}
	// match: (SETBCR [1] cmp)
	// cond: buildcfg.GOPPC64 <= 9
	// result: (ISELZ [5] (MOVDconst [1]) cmp)
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		cmp := v_0
		if !(buildcfg.GOPPC64 <= 9) {
			break
		}
		v.reset(OpPPC64ISELZ)
		v.AuxInt = int32ToAuxInt(5)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v0.AuxInt = int64ToAuxInt(1)
		v.AddArg2(v0, cmp)
		return true
	}
	return false
}
func rewriteBlockPPC64latelower(b *Block) bool {
	return false
}
