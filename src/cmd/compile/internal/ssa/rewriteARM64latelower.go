// Code generated from _gen/ARM64latelower.rules using 'go generate'; DO NOT EDIT.

package ssa

func rewriteValueARM64latelower(v *Value) bool {
	switch v.Op {
	case OpARM64ADDSconstflags:
		return rewriteValueARM64latelower_OpARM64ADDSconstflags(v)
	case OpARM64ADDconst:
		return rewriteValueARM64latelower_OpARM64ADDconst(v)
	case OpARM64ANDconst:
		return rewriteValueARM64latelower_OpARM64ANDconst(v)
	case OpARM64CMNWconst:
		return rewriteValueARM64latelower_OpARM64CMNWconst(v)
	case OpARM64CMNconst:
		return rewriteValueARM64latelower_OpARM64CMNconst(v)
	case OpARM64CMPWconst:
		return rewriteValueARM64latelower_OpARM64CMPWconst(v)
	case OpARM64CMPconst:
		return rewriteValueARM64latelower_OpARM64CMPconst(v)
	case OpARM64MOVBUreg:
		return rewriteValueARM64latelower_OpARM64MOVBUreg(v)
	case OpARM64MOVBreg:
		return rewriteValueARM64latelower_OpARM64MOVBreg(v)
	case OpARM64MOVDconst:
		return rewriteValueARM64latelower_OpARM64MOVDconst(v)
	case OpARM64MOVDnop:
		return rewriteValueARM64latelower_OpARM64MOVDnop(v)
	case OpARM64MOVDreg:
		return rewriteValueARM64latelower_OpARM64MOVDreg(v)
	case OpARM64MOVHUreg:
		return rewriteValueARM64latelower_OpARM64MOVHUreg(v)
	case OpARM64MOVHreg:
		return rewriteValueARM64latelower_OpARM64MOVHreg(v)
	case OpARM64MOVWUreg:
		return rewriteValueARM64latelower_OpARM64MOVWUreg(v)
	case OpARM64MOVWreg:
		return rewriteValueARM64latelower_OpARM64MOVWreg(v)
	case OpARM64ORconst:
		return rewriteValueARM64latelower_OpARM64ORconst(v)
	case OpARM64SUBconst:
		return rewriteValueARM64latelower_OpARM64SUBconst(v)
	case OpARM64TSTWconst:
		return rewriteValueARM64latelower_OpARM64TSTWconst(v)
	case OpARM64TSTconst:
		return rewriteValueARM64latelower_OpARM64TSTconst(v)
	case OpARM64XORconst:
		return rewriteValueARM64latelower_OpARM64XORconst(v)
	}
	return false
}
func rewriteValueARM64latelower_OpARM64ADDSconstflags(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADDSconstflags [c] x)
	// cond: !isARM64addcon(c)
	// result: (ADDSflags x (MOVDconst [c]))
	for {
		c := auxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64addcon(c)) {
			break
		}
		v.reset(OpARM64ADDSflags)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64ADDconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADDconst [c] x)
	// cond: !isARM64addcon(c)
	// result: (ADD x (MOVDconst [c]))
	for {
		c := auxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64addcon(c)) {
			break
		}
		v.reset(OpARM64ADD)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64ANDconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ANDconst [c] x)
	// cond: !isARM64bitcon(uint64(c))
	// result: (AND x (MOVDconst [c]))
	for {
		c := auxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64bitcon(uint64(c))) {
			break
		}
		v.reset(OpARM64AND)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64CMNWconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMNWconst [c] x)
	// cond: !isARM64addcon(int64(c))
	// result: (CMNW x (MOVDconst [int64(c)]))
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(!isARM64addcon(int64(c))) {
			break
		}
		v.reset(OpARM64CMNW)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(int64(c))
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64CMNconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMNconst [c] x)
	// cond: !isARM64addcon(c)
	// result: (CMN x (MOVDconst [c]))
	for {
		c := auxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64addcon(c)) {
			break
		}
		v.reset(OpARM64CMN)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64CMPWconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPWconst [c] x)
	// cond: !isARM64addcon(int64(c))
	// result: (CMPW x (MOVDconst [int64(c)]))
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(!isARM64addcon(int64(c))) {
			break
		}
		v.reset(OpARM64CMPW)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(int64(c))
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64CMPconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPconst [c] x)
	// cond: !isARM64addcon(c)
	// result: (CMP x (MOVDconst [c]))
	for {
		c := auxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64addcon(c)) {
			break
		}
		v.reset(OpARM64CMP)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVBUreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBUreg x:(Equal _))
	// result: x
	for {
		x := v_0
		if x.Op != OpARM64Equal {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(NotEqual _))
	// result: x
	for {
		x := v_0
		if x.Op != OpARM64NotEqual {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(LessThan _))
	// result: x
	for {
		x := v_0
		if x.Op != OpARM64LessThan {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(LessThanU _))
	// result: x
	for {
		x := v_0
		if x.Op != OpARM64LessThanU {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(LessThanF _))
	// result: x
	for {
		x := v_0
		if x.Op != OpARM64LessThanF {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(LessEqual _))
	// result: x
	for {
		x := v_0
		if x.Op != OpARM64LessEqual {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(LessEqualU _))
	// result: x
	for {
		x := v_0
		if x.Op != OpARM64LessEqualU {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(LessEqualF _))
	// result: x
	for {
		x := v_0
		if x.Op != OpARM64LessEqualF {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(GreaterThan _))
	// result: x
	for {
		x := v_0
		if x.Op != OpARM64GreaterThan {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(GreaterThanU _))
	// result: x
	for {
		x := v_0
		if x.Op != OpARM64GreaterThanU {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(GreaterThanF _))
	// result: x
	for {
		x := v_0
		if x.Op != OpARM64GreaterThanF {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(GreaterEqual _))
	// result: x
	for {
		x := v_0
		if x.Op != OpARM64GreaterEqual {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(GreaterEqualU _))
	// result: x
	for {
		x := v_0
		if x.Op != OpARM64GreaterEqualU {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(GreaterEqualF _))
	// result: x
	for {
		x := v_0
		if x.Op != OpARM64GreaterEqualF {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVBreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBreg x:(MOVBload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVDconst(v *Value) bool {
	// match: (MOVDconst [0])
	// result: (ZERO)
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		v.reset(OpARM64ZERO)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVDnop(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVDnop (MOVDconst [c]))
	// result: (MOVDconst [c])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVDreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVDreg x)
	// cond: x.Uses == 1
	// result: (MOVDnop x)
	for {
		x := v_0
		if !(x.Uses == 1) {
			break
		}
		v.reset(OpARM64MOVDnop)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVHUreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHUreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUloadidx2 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUloadidx2 {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVHreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHreg x:(MOVBload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHloadidx2 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHloadidx2 {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVWUreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWUreg x)
	// cond: zeroUpper32Bits(x, 3)
	// result: x
	for {
		x := v_0
		if !(zeroUpper32Bits(x, 3)) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVWUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVWUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUloadidx2 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUloadidx2 {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUloadidx4 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVWUloadidx4 {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVWUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVWreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWreg x:(MOVBload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVWload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVWloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHloadidx2 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHloadidx2 {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUloadidx2 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUloadidx2 {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWloadidx4 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVWloadidx4 {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVWreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64ORconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ORconst [c] x)
	// cond: !isARM64bitcon(uint64(c))
	// result: (OR x (MOVDconst [c]))
	for {
		c := auxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64bitcon(uint64(c))) {
			break
		}
		v.reset(OpARM64OR)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64SUBconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SUBconst [c] x)
	// cond: !isARM64addcon(c)
	// result: (SUB x (MOVDconst [c]))
	for {
		c := auxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64addcon(c)) {
			break
		}
		v.reset(OpARM64SUB)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64TSTWconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (TSTWconst [c] x)
	// cond: !isARM64bitcon(uint64(c)|uint64(c)<<32)
	// result: (TSTW x (MOVDconst [int64(c)]))
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(!isARM64bitcon(uint64(c) | uint64(c)<<32)) {
			break
		}
		v.reset(OpARM64TSTW)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(int64(c))
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64TSTconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (TSTconst [c] x)
	// cond: !isARM64bitcon(uint64(c))
	// result: (TST x (MOVDconst [c]))
	for {
		c := auxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64bitcon(uint64(c))) {
			break
		}
		v.reset(OpARM64TST)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64XORconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (XORconst [c] x)
	// cond: !isARM64bitcon(uint64(c))
	// result: (XOR x (MOVDconst [c]))
	for {
		c := auxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64bitcon(uint64(c))) {
			break
		}
		v.reset(OpARM64XOR)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteBlockARM64latelower(b *Block) bool {
	return false
}
