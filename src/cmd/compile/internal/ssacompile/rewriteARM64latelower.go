// Code generated from _gen/ARM64latelower.rules using 'go generate'; DO NOT EDIT.

package ssacompile

import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa/ssacore"

func rewriteValueARM64latelower(v *ssacore.Value) bool {
	switch v.Op {
	case ssaop.OpARM64ADDSconstflags:
		return rewriteValueARM64latelower_OpARM64ADDSconstflags(v)
	case ssaop.OpARM64ADDconst:
		return rewriteValueARM64latelower_OpARM64ADDconst(v)
	case ssaop.OpARM64ANDconst:
		return rewriteValueARM64latelower_OpARM64ANDconst(v)
	case ssaop.OpARM64CMNWconst:
		return rewriteValueARM64latelower_OpARM64CMNWconst(v)
	case ssaop.OpARM64CMNconst:
		return rewriteValueARM64latelower_OpARM64CMNconst(v)
	case ssaop.OpARM64CMPWconst:
		return rewriteValueARM64latelower_OpARM64CMPWconst(v)
	case ssaop.OpARM64CMPconst:
		return rewriteValueARM64latelower_OpARM64CMPconst(v)
	case ssaop.OpARM64MOVBUreg:
		return rewriteValueARM64latelower_OpARM64MOVBUreg(v)
	case ssaop.OpARM64MOVBreg:
		return rewriteValueARM64latelower_OpARM64MOVBreg(v)
	case ssaop.OpARM64MOVDconst:
		return rewriteValueARM64latelower_OpARM64MOVDconst(v)
	case ssaop.OpARM64MOVDnop:
		return rewriteValueARM64latelower_OpARM64MOVDnop(v)
	case ssaop.OpARM64MOVDreg:
		return rewriteValueARM64latelower_OpARM64MOVDreg(v)
	case ssaop.OpARM64MOVHUreg:
		return rewriteValueARM64latelower_OpARM64MOVHUreg(v)
	case ssaop.OpARM64MOVHreg:
		return rewriteValueARM64latelower_OpARM64MOVHreg(v)
	case ssaop.OpARM64MOVWUreg:
		return rewriteValueARM64latelower_OpARM64MOVWUreg(v)
	case ssaop.OpARM64MOVWreg:
		return rewriteValueARM64latelower_OpARM64MOVWreg(v)
	case ssaop.OpARM64ORconst:
		return rewriteValueARM64latelower_OpARM64ORconst(v)
	case ssaop.OpARM64SLLconst:
		return rewriteValueARM64latelower_OpARM64SLLconst(v)
	case ssaop.OpARM64SUBconst:
		return rewriteValueARM64latelower_OpARM64SUBconst(v)
	case ssaop.OpARM64TSTWconst:
		return rewriteValueARM64latelower_OpARM64TSTWconst(v)
	case ssaop.OpARM64TSTconst:
		return rewriteValueARM64latelower_OpARM64TSTconst(v)
	case ssaop.OpARM64XORconst:
		return rewriteValueARM64latelower_OpARM64XORconst(v)
	}
	return false
}
func rewriteValueARM64latelower_OpARM64ADDSconstflags(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADDSconstflags [c] x)
	// cond: !isARM64addcon(c)
	// result: (ADDSflags x (MOVDconst [c]))
	for {
		c := AuxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64addcon(c)) {
			break
		}
		v.Reset(ssaop.OpARM64ADDSflags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64ADDconst(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADDconst [c] x)
	// cond: !isARM64addcon(c)
	// result: (ADD x (MOVDconst [c]))
	for {
		c := AuxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64addcon(c)) {
			break
		}
		v.Reset(ssaop.OpARM64ADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64ANDconst(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ANDconst [c] x)
	// cond: !isARM64bitcon(uint64(c))
	// result: (AND x (MOVDconst [c]))
	for {
		c := AuxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64bitcon(uint64(c))) {
			break
		}
		v.Reset(ssaop.OpARM64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64CMNWconst(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMNWconst [c] x)
	// cond: !isARM64addcon(int64(c))
	// result: (CMNW x (MOVDconst [int64(c)]))
	for {
		c := AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(!isARM64addcon(int64(c))) {
			break
		}
		v.Reset(ssaop.OpARM64CMNW)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(int64(c))
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64CMNconst(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMNconst [c] x)
	// cond: !isARM64addcon(c)
	// result: (CMN x (MOVDconst [c]))
	for {
		c := AuxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64addcon(c)) {
			break
		}
		v.Reset(ssaop.OpARM64CMN)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64CMPWconst(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPWconst [c] x)
	// cond: !isARM64addcon(int64(c))
	// result: (CMPW x (MOVDconst [int64(c)]))
	for {
		c := AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(!isARM64addcon(int64(c))) {
			break
		}
		v.Reset(ssaop.OpARM64CMPW)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(int64(c))
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64CMPconst(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CMPconst [c] x)
	// cond: !isARM64addcon(c)
	// result: (CMP x (MOVDconst [c]))
	for {
		c := AuxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64addcon(c)) {
			break
		}
		v.Reset(ssaop.OpARM64CMP)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVBUreg(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBUreg x:(Equal _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpARM64Equal {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(NotEqual _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpARM64NotEqual {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(LessThan _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpARM64LessThan {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(LessThanU _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpARM64LessThanU {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(LessThanF _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpARM64LessThanF {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(LessEqual _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpARM64LessEqual {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(LessEqualU _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpARM64LessEqualU {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(LessEqualF _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpARM64LessEqualF {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(GreaterThan _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpARM64GreaterThan {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(GreaterThanU _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpARM64GreaterThanU {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(GreaterThanF _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpARM64GreaterThanF {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(GreaterEqual _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpARM64GreaterEqual {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(GreaterEqualU _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpARM64GreaterEqualU {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(GreaterEqualF _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpARM64GreaterEqualF {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x)
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
	// match: (MOVBUreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBUload {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBUloadidx {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVBreg(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBreg x:(MOVBload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBload {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBloadidx {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBreg {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVDconst(v *ssacore.Value) bool {
	// match: (MOVDconst [0])
	// result: (ZERO)
	for {
		if AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpARM64ZERO)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVDnop(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVDnop (MOVDconst [c]))
	// result: (MOVDconst [c])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = Int64ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVDreg(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVDreg x)
	// cond: x.Uses == 1
	// result: (MOVDnop x)
	for {
		x := v_0
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDnop)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVHUreg(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHUreg x)
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
	// match: (MOVHUreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBUload {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHUload {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBUloadidx {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHUloadidx {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUloadidx2 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHUloadidx2 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHUreg {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVHreg(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHreg x:(MOVBload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBload {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBUload {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHload {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBloadidx {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBUloadidx {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHloadidx {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHloadidx2 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHloadidx2 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBreg {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHreg {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVWUreg(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWUreg x)
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
	// match: (MOVWUreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBUload {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHUload {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVWUload {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBUloadidx {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHUloadidx {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVWUloadidx {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUloadidx2 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHUloadidx2 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUloadidx4 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVWUloadidx4 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHUreg {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVWUreg {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64MOVWreg(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWreg x:(MOVBload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBload {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBUload {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHload {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHUload {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVWload {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBloadidx {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBUloadidx {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHloadidx {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHUloadidx {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVWloadidx {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHloadidx2 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHloadidx2 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUloadidx2 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHUloadidx2 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWloadidx4 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVWloadidx4 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBreg {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVHreg {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64MOVWreg {
			break
		}
		v.Reset(ssaop.OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64ORconst(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ORconst [c] x)
	// cond: !isARM64bitcon(uint64(c))
	// result: (OR x (MOVDconst [c]))
	for {
		c := AuxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64bitcon(uint64(c))) {
			break
		}
		v.Reset(ssaop.OpARM64OR)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64SLLconst(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	// match: (SLLconst [1] x)
	// result: (ADD x x)
	for {
		if AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		x := v_0
		v.Reset(ssaop.OpARM64ADD)
		v.AddArg2(x, x)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64SUBconst(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SUBconst [c] x)
	// cond: !isARM64addcon(c)
	// result: (SUB x (MOVDconst [c]))
	for {
		c := AuxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64addcon(c)) {
			break
		}
		v.Reset(ssaop.OpARM64SUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64TSTWconst(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (TSTWconst [c] x)
	// cond: !isARM64bitcon(uint64(c)|uint64(c)<<32)
	// result: (TSTW x (MOVDconst [int64(c)]))
	for {
		c := AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(!isARM64bitcon(uint64(c) | uint64(c)<<32)) {
			break
		}
		v.Reset(ssaop.OpARM64TSTW)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(int64(c))
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64TSTconst(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (TSTconst [c] x)
	// cond: !isARM64bitcon(uint64(c))
	// result: (TST x (MOVDconst [c]))
	for {
		c := AuxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64bitcon(uint64(c))) {
			break
		}
		v.Reset(ssaop.OpARM64TST)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64latelower_OpARM64XORconst(v *ssacore.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (XORconst [c] x)
	// cond: !isARM64bitcon(uint64(c))
	// result: (XOR x (MOVDconst [c]))
	for {
		c := AuxIntToInt64(v.AuxInt)
		x := v_0
		if !(!isARM64bitcon(uint64(c))) {
			break
		}
		v.Reset(ssaop.OpARM64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteBlockARM64latelower(b *ssacore.Block) bool {
	return false
}
