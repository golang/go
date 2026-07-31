// Code generated from _gen/divmod.rules using 'go generate'; DO NOT EDIT.

package ssacompile

import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa"

func rewriteValuedivmod(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpDiv16:
		return rewriteValuedivmod_OpDiv16(v)
	case ssaop.OpDiv16u:
		return rewriteValuedivmod_OpDiv16u(v)
	case ssaop.OpDiv32:
		return rewriteValuedivmod_OpDiv32(v)
	case ssaop.OpDiv32u:
		return rewriteValuedivmod_OpDiv32u(v)
	case ssaop.OpDiv64:
		return rewriteValuedivmod_OpDiv64(v)
	case ssaop.OpDiv64u:
		return rewriteValuedivmod_OpDiv64u(v)
	case ssaop.OpDiv8:
		return rewriteValuedivmod_OpDiv8(v)
	case ssaop.OpDiv8u:
		return rewriteValuedivmod_OpDiv8u(v)
	}
	return false
}
func rewriteValuedivmod_OpDiv16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 <t> n (Const16 [c]))
	// cond: ssa.IsPowerOfTwo(c)
	// result: (Rsh16x64 (Add16 <t> n (Rsh16Ux64 <t> (Rsh16x64 <t> n (Const64 <typ.UInt64> [15])) (Const64 <typ.UInt64> [int64(16-ssa.Log16(c))]))) (Const64 <typ.UInt64> [int64(ssa.Log16(c))]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != ssaop.OpConst16 {
			break
		}
		c := ssa.AuxIntToInt16(v_1.AuxInt)
		if !(ssa.IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpRsh16x64)
		v0 := b.NewValue0(v.Pos, ssaop.OpAdd16, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpRsh16Ux64, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh16x64, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(15)
		v2.AddArg2(n, v3)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(int64(16 - ssa.Log16(c)))
		v1.AddArg2(v2, v4)
		v0.AddArg2(n, v1)
		v5 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v5.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Log16(c)))
		v.AddArg2(v0, v5)
		return true
	}
	// match: (Div16 <t> x (Const16 [c]))
	// cond: smagicOK16(c)
	// result: (Sub16 <t> (Rsh32x64 <t> (Mul32 <typ.UInt32> (SignExt16to32 x) (Const32 <typ.UInt32> [int32(smagic16(c).M)])) (Const64 <typ.UInt64> [16 + smagic16(c).S])) (Rsh32x64 <t> (SignExt16to32 x) (Const64 <typ.UInt64> [31])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst16 {
			break
		}
		c := ssa.AuxIntToInt16(v_1.AuxInt)
		if !(smagicOK16(c)) {
			break
		}
		v.Reset(ssaop.OpSub16)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRsh32x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst32, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(int32(smagic16(c).M))
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(16 + smagic16(c).S)
		v0.AddArg2(v1, v4)
		v5 := b.NewValue0(v.Pos, ssaop.OpRsh32x64, t)
		v6 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v6.AuxInt = ssa.Int64ToAuxInt(31)
		v5.AddArg2(v2, v6)
		v.AddArg2(v0, v5)
		return true
	}
	return false
}
func rewriteValuedivmod_OpDiv16u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Div16u <t> x (Const16 [c]))
	// cond: t.IsSigned() && smagicOK16(c)
	// result: (Rsh32Ux64 <t> (Mul32 <typ.UInt32> (SignExt16to32 x) (Const32 <typ.UInt32> [int32(smagic16(c).M)])) (Const64 <typ.UInt64> [16 + smagic16(c).S]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst16 {
			break
		}
		c := ssa.AuxIntToInt16(v_1.AuxInt)
		if !(t.IsSigned() && smagicOK16(c)) {
			break
		}
		v.Reset(ssaop.OpRsh32Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpMul32, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpConst32, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(int32(smagic16(c).M))
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(16 + smagic16(c).S)
		v.AddArg2(v0, v3)
		return true
	}
	// match: (Div16u <t> x (Const16 [c]))
	// cond: umagicOK16(c) && config.RegSize == 8
	// result: (Trunc64to16 <t> (Rsh64Ux64 <typ.UInt64> (Mul64 <typ.UInt64> (ZeroExt16to64 x) (Const64 <typ.UInt64> [int64(1<<16 + umagic16(c).M)])) (Const64 <typ.UInt64> [16 + umagic16(c).S])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst16 {
			break
		}
		c := ssa.AuxIntToInt16(v_1.AuxInt)
		if !(umagicOK16(c) && config.RegSize == 8) {
			break
		}
		v.Reset(ssaop.OpTrunc64to16)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRsh64Ux64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpMul64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(int64(1<<16 + umagic16(c).M))
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(16 + umagic16(c).S)
		v0.AddArg2(v1, v4)
		v.AddArg(v0)
		return true
	}
	// match: (Div16u <t> x (Const16 [c]))
	// cond: umagicOK16(c) && umagic16(c).M&1 == 0
	// result: (Trunc32to16 <t> (Rsh32Ux64 <typ.UInt32> (Mul32 <typ.UInt32> (ZeroExt16to32 x) (Const32 <typ.UInt32> [int32(1<<15 + umagic16(c).M/2)])) (Const64 <typ.UInt64> [16 + umagic16(c).S - 1])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst16 {
			break
		}
		c := ssa.AuxIntToInt16(v_1.AuxInt)
		if !(umagicOK16(c) && umagic16(c).M&1 == 0) {
			break
		}
		v.Reset(ssaop.OpTrunc32to16)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRsh32Ux64, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst32, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(int32(1<<15 + umagic16(c).M/2))
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(16 + umagic16(c).S - 1)
		v0.AddArg2(v1, v4)
		v.AddArg(v0)
		return true
	}
	// match: (Div16u <t> x (Const16 [c]))
	// cond: umagicOK16(c) && config.RegSize == 4 && c&1 == 0
	// result: (Trunc32to16 <t> (Rsh32Ux64 <typ.UInt32> (Mul32 <typ.UInt32> (Rsh32Ux64 <typ.UInt32> (ZeroExt16to32 x) (Const64 <typ.UInt64> [1])) (Const32 <typ.UInt32> [int32(1<<15 + (umagic16(c).M+1)/2)])) (Const64 <typ.UInt64> [16 + umagic16(c).S - 2])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst16 {
			break
		}
		c := ssa.AuxIntToInt16(v_1.AuxInt)
		if !(umagicOK16(c) && config.RegSize == 4 && c&1 == 0) {
			break
		}
		v.Reset(ssaop.OpTrunc32to16)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRsh32Ux64, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh32Ux64, typ.UInt32)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(1)
		v2.AddArg2(v3, v4)
		v5 := b.NewValue0(v.Pos, ssaop.OpConst32, typ.UInt32)
		v5.AuxInt = ssa.Int32ToAuxInt(int32(1<<15 + (umagic16(c).M+1)/2))
		v1.AddArg2(v2, v5)
		v6 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v6.AuxInt = ssa.Int64ToAuxInt(16 + umagic16(c).S - 2)
		v0.AddArg2(v1, v6)
		v.AddArg(v0)
		return true
	}
	// match: (Div16u <t> x (Const16 [c]))
	// cond: umagicOK16(c) && config.RegSize == 4
	// result: (Trunc32to16 <t> (Rsh32Ux64 <typ.UInt32> (Avg32u (Lsh32x64 <typ.UInt32> (ZeroExt16to32 x) (Const64 <typ.UInt64> [16])) (Mul32 <typ.UInt32> (ZeroExt16to32 x) (Const32 <typ.UInt32> [int32(umagic16(c).M)]))) (Const64 <typ.UInt64> [16 + umagic16(c).S - 1])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst16 {
			break
		}
		c := ssa.AuxIntToInt16(v_1.AuxInt)
		if !(umagicOK16(c) && config.RegSize == 4) {
			break
		}
		v.Reset(ssaop.OpTrunc32to16)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRsh32Ux64, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpAvg32u, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpLsh32x64, typ.UInt32)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(16)
		v2.AddArg2(v3, v4)
		v5 := b.NewValue0(v.Pos, ssaop.OpMul32, typ.UInt32)
		v6 := b.NewValue0(v.Pos, ssaop.OpConst32, typ.UInt32)
		v6.AuxInt = ssa.Int32ToAuxInt(int32(umagic16(c).M))
		v5.AddArg2(v3, v6)
		v1.AddArg2(v2, v5)
		v7 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v7.AuxInt = ssa.Int64ToAuxInt(16 + umagic16(c).S - 1)
		v0.AddArg2(v1, v7)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuedivmod_OpDiv32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Div32 <t> n (Const32 [c]))
	// cond: ssa.IsPowerOfTwo(c)
	// result: (Rsh32x64 (Add32 <t> n (Rsh32Ux64 <t> (Rsh32x64 <t> n (Const64 <typ.UInt64> [31])) (Const64 <typ.UInt64> [int64(32-ssa.Log32(c))]))) (Const64 <typ.UInt64> [int64(ssa.Log32(c))]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != ssaop.OpConst32 {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(ssa.IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpRsh32x64)
		v0 := b.NewValue0(v.Pos, ssaop.OpAdd32, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpRsh32Ux64, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh32x64, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(31)
		v2.AddArg2(n, v3)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(int64(32 - ssa.Log32(c)))
		v1.AddArg2(v2, v4)
		v0.AddArg2(n, v1)
		v5 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v5.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Log32(c)))
		v.AddArg2(v0, v5)
		return true
	}
	// match: (Div32 <t> x (Const32 [c]))
	// cond: smagicOK32(c) && config.RegSize == 8
	// result: (Sub32 <t> (Rsh64x64 <t> (Mul64 <typ.UInt64> (SignExt32to64 x) (Const64 <typ.UInt64> [int64(smagic32(c).M)])) (Const64 <typ.UInt64> [32 + smagic32(c).S])) (Rsh64x64 <t> (SignExt32to64 x) (Const64 <typ.UInt64> [63])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst32 {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(smagicOK32(c) && config.RegSize == 8) {
			break
		}
		v.Reset(ssaop.OpSub32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRsh64x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMul64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(int64(smagic32(c).M))
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(32 + smagic32(c).S)
		v0.AddArg2(v1, v4)
		v5 := b.NewValue0(v.Pos, ssaop.OpRsh64x64, t)
		v6 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v6.AuxInt = ssa.Int64ToAuxInt(63)
		v5.AddArg2(v2, v6)
		v.AddArg2(v0, v5)
		return true
	}
	// match: (Div32 <t> x (Const32 [c]))
	// cond: smagicOK32(c) && config.RegSize == 4 && smagic32(c).M&1 == 0
	// result: (Sub32 <t> (Rsh32x64 <t> (Hmul32 <t> x (Const32 <typ.UInt32> [int32(smagic32(c).M/2)])) (Const64 <typ.UInt64> [smagic32(c).S - 1])) (Rsh32x64 <t> x (Const64 <typ.UInt64> [31])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst32 {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(smagicOK32(c) && config.RegSize == 4 && smagic32(c).M&1 == 0) {
			break
		}
		v.Reset(ssaop.OpSub32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRsh32x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpHmul32, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpConst32, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(int32(smagic32(c).M / 2))
		v1.AddArg2(x, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(smagic32(c).S - 1)
		v0.AddArg2(v1, v3)
		v4 := b.NewValue0(v.Pos, ssaop.OpRsh32x64, t)
		v5 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v5.AuxInt = ssa.Int64ToAuxInt(31)
		v4.AddArg2(x, v5)
		v.AddArg2(v0, v4)
		return true
	}
	// match: (Div32 <t> x (Const32 [c]))
	// cond: smagicOK32(c) && config.RegSize == 4 && smagic32(c).M&1 != 0
	// result: (Sub32 <t> (Rsh32x64 <t> (Add32 <t> x (Hmul32 <t> x (Const32 <typ.UInt32> [int32(smagic32(c).M)]))) (Const64 <typ.UInt64> [smagic32(c).S])) (Rsh32x64 <t> x (Const64 <typ.UInt64> [31])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst32 {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(smagicOK32(c) && config.RegSize == 4 && smagic32(c).M&1 != 0) {
			break
		}
		v.Reset(ssaop.OpSub32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRsh32x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpAdd32, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpHmul32, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst32, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(int32(smagic32(c).M))
		v2.AddArg2(x, v3)
		v1.AddArg2(x, v2)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(smagic32(c).S)
		v0.AddArg2(v1, v4)
		v5 := b.NewValue0(v.Pos, ssaop.OpRsh32x64, t)
		v6 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v6.AuxInt = ssa.Int64ToAuxInt(31)
		v5.AddArg2(x, v6)
		v.AddArg2(v0, v5)
		return true
	}
	return false
}
func rewriteValuedivmod_OpDiv32u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Div32u <t> x (Const32 [c]))
	// cond: t.IsSigned() && smagicOK32(c) && config.RegSize == 8
	// result: (Rsh64Ux64 <t> (Mul64 <typ.UInt64> (SignExt32to64 x) (Const64 <typ.UInt64> [int64(smagic32(c).M)])) (Const64 <typ.UInt64> [32 + smagic32(c).S]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst32 {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(t.IsSigned() && smagicOK32(c) && config.RegSize == 8) {
			break
		}
		v.Reset(ssaop.OpRsh64Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpMul64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(int64(smagic32(c).M))
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(32 + smagic32(c).S)
		v.AddArg2(v0, v3)
		return true
	}
	// match: (Div32u <t> x (Const32 [c]))
	// cond: t.IsSigned() && smagicOK32(c) && config.RegSize == 4
	// result: (Rsh32Ux64 <t> (Hmul32u <typ.UInt32> x (Const32 <typ.UInt32> [int32(smagic32(c).M)])) (Const64 <typ.UInt64> [smagic32(c).S]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst32 {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(t.IsSigned() && smagicOK32(c) && config.RegSize == 4) {
			break
		}
		v.Reset(ssaop.OpRsh32Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpHmul32u, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpConst32, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(int32(smagic32(c).M))
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(smagic32(c).S)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Div32u <t> x (Const32 [c]))
	// cond: umagicOK32(c) && umagic32(c).M&1 == 0 && config.RegSize == 8
	// result: (Trunc64to32 <t> (Rsh64Ux64 <typ.UInt64> (Mul64 <typ.UInt64> (ZeroExt32to64 x) (Const64 <typ.UInt64> [int64(1<<31 + umagic32(c).M/2)])) (Const64 <typ.UInt64> [32 + umagic32(c).S - 1])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst32 {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && umagic32(c).M&1 == 0 && config.RegSize == 8) {
			break
		}
		v.Reset(ssaop.OpTrunc64to32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRsh64Ux64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpMul64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(int64(1<<31 + umagic32(c).M/2))
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(32 + umagic32(c).S - 1)
		v0.AddArg2(v1, v4)
		v.AddArg(v0)
		return true
	}
	// match: (Div32u <t> x (Const32 [c]))
	// cond: umagicOK32(c) && umagic32(c).M&1 == 0 && config.RegSize == 4
	// result: (Rsh32Ux64 <t> (Hmul32u <typ.UInt32> x (Const32 <typ.UInt32> [int32(1<<31 + umagic32(c).M/2)])) (Const64 <typ.UInt64> [umagic32(c).S - 1]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst32 {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && umagic32(c).M&1 == 0 && config.RegSize == 4) {
			break
		}
		v.Reset(ssaop.OpRsh32Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpHmul32u, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpConst32, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(int32(1<<31 + umagic32(c).M/2))
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(umagic32(c).S - 1)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Div32u <t> x (Const32 [c]))
	// cond: umagicOK32(c) && config.RegSize == 8 && config.Ctxt.Arch.Name != "wasm" && umagic32(c).M&1 != 0
	// result: (Trunc64to32 <t> (Hmul64u <typ.UInt64> (ZeroExt32to64 x) (Const64 <typ.UInt64> [int64(umagic32PreShifted(c))])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst32 {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && config.RegSize == 8 && config.Ctxt.Arch.Name != "wasm" && umagic32(c).M&1 != 0) {
			break
		}
		v.Reset(ssaop.OpTrunc64to32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpHmul64u, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(int64(umagic32PreShifted(c)))
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
	// match: (Div32u <t> x (Const32 [c]))
	// cond: umagicOK32(c) && config.RegSize == 8 && c&1 == 0
	// result: (Trunc64to32 <t> (Rsh64Ux64 <typ.UInt64> (Mul64 <typ.UInt64> (Rsh64Ux64 <typ.UInt64> (ZeroExt32to64 x) (Const64 <typ.UInt64> [1])) (Const64 <typ.UInt64> [int64(1<<31 + (umagic32(c).M+1)/2)])) (Const64 <typ.UInt64> [32 + umagic32(c).S - 2])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst32 {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && config.RegSize == 8 && c&1 == 0) {
			break
		}
		v.Reset(ssaop.OpTrunc64to32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRsh64Ux64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpMul64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh64Ux64, typ.UInt64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(1)
		v2.AddArg2(v3, v4)
		v5 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v5.AuxInt = ssa.Int64ToAuxInt(int64(1<<31 + (umagic32(c).M+1)/2))
		v1.AddArg2(v2, v5)
		v6 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v6.AuxInt = ssa.Int64ToAuxInt(32 + umagic32(c).S - 2)
		v0.AddArg2(v1, v6)
		v.AddArg(v0)
		return true
	}
	// match: (Div32u <t> x (Const32 [c]))
	// cond: umagicOK32(c) && config.RegSize == 4 && c&1 == 0
	// result: (Rsh32Ux64 <t> (Hmul32u <typ.UInt32> (Rsh32Ux64 <typ.UInt32> x (Const64 <typ.UInt64> [1])) (Const32 <typ.UInt32> [int32(1<<31 + (umagic32(c).M+1)/2)])) (Const64 <typ.UInt64> [umagic32(c).S - 2]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst32 {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && config.RegSize == 4 && c&1 == 0) {
			break
		}
		v.Reset(ssaop.OpRsh32Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpHmul32u, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpRsh32Ux64, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(1)
		v1.AddArg2(x, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst32, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(int32(1<<31 + (umagic32(c).M+1)/2))
		v0.AddArg2(v1, v3)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(umagic32(c).S - 2)
		v.AddArg2(v0, v4)
		return true
	}
	// match: (Div32u <t> x (Const32 [c]))
	// cond: umagicOK32(c) && config.RegSize == 8 && config.Ctxt.Arch.Name == "wasm"
	// result: (Trunc64to32 <t> (Rsh64Ux64 <typ.UInt64> (Avg64u (Lsh64x64 <typ.UInt64> (ZeroExt32to64 x) (Const64 <typ.UInt64> [32])) (Mul64 <typ.UInt64> (ZeroExt32to64 x) (Const64 <typ.UInt64> [int64(umagic32(c).M)]))) (Const64 <typ.UInt64> [32 + umagic32(c).S - 1])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst32 {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && config.RegSize == 8 && config.Ctxt.Arch.Name == "wasm") {
			break
		}
		v.Reset(ssaop.OpTrunc64to32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRsh64Ux64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpAvg64u, typ.UInt64)
		v2 := b.NewValue0(v.Pos, ssaop.OpLsh64x64, typ.UInt64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(32)
		v2.AddArg2(v3, v4)
		v5 := b.NewValue0(v.Pos, ssaop.OpMul64, typ.UInt64)
		v6 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v6.AuxInt = ssa.Int64ToAuxInt(int64(umagic32(c).M))
		v5.AddArg2(v3, v6)
		v1.AddArg2(v2, v5)
		v7 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v7.AuxInt = ssa.Int64ToAuxInt(32 + umagic32(c).S - 1)
		v0.AddArg2(v1, v7)
		v.AddArg(v0)
		return true
	}
	// match: (Div32u <t> x (Const32 [c]))
	// cond: umagicOK32(c) && config.RegSize == 4
	// result: (Rsh32Ux64 <t> (Avg32u x (Hmul32u <typ.UInt32> x (Const32 <typ.UInt32> [int32(umagic32(c).M)]))) (Const64 <typ.UInt64> [umagic32(c).S - 1]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst32 {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && config.RegSize == 4) {
			break
		}
		v.Reset(ssaop.OpRsh32Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpAvg32u, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpHmul32u, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpConst32, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(int32(umagic32(c).M))
		v1.AddArg2(x, v2)
		v0.AddArg2(x, v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(umagic32(c).S - 1)
		v.AddArg2(v0, v3)
		return true
	}
	return false
}
func rewriteValuedivmod_OpDiv64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div64 <t> n (Const64 [c]))
	// cond: ssa.IsPowerOfTwo(c)
	// result: (Rsh64x64 (Add64 <t> n (Rsh64Ux64 <t> (Rsh64x64 <t> n (Const64 <typ.UInt64> [63])) (Const64 <typ.UInt64> [int64(64-ssa.Log64(c))]))) (Const64 <typ.UInt64> [int64(ssa.Log64(c))]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(ssa.IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpRsh64x64)
		v0 := b.NewValue0(v.Pos, ssaop.OpAdd64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpRsh64Ux64, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh64x64, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(63)
		v2.AddArg2(n, v3)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(int64(64 - ssa.Log64(c)))
		v1.AddArg2(v2, v4)
		v0.AddArg2(n, v1)
		v5 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v5.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Log64(c)))
		v.AddArg2(v0, v5)
		return true
	}
	// match: (Div64 <t> x (Const64 [c]))
	// cond: smagicOK64(c) && smagic64(c).M&1 == 0
	// result: (Sub64 <t> (Rsh64x64 <t> (Hmul64 <t> x (Const64 <typ.UInt64> [int64(smagic64(c).M/2)])) (Const64 <typ.UInt64> [smagic64(c).S - 1])) (Rsh64x64 <t> x (Const64 <typ.UInt64> [63])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(smagicOK64(c) && smagic64(c).M&1 == 0) {
			break
		}
		v.Reset(ssaop.OpSub64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRsh64x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpHmul64, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(int64(smagic64(c).M / 2))
		v1.AddArg2(x, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(smagic64(c).S - 1)
		v0.AddArg2(v1, v3)
		v4 := b.NewValue0(v.Pos, ssaop.OpRsh64x64, t)
		v5 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v5.AuxInt = ssa.Int64ToAuxInt(63)
		v4.AddArg2(x, v5)
		v.AddArg2(v0, v4)
		return true
	}
	// match: (Div64 <t> x (Const64 [c]))
	// cond: smagicOK64(c) && smagic64(c).M&1 != 0
	// result: (Sub64 <t> (Rsh64x64 <t> (Add64 <t> x (Hmul64 <t> x (Const64 <typ.UInt64> [int64(smagic64(c).M)]))) (Const64 <typ.UInt64> [smagic64(c).S])) (Rsh64x64 <t> x (Const64 <typ.UInt64> [63])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(smagicOK64(c) && smagic64(c).M&1 != 0) {
			break
		}
		v.Reset(ssaop.OpSub64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRsh64x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpAdd64, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpHmul64, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(int64(smagic64(c).M))
		v2.AddArg2(x, v3)
		v1.AddArg2(x, v2)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(smagic64(c).S)
		v0.AddArg2(v1, v4)
		v5 := b.NewValue0(v.Pos, ssaop.OpRsh64x64, t)
		v6 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v6.AuxInt = ssa.Int64ToAuxInt(63)
		v5.AddArg2(x, v6)
		v.AddArg2(v0, v5)
		return true
	}
	return false
}
func rewriteValuedivmod_OpDiv64u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div64u <t> x (Const64 [c]))
	// cond: t.IsSigned() && smagicOK64(c)
	// result: (Rsh64Ux64 <t> (Hmul64u <typ.UInt64> x (Const64 <typ.UInt64> [int64(smagic64(c).M)])) (Const64 <typ.UInt64> [smagic64(c).S]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(t.IsSigned() && smagicOK64(c)) {
			break
		}
		v.Reset(ssaop.OpRsh64Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpHmul64u, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v1.AuxInt = ssa.Int64ToAuxInt(int64(smagic64(c).M))
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(smagic64(c).S)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Div64u <t> x (Const64 [c]))
	// cond: umagicOK64(c) && umagic64(c).M&1 == 0
	// result: (Rsh64Ux64 <t> (Hmul64u <typ.UInt64> x (Const64 <typ.UInt64> [int64(1<<63 + umagic64(c).M/2)])) (Const64 <typ.UInt64> [umagic64(c).S - 1]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(umagicOK64(c) && umagic64(c).M&1 == 0) {
			break
		}
		v.Reset(ssaop.OpRsh64Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpHmul64u, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v1.AuxInt = ssa.Int64ToAuxInt(int64(1<<63 + umagic64(c).M/2))
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(umagic64(c).S - 1)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Div64u <t> x (Const64 [c]))
	// cond: umagicOK64(c) && c&1 == 0
	// result: (Rsh64Ux64 <t> (Hmul64u <typ.UInt64> (Rsh64Ux64 <typ.UInt64> x (Const64 <typ.UInt64> [1])) (Const64 <typ.UInt64> [int64(1<<63 + (umagic64(c).M+1)/2)])) (Const64 <typ.UInt64> [umagic64(c).S - 2]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(umagicOK64(c) && c&1 == 0) {
			break
		}
		v.Reset(ssaop.OpRsh64Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpHmul64u, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpRsh64Ux64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(1)
		v1.AddArg2(x, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(int64(1<<63 + (umagic64(c).M+1)/2))
		v0.AddArg2(v1, v3)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(umagic64(c).S - 2)
		v.AddArg2(v0, v4)
		return true
	}
	// match: (Div64u <t> x (Const64 [c]))
	// cond: umagicOK64(c)
	// result: (Rsh64Ux64 <t> (Avg64u x (Hmul64u <typ.UInt64> x (Const64 <typ.UInt64> [int64(umagic64(c).M)]))) (Const64 <typ.UInt64> [umagic64(c).S - 1]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(umagicOK64(c)) {
			break
		}
		v.Reset(ssaop.OpRsh64Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpAvg64u, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpHmul64u, typ.UInt64)
		v2 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(int64(umagic64(c).M))
		v1.AddArg2(x, v2)
		v0.AddArg2(x, v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(umagic64(c).S - 1)
		v.AddArg2(v0, v3)
		return true
	}
	return false
}
func rewriteValuedivmod_OpDiv8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 <t> n (Const8 [c]))
	// cond: ssa.IsPowerOfTwo(c)
	// result: (Rsh8x64 (Add8 <t> n (Rsh8Ux64 <t> (Rsh8x64 <t> n (Const64 <typ.UInt64> [ 7])) (Const64 <typ.UInt64> [int64( 8-log8(c))]))) (Const64 <typ.UInt64> [int64(log8(c))]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != ssaop.OpConst8 {
			break
		}
		c := ssa.AuxIntToInt8(v_1.AuxInt)
		if !(ssa.IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpRsh8x64)
		v0 := b.NewValue0(v.Pos, ssaop.OpAdd8, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpRsh8Ux64, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh8x64, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(7)
		v2.AddArg2(n, v3)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(int64(8 - log8(c)))
		v1.AddArg2(v2, v4)
		v0.AddArg2(n, v1)
		v5 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v5.AuxInt = ssa.Int64ToAuxInt(int64(log8(c)))
		v.AddArg2(v0, v5)
		return true
	}
	// match: (Div8 <t> x (Const8 [c]))
	// cond: smagicOK8(c)
	// result: (Sub8 <t> (Rsh32x64 <t> (Mul32 <typ.UInt32> (SignExt8to32 x) (Const32 <typ.UInt32> [int32(smagic8(c).M)])) (Const64 <typ.UInt64> [8 + smagic8(c).S])) (Rsh32x64 <t> (SignExt8to32 x) (Const64 <typ.UInt64> [31])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst8 {
			break
		}
		c := ssa.AuxIntToInt8(v_1.AuxInt)
		if !(smagicOK8(c)) {
			break
		}
		v.Reset(ssaop.OpSub8)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRsh32x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst32, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(int32(smagic8(c).M))
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(8 + smagic8(c).S)
		v0.AddArg2(v1, v4)
		v5 := b.NewValue0(v.Pos, ssaop.OpRsh32x64, t)
		v6 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v6.AuxInt = ssa.Int64ToAuxInt(31)
		v5.AddArg2(v2, v6)
		v.AddArg2(v0, v5)
		return true
	}
	return false
}
func rewriteValuedivmod_OpDiv8u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u <t> x (Const8 [c]))
	// cond: umagicOK8(c)
	// result: (Trunc32to8 <t> (Rsh32Ux64 <typ.UInt32> (Mul32 <typ.UInt32> (ZeroExt8to32 x) (Const32 <typ.UInt32> [int32(1<<8 + umagic8(c).M)])) (Const64 <typ.UInt64> [8 + umagic8(c).S])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpConst8 {
			break
		}
		c := ssa.AuxIntToInt8(v_1.AuxInt)
		if !(umagicOK8(c)) {
			break
		}
		v.Reset(ssaop.OpTrunc32to8)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRsh32Ux64, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpConst32, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(int32(1<<8 + umagic8(c).M))
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, ssaop.OpConst64, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(8 + umagic8(c).S)
		v0.AddArg2(v1, v4)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteBlockdivmod(b *ssa.Block) bool {
	return false
}
