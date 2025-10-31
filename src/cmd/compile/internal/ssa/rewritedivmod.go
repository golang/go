// Code generated from _gen/divmod.rules using 'go generate'; DO NOT EDIT.

package ssa

func rewriteValuedivmod(v *Value) bool {
	switch v.Op {
	case OpDiv16:
		return rewriteValuedivmod_OpDiv16(v)
	case OpDiv16u:
		return rewriteValuedivmod_OpDiv16u(v)
	case OpDiv32:
		return rewriteValuedivmod_OpDiv32(v)
	case OpDiv32u:
		return rewriteValuedivmod_OpDiv32u(v)
	case OpDiv64:
		return rewriteValuedivmod_OpDiv64(v)
	case OpDiv64u:
		return rewriteValuedivmod_OpDiv64u(v)
	case OpDiv8:
		return rewriteValuedivmod_OpDiv8(v)
	case OpDiv8u:
		return rewriteValuedivmod_OpDiv8u(v)
	}
	return false
}
func rewriteValuedivmod_OpDiv16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 <t> n (Const16 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Rsh16x64 (Add16 <t> n (Rsh16Ux64 <t> (Rsh16x64 <t> n (Const64 <typ.UInt64> [15])) (Const64 <typ.UInt64> [int64(16-log16(c))]))) (Const64 <typ.UInt64> [int64(log16(c))]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh16x64)
		v0 := b.NewValue0(v.Pos, OpAdd16, t)
		v1 := b.NewValue0(v.Pos, OpRsh16Ux64, t)
		v2 := b.NewValue0(v.Pos, OpRsh16x64, t)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(15)
		v2.AddArg2(n, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(int64(16 - log16(c)))
		v1.AddArg2(v2, v4)
		v0.AddArg2(n, v1)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(int64(log16(c)))
		v.AddArg2(v0, v5)
		return true
	}
	// match: (Div16 <t> x (Const16 [c]))
	// cond: smagicOK16(c)
	// result: (Sub16 <t> (Rsh32x64 <t> (Mul32 <typ.UInt32> (SignExt16to32 x) (Const32 <typ.UInt32> [int32(smagic16(c).m)])) (Const64 <typ.UInt64> [16 + smagic16(c).s])) (Rsh32x64 <t> (SignExt16to32 x) (Const64 <typ.UInt64> [31])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(smagicOK16(c)) {
			break
		}
		v.reset(OpSub16)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v3.AuxInt = int32ToAuxInt(int32(smagic16(c).m))
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(16 + smagic16(c).s)
		v0.AddArg2(v1, v4)
		v5 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = int64ToAuxInt(31)
		v5.AddArg2(v2, v6)
		v.AddArg2(v0, v5)
		return true
	}
	return false
}
func rewriteValuedivmod_OpDiv16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Div16u <t> x (Const16 [c]))
	// cond: t.IsSigned() && smagicOK16(c)
	// result: (Rsh32Ux64 <t> (Mul32 <typ.UInt32> (SignExt16to32 x) (Const32 <typ.UInt32> [int32(smagic16(c).m)])) (Const64 <typ.UInt64> [16 + smagic16(c).s]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(t.IsSigned() && smagicOK16(c)) {
			break
		}
		v.reset(OpRsh32Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v2.AuxInt = int32ToAuxInt(int32(smagic16(c).m))
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(16 + smagic16(c).s)
		v.AddArg2(v0, v3)
		return true
	}
	// match: (Div16u <t> x (Const16 [c]))
	// cond: umagicOK16(c) && config.RegSize == 8
	// result: (Trunc64to16 <t> (Rsh64Ux64 <typ.UInt64> (Mul64 <typ.UInt64> (ZeroExt16to64 x) (Const64 <typ.UInt64> [int64(1<<16 + umagic16(c).m)])) (Const64 <typ.UInt64> [16 + umagic16(c).s])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(umagicOK16(c) && config.RegSize == 8) {
			break
		}
		v.reset(OpTrunc64to16)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(int64(1<<16 + umagic16(c).m))
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(16 + umagic16(c).s)
		v0.AddArg2(v1, v4)
		v.AddArg(v0)
		return true
	}
	// match: (Div16u <t> x (Const16 [c]))
	// cond: umagicOK16(c) && umagic16(c).m&1 == 0
	// result: (Trunc32to16 <t> (Rsh32Ux64 <typ.UInt32> (Mul32 <typ.UInt32> (ZeroExt16to32 x) (Const32 <typ.UInt32> [int32(1<<15 + umagic16(c).m/2)])) (Const64 <typ.UInt64> [16 + umagic16(c).s - 1])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(umagicOK16(c) && umagic16(c).m&1 == 0) {
			break
		}
		v.reset(OpTrunc32to16)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v3.AuxInt = int32ToAuxInt(int32(1<<15 + umagic16(c).m/2))
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(16 + umagic16(c).s - 1)
		v0.AddArg2(v1, v4)
		v.AddArg(v0)
		return true
	}
	// match: (Div16u <t> x (Const16 [c]))
	// cond: umagicOK16(c) && config.RegSize == 4 && c&1 == 0
	// result: (Trunc32to16 <t> (Rsh32Ux64 <typ.UInt32> (Mul32 <typ.UInt32> (Rsh32Ux64 <typ.UInt32> (ZeroExt16to32 x) (Const64 <typ.UInt64> [1])) (Const32 <typ.UInt32> [int32(1<<15 + (umagic16(c).m+1)/2)])) (Const64 <typ.UInt64> [16 + umagic16(c).s - 2])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(umagicOK16(c) && config.RegSize == 4 && c&1 == 0) {
			break
		}
		v.reset(OpTrunc32to16)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(1)
		v2.AddArg2(v3, v4)
		v5 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v5.AuxInt = int32ToAuxInt(int32(1<<15 + (umagic16(c).m+1)/2))
		v1.AddArg2(v2, v5)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = int64ToAuxInt(16 + umagic16(c).s - 2)
		v0.AddArg2(v1, v6)
		v.AddArg(v0)
		return true
	}
	// match: (Div16u <t> x (Const16 [c]))
	// cond: umagicOK16(c) && config.RegSize == 4 && config.useAvg
	// result: (Trunc32to16 <t> (Rsh32Ux64 <typ.UInt32> (Avg32u (Lsh32x64 <typ.UInt32> (ZeroExt16to32 x) (Const64 <typ.UInt64> [16])) (Mul32 <typ.UInt32> (ZeroExt16to32 x) (Const32 <typ.UInt32> [int32(umagic16(c).m)]))) (Const64 <typ.UInt64> [16 + umagic16(c).s - 1])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst16 {
			break
		}
		c := auxIntToInt16(v_1.AuxInt)
		if !(umagicOK16(c) && config.RegSize == 4 && config.useAvg) {
			break
		}
		v.reset(OpTrunc32to16)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpAvg32u, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpLsh32x64, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(16)
		v2.AddArg2(v3, v4)
		v5 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v6 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v6.AuxInt = int32ToAuxInt(int32(umagic16(c).m))
		v5.AddArg2(v3, v6)
		v1.AddArg2(v2, v5)
		v7 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v7.AuxInt = int64ToAuxInt(16 + umagic16(c).s - 1)
		v0.AddArg2(v1, v7)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuedivmod_OpDiv32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Div32 <t> n (Const32 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Rsh32x64 (Add32 <t> n (Rsh32Ux64 <t> (Rsh32x64 <t> n (Const64 <typ.UInt64> [31])) (Const64 <typ.UInt64> [int64(32-log32(c))]))) (Const64 <typ.UInt64> [int64(log32(c))]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh32x64)
		v0 := b.NewValue0(v.Pos, OpAdd32, t)
		v1 := b.NewValue0(v.Pos, OpRsh32Ux64, t)
		v2 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(31)
		v2.AddArg2(n, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(int64(32 - log32(c)))
		v1.AddArg2(v2, v4)
		v0.AddArg2(n, v1)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(int64(log32(c)))
		v.AddArg2(v0, v5)
		return true
	}
	// match: (Div32 <t> x (Const32 [c]))
	// cond: smagicOK32(c) && config.RegSize == 8
	// result: (Sub32 <t> (Rsh64x64 <t> (Mul64 <typ.UInt64> (SignExt32to64 x) (Const64 <typ.UInt64> [int64(smagic32(c).m)])) (Const64 <typ.UInt64> [32 + smagic32(c).s])) (Rsh64x64 <t> (SignExt32to64 x) (Const64 <typ.UInt64> [63])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(smagicOK32(c) && config.RegSize == 8) {
			break
		}
		v.reset(OpSub32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v1 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(int64(smagic32(c).m))
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(32 + smagic32(c).s)
		v0.AddArg2(v1, v4)
		v5 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = int64ToAuxInt(63)
		v5.AddArg2(v2, v6)
		v.AddArg2(v0, v5)
		return true
	}
	// match: (Div32 <t> x (Const32 [c]))
	// cond: smagicOK32(c) && config.RegSize == 4 && smagic32(c).m&1 == 0 && config.useHmul
	// result: (Sub32 <t> (Rsh32x64 <t> (Hmul32 <t> x (Const32 <typ.UInt32> [int32(smagic32(c).m/2)])) (Const64 <typ.UInt64> [smagic32(c).s - 1])) (Rsh32x64 <t> x (Const64 <typ.UInt64> [31])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(smagicOK32(c) && config.RegSize == 4 && smagic32(c).m&1 == 0 && config.useHmul) {
			break
		}
		v.reset(OpSub32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v1 := b.NewValue0(v.Pos, OpHmul32, t)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v2.AuxInt = int32ToAuxInt(int32(smagic32(c).m / 2))
		v1.AddArg2(x, v2)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(smagic32(c).s - 1)
		v0.AddArg2(v1, v3)
		v4 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(31)
		v4.AddArg2(x, v5)
		v.AddArg2(v0, v4)
		return true
	}
	// match: (Div32 <t> x (Const32 [c]))
	// cond: smagicOK32(c) && config.RegSize == 4 && smagic32(c).m&1 != 0 && config.useHmul
	// result: (Sub32 <t> (Rsh32x64 <t> (Add32 <t> x (Hmul32 <t> x (Const32 <typ.UInt32> [int32(smagic32(c).m)]))) (Const64 <typ.UInt64> [smagic32(c).s])) (Rsh32x64 <t> x (Const64 <typ.UInt64> [31])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(smagicOK32(c) && config.RegSize == 4 && smagic32(c).m&1 != 0 && config.useHmul) {
			break
		}
		v.reset(OpSub32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v1 := b.NewValue0(v.Pos, OpAdd32, t)
		v2 := b.NewValue0(v.Pos, OpHmul32, t)
		v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v3.AuxInt = int32ToAuxInt(int32(smagic32(c).m))
		v2.AddArg2(x, v3)
		v1.AddArg2(x, v2)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(smagic32(c).s)
		v0.AddArg2(v1, v4)
		v5 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = int64ToAuxInt(31)
		v5.AddArg2(x, v6)
		v.AddArg2(v0, v5)
		return true
	}
	return false
}
func rewriteValuedivmod_OpDiv32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Div32u <t> x (Const32 [c]))
	// cond: t.IsSigned() && smagicOK32(c) && config.RegSize == 8
	// result: (Rsh64Ux64 <t> (Mul64 <typ.UInt64> (SignExt32to64 x) (Const64 <typ.UInt64> [int64(smagic32(c).m)])) (Const64 <typ.UInt64> [32 + smagic32(c).s]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(t.IsSigned() && smagicOK32(c) && config.RegSize == 8) {
			break
		}
		v.reset(OpRsh64Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(int64(smagic32(c).m))
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(32 + smagic32(c).s)
		v.AddArg2(v0, v3)
		return true
	}
	// match: (Div32u <t> x (Const32 [c]))
	// cond: t.IsSigned() && smagicOK32(c) && config.RegSize == 4 && config.useHmul
	// result: (Rsh32Ux64 <t> (Hmul32u <typ.UInt32> x (Const32 <typ.UInt32> [int32(smagic32(c).m)])) (Const64 <typ.UInt64> [smagic32(c).s]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(t.IsSigned() && smagicOK32(c) && config.RegSize == 4 && config.useHmul) {
			break
		}
		v.reset(OpRsh32Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpHmul32u, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v1.AuxInt = int32ToAuxInt(int32(smagic32(c).m))
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(smagic32(c).s)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Div32u <t> x (Const32 [c]))
	// cond: umagicOK32(c) && umagic32(c).m&1 == 0 && config.RegSize == 8
	// result: (Trunc64to32 <t> (Rsh64Ux64 <typ.UInt64> (Mul64 <typ.UInt64> (ZeroExt32to64 x) (Const64 <typ.UInt64> [int64(1<<31 + umagic32(c).m/2)])) (Const64 <typ.UInt64> [32 + umagic32(c).s - 1])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && umagic32(c).m&1 == 0 && config.RegSize == 8) {
			break
		}
		v.reset(OpTrunc64to32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(int64(1<<31 + umagic32(c).m/2))
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(32 + umagic32(c).s - 1)
		v0.AddArg2(v1, v4)
		v.AddArg(v0)
		return true
	}
	// match: (Div32u <t> x (Const32 [c]))
	// cond: umagicOK32(c) && umagic32(c).m&1 == 0 && config.RegSize == 4 && config.useHmul
	// result: (Rsh32Ux64 <t> (Hmul32u <typ.UInt32> x (Const32 <typ.UInt32> [int32(1<<31 + umagic32(c).m/2)])) (Const64 <typ.UInt64> [umagic32(c).s - 1]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && umagic32(c).m&1 == 0 && config.RegSize == 4 && config.useHmul) {
			break
		}
		v.reset(OpRsh32Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpHmul32u, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v1.AuxInt = int32ToAuxInt(int32(1<<31 + umagic32(c).m/2))
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(umagic32(c).s - 1)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Div32u <t> x (Const32 [c]))
	// cond: umagicOK32(c) && config.RegSize == 8 && c&1 == 0
	// result: (Trunc64to32 <t> (Rsh64Ux64 <typ.UInt64> (Mul64 <typ.UInt64> (Rsh64Ux64 <typ.UInt64> (ZeroExt32to64 x) (Const64 <typ.UInt64> [1])) (Const64 <typ.UInt64> [int64(1<<31 + (umagic32(c).m+1)/2)])) (Const64 <typ.UInt64> [32 + umagic32(c).s - 2])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && config.RegSize == 8 && c&1 == 0) {
			break
		}
		v.reset(OpTrunc64to32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(1)
		v2.AddArg2(v3, v4)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(int64(1<<31 + (umagic32(c).m+1)/2))
		v1.AddArg2(v2, v5)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = int64ToAuxInt(32 + umagic32(c).s - 2)
		v0.AddArg2(v1, v6)
		v.AddArg(v0)
		return true
	}
	// match: (Div32u <t> x (Const32 [c]))
	// cond: umagicOK32(c) && config.RegSize == 4 && c&1 == 0 && config.useHmul
	// result: (Rsh32Ux64 <t> (Hmul32u <typ.UInt32> (Rsh32Ux64 <typ.UInt32> x (Const64 <typ.UInt64> [1])) (Const32 <typ.UInt32> [int32(1<<31 + (umagic32(c).m+1)/2)])) (Const64 <typ.UInt64> [umagic32(c).s - 2]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && config.RegSize == 4 && c&1 == 0 && config.useHmul) {
			break
		}
		v.reset(OpRsh32Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpHmul32u, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(1)
		v1.AddArg2(x, v2)
		v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v3.AuxInt = int32ToAuxInt(int32(1<<31 + (umagic32(c).m+1)/2))
		v0.AddArg2(v1, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(umagic32(c).s - 2)
		v.AddArg2(v0, v4)
		return true
	}
	// match: (Div32u <t> x (Const32 [c]))
	// cond: umagicOK32(c) && config.RegSize == 8 && config.useAvg
	// result: (Trunc64to32 <t> (Rsh64Ux64 <typ.UInt64> (Avg64u (Lsh64x64 <typ.UInt64> (ZeroExt32to64 x) (Const64 <typ.UInt64> [32])) (Mul64 <typ.UInt64> (ZeroExt32to64 x) (Const64 <typ.UInt32> [int64(umagic32(c).m)]))) (Const64 <typ.UInt64> [32 + umagic32(c).s - 1])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && config.RegSize == 8 && config.useAvg) {
			break
		}
		v.reset(OpTrunc64to32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpAvg64u, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpLsh64x64, typ.UInt64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(32)
		v2.AddArg2(v3, v4)
		v5 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt32)
		v6.AuxInt = int64ToAuxInt(int64(umagic32(c).m))
		v5.AddArg2(v3, v6)
		v1.AddArg2(v2, v5)
		v7 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v7.AuxInt = int64ToAuxInt(32 + umagic32(c).s - 1)
		v0.AddArg2(v1, v7)
		v.AddArg(v0)
		return true
	}
	// match: (Div32u <t> x (Const32 [c]))
	// cond: umagicOK32(c) && config.RegSize == 4 && config.useAvg && config.useHmul
	// result: (Rsh32Ux64 <t> (Avg32u x (Hmul32u <typ.UInt32> x (Const32 <typ.UInt32> [int32(umagic32(c).m)]))) (Const64 <typ.UInt64> [umagic32(c).s - 1]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst32 {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(umagicOK32(c) && config.RegSize == 4 && config.useAvg && config.useHmul) {
			break
		}
		v.reset(OpRsh32Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAvg32u, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpHmul32u, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v2.AuxInt = int32ToAuxInt(int32(umagic32(c).m))
		v1.AddArg2(x, v2)
		v0.AddArg2(x, v1)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(umagic32(c).s - 1)
		v.AddArg2(v0, v3)
		return true
	}
	return false
}
func rewriteValuedivmod_OpDiv64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Div64 <t> n (Const64 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Rsh64x64 (Add64 <t> n (Rsh64Ux64 <t> (Rsh64x64 <t> n (Const64 <typ.UInt64> [63])) (Const64 <typ.UInt64> [int64(64-log64(c))]))) (Const64 <typ.UInt64> [int64(log64(c))]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh64x64)
		v0 := b.NewValue0(v.Pos, OpAdd64, t)
		v1 := b.NewValue0(v.Pos, OpRsh64Ux64, t)
		v2 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(63)
		v2.AddArg2(n, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(int64(64 - log64(c)))
		v1.AddArg2(v2, v4)
		v0.AddArg2(n, v1)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(int64(log64(c)))
		v.AddArg2(v0, v5)
		return true
	}
	// match: (Div64 <t> x (Const64 [c]))
	// cond: smagicOK64(c) && smagic64(c).m&1 == 0 && config.useHmul
	// result: (Sub64 <t> (Rsh64x64 <t> (Hmul64 <t> x (Const64 <typ.UInt64> [int64(smagic64(c).m/2)])) (Const64 <typ.UInt64> [smagic64(c).s - 1])) (Rsh64x64 <t> x (Const64 <typ.UInt64> [63])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(smagicOK64(c) && smagic64(c).m&1 == 0 && config.useHmul) {
			break
		}
		v.reset(OpSub64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v1 := b.NewValue0(v.Pos, OpHmul64, t)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(int64(smagic64(c).m / 2))
		v1.AddArg2(x, v2)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(smagic64(c).s - 1)
		v0.AddArg2(v1, v3)
		v4 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(63)
		v4.AddArg2(x, v5)
		v.AddArg2(v0, v4)
		return true
	}
	// match: (Div64 <t> x (Const64 [c]))
	// cond: smagicOK64(c) && smagic64(c).m&1 != 0 && config.useHmul
	// result: (Sub64 <t> (Rsh64x64 <t> (Add64 <t> x (Hmul64 <t> x (Const64 <typ.UInt64> [int64(smagic64(c).m)]))) (Const64 <typ.UInt64> [smagic64(c).s])) (Rsh64x64 <t> x (Const64 <typ.UInt64> [63])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(smagicOK64(c) && smagic64(c).m&1 != 0 && config.useHmul) {
			break
		}
		v.reset(OpSub64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v1 := b.NewValue0(v.Pos, OpAdd64, t)
		v2 := b.NewValue0(v.Pos, OpHmul64, t)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(int64(smagic64(c).m))
		v2.AddArg2(x, v3)
		v1.AddArg2(x, v2)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(smagic64(c).s)
		v0.AddArg2(v1, v4)
		v5 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = int64ToAuxInt(63)
		v5.AddArg2(x, v6)
		v.AddArg2(v0, v5)
		return true
	}
	return false
}
func rewriteValuedivmod_OpDiv64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Div64u <t> x (Const64 [c]))
	// cond: t.IsSigned() && smagicOK64(c) && config.useHmul
	// result: (Rsh64Ux64 <t> (Hmul64u <typ.UInt64> x (Const64 <typ.UInt64> [int64(smagic64(c).m)])) (Const64 <typ.UInt64> [smagic64(c).s]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(t.IsSigned() && smagicOK64(c) && config.useHmul) {
			break
		}
		v.reset(OpRsh64Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpHmul64u, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(int64(smagic64(c).m))
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(smagic64(c).s)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Div64u <t> x (Const64 [c]))
	// cond: umagicOK64(c) && umagic64(c).m&1 == 0 && config.useHmul
	// result: (Rsh64Ux64 <t> (Hmul64u <typ.UInt64> x (Const64 <typ.UInt64> [int64(1<<63 + umagic64(c).m/2)])) (Const64 <typ.UInt64> [umagic64(c).s - 1]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(umagicOK64(c) && umagic64(c).m&1 == 0 && config.useHmul) {
			break
		}
		v.reset(OpRsh64Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpHmul64u, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(int64(1<<63 + umagic64(c).m/2))
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(umagic64(c).s - 1)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Div64u <t> x (Const64 [c]))
	// cond: umagicOK64(c) && c&1 == 0 && config.useHmul
	// result: (Rsh64Ux64 <t> (Hmul64u <typ.UInt64> (Rsh64Ux64 <typ.UInt64> x (Const64 <typ.UInt64> [1])) (Const64 <typ.UInt64> [int64(1<<63 + (umagic64(c).m+1)/2)])) (Const64 <typ.UInt64> [umagic64(c).s - 2]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(umagicOK64(c) && c&1 == 0 && config.useHmul) {
			break
		}
		v.reset(OpRsh64Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpHmul64u, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(1)
		v1.AddArg2(x, v2)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(int64(1<<63 + (umagic64(c).m+1)/2))
		v0.AddArg2(v1, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(umagic64(c).s - 2)
		v.AddArg2(v0, v4)
		return true
	}
	// match: (Div64u <t> x (Const64 [c]))
	// cond: umagicOK64(c) && config.useAvg && config.useHmul
	// result: (Rsh64Ux64 <t> (Avg64u x (Hmul64u <typ.UInt64> x (Const64 <typ.UInt64> [int64(umagic64(c).m)]))) (Const64 <typ.UInt64> [umagic64(c).s - 1]))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(umagicOK64(c) && config.useAvg && config.useHmul) {
			break
		}
		v.reset(OpRsh64Ux64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAvg64u, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpHmul64u, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(int64(umagic64(c).m))
		v1.AddArg2(x, v2)
		v0.AddArg2(x, v1)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(umagic64(c).s - 1)
		v.AddArg2(v0, v3)
		return true
	}
	return false
}
func rewriteValuedivmod_OpDiv8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 <t> n (Const8 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Rsh8x64 (Add8 <t> n (Rsh8Ux64 <t> (Rsh8x64 <t> n (Const64 <typ.UInt64> [ 7])) (Const64 <typ.UInt64> [int64( 8-log8(c))]))) (Const64 <typ.UInt64> [int64(log8(c))]))
	for {
		t := v.Type
		n := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh8x64)
		v0 := b.NewValue0(v.Pos, OpAdd8, t)
		v1 := b.NewValue0(v.Pos, OpRsh8Ux64, t)
		v2 := b.NewValue0(v.Pos, OpRsh8x64, t)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(7)
		v2.AddArg2(n, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(int64(8 - log8(c)))
		v1.AddArg2(v2, v4)
		v0.AddArg2(n, v1)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(int64(log8(c)))
		v.AddArg2(v0, v5)
		return true
	}
	// match: (Div8 <t> x (Const8 [c]))
	// cond: smagicOK8(c)
	// result: (Sub8 <t> (Rsh32x64 <t> (Mul32 <typ.UInt32> (SignExt8to32 x) (Const32 <typ.UInt32> [int32(smagic8(c).m)])) (Const64 <typ.UInt64> [8 + smagic8(c).s])) (Rsh32x64 <t> (SignExt8to32 x) (Const64 <typ.UInt64> [31])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		if !(smagicOK8(c)) {
			break
		}
		v.reset(OpSub8)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v3.AuxInt = int32ToAuxInt(int32(smagic8(c).m))
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(8 + smagic8(c).s)
		v0.AddArg2(v1, v4)
		v5 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = int64ToAuxInt(31)
		v5.AddArg2(v2, v6)
		v.AddArg2(v0, v5)
		return true
	}
	return false
}
func rewriteValuedivmod_OpDiv8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u <t> x (Const8 [c]))
	// cond: umagicOK8(c)
	// result: (Trunc32to8 <t> (Rsh32Ux64 <typ.UInt32> (Mul32 <typ.UInt32> (ZeroExt8to32 x) (Const32 <typ.UInt32> [int32(1<<8 + umagic8(c).m)])) (Const64 <typ.UInt64> [8 + umagic8(c).s])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpConst8 {
			break
		}
		c := auxIntToInt8(v_1.AuxInt)
		if !(umagicOK8(c)) {
			break
		}
		v.reset(OpTrunc32to8)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v3.AuxInt = int32ToAuxInt(int32(1<<8 + umagic8(c).m))
		v1.AddArg2(v2, v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(8 + umagic8(c).s)
		v0.AddArg2(v1, v4)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteBlockdivmod(b *Block) bool {
	return false
}
