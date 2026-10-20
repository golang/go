// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/sys"
)

type logosMaskedLoadConfig struct {
	name     string
	op       ssa.Op
	lanes    int64
	bits     int64
	vec      *types.Type
	trunc    ssa.Op
	setElem  ssa.Op
	setConst ssa.Op
	bcast    ssa.Op
	and      ssa.Op
	greater  ssa.Op
	lo       [2]ssa.Op
	hi       [2]ssa.Op
}

var logosMaskedLoads = []logosMaskedLoadConfig{
	{
		name: "LoadFloat32x8SliceMasked", op: ssa.OpLoadMasked32,
		lanes: 8, bits: 32, vec: types.TypeVec256,
		trunc: ssa.OpTrunc64to32, setElem: ssa.OpSetElemInt32x4, setConst: ssa.OpConst32,
		bcast: ssa.Opbroadcast1To8Int32x4, and: ssa.OpAndInt32x8, greater: ssa.OpGreaterInt32x8,
		lo: [2]ssa.Op{ssa.OpSetLoInt32x8, 0}, hi: [2]ssa.Op{ssa.OpSetHiInt32x8, 0},
	},
	{
		name: "LoadInt32x8SliceMasked", op: ssa.OpLoadMasked32,
		lanes: 8, bits: 32, vec: types.TypeVec256,
		trunc: ssa.OpTrunc64to32, setElem: ssa.OpSetElemInt32x4, setConst: ssa.OpConst32,
		bcast: ssa.Opbroadcast1To8Int32x4, and: ssa.OpAndInt32x8, greater: ssa.OpGreaterInt32x8,
		lo: [2]ssa.Op{ssa.OpSetLoInt32x8, 0}, hi: [2]ssa.Op{ssa.OpSetHiInt32x8, 0},
	},
	{
		name: "LoadUint32x8SliceMasked", op: ssa.OpLoadMasked32,
		lanes: 8, bits: 32, vec: types.TypeVec256,
		trunc: ssa.OpTrunc64to32, setElem: ssa.OpSetElemInt32x4, setConst: ssa.OpConst32,
		bcast: ssa.Opbroadcast1To8Int32x4, and: ssa.OpAndInt32x8, greater: ssa.OpGreaterInt32x8,
		lo: [2]ssa.Op{ssa.OpSetLoInt32x8, 0}, hi: [2]ssa.Op{ssa.OpSetHiInt32x8, 0},
	},
	{
		name: "LoadFloat64x4SliceMasked", op: ssa.OpLoadMasked64,
		lanes: 4, bits: 64, vec: types.TypeVec256,
		setElem: ssa.OpSetElemInt64x2, setConst: ssa.OpConst64,
		bcast: ssa.Opbroadcast1To4Int64x2, and: ssa.OpAndInt32x8, greater: ssa.OpGreaterInt64x4,
		lo: [2]ssa.Op{ssa.OpSetLoInt64x4, 0}, hi: [2]ssa.Op{ssa.OpSetHiInt64x4, 0},
	},
	{
		name: "LoadInt64x4SliceMasked", op: ssa.OpLoadMasked64,
		lanes: 4, bits: 64, vec: types.TypeVec256,
		setElem: ssa.OpSetElemInt64x2, setConst: ssa.OpConst64,
		bcast: ssa.Opbroadcast1To4Int64x2, and: ssa.OpAndInt32x8, greater: ssa.OpGreaterInt64x4,
		lo: [2]ssa.Op{ssa.OpSetLoInt64x4, 0}, hi: [2]ssa.Op{ssa.OpSetHiInt64x4, 0},
	},
	{
		name: "LoadUint64x4SliceMasked", op: ssa.OpLoadMasked64,
		lanes: 4, bits: 64, vec: types.TypeVec256,
		setElem: ssa.OpSetElemInt64x2, setConst: ssa.OpConst64,
		bcast: ssa.Opbroadcast1To4Int64x2, and: ssa.OpAndInt32x8, greater: ssa.OpGreaterInt64x4,
		lo: [2]ssa.Op{ssa.OpSetLoInt64x4, 0}, hi: [2]ssa.Op{ssa.OpSetHiInt64x4, 0},
	},
	{
		name: "LoadInt16x32SliceMasked", op: ssa.OpLoadMasked16,
		lanes: 32, bits: 16, vec: types.TypeVec512,
		trunc: ssa.OpTrunc64to16, setElem: ssa.OpSetElemInt16x8, setConst: ssa.OpConst16,
		bcast: ssa.Opbroadcast1To32Int16x8, and: ssa.OpAndInt32x16, greater: ssa.OpGreaterInt16x32,
		lo: [2]ssa.Op{ssa.OpSetLoInt16x16, ssa.OpSetLoInt16x32},
		hi: [2]ssa.Op{ssa.OpSetHiInt16x16, ssa.OpSetHiInt16x32},
	},
	{
		name: "LoadUint16x32SliceMasked", op: ssa.OpLoadMasked16,
		lanes: 32, bits: 16, vec: types.TypeVec512,
		trunc: ssa.OpTrunc64to16, setElem: ssa.OpSetElemInt16x8, setConst: ssa.OpConst16,
		bcast: ssa.Opbroadcast1To32Int16x8, and: ssa.OpAndInt32x16, greater: ssa.OpGreaterInt16x32,
		lo: [2]ssa.Op{ssa.OpSetLoInt16x16, ssa.OpSetLoInt16x32},
		hi: [2]ssa.Op{ssa.OpSetHiInt16x16, ssa.OpSetHiInt16x32},
	},
	{
		name: "LoadInt8x64SliceMasked", op: ssa.OpLoadMasked8,
		lanes: 64, bits: 8, vec: types.TypeVec512,
		trunc: ssa.OpTrunc64to8, setElem: ssa.OpSetElemInt8x16, setConst: ssa.OpConst8,
		bcast: ssa.Opbroadcast1To64Int8x16, and: ssa.OpAndInt32x16, greater: ssa.OpGreaterInt8x64,
		lo: [2]ssa.Op{ssa.OpSetLoInt8x32, ssa.OpSetLoInt8x64},
		hi: [2]ssa.Op{ssa.OpSetHiInt8x32, ssa.OpSetHiInt8x64},
	},
	{
		name: "LoadUint8x64SliceMasked", op: ssa.OpLoadMasked8,
		lanes: 64, bits: 8, vec: types.TypeVec512,
		trunc: ssa.OpTrunc64to8, setElem: ssa.OpSetElemInt8x16, setConst: ssa.OpConst8,
		bcast: ssa.Opbroadcast1To64Int8x16, and: ssa.OpAndInt32x16, greater: ssa.OpGreaterInt8x64,
		lo: [2]ssa.Op{ssa.OpSetLoInt8x32, ssa.OpSetLoInt8x64},
		hi: [2]ssa.Op{ssa.OpSetHiInt8x32, ssa.OpSetHiInt8x64},
	},
}

func logosMaskedLoadSlice(cfg logosMaskedLoadConfig) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		tint := types.Types[types.TINT]
		scalar := types.Types[types.TINT64]
		switch cfg.bits {
		case 8:
			scalar = types.Types[types.TINT8]
		case 16:
			scalar = types.Types[types.TINT16]
		case 32:
			scalar = types.Types[types.TINT32]
		}
		ptr := s.newValue1(ssa.OpSlicePtr, types.NewPtr(args[0].Type.Elem()), args[0])
		ln := s.newValue1(ssa.OpSliceLen, tint, args[0])
		count := s.newValue0I(ssa.OpConst64, tint, cfg.lanes)
		d := s.newValue2(ssa.OpSub64, tint, ln, count)
		sh := s.newValue0I(ssa.OpConst64, types.Types[types.TUINT64], 63)
		m := s.newValue2(ssa.OpRsh64x64, tint, d, sh)
		nc := s.newValue2(ssa.OpAdd64, tint, count, s.newValue2(ssa.OpAnd64, tint, d, m))
		nv := nc
		if cfg.trunc != 0 {
			nv = s.newValue1(cfg.trunc, scalar, nc)
		}
		z128 := s.newValue0(ssa.OpZeroSIMD, types.TypeVec128)
		splat := s.newValue1(cfg.bcast, cfg.vec, s.newValue2I(cfg.setElem, types.TypeVec128, 0, z128, nv))
		per := 128 / cfg.bits
		var chunks []*ssa.Value
		for base := int64(0); base < cfg.lanes; base += per {
			v := z128
			for i := int64(0); i < per; i++ {
				v = s.newValue2I(cfg.setElem, types.TypeVec128, i, v, s.newValue0I(cfg.setConst, scalar, base+i))
			}
			chunks = append(chunks, v)
		}
		widths := [2]*types.Type{types.TypeVec256, types.TypeVec512}
		level := 0
		for len(chunks) > 1 {
			zt := widths[level]
			z := s.newValue0(ssa.OpZeroSIMD, zt)
			var next []*ssa.Value
			for i := 0; i < len(chunks); i += 2 {
				next = append(next, s.newValue2(cfg.hi[level], zt, s.newValue2(cfg.lo[level], zt, z, chunks[i]), chunks[i+1]))
			}
			chunks = next
			level++
		}
		bound := s.newValue2(cfg.greater, cfg.vec, splat, chunks[0])
		masked := s.newValue2(cfg.and, cfg.vec, args[1], bound)
		return s.newValue3(cfg.op, n.Type(), ptr, masked, s.mem())
	}
}

// logosSimdIntrinsics registers the masked slice loaders.
func logosSimdIntrinsics(addF func(pkg, fn string, b intrinsicBuilder, archFamilies ...sys.ArchFamily)) {
	for _, l := range logosMaskedLoads {
		addF(simdPackage, l.name, logosMaskedLoadSlice(l), sys.AMD64)
	}
}
