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

// logosMaskedLoads lists the fault-suppressing slice loaders of
// simd/archsimd (logos_maskload_amd64.go) with the generic masked-load
// operation each one lowers to. The generic operations and their amd64
// rules (VPMASKMOVD and VPMASKMOVQ on AVX2, the zeroing K-masked
// VMOVDQU8 and VMOVDQU16 on AVX512) already exist; the upstream simdgen
// output registers no load that takes a mask, so the loaders register
// here.
var logosMaskedLoads = []struct {
	name string
	op   ssa.Op
}{
	{"LoadFloat32x8SliceMasked", ssa.OpLoadMasked32},
	{"LoadInt32x8SliceMasked", ssa.OpLoadMasked32},
	{"LoadUint32x8SliceMasked", ssa.OpLoadMasked32},
	{"LoadFloat64x4SliceMasked", ssa.OpLoadMasked64},
	{"LoadInt64x4SliceMasked", ssa.OpLoadMasked64},
	{"LoadUint64x4SliceMasked", ssa.OpLoadMasked64},
	{"LoadInt16x32SliceMasked", ssa.OpLoadMasked16},
	{"LoadUint16x32SliceMasked", ssa.OpLoadMasked16},
	{"LoadInt8x64SliceMasked", ssa.OpLoadMasked8},
	{"LoadUint8x64SliceMasked", ssa.OpLoadMasked8},
}

// logosMaskedLoadSlice builds a masked load from the first element of
// its slice argument. A slice taken at its length has a pointer that
// never points past the backing array, and a masked-off lane reads no
// memory, so an empty slice and a zero mask read nothing.
func logosMaskedLoadSlice(op ssa.Op) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		ptr := s.newValue1(ssa.OpSlicePtr, types.NewPtr(args[0].Type.Elem()), args[0])
		return s.newValue3(op, n.Type(), ptr, args[1], s.mem())
	}
}

// logosSimdIntrinsics registers the masked slice loaders.
func logosSimdIntrinsics(addF func(pkg, fn string, b intrinsicBuilder, archFamilies ...sys.ArchFamily)) {
	for _, l := range logosMaskedLoads {
		addF(simdPackage, l.name, logosMaskedLoadSlice(l.op), sys.AMD64)
	}
}
