// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"internal/goarch"

	"cmd/compile/internal/ssa"
)

// simdEnlarge bypasses Enlarge for SIMD operations that read only the low part
// of an input.
//
// x.Enlarge() puts x in the low half of a vector twice as wide, zeroing the
// upper half. Some operations, such as the widening ExtendTo* and
// ConvertToFloat64 conversions, read only the low part of an input. If that
// part lies within x, the operation can read x directly and the Enlarge is
// unnecessary. For example, for an Int8x16 x, Int8x32.ExtendToInt32 reads only
// the low 128 bits of x.Enlarge(), which are x, so this rewrites
//
//	(ExtendToInt32Int8x32 (EnlargeInt8x16 x)) -> (ExtendToInt32Int8x32 x)
//
// This also looks through chains of Enlarge, as in
// x.Enlarge().Enlarge().ExtendToInt64().
//
// This works on generic ops, just before lowering, so the lowering rules
// (including the rules that combine loads with ops, and those that form
// masked and merging machine ops) see the results of this optimization.
//
// Bypassing an Enlarge gives the op an input whose type is narrower than
// the op's input type. That's safe because lowering passes the input
// through to the machine op unchanged, and codegen names the input's
// register by the size the instruction reads, not by the input's type.
//
// simdgen generates the tables of Enlarge ops and ops with narrow inputs.
func simdEnlarge(f *ssa.Func) {
	if f.MaxCPUFeatures == ssa.CPUNone {
		return
	}
	// AMD64 is the only architecture with multiple fixed-width vector
	// sizes, so it's the only one with Enlarge.
	if f.Config.Ctxt.Arch.Family != goarch.AMD64 {
		return
	}

	for _, b := range f.Blocks {
		for _, v := range b.Values {
			i, bits := simdAMD64NarrowArg(v.Op)
			if i < 0 {
				continue
			}
			// v reads only the low bits of its i'th argument.
			// Unwind any Enlarges of that argument that don't
			// affect those bits.
			x := v.Args[i]
			for simdAMD64IsEnlarge(x.Op) && x.Args[0].Type.Size()*8 >= bits {
				x = x.Args[0]
			}
			if x != v.Args[i] {
				if f.Pass.Debug > 0 {
					f.Warnl(v.Pos, "bypassed Enlarge for %v", v.Op)
				}
				v.SetArg(i, x)
			}
		}
	}
}
