// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
)

// simdEqualOp returns the generic op that implements == on values of SIMD
// vector or mask type t. == compares SIMD values as if they were structs with
// one field per element, or one bool per element for masks.
func simdEqualOp(t *types.Type) ssaop.Op {
	shape, ok := t.SIMDShape()
	if !ok {
		base.Fatalf("== on non-SIMD type %v", t)
	}
	if shape.Scalable {
		// SVE types aren't comparable yet.
		base.Fatalf("== on SVE type %v", t)
	}

	// Integer vectors compare bitwise. Fixed-width masks are simply vectors
	// whose elements are 0 or -1, so they also compare bitwise. Float vectors
	// need to compare the individual elements. Hence, the op depends only on
	// whether the elements are floats, and on the vector width.
	elem := shape.Elem.Kind()
	switch t.Size() {
	case 16:
		switch elem {
		case types.TFLOAT32:
			return ssaop.OpEqualVecFloat32x4
		case types.TFLOAT64:
			return ssaop.OpEqualVecFloat64x2
		}
		return ssaop.OpEqualVec128
	case 32:
		switch elem {
		case types.TFLOAT32:
			return ssaop.OpEqualVecFloat32x8
		case types.TFLOAT64:
			return ssaop.OpEqualVecFloat64x4
		}
		return ssaop.OpEqualVec256
	case 64:
		switch elem {
		case types.TFLOAT32:
			return ssaop.OpEqualVecFloat32x16
		case types.TFLOAT64:
			return ssaop.OpEqualVecFloat64x8
		}
		return ssaop.OpEqualVec512
	}
	base.Fatalf("== on SIMD type %v of unexpected size %d", t, t.Size())
	panic("unreachable")
}
