// asmcheck

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check code generation of simd peephole optimizations.

//go:build goexperiment.simd

package codegen

import "simd"

func vptest1() bool {
	v1 := simd.LoadUint64x2Slice([]uint64{0, 1})
	v2 := simd.LoadUint64x2Slice([]uint64{0, 0})
	// amd64:`VPTEST\s(.*)(.*)$`
	// amd64:`SETCS\s(.*)$`
	return v1.AndNot(v2).IsZero()
}

func vptest2() bool {
	v1 := simd.LoadUint64x2Slice([]uint64{0, 1})
	v2 := simd.LoadUint64x2Slice([]uint64{0, 0})
	// amd64:`VPTEST\s(.*)(.*)$`
	// amd64:`SETEQ\s(.*)$`
	return v1.And(v2).IsZero()
}
