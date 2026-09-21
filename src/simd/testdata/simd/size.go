// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd

// This is "simd/testdata/simd", not really "simd", hence not subject
// to the sizeof restrictions on simd types despite matching in every
// other detail.

// _simd mimics the same type definition in the top-level "simd" package
// to test the non-constant-sizeof test.
type _simd struct {
	_ [0]func(*_simd) *_simd
}

type HasConstantSize24 struct {
	_       _simd
	a, b, c uint64
}
