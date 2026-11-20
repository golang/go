// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable && wasm

package simd

// archClmulImpl returns how archsimd implements carryless multiply on vectors
// of the given bit size (see clmulImpl), and whether that is a single
// instruction on the whole vector. WebAssembly has no carryless multiply
// instruction, so archsimd implements it in software.
func archClmulImpl(size int) (impl string, full bool) {
	return "software", false
}

// archMaxVectorSize returns the bit size of the largest vectors the CPU
// supports.
func archMaxVectorSize() (size int) {
	return 128
}
