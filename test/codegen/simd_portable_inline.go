// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that portable simd operations inline completely in a package that
// imports only simd. The bridge between simd and simd/archsimd inlines bodies
// from simd/archsimd (and, in its emulated fallbacks, from math and
// math/bits), and simd's export data must carry those bodies, too.
//
// Do not import simd/archsimd, math, or math/bits here: that would mask
// missing bodies.

//go:build goexperiment.simdstable && (amd64 || arm64 || wasm)

package codegen

import "simd"

func portableLoadStore(s []uint32) {
	// amd64:-`CALL simd/archsimd` arm64:-`CALL simd/archsimd` wasm:-`CALL simd/archsimd`
	x := simd.LoadUint32s(s)
	// amd64:-`CALL simd/archsimd` arm64:-`CALL simd/archsimd` wasm:-`CALL simd/archsimd`
	x.Add(x).Store(s)
}

func portableLoadStorePart(s []uint32) int {
	// amd64:-`CALL simd/archsimd` arm64:-`CALL simd/archsimd` wasm:-`CALL simd/archsimd`
	x, n := simd.LoadUint32sPart(s)
	// amd64:-`CALL simd/archsimd` arm64:-`CALL simd/archsimd` wasm:-`CALL simd/archsimd`
	return x.StorePart(s) + n
}

func portableLenReduceSum(s []int16) int16 {
	// amd64:-`CALL simd/archsimd` arm64:-`CALL simd/archsimd` wasm:-`CALL simd/archsimd`
	return simd.LoadInt16s(s[:simd.Int16s{}.Len()]).ReduceSum()
}

// Like any function that uses portable simd types, these are multicompiled
// for each vector width. That includes an emulated (no SIMD) variant, which
// uses the bridge's pure-Go fallbacks. Those call math.Float32bits and
// bits.OnesCount8.

func portableToBits(s []float32, d []uint32) {
	// amd64:-`CALL math` arm64:-`CALL math` wasm:-`CALL math`
	simd.LoadFloat32s(s).ToBits().Store(d)
}

func portableOnesCount(s []int8) {
	// amd64:-`CALL math` arm64:-`CALL math` wasm:-`CALL math`
	simd.LoadInt8s(s).OnesCount().Store(s)
}
