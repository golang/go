// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && wasm

package simd

// archHasHwClmul reports whether carryless multiply on vectors of the given
// bit size is a single hardware instruction. WebAssembly has no such
// instruction.
func archHasHwClmul(size int) bool {
	return false
}

func archMaxVectorSize() (size, allFeatureSize int, arch string) {
	return 128, 128, "wasm"
}
