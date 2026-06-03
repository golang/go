// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && wasm

package simd

const archHasHwClmul = false

func archMaxVectorSize() (size, allFeatureSize int) {
	return 128, 128
}
