// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package main

import "simd/archsimd"

func sveVectorBits() (int, bool) {
	if !archsimd.ARM64.SVE() {
		return 0, false
	}
	// Int8s.Len() returns the runtime vector length in bytes (vl()).
	return archsimd.Int8s{}.Len() * 8, true
}
