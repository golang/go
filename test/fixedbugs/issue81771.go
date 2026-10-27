// build

//go:build goexperiment.simd && amd64

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "simd/archsimd"

type S struct {
	f archsimd.Float32x8
}

type I interface {
	M(S) rune
}

func f(i I) rune {
	for range []int{} {
		_ = i.M(S{(archsimd.Float32x8{})})
	}
	return i.M(S{(archsimd.Float32x8{})})
}
