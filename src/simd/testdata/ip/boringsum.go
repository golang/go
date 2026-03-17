// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"simd"
)

func boringSum(x simd.Float32s) float32 {
	s := make([]float32, x.Len())
	x.Store(s)
	var r float32
	for _, e := range s {
		r += e
	}
	return r
}
