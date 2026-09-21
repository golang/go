// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && !amd64

package simd_test

import (
	"simd"
)

func sum(x simd.Float32s) float32 {
	return boringSum(x)
}
