// run -gcflags=all=-d=checkptr

//go:build goexperiment.simd && amd64

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case for issue #78413.

package main

import (
	"simd/archsimd"
)

//go:noinline
func F() []int32 {
	return []int32{0}
}

func main() {
	archsimd.LoadInt32x8SlicePart(F())
}
