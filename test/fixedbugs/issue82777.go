// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/bits"
)

//go:noinline
func f(x, y int64) int64 {
	r := x >> 33
	if bits.OnesCount8(128) > 0 {
		r = ^y
	}
	return r
}

func main() {
	if got := f(0, 2); got != -3 {
		fmt.Printf("selected ^2 = %d, want -3\n", got)
		panic("FAIL")
	}
}
