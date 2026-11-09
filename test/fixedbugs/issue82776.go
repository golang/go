// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

//go:noinline
func f(x, y int8) int8 { return (x >> 7) >> (uint64(y) | 0xFFFFFFFF) }

func main() {
	if got := f(127, -128); got != 0 {
		fmt.Printf("(int8(127)>>7)>>(uint64(-128)|0xFFFFFFFF) = %d, want 0\n", got)
		panic("FAIL")
	}
}
