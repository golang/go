// +build !386
// run

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 12411. Loss of AX during %.

package main

func main() {
	x := f(4)
	if x != 0 {
		println("BUG: x=", x)
	}
}

//go:noinline
func f(x int) int {
	// AX was live on entry to one of the % code generations,
	// and the % code generation smashed it.
	return ((2 * x) % 3) % (2 % ((x << 2) ^ (x % 3)))
}
