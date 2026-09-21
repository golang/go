// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Folding a paired modulo-count shift into a double-register shift
// (SHRD/SHLD on amd64) must not drop the second operand when the
// shift count is zero modulo 64.

package main

//go:noinline
func shrd(lo, hi uint64, bits uint) uint64 {
	return lo>>(bits&63) | hi<<((-bits)&63)
}

func main() {
	got := shrd(0x1111000000000000, 0x2222, 0)
	want := uint64(0x1111000000002222)
	if got != want {
		println("shrd zero count: got", got, "want", want)
		panic("FAIL")
	}
}
