// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that 32-bit results are zero-extended
// when an operand becomes constant late in compilation.

package main

import "fmt"

// Indexing a string with idx0() or idx1() yields a constant byte,
// but only once the load is lowered, after early constant folding.
func idx0() int { return 0 }
func idx1() int { return 1 }

//go:noinline
func addConst(x uint64) uint64 {
	k := uint32("\x00"[idx0()])
	return uint64(uint32(x) + k)
}

//go:noinline
func negMulConst(x uint32) uint64 {
	k := uint32("\x00\x02"[idx1()])
	return uint64(-(x * k))
}

func main() {
	if got, want := addConst(0xffffffff00000001), uint64(1); got != want {
		panic(fmt.Sprintf("addConst(0xffffffff00000001) = %#x, want %#x", got, want))
	}
	if got, want := negMulConst(1), uint64(0xfffffffe); got != want {
		panic(fmt.Sprintf("negMulConst(1) = %#x, want %#x", got, want))
	}
}
