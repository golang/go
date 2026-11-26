// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func zero(n int) uint {
	s := uint(0)
	for i := 0; i < n; i++ {
		s *= 3
	}
	return s
}

//go:noinline
func shl(x uint64, n int) uint64 { return uint64(uint32(x) << zero(n)) }

//go:noinline
func shr(x uint64, n int) uint64 { return uint64(uint32(x) >> zero(n)) }

//go:noinline
func shr64(x uint64, n int) uint64 { return uint64(uint32(x)) >> zero(n) }

func main() {
	const x, want = 0x1234567880000001, 0x80000001
	if got := shl(x, 3); got != want {
		panic(got)
	}
	if got := shr(x, 3); got != want {
		panic(got)
	}
	if got := shr64(x, 3); got != want {
		panic(got)
	}
}
