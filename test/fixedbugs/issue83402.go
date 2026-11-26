// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "math/bits"

//go:noinline
func andNot(p *uint32) uint64 {
	return uint64(*p &^ 128)
}

//go:noinline
func and(p *uint32) uint64 {
	return uint64(*p & 0xffff00ff)
}

// zero returns 0 in a way only sccp can see once it is inlined.
func zero(n int) uint {
	s := uint(0)
	for i := 0; i < n; i++ {
		s *= 3
	}
	return s
}

//go:noinline
func shiftAnd(x uint32, n int) uint64 {
	s := zero(n)
	return uint64(x>>s) & 0xffffff7f
}

//go:noinline
func rotateShift(x uint32, n int) uint64 {
	s := zero(n)
	return uint64(bits.RotateLeft32(x, 4)&0xffffff7f) << s
}

func main() {
	v := uint32(0x80000000)
	if got := andNot(&v); got != 0x80000000 {
		panic(got)
	}
	if got := and(&v); got != 0x80000000 {
		panic(got)
	}
	if got := shiftAnd(v, 3); got != 0x80000000 {
		panic(got)
	}
	if got := rotateShift(0x08000000, 3); got != 0x80000000 {
		panic(got)
	}
}
