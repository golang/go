// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func f32(_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, x int32) uint64 {
	return uint64(uint32(x))
}

//go:noinline
func f16(_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, x int16) uint64 {
	return uint64(uint16(x))
}

//go:noinline
func f8(_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, x int8) uint64 {
	return uint64(uint8(x))
}

//go:noinline
func g32(_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, x uint32) int64 {
	return int64(int32(x))
}

//go:noinline
func g16(_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, x uint16) int64 {
	return int64(int16(x))
}

//go:noinline
func g8(_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, x uint8) int64 {
	return int64(int8(x))
}

func main() {
	if got := f32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1); got != 0xffffffff {
		println("bad f32", got)
	}
	if got := f16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1); got != 0xffff {
		println("bad f16", got)
	}
	if got := f8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1); got != 0xff {
		println("bad f8", got)
	}
	if got := g32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xffffffff); got != -1 {
		println("bad g32", got)
	}
	if got := g16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xffff); got != -1 {
		println("bad g16", got)
	}
	if got := g8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff); got != -1 {
		println("bad g8", got)
	}
}
