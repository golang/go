// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file tests the splitting of constants into
// multiple immediates on arm.

package main

import "fmt"

const c32 = 0xaa00dd
const c64 = 0xaa00dd55000066

//go:noinline
func add32(x uint32) uint32 {
	return x + c32
}

//go:noinline
func sub32(x uint32) uint32 {
	return x - c32
}

//go:noinline
func or32(x uint32) uint32 {
	return x | c32
}

//go:noinline
func xor32(x uint32) uint32 {
	return x ^ c32
}

//go:noinline
func subr32(x uint32) uint32 {
	return c32 - x
}

//go:noinline
func add64(x uint64) uint64 {
	return x + c64
}

//go:noinline
func sub64(x uint64) uint64 {
	return x - c64
}

//go:noinline
func or64(x uint64) uint64 {
	return x | c64
}

//go:noinline
func xor64(x uint64) uint64 {
	return x ^ c64
}

//go:noinline
func subr64(x uint64) uint64 {
	return c64 - x
}

// Note: x-c gets rewritten to x+(-c), so SUB and SBC are not directly testable.
// I disabled that rewrite rule before running this test.

func main() {
	test32()
	test64()
}

func test32() {
	var a uint32 = 0x11111111
	var want, got uint32
	if want, got = a+c32, add32(a); got != want {
		panic(fmt.Sprintf("add32(%x) = %x, want %x", a, got, want))
	}
	if want, got = a-c32, sub32(a); got != want {
		panic(fmt.Sprintf("sub32(%x) = %x, want %x", a, got, want))
	}
	if want, got = a|c32, or32(a); got != want {
		panic(fmt.Sprintf("or32(%x) = %x, want %x", a, got, want))
	}
	if want, got = a^c32, xor32(a); got != want {
		panic(fmt.Sprintf("xor32(%x) = %x, want %x", a, got, want))
	}
	if want, got = c32-a, subr32(a); got != want {
		panic(fmt.Sprintf("subr32(%x) = %x, want %x", a, got, want))
	}
}

func test64() {
	var a uint64 = 0x1111111111111111
	var want, got uint64
	if want, got = a+c64, add64(a); got != want {
		panic(fmt.Sprintf("add64(%x) = %x, want %x", a, got, want))
	}
	if want, got = a-c64, sub64(a); got != want {
		panic(fmt.Sprintf("sub64(%x) = %x, want %x", a, got, want))
	}
	if want, got = a|c64, or64(a); got != want {
		panic(fmt.Sprintf("or64(%x) = %x, want %x", a, got, want))
	}
	if want, got = a^c64, xor64(a); got != want {
		panic(fmt.Sprintf("xor64(%x) = %x, want %x", a, got, want))
	}
	if want, got = c64-a, subr64(a); got != want {
		panic(fmt.Sprintf("subr64(%x) = %x, want %x", a, got, want))
	}
}
