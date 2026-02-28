// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file tests the splitting of constants into
// multiple immediates on arm.

package main

import "fmt"

const c32a = 0x00aa00dd
const c32s = 0x00ffff00
const c64a = 0x00aa00dd55000066
const c64s = 0x00ffff00004fff00

//go:noinline
func add32a(x uint32) uint32 {
	return x + c32a
}

//go:noinline
func add32s(x uint32) uint32 {
	return x + c32s
}

//go:noinline
func sub32a(x uint32) uint32 {
	return x - c32a
}

//go:noinline
func sub32s(x uint32) uint32 {
	return x - c32s
}

//go:noinline
func or32(x uint32) uint32 {
	return x | c32a
}

//go:noinline
func xor32(x uint32) uint32 {
	return x ^ c32a
}

//go:noinline
func subr32a(x uint32) uint32 {
	return c32a - x
}

//go:noinline
func subr32s(x uint32) uint32 {
	return c32s - x
}

//go:noinline
func bic32(x uint32) uint32 {
	return x &^ c32a
}

//go:noinline
func add64a(x uint64) uint64 {
	return x + c64a
}

//go:noinline
func add64s(x uint64) uint64 {
	return x + c64s
}

//go:noinline
func sub64a(x uint64) uint64 {
	return x - c64a
}

//go:noinline
func sub64s(x uint64) uint64 {
	return x - c64s
}

//go:noinline
func or64(x uint64) uint64 {
	return x | c64a
}

//go:noinline
func xor64(x uint64) uint64 {
	return x ^ c64a
}

//go:noinline
func subr64a(x uint64) uint64 {
	return c64a - x
}

//go:noinline
func subr64s(x uint64) uint64 {
	return c64s - x
}

//go:noinline
func bic64(x uint64) uint64 {
	return x &^ c64a
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
	if want, got = a+c32a, add32a(a); got != want {
		panic(fmt.Sprintf("add32a(%x) = %x, want %x", a, got, want))
	}
	if want, got = a+c32s, add32s(a); got != want {
		panic(fmt.Sprintf("add32s(%x) = %x, want %x", a, got, want))
	}
	if want, got = a-c32a, sub32a(a); got != want {
		panic(fmt.Sprintf("sub32a(%x) = %x, want %x", a, got, want))
	}
	if want, got = a-c32s, sub32s(a); got != want {
		panic(fmt.Sprintf("sub32s(%x) = %x, want %x", a, got, want))
	}
	if want, got = a|c32a, or32(a); got != want {
		panic(fmt.Sprintf("or32(%x) = %x, want %x", a, got, want))
	}
	if want, got = a^c32a, xor32(a); got != want {
		panic(fmt.Sprintf("xor32(%x) = %x, want %x", a, got, want))
	}
	if want, got = c32a-a, subr32a(a); got != want {
		panic(fmt.Sprintf("subr32a(%x) = %x, want %x", a, got, want))
	}
	if want, got = c32s-a, subr32s(a); got != want {
		panic(fmt.Sprintf("subr32s(%x) = %x, want %x", a, got, want))
	}
	if want, got = a&^c32a, bic32(a); got != want {
		panic(fmt.Sprintf("bic32(%x) = %x, want %x", a, got, want))
	}
}

func test64() {
	var a uint64 = 0x1111111111111111
	var want, got uint64
	if want, got = a+c64a, add64a(a); got != want {
		panic(fmt.Sprintf("add64a(%x) = %x, want %x", a, got, want))
	}
	if want, got = a+c64s, add64s(a); got != want {
		panic(fmt.Sprintf("add64s(%x) = %x, want %x", a, got, want))
	}
	if want, got = a-c64a, sub64a(a); got != want {
		panic(fmt.Sprintf("sub64a(%x) = %x, want %x", a, got, want))
	}
	if want, got = a-c64s, sub64s(a); got != want {
		panic(fmt.Sprintf("sub64s(%x) = %x, want %x", a, got, want))
	}
	if want, got = a|c64a, or64(a); got != want {
		panic(fmt.Sprintf("or64(%x) = %x, want %x", a, got, want))
	}
	if want, got = a^c64a, xor64(a); got != want {
		panic(fmt.Sprintf("xor64(%x) = %x, want %x", a, got, want))
	}
	if want, got = c64a-a, subr64a(a); got != want {
		panic(fmt.Sprintf("subr64a(%x) = %x, want %x", a, got, want))
	}
	if want, got = c64s-a, subr64s(a); got != want {
		panic(fmt.Sprintf("subr64s(%x) = %x, want %x", a, got, want))
	}
	if want, got = a&^c64a, bic64(a); got != want {
		panic(fmt.Sprintf("bic64(%x) = %x, want %x", a, got, want))
	}
}
