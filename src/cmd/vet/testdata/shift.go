// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the suspicious shift checker.

package testdata

import "unsafe"

func ShiftTest() {
	var i8 int8
	_ = i8 << 7
	_ = (i8 + 1) << 8 // ERROR ".i8 . 1. .8 bits. too small for shift of 8"
	_ = i8 << (7 + 1) // ERROR "i8 .8 bits. too small for shift of 8"
	_ = i8 >> 8       // ERROR "i8 .8 bits. too small for shift of 8"
	i8 <<= 8          // ERROR "i8 .8 bits. too small for shift of 8"
	i8 >>= 8          // ERROR "i8 .8 bits. too small for shift of 8"
	var i16 int16
	_ = i16 << 15
	_ = i16 << 16 // ERROR "i16 .16 bits. too small for shift of 16"
	_ = i16 >> 16 // ERROR "i16 .16 bits. too small for shift of 16"
	i16 <<= 16    // ERROR "i16 .16 bits. too small for shift of 16"
	i16 >>= 16    // ERROR "i16 .16 bits. too small for shift of 16"
	var i32 int32
	_ = i32 << 31
	_ = i32 << 32 // ERROR "i32 .32 bits. too small for shift of 32"
	_ = i32 >> 32 // ERROR "i32 .32 bits. too small for shift of 32"
	i32 <<= 32    // ERROR "i32 .32 bits. too small for shift of 32"
	i32 >>= 32    // ERROR "i32 .32 bits. too small for shift of 32"
	var i64 int64
	_ = i64 << 63
	_ = i64 << 64 // ERROR "i64 .64 bits. too small for shift of 64"
	_ = i64 >> 64 // ERROR "i64 .64 bits. too small for shift of 64"
	i64 <<= 64    // ERROR "i64 .64 bits. too small for shift of 64"
	i64 >>= 64    // ERROR "i64 .64 bits. too small for shift of 64"
	var u8 uint8
	_ = u8 << 7
	_ = u8 << 8 // ERROR "u8 .8 bits. too small for shift of 8"
	_ = u8 >> 8 // ERROR "u8 .8 bits. too small for shift of 8"
	u8 <<= 8    // ERROR "u8 .8 bits. too small for shift of 8"
	u8 >>= 8    // ERROR "u8 .8 bits. too small for shift of 8"
	var u16 uint16
	_ = u16 << 15
	_ = u16 << 16 // ERROR "u16 .16 bits. too small for shift of 16"
	_ = u16 >> 16 // ERROR "u16 .16 bits. too small for shift of 16"
	u16 <<= 16    // ERROR "u16 .16 bits. too small for shift of 16"
	u16 >>= 16    // ERROR "u16 .16 bits. too small for shift of 16"
	var u32 uint32
	_ = u32 << 31
	_ = u32 << 32 // ERROR "u32 .32 bits. too small for shift of 32"
	_ = u32 >> 32 // ERROR "u32 .32 bits. too small for shift of 32"
	u32 <<= 32    // ERROR "u32 .32 bits. too small for shift of 32"
	u32 >>= 32    // ERROR "u32 .32 bits. too small for shift of 32"
	var u64 uint64
	_ = u64 << 63
	_ = u64 << 64  // ERROR "u64 .64 bits. too small for shift of 64"
	_ = u64 >> 64  // ERROR "u64 .64 bits. too small for shift of 64"
	u64 <<= 64     // ERROR "u64 .64 bits. too small for shift of 64"
	u64 >>= 64     // ERROR "u64 .64 bits. too small for shift of 64"
	_ = u64 << u64 // Non-constant shifts should succeed.

	var i int
	_ = i << 31
	const in = 8 * unsafe.Sizeof(i)
	_ = i << in // ERROR "too small for shift"
	_ = i >> in // ERROR "too small for shift"
	i <<= in    // ERROR "too small for shift"
	i >>= in    // ERROR "too small for shift"
	const ix = 8*unsafe.Sizeof(i) - 1
	_ = i << ix
	_ = i >> ix
	i <<= ix
	i >>= ix

	var u uint
	_ = u << 31
	const un = 8 * unsafe.Sizeof(u)
	_ = u << un // ERROR "too small for shift"
	_ = u >> un // ERROR "too small for shift"
	u <<= un    // ERROR "too small for shift"
	u >>= un    // ERROR "too small for shift"
	const ux = 8*unsafe.Sizeof(u) - 1
	_ = u << ux
	_ = u >> ux
	u <<= ux
	u >>= ux

	var p uintptr
	_ = p << 31
	const pn = 8 * unsafe.Sizeof(p)
	_ = p << pn // ERROR "too small for shift"
	_ = p >> pn // ERROR "too small for shift"
	p <<= pn    // ERROR "too small for shift"
	p >>= pn    // ERROR "too small for shift"
	const px = 8*unsafe.Sizeof(p) - 1
	_ = p << px
	_ = p >> px
	p <<= px
	p >>= px

	const oneIf64Bit = ^uint(0) >> 63 // allow large shifts of constants; they are used for 32/64 bit compatibility tricks

	var h uintptr
	h = h<<8 | (h >> (8 * (unsafe.Sizeof(h) - 1)))
	h <<= 8 * unsafe.Sizeof(h) // ERROR "too small for shift"
	h >>= 7 * unsafe.Alignof(h)
	h >>= 8 * unsafe.Alignof(h) // ERROR "too small for shift"
}

func ShiftDeadCode() {
	var i int
	const iBits = 8 * unsafe.Sizeof(i)

	if iBits <= 32 {
		if iBits == 16 {
			_ = i >> 8
		} else {
			_ = i >> 16
		}
	} else {
		_ = i >> 32
	}

	if iBits >= 64 {
		_ = i << 32
		if iBits == 128 {
			_ = i << 64
		}
	} else {
		_ = i << 16
	}

	if iBits == 64 {
		_ = i << 32
	}

	switch iBits {
	case 128, 64:
		_ = i << 32
	default:
		_ = i << 16
	}

	switch {
	case iBits < 32:
		_ = i << 16
	case iBits > 64:
		_ = i << 64
	default:
		_ = i << 64 // ERROR "too small for shift"
	}

	// Make sure other vet checks work in dead code.
	if iBits == 1024 {
		_ = i << 512                  // OK
		fmt.Printf("foo %s bar", 123) // ERROR "arg 123 for printf verb %s of wrong type: untyped int"
	}
}
