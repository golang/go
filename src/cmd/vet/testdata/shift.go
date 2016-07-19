// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the suspicious shift checker.

package testdata

func ShiftTest() {
	var i8 int8
	_ = i8 << 7
	_ = (i8 + 1) << 8 // ERROR "\(i8 \+ 1\) too small for shift of 8"
	_ = i8 << (7 + 1) // ERROR "i8 too small for shift of 8"
	_ = i8 >> 8       // ERROR "i8 too small for shift of 8"
	i8 <<= 8          // ERROR "i8 too small for shift of 8"
	i8 >>= 8          // ERROR "i8 too small for shift of 8"
	var i16 int16
	_ = i16 << 15
	_ = i16 << 16 // ERROR "i16 too small for shift of 16"
	_ = i16 >> 16 // ERROR "i16 too small for shift of 16"
	i16 <<= 16    // ERROR "i16 too small for shift of 16"
	i16 >>= 16    // ERROR "i16 too small for shift of 16"
	var i32 int32
	_ = i32 << 31
	_ = i32 << 32 // ERROR "i32 too small for shift of 32"
	_ = i32 >> 32 // ERROR "i32 too small for shift of 32"
	i32 <<= 32    // ERROR "i32 too small for shift of 32"
	i32 >>= 32    // ERROR "i32 too small for shift of 32"
	var i64 int64
	_ = i64 << 63
	_ = i64 << 64 // ERROR "i64 too small for shift of 64"
	_ = i64 >> 64 // ERROR "i64 too small for shift of 64"
	i64 <<= 64    // ERROR "i64 too small for shift of 64"
	i64 >>= 64    // ERROR "i64 too small for shift of 64"
	var u8 uint8
	_ = u8 << 7
	_ = u8 << 8 // ERROR "u8 too small for shift of 8"
	_ = u8 >> 8 // ERROR "u8 too small for shift of 8"
	u8 <<= 8    // ERROR "u8 too small for shift of 8"
	u8 >>= 8    // ERROR "u8 too small for shift of 8"
	var u16 uint16
	_ = u16 << 15
	_ = u16 << 16 // ERROR "u16 too small for shift of 16"
	_ = u16 >> 16 // ERROR "u16 too small for shift of 16"
	u16 <<= 16    // ERROR "u16 too small for shift of 16"
	u16 >>= 16    // ERROR "u16 too small for shift of 16"
	var u32 uint32
	_ = u32 << 31
	_ = u32 << 32 // ERROR "u32 too small for shift of 32"
	_ = u32 >> 32 // ERROR "u32 too small for shift of 32"
	u32 <<= 32    // ERROR "u32 too small for shift of 32"
	u32 >>= 32    // ERROR "u32 too small for shift of 32"
	var u64 uint64
	_ = u64 << 63
	_ = u64 << 64  // ERROR "u64 too small for shift of 64"
	_ = u64 >> 64  // ERROR "u64 too small for shift of 64"
	u64 <<= 64     // ERROR "u64 too small for shift of 64"
	u64 >>= 64     // ERROR "u64 too small for shift of 64"
	_ = u64 << u64 // Non-constant shifts should succeed.
	var i int
	_ = i << 31
	_ = i << 32 // ERROR "i might be too small for shift of 32"
	_ = i >> 32 // ERROR "i might be too small for shift of 32"
	i <<= 32    // ERROR "i might be too small for shift of 32"
	i >>= 32    // ERROR "i might be too small for shift of 32"
	var u uint
	_ = u << 31
	_ = u << 32 // ERROR "u might be too small for shift of 32"
	_ = u >> 32 // ERROR "u might be too small for shift of 32"
	u <<= 32    // ERROR "u might be too small for shift of 32"
	u >>= 32    // ERROR "u might be too small for shift of 32"
	var p uintptr
	_ = p << 31
	_ = p << 32 // ERROR "p might be too small for shift of 32"
	_ = p >> 32 // ERROR "p might be too small for shift of 32"
	p <<= 32    // ERROR "p might be too small for shift of 32"
	p >>= 32    // ERROR "p might be too small for shift of 32"

	const oneIf64Bit = ^uint(0) >> 63 // allow large shifts of constants; they are used for 32/64 bit compatibility tricks
}
