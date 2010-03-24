// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "rand"

const Count = 1e5

func i64rand() int64 {
	for {
		a := int64(rand.Uint32())
		a = (a << 32) | int64(rand.Uint32())
		a >>= uint(rand.Intn(64))
		if -a != a {
			return a
		}
	}
	return 0 // impossible
}

func i64test(a, b, c int64) {
	d := a / c
	if d != b {
		println("i64", a, b, c, d)
		panic("fail")
	}
}

func i64run() {
	var a, b int64

	for i := 0; i < Count; i++ {
		a = i64rand()

		b = a / 1
		i64test(a, b, 1)
		b = a / 2
		i64test(a, b, 2)
		b = a / 3
		i64test(a, b, 3)
		b = a / 4
		i64test(a, b, 4)
		b = a / 5
		i64test(a, b, 5)
		b = a / 6
		i64test(a, b, 6)
		b = a / 7
		i64test(a, b, 7)
		b = a / 8
		i64test(a, b, 8)
		b = a / 10
		i64test(a, b, 10)
		b = a / 16
		i64test(a, b, 16)
		b = a / 20
		i64test(a, b, 20)
		b = a / 32
		i64test(a, b, 32)
		b = a / 60
		i64test(a, b, 60)
		b = a / 64
		i64test(a, b, 64)
		b = a / 128
		i64test(a, b, 128)
		b = a / 256
		i64test(a, b, 256)
		b = a / 16384
		i64test(a, b, 16384)

		b = a / -1
		i64test(a, b, -1)
		b = a / -2
		i64test(a, b, -2)
		b = a / -3
		i64test(a, b, -3)
		b = a / -4
		i64test(a, b, -4)
		b = a / -5
		i64test(a, b, -5)
		b = a / -6
		i64test(a, b, -6)
		b = a / -7
		i64test(a, b, -7)
		b = a / -8
		i64test(a, b, -8)
		b = a / -10
		i64test(a, b, -10)
		b = a / -16
		i64test(a, b, -16)
		b = a / -20
		i64test(a, b, -20)
		b = a / -32
		i64test(a, b, -32)
		b = a / -60
		i64test(a, b, -60)
		b = a / -64
		i64test(a, b, -64)
		b = a / -128
		i64test(a, b, -128)
		b = a / -256
		i64test(a, b, -256)
		b = a / -16384
		i64test(a, b, -16384)
	}
}

func u64rand() uint64 {
	a := uint64(rand.Uint32())
	a = (a << 32) | uint64(rand.Uint32())
	a >>= uint(rand.Intn(64))
	return a
}

func u64test(a, b, c uint64) {
	d := a / c
	if d != b {
		println("u64", a, b, c, d)
		panic("fail")
	}
}

func u64run() {
	var a, b uint64

	for i := 0; i < Count; i++ {
		a = u64rand()

		b = a / 1
		u64test(a, b, 1)
		b = a / 2
		u64test(a, b, 2)
		b = a / 3
		u64test(a, b, 3)
		b = a / 4
		u64test(a, b, 4)
		b = a / 5
		u64test(a, b, 5)
		b = a / 6
		u64test(a, b, 6)
		b = a / 7
		u64test(a, b, 7)
		b = a / 8
		u64test(a, b, 8)
		b = a / 10
		u64test(a, b, 10)
		b = a / 16
		u64test(a, b, 16)
		b = a / 20
		u64test(a, b, 20)
		b = a / 32
		u64test(a, b, 32)
		b = a / 60
		u64test(a, b, 60)
		b = a / 64
		u64test(a, b, 64)
		b = a / 128
		u64test(a, b, 128)
		b = a / 256
		u64test(a, b, 256)
		b = a / 16384
		u64test(a, b, 16384)
	}
}

func i32rand() int32 {
	for {
		a := int32(rand.Uint32())
		a >>= uint(rand.Intn(32))
		if -a != a {
			return a
		}
	}
	return 0 // impossible
}

func i32test(a, b, c int32) {
	d := a / c
	if d != b {
		println("i32", a, b, c, d)
		panic("fail")
	}
}

func i32run() {
	var a, b int32

	for i := 0; i < Count; i++ {
		a = i32rand()

		b = a / 1
		i32test(a, b, 1)
		b = a / 2
		i32test(a, b, 2)
		b = a / 3
		i32test(a, b, 3)
		b = a / 4
		i32test(a, b, 4)
		b = a / 5
		i32test(a, b, 5)
		b = a / 6
		i32test(a, b, 6)
		b = a / 7
		i32test(a, b, 7)
		b = a / 8
		i32test(a, b, 8)
		b = a / 10
		i32test(a, b, 10)
		b = a / 16
		i32test(a, b, 16)
		b = a / 20
		i32test(a, b, 20)
		b = a / 32
		i32test(a, b, 32)
		b = a / 60
		i32test(a, b, 60)
		b = a / 64
		i32test(a, b, 64)
		b = a / 128
		i32test(a, b, 128)
		b = a / 256
		i32test(a, b, 256)
		b = a / 16384
		i32test(a, b, 16384)

		b = a / -1
		i32test(a, b, -1)
		b = a / -2
		i32test(a, b, -2)
		b = a / -3
		i32test(a, b, -3)
		b = a / -4
		i32test(a, b, -4)
		b = a / -5
		i32test(a, b, -5)
		b = a / -6
		i32test(a, b, -6)
		b = a / -7
		i32test(a, b, -7)
		b = a / -8
		i32test(a, b, -8)
		b = a / -10
		i32test(a, b, -10)
		b = a / -16
		i32test(a, b, -16)
		b = a / -20
		i32test(a, b, -20)
		b = a / -32
		i32test(a, b, -32)
		b = a / -60
		i32test(a, b, -60)
		b = a / -64
		i32test(a, b, -64)
		b = a / -128
		i32test(a, b, -128)
		b = a / -256
		i32test(a, b, -256)
	}
}

func u32rand() uint32 {
	a := uint32(rand.Uint32())
	a >>= uint(rand.Intn(32))
	return a
}

func u32test(a, b, c uint32) {
	d := a / c
	if d != b {
		println("u32", a, b, c, d)
		panic("fail")
	}
}

func u32run() {
	var a, b uint32

	for i := 0; i < Count; i++ {
		a = u32rand()

		b = a / 1
		u32test(a, b, 1)
		b = a / 2
		u32test(a, b, 2)
		b = a / 3
		u32test(a, b, 3)
		b = a / 4
		u32test(a, b, 4)
		b = a / 5
		u32test(a, b, 5)
		b = a / 6
		u32test(a, b, 6)
		b = a / 7
		u32test(a, b, 7)
		b = a / 8
		u32test(a, b, 8)
		b = a / 10
		u32test(a, b, 10)
		b = a / 16
		u32test(a, b, 16)
		b = a / 20
		u32test(a, b, 20)
		b = a / 32
		u32test(a, b, 32)
		b = a / 60
		u32test(a, b, 60)
		b = a / 64
		u32test(a, b, 64)
		b = a / 128
		u32test(a, b, 128)
		b = a / 256
		u32test(a, b, 256)
		b = a / 16384
		u32test(a, b, 16384)
	}
}

func i16rand() int16 {
	for {
		a := int16(rand.Uint32())
		a >>= uint(rand.Intn(16))
		if -a != a {
			return a
		}
	}
	return 0 // impossible
}

func i16test(a, b, c int16) {
	d := a / c
	if d != b {
		println("i16", a, b, c, d)
		panic("fail")
	}
}

func i16run() {
	var a, b int16

	for i := 0; i < Count; i++ {
		a = i16rand()

		b = a / 1
		i16test(a, b, 1)
		b = a / 2
		i16test(a, b, 2)
		b = a / 3
		i16test(a, b, 3)
		b = a / 4
		i16test(a, b, 4)
		b = a / 5
		i16test(a, b, 5)
		b = a / 6
		i16test(a, b, 6)
		b = a / 7
		i16test(a, b, 7)
		b = a / 8
		i16test(a, b, 8)
		b = a / 10
		i16test(a, b, 10)
		b = a / 16
		i16test(a, b, 16)
		b = a / 20
		i16test(a, b, 20)
		b = a / 32
		i16test(a, b, 32)
		b = a / 60
		i16test(a, b, 60)
		b = a / 64
		i16test(a, b, 64)
		b = a / 128
		i16test(a, b, 128)
		b = a / 256
		i16test(a, b, 256)
		b = a / 16384
		i16test(a, b, 16384)

		b = a / -1
		i16test(a, b, -1)
		b = a / -2
		i16test(a, b, -2)
		b = a / -3
		i16test(a, b, -3)
		b = a / -4
		i16test(a, b, -4)
		b = a / -5
		i16test(a, b, -5)
		b = a / -6
		i16test(a, b, -6)
		b = a / -7
		i16test(a, b, -7)
		b = a / -8
		i16test(a, b, -8)
		b = a / -10
		i16test(a, b, -10)
		b = a / -16
		i16test(a, b, -16)
		b = a / -20
		i16test(a, b, -20)
		b = a / -32
		i16test(a, b, -32)
		b = a / -60
		i16test(a, b, -60)
		b = a / -64
		i16test(a, b, -64)
		b = a / -128
		i16test(a, b, -128)
		b = a / -256
		i16test(a, b, -256)
		b = a / -16384
		i16test(a, b, -16384)
	}
}

func u16rand() uint16 {
	a := uint16(rand.Uint32())
	a >>= uint(rand.Intn(16))
	return a
}

func u16test(a, b, c uint16) {
	d := a / c
	if d != b {
		println("u16", a, b, c, d)
		panic("fail")
	}
}

func u16run() {
	var a, b uint16

	for i := 0; i < Count; i++ {
		a = u16rand()

		b = a / 1
		u16test(a, b, 1)
		b = a / 2
		u16test(a, b, 2)
		b = a / 3
		u16test(a, b, 3)
		b = a / 4
		u16test(a, b, 4)
		b = a / 5
		u16test(a, b, 5)
		b = a / 6
		u16test(a, b, 6)
		b = a / 7
		u16test(a, b, 7)
		b = a / 8
		u16test(a, b, 8)
		b = a / 10
		u16test(a, b, 10)
		b = a / 16
		u16test(a, b, 16)
		b = a / 20
		u16test(a, b, 20)
		b = a / 32
		u16test(a, b, 32)
		b = a / 60
		u16test(a, b, 60)
		b = a / 64
		u16test(a, b, 64)
		b = a / 128
		u16test(a, b, 128)
		b = a / 256
		u16test(a, b, 256)
		b = a / 16384
		u16test(a, b, 16384)
	}
}

func i8rand() int8 {
	for {
		a := int8(rand.Uint32())
		a >>= uint(rand.Intn(8))
		if -a != a {
			return a
		}
	}
	return 0 // impossible
}

func i8test(a, b, c int8) {
	d := a / c
	if d != b {
		println("i8", a, b, c, d)
		panic("fail")
	}
}

func i8run() {
	var a, b int8

	for i := 0; i < Count; i++ {
		a = i8rand()

		b = a / 1
		i8test(a, b, 1)
		b = a / 2
		i8test(a, b, 2)
		b = a / 3
		i8test(a, b, 3)
		b = a / 4
		i8test(a, b, 4)
		b = a / 5
		i8test(a, b, 5)
		b = a / 6
		i8test(a, b, 6)
		b = a / 7
		i8test(a, b, 7)
		b = a / 8
		i8test(a, b, 8)
		b = a / 10
		i8test(a, b, 10)
		b = a / 8
		i8test(a, b, 8)
		b = a / 20
		i8test(a, b, 20)
		b = a / 32
		i8test(a, b, 32)
		b = a / 60
		i8test(a, b, 60)
		b = a / 64
		i8test(a, b, 64)
		b = a / 127
		i8test(a, b, 127)

		b = a / -1
		i8test(a, b, -1)
		b = a / -2
		i8test(a, b, -2)
		b = a / -3
		i8test(a, b, -3)
		b = a / -4
		i8test(a, b, -4)
		b = a / -5
		i8test(a, b, -5)
		b = a / -6
		i8test(a, b, -6)
		b = a / -7
		i8test(a, b, -7)
		b = a / -8
		i8test(a, b, -8)
		b = a / -10
		i8test(a, b, -10)
		b = a / -8
		i8test(a, b, -8)
		b = a / -20
		i8test(a, b, -20)
		b = a / -32
		i8test(a, b, -32)
		b = a / -60
		i8test(a, b, -60)
		b = a / -64
		i8test(a, b, -64)
		b = a / -128
		i8test(a, b, -128)
	}
}

func u8rand() uint8 {
	a := uint8(rand.Uint32())
	a >>= uint(rand.Intn(8))
	return a
}

func u8test(a, b, c uint8) {
	d := a / c
	if d != b {
		println("u8", a, b, c, d)
		panic("fail")
	}
}

func u8run() {
	var a, b uint8

	for i := 0; i < Count; i++ {
		a = u8rand()

		b = a / 1
		u8test(a, b, 1)
		b = a / 2
		u8test(a, b, 2)
		b = a / 3
		u8test(a, b, 3)
		b = a / 4
		u8test(a, b, 4)
		b = a / 5
		u8test(a, b, 5)
		b = a / 6
		u8test(a, b, 6)
		b = a / 7
		u8test(a, b, 7)
		b = a / 8
		u8test(a, b, 8)
		b = a / 10
		u8test(a, b, 10)
		b = a / 8
		u8test(a, b, 8)
		b = a / 20
		u8test(a, b, 20)
		b = a / 32
		u8test(a, b, 32)
		b = a / 60
		u8test(a, b, 60)
		b = a / 64
		u8test(a, b, 64)
		b = a / 128
		u8test(a, b, 128)
		b = a / 184
		u8test(a, b, 184)
	}
}

func main() {
	xtest()
	i64run()
	u64run()
	i32run()
	u32run()
	i16run()
	u16run()
	i8run()
	u8run()
}

func xtest() {
}
