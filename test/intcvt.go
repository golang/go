// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const (
	ci8  = -1 << 7
	ci16 = -1<<15 + 100
	ci32 = -1<<31 + 100000
	ci64 = -1<<63 + 10000000001

	cu8  = 1<<8 - 1
	cu16 = 1<<16 - 1234
	cu32 = 1<<32 - 1234567
	cu64 = 1<<64 - 1234567890123

	cf32 = 1e8 + 0.5
	cf64 = -1e8 + 0.5
)

var (
	i8  int8  = ci8
	i16 int16 = ci16
	i32 int32 = ci32
	i64 int64 = ci64

	u8  uint8  = cu8
	u16 uint16 = cu16
	u32 uint32 = cu32
	u64 uint64 = cu64

	//	f32 float32 = 1e8 + 0.5
	//	f64 float64 = -1e8 + 0.5
)

func chki8(i, v int8) {
	if i != v {
		println(i, "!=", v)
		panic("fail")
	}
}
func chki16(i, v int16) {
	if i != v {
		println(i, "!=", v)
		panic("fail")
	}
}
func chki32(i, v int32) {
	if i != v {
		println(i, "!=", v)
		panic("fail")
	}
}
func chki64(i, v int64) {
	if i != v {
		println(i, "!=", v)
		panic("fail")
	}
}
func chku8(i, v uint8) {
	if i != v {
		println(i, "!=", v)
		panic("fail")
	}
}
func chku16(i, v uint16) {
	if i != v {
		println(i, "!=", v)
		panic("fail")
	}
}
func chku32(i, v uint32) {
	if i != v {
		println(i, "!=", v)
		panic("fail")
	}
}
func chku64(i, v uint64) {
	if i != v {
		println(i, "!=", v)
		panic("fail")
	}
}
//func chkf32(f, v float32) { if f != v { println(f, "!=", v); panic("fail") } }
//func chkf64(f, v float64) { if f != v { println(f, "!=", v); panic("fail") } }

func main() {
	chki8(int8(i8), ci8&0xff-1<<8)
	chki8(int8(i16), ci16&0xff)
	chki8(int8(i32), ci32&0xff-1<<8)
	chki8(int8(i64), ci64&0xff)
	chki8(int8(u8), cu8&0xff-1<<8)
	chki8(int8(u16), cu16&0xff)
	chki8(int8(u32), cu32&0xff)
	chki8(int8(u64), cu64&0xff)
	//	chki8(int8(f32), 0)
	//	chki8(int8(f64), 0)

	chki16(int16(i8), ci8&0xffff-1<<16)
	chki16(int16(i16), ci16&0xffff-1<<16)
	chki16(int16(i32), ci32&0xffff-1<<16)
	chki16(int16(i64), ci64&0xffff-1<<16)
	chki16(int16(u8), cu8&0xffff)
	chki16(int16(u16), cu16&0xffff-1<<16)
	chki16(int16(u32), cu32&0xffff)
	chki16(int16(u64), cu64&0xffff-1<<16)
	//	chki16(int16(f32), 0)
	//	chki16(int16(f64), 0)

	chki32(int32(i8), ci8&0xffffffff-1<<32)
	chki32(int32(i16), ci16&0xffffffff-1<<32)
	chki32(int32(i32), ci32&0xffffffff-1<<32)
	chki32(int32(i64), ci64&0xffffffff)
	chki32(int32(u8), cu8&0xffffffff)
	chki32(int32(u16), cu16&0xffffffff)
	chki32(int32(u32), cu32&0xffffffff-1<<32)
	chki32(int32(u64), cu64&0xffffffff-1<<32)
	//	chki32(int32(f32), 0)
	//	chki32(int32(f64), 0)

	chki64(int64(i8), ci8&0xffffffffffffffff-1<<64)
	chki64(int64(i16), ci16&0xffffffffffffffff-1<<64)
	chki64(int64(i32), ci32&0xffffffffffffffff-1<<64)
	chki64(int64(i64), ci64&0xffffffffffffffff-1<<64)
	chki64(int64(u8), cu8&0xffffffffffffffff)
	chki64(int64(u16), cu16&0xffffffffffffffff)
	chki64(int64(u32), cu32&0xffffffffffffffff)
	chki64(int64(u64), cu64&0xffffffffffffffff-1<<64)
	//	chki64(int64(f32), 0)
	//	chki64(int64(f64), 0)


	chku8(uint8(i8), ci8&0xff)
	chku8(uint8(i16), ci16&0xff)
	chku8(uint8(i32), ci32&0xff)
	chku8(uint8(i64), ci64&0xff)
	chku8(uint8(u8), cu8&0xff)
	chku8(uint8(u16), cu16&0xff)
	chku8(uint8(u32), cu32&0xff)
	chku8(uint8(u64), cu64&0xff)
	//	chku8(uint8(f32), 0)
	//	chku8(uint8(f64), 0)

	chku16(uint16(i8), ci8&0xffff)
	chku16(uint16(i16), ci16&0xffff)
	chku16(uint16(i32), ci32&0xffff)
	chku16(uint16(i64), ci64&0xffff)
	chku16(uint16(u8), cu8&0xffff)
	chku16(uint16(u16), cu16&0xffff)
	chku16(uint16(u32), cu32&0xffff)
	chku16(uint16(u64), cu64&0xffff)
	//	chku16(uint16(f32), 0)
	//	chku16(uint16(f64), 0)

	chku32(uint32(i8), ci8&0xffffffff)
	chku32(uint32(i16), ci16&0xffffffff)
	chku32(uint32(i32), ci32&0xffffffff)
	chku32(uint32(i64), ci64&0xffffffff)
	chku32(uint32(u8), cu8&0xffffffff)
	chku32(uint32(u16), cu16&0xffffffff)
	chku32(uint32(u32), cu32&0xffffffff)
	chku32(uint32(u64), cu64&0xffffffff)
	//	chku32(uint32(f32), 0)
	//	chku32(uint32(f64), 0)

	chku64(uint64(i8), ci8&0xffffffffffffffff)
	chku64(uint64(i16), ci16&0xffffffffffffffff)
	chku64(uint64(i32), ci32&0xffffffffffffffff)
	chku64(uint64(i64), ci64&0xffffffffffffffff)
	chku64(uint64(u8), cu8&0xffffffffffffffff)
	chku64(uint64(u16), cu16&0xffffffffffffffff)
	chku64(uint64(u32), cu32&0xffffffffffffffff)
	chku64(uint64(u64), cu64&0xffffffffffffffff)
	//	chku64(uint64(f32), 0)
	//	chku64(uint64(f64), 0)
}
