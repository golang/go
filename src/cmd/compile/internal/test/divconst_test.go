// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"testing"
)

var boolres bool

var i64res int64

func BenchmarkDivconstI64(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i64res = int64(i) / 7
	}
}

func BenchmarkModconstI64(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i64res = int64(i) % 7
	}
}

func BenchmarkDivisiblePow2constI64(b *testing.B) {
	for i := 0; i < b.N; i++ {
		boolres = int64(i)%16 == 0
	}
}
func BenchmarkDivisibleconstI64(b *testing.B) {
	for i := 0; i < b.N; i++ {
		boolres = int64(i)%7 == 0
	}
}

func BenchmarkDivisibleWDivconstI64(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i64res = int64(i) / 7
		boolres = int64(i)%7 == 0
	}
}

var u64res uint64

func BenchmarkDivconstU64(b *testing.B) {
	for i := 0; i < b.N; i++ {
		u64res = uint64(i) / 7
	}
}

func BenchmarkModconstU64(b *testing.B) {
	for i := 0; i < b.N; i++ {
		u64res = uint64(i) % 7
	}
}

func BenchmarkDivisibleconstU64(b *testing.B) {
	for i := 0; i < b.N; i++ {
		boolres = uint64(i)%7 == 0
	}
}

func BenchmarkDivisibleWDivconstU64(b *testing.B) {
	for i := 0; i < b.N; i++ {
		u64res = uint64(i) / 7
		boolres = uint64(i)%7 == 0
	}
}

var i32res int32

func BenchmarkDivconstI32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i32res = int32(i) / 7
	}
}

func BenchmarkModconstI32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i32res = int32(i) % 7
	}
}

func BenchmarkDivisiblePow2constI32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		boolres = int32(i)%16 == 0
	}
}

func BenchmarkDivisibleconstI32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		boolres = int32(i)%7 == 0
	}
}

func BenchmarkDivisibleWDivconstI32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i32res = int32(i) / 7
		boolres = int32(i)%7 == 0
	}
}

var u32res uint32

func BenchmarkDivconstU32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		u32res = uint32(i) / 7
	}
}

func BenchmarkModconstU32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		u32res = uint32(i) % 7
	}
}

func BenchmarkDivisibleconstU32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		boolres = uint32(i)%7 == 0
	}
}

func BenchmarkDivisibleWDivconstU32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		u32res = uint32(i) / 7
		boolres = uint32(i)%7 == 0
	}
}

var i16res int16

func BenchmarkDivconstI16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i16res = int16(i) / 7
	}
}

func BenchmarkModconstI16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i16res = int16(i) % 7
	}
}

func BenchmarkDivisiblePow2constI16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		boolres = int16(i)%16 == 0
	}
}

func BenchmarkDivisibleconstI16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		boolres = int16(i)%7 == 0
	}
}

func BenchmarkDivisibleWDivconstI16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i16res = int16(i) / 7
		boolres = int16(i)%7 == 0
	}
}

var u16res uint16

func BenchmarkDivconstU16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		u16res = uint16(i) / 7
	}
}

func BenchmarkModconstU16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		u16res = uint16(i) % 7
	}
}

func BenchmarkDivisibleconstU16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		boolres = uint16(i)%7 == 0
	}
}

func BenchmarkDivisibleWDivconstU16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		u16res = uint16(i) / 7
		boolres = uint16(i)%7 == 0
	}
}

var i8res int8

func BenchmarkDivconstI8(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i8res = int8(i) / 7
	}
}

func BenchmarkModconstI8(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i8res = int8(i) % 7
	}
}

func BenchmarkDivisiblePow2constI8(b *testing.B) {
	for i := 0; i < b.N; i++ {
		boolres = int8(i)%16 == 0
	}
}

func BenchmarkDivisibleconstI8(b *testing.B) {
	for i := 0; i < b.N; i++ {
		boolres = int8(i)%7 == 0
	}
}

func BenchmarkDivisibleWDivconstI8(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i8res = int8(i) / 7
		boolres = int8(i)%7 == 0
	}
}

var u8res uint8

func BenchmarkDivconstU8(b *testing.B) {
	for i := 0; i < b.N; i++ {
		u8res = uint8(i) / 7
	}
}

func BenchmarkModconstU8(b *testing.B) {
	for i := 0; i < b.N; i++ {
		u8res = uint8(i) % 7
	}
}

func BenchmarkDivisibleconstU8(b *testing.B) {
	for i := 0; i < b.N; i++ {
		boolres = uint8(i)%7 == 0
	}
}

func BenchmarkDivisibleWDivconstU8(b *testing.B) {
	for i := 0; i < b.N; i++ {
		u8res = uint8(i) / 7
		boolres = uint8(i)%7 == 0
	}
}
