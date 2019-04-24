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

func BenchmarkDivisiblePow2constI64(b *testing.B) {
	for i := 0; i < b.N; i++ {
		boolres = int64(i)%16 == 0
	}
}

var u64res uint64

func BenchmarkDivconstU64(b *testing.B) {
	for i := 0; i < b.N; i++ {
		u64res = uint64(i) / 7
	}
}

var i32res int32

func BenchmarkDivconstI32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i32res = int32(i) / 7
	}
}

func BenchmarkDivisiblePow2constI32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		boolres = int32(i)%16 == 0
	}
}

var u32res uint32

func BenchmarkDivconstU32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		u32res = uint32(i) / 7
	}
}

var i16res int16

func BenchmarkDivconstI16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i16res = int16(i) / 7
	}
}

func BenchmarkDivisiblePow2constI16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		boolres = int16(i)%16 == 0
	}
}

var u16res uint16

func BenchmarkDivconstU16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		u16res = uint16(i) / 7
	}
}

var i8res int8

func BenchmarkDivconstI8(b *testing.B) {
	for i := 0; i < b.N; i++ {
		i8res = int8(i) / 7
	}
}

func BenchmarkDivisiblePow2constI8(b *testing.B) {
	for i := 0; i < b.N; i++ {
		boolres = int8(i)%16 == 0
	}
}

var u8res uint8

func BenchmarkDivconstU8(b *testing.B) {
	for i := 0; i < b.N; i++ {
		u8res = uint8(i) / 7
	}
}
