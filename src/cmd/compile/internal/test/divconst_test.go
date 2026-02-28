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

func TestDivmodConstU64(t *testing.T) {
	// Test division by c. Function f must be func(n) { return n/c, n%c }
	testdiv := func(c uint64, f func(uint64) (uint64, uint64)) func(*testing.T) {
		return func(t *testing.T) {
			x := uint64(12345)
			for i := 0; i < 10000; i++ {
				x += x << 2
				q, r := f(x)
				if r < 0 || r >= c || q*c+r != x {
					t.Errorf("divmod(%d, %d) returned incorrect (%d, %d)", x, c, q, r)
				}
			}
			max := uint64(1<<64-1) / c * c
			xs := []uint64{0, 1, c - 1, c, c + 1, 2*c - 1, 2 * c, 2*c + 1,
				c*c - 1, c * c, c*c + 1, max - 1, max, max + 1, 1<<64 - 1}
			for _, x := range xs {
				q, r := f(x)
				if r < 0 || r >= c || q*c+r != x {
					t.Errorf("divmod(%d, %d) returned incorrect (%d, %d)", x, c, q, r)
				}
			}
		}
	}
	t.Run("2", testdiv(2, func(n uint64) (uint64, uint64) { return n / 2, n % 2 }))
	t.Run("3", testdiv(3, func(n uint64) (uint64, uint64) { return n / 3, n % 3 }))
	t.Run("4", testdiv(4, func(n uint64) (uint64, uint64) { return n / 4, n % 4 }))
	t.Run("5", testdiv(5, func(n uint64) (uint64, uint64) { return n / 5, n % 5 }))
	t.Run("6", testdiv(6, func(n uint64) (uint64, uint64) { return n / 6, n % 6 }))
	t.Run("7", testdiv(7, func(n uint64) (uint64, uint64) { return n / 7, n % 7 }))
	t.Run("8", testdiv(8, func(n uint64) (uint64, uint64) { return n / 8, n % 8 }))
	t.Run("9", testdiv(9, func(n uint64) (uint64, uint64) { return n / 9, n % 9 }))
	t.Run("10", testdiv(10, func(n uint64) (uint64, uint64) { return n / 10, n % 10 }))
	t.Run("11", testdiv(11, func(n uint64) (uint64, uint64) { return n / 11, n % 11 }))
	t.Run("12", testdiv(12, func(n uint64) (uint64, uint64) { return n / 12, n % 12 }))
	t.Run("13", testdiv(13, func(n uint64) (uint64, uint64) { return n / 13, n % 13 }))
	t.Run("14", testdiv(14, func(n uint64) (uint64, uint64) { return n / 14, n % 14 }))
	t.Run("15", testdiv(15, func(n uint64) (uint64, uint64) { return n / 15, n % 15 }))
	t.Run("16", testdiv(16, func(n uint64) (uint64, uint64) { return n / 16, n % 16 }))
	t.Run("17", testdiv(17, func(n uint64) (uint64, uint64) { return n / 17, n % 17 }))
	t.Run("255", testdiv(255, func(n uint64) (uint64, uint64) { return n / 255, n % 255 }))
	t.Run("256", testdiv(256, func(n uint64) (uint64, uint64) { return n / 256, n % 256 }))
	t.Run("257", testdiv(257, func(n uint64) (uint64, uint64) { return n / 257, n % 257 }))
	t.Run("65535", testdiv(65535, func(n uint64) (uint64, uint64) { return n / 65535, n % 65535 }))
	t.Run("65536", testdiv(65536, func(n uint64) (uint64, uint64) { return n / 65536, n % 65536 }))
	t.Run("65537", testdiv(65537, func(n uint64) (uint64, uint64) { return n / 65537, n % 65537 }))
	t.Run("1<<32-1", testdiv(1<<32-1, func(n uint64) (uint64, uint64) { return n / (1<<32 - 1), n % (1<<32 - 1) }))
	t.Run("1<<32+1", testdiv(1<<32+1, func(n uint64) (uint64, uint64) { return n / (1<<32 + 1), n % (1<<32 + 1) }))
	t.Run("1<<64-1", testdiv(1<<64-1, func(n uint64) (uint64, uint64) { return n / (1<<64 - 1), n % (1<<64 - 1) }))
}

func BenchmarkDivconstU64(b *testing.B) {
	b.Run("3", func(b *testing.B) {
		x := uint64(123456789123456789)
		for i := 0; i < b.N; i++ {
			x += x << 4
			u64res = x / 3
		}
	})
	b.Run("5", func(b *testing.B) {
		x := uint64(123456789123456789)
		for i := 0; i < b.N; i++ {
			x += x << 4
			u64res = x / 5
		}
	})
	b.Run("37", func(b *testing.B) {
		x := uint64(123456789123456789)
		for i := 0; i < b.N; i++ {
			x += x << 4
			u64res = x / 37
		}
	})
	b.Run("1234567", func(b *testing.B) {
		x := uint64(123456789123456789)
		for i := 0; i < b.N; i++ {
			x += x << 4
			u64res = x / 1234567
		}
	})
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
