// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bits_test

import (
	. "math/bits"
	"runtime"
	"testing"
	"unsafe"
)

func TestUintSize(t *testing.T) {
	var x uint
	if want := unsafe.Sizeof(x) * 8; UintSize != want {
		t.Fatalf("UintSize = %d; want %d", UintSize, want)
	}
}

func TestLeadingZeros(t *testing.T) {
	for i := 0; i < 256; i++ {
		nlz := tab[i].nlz
		for k := 0; k < 64-8; k++ {
			x := uint64(i) << uint(k)
			if x <= 1<<8-1 {
				got := LeadingZeros8(uint8(x))
				want := nlz - k + (8 - 8)
				if x == 0 {
					want = 8
				}
				if got != want {
					t.Fatalf("LeadingZeros8(%#02x) == %d; want %d", x, got, want)
				}
			}

			if x <= 1<<16-1 {
				got := LeadingZeros16(uint16(x))
				want := nlz - k + (16 - 8)
				if x == 0 {
					want = 16
				}
				if got != want {
					t.Fatalf("LeadingZeros16(%#04x) == %d; want %d", x, got, want)
				}
			}

			if x <= 1<<32-1 {
				got := LeadingZeros32(uint32(x))
				want := nlz - k + (32 - 8)
				if x == 0 {
					want = 32
				}
				if got != want {
					t.Fatalf("LeadingZeros32(%#08x) == %d; want %d", x, got, want)
				}
				if UintSize == 32 {
					got = LeadingZeros(uint(x))
					if got != want {
						t.Fatalf("LeadingZeros(%#08x) == %d; want %d", x, got, want)
					}
				}
			}

			if x <= 1<<64-1 {
				got := LeadingZeros64(uint64(x))
				want := nlz - k + (64 - 8)
				if x == 0 {
					want = 64
				}
				if got != want {
					t.Fatalf("LeadingZeros64(%#016x) == %d; want %d", x, got, want)
				}
				if UintSize == 64 {
					got = LeadingZeros(uint(x))
					if got != want {
						t.Fatalf("LeadingZeros(%#016x) == %d; want %d", x, got, want)
					}
				}
			}
		}
	}
}

// Exported (global) variable serving as input for some
// of the benchmarks to ensure side-effect free calls
// are not optimized away.
var Input uint64 = DeBruijn64

// Exported (global) variable to store function results
// during benchmarking to ensure side-effect free calls
// are not optimized away.
var Output int

func BenchmarkLeadingZeros(b *testing.B) {
	var s int
	for i := 0; i < b.N; i++ {
		s += LeadingZeros(uint(Input) >> (uint(i) % UintSize))
	}
	Output = s
}

func BenchmarkLeadingZeros8(b *testing.B) {
	var s int
	for i := 0; i < b.N; i++ {
		s += LeadingZeros8(uint8(Input) >> (uint(i) % 8))
	}
	Output = s
}

func BenchmarkLeadingZeros16(b *testing.B) {
	var s int
	for i := 0; i < b.N; i++ {
		s += LeadingZeros16(uint16(Input) >> (uint(i) % 16))
	}
	Output = s
}

func BenchmarkLeadingZeros32(b *testing.B) {
	var s int
	for i := 0; i < b.N; i++ {
		s += LeadingZeros32(uint32(Input) >> (uint(i) % 32))
	}
	Output = s
}

func BenchmarkLeadingZeros64(b *testing.B) {
	var s int
	for i := 0; i < b.N; i++ {
		s += LeadingZeros64(uint64(Input) >> (uint(i) % 64))
	}
	Output = s
}

func TestTrailingZeros(t *testing.T) {
	for i := 0; i < 256; i++ {
		ntz := tab[i].ntz
		for k := 0; k < 64-8; k++ {
			x := uint64(i) << uint(k)
			want := ntz + k
			if x <= 1<<8-1 {
				got := TrailingZeros8(uint8(x))
				if x == 0 {
					want = 8
				}
				if got != want {
					t.Fatalf("TrailingZeros8(%#02x) == %d; want %d", x, got, want)
				}
			}

			if x <= 1<<16-1 {
				got := TrailingZeros16(uint16(x))
				if x == 0 {
					want = 16
				}
				if got != want {
					t.Fatalf("TrailingZeros16(%#04x) == %d; want %d", x, got, want)
				}
			}

			if x <= 1<<32-1 {
				got := TrailingZeros32(uint32(x))
				if x == 0 {
					want = 32
				}
				if got != want {
					t.Fatalf("TrailingZeros32(%#08x) == %d; want %d", x, got, want)
				}
				if UintSize == 32 {
					got = TrailingZeros(uint(x))
					if got != want {
						t.Fatalf("TrailingZeros(%#08x) == %d; want %d", x, got, want)
					}
				}
			}

			if x <= 1<<64-1 {
				got := TrailingZeros64(uint64(x))
				if x == 0 {
					want = 64
				}
				if got != want {
					t.Fatalf("TrailingZeros64(%#016x) == %d; want %d", x, got, want)
				}
				if UintSize == 64 {
					got = TrailingZeros(uint(x))
					if got != want {
						t.Fatalf("TrailingZeros(%#016x) == %d; want %d", x, got, want)
					}
				}
			}
		}
	}
}

func BenchmarkTrailingZeros(b *testing.B) {
	var s int
	for i := 0; i < b.N; i++ {
		s += TrailingZeros(uint(Input) << (uint(i) % UintSize))
	}
	Output = s
}

func BenchmarkTrailingZeros8(b *testing.B) {
	var s int
	for i := 0; i < b.N; i++ {
		s += TrailingZeros8(uint8(Input) << (uint(i) % 8))
	}
	Output = s
}

func BenchmarkTrailingZeros16(b *testing.B) {
	var s int
	for i := 0; i < b.N; i++ {
		s += TrailingZeros16(uint16(Input) << (uint(i) % 16))
	}
	Output = s
}

func BenchmarkTrailingZeros32(b *testing.B) {
	var s int
	for i := 0; i < b.N; i++ {
		s += TrailingZeros32(uint32(Input) << (uint(i) % 32))
	}
	Output = s
}

func BenchmarkTrailingZeros64(b *testing.B) {
	var s int
	for i := 0; i < b.N; i++ {
		s += TrailingZeros64(uint64(Input) << (uint(i) % 64))
	}
	Output = s
}

func TestOnesCount(t *testing.T) {
	var x uint64
	for i := 0; i <= 64; i++ {
		testOnesCount(t, x, i)
		x = x<<1 | 1
	}

	for i := 64; i >= 0; i-- {
		testOnesCount(t, x, i)
		x = x << 1
	}

	for i := 0; i < 256; i++ {
		for k := 0; k < 64-8; k++ {
			testOnesCount(t, uint64(i)<<uint(k), tab[i].pop)
		}
	}
}

func testOnesCount(t *testing.T, x uint64, want int) {
	if x <= 1<<8-1 {
		got := OnesCount8(uint8(x))
		if got != want {
			t.Fatalf("OnesCount8(%#02x) == %d; want %d", uint8(x), got, want)
		}
	}

	if x <= 1<<16-1 {
		got := OnesCount16(uint16(x))
		if got != want {
			t.Fatalf("OnesCount16(%#04x) == %d; want %d", uint16(x), got, want)
		}
	}

	if x <= 1<<32-1 {
		got := OnesCount32(uint32(x))
		if got != want {
			t.Fatalf("OnesCount32(%#08x) == %d; want %d", uint32(x), got, want)
		}
		if UintSize == 32 {
			got = OnesCount(uint(x))
			if got != want {
				t.Fatalf("OnesCount(%#08x) == %d; want %d", uint32(x), got, want)
			}
		}
	}

	if x <= 1<<64-1 {
		got := OnesCount64(uint64(x))
		if got != want {
			t.Fatalf("OnesCount64(%#016x) == %d; want %d", x, got, want)
		}
		if UintSize == 64 {
			got = OnesCount(uint(x))
			if got != want {
				t.Fatalf("OnesCount(%#016x) == %d; want %d", x, got, want)
			}
		}
	}
}

func BenchmarkOnesCount(b *testing.B) {
	var s int
	for i := 0; i < b.N; i++ {
		s += OnesCount(uint(Input))
	}
	Output = s
}

func BenchmarkOnesCount8(b *testing.B) {
	var s int
	for i := 0; i < b.N; i++ {
		s += OnesCount8(uint8(Input))
	}
	Output = s
}

func BenchmarkOnesCount16(b *testing.B) {
	var s int
	for i := 0; i < b.N; i++ {
		s += OnesCount16(uint16(Input))
	}
	Output = s
}

func BenchmarkOnesCount32(b *testing.B) {
	var s int
	for i := 0; i < b.N; i++ {
		s += OnesCount32(uint32(Input))
	}
	Output = s
}

func BenchmarkOnesCount64(b *testing.B) {
	var s int
	for i := 0; i < b.N; i++ {
		s += OnesCount64(uint64(Input))
	}
	Output = s
}

func TestRotateLeft(t *testing.T) {
	var m uint64 = DeBruijn64

	for k := uint(0); k < 128; k++ {
		x8 := uint8(m)
		got8 := RotateLeft8(x8, int(k))
		want8 := x8<<(k&0x7) | x8>>(8-k&0x7)
		if got8 != want8 {
			t.Fatalf("RotateLeft8(%#02x, %d) == %#02x; want %#02x", x8, k, got8, want8)
		}
		got8 = RotateLeft8(want8, -int(k))
		if got8 != x8 {
			t.Fatalf("RotateLeft8(%#02x, -%d) == %#02x; want %#02x", want8, k, got8, x8)
		}

		x16 := uint16(m)
		got16 := RotateLeft16(x16, int(k))
		want16 := x16<<(k&0xf) | x16>>(16-k&0xf)
		if got16 != want16 {
			t.Fatalf("RotateLeft16(%#04x, %d) == %#04x; want %#04x", x16, k, got16, want16)
		}
		got16 = RotateLeft16(want16, -int(k))
		if got16 != x16 {
			t.Fatalf("RotateLeft16(%#04x, -%d) == %#04x; want %#04x", want16, k, got16, x16)
		}

		x32 := uint32(m)
		got32 := RotateLeft32(x32, int(k))
		want32 := x32<<(k&0x1f) | x32>>(32-k&0x1f)
		if got32 != want32 {
			t.Fatalf("RotateLeft32(%#08x, %d) == %#08x; want %#08x", x32, k, got32, want32)
		}
		got32 = RotateLeft32(want32, -int(k))
		if got32 != x32 {
			t.Fatalf("RotateLeft32(%#08x, -%d) == %#08x; want %#08x", want32, k, got32, x32)
		}
		if UintSize == 32 {
			x := uint(m)
			got := RotateLeft(x, int(k))
			want := x<<(k&0x1f) | x>>(32-k&0x1f)
			if got != want {
				t.Fatalf("RotateLeft(%#08x, %d) == %#08x; want %#08x", x, k, got, want)
			}
			got = RotateLeft(want, -int(k))
			if got != x {
				t.Fatalf("RotateLeft(%#08x, -%d) == %#08x; want %#08x", want, k, got, x)
			}
		}

		x64 := uint64(m)
		got64 := RotateLeft64(x64, int(k))
		want64 := x64<<(k&0x3f) | x64>>(64-k&0x3f)
		if got64 != want64 {
			t.Fatalf("RotateLeft64(%#016x, %d) == %#016x; want %#016x", x64, k, got64, want64)
		}
		got64 = RotateLeft64(want64, -int(k))
		if got64 != x64 {
			t.Fatalf("RotateLeft64(%#016x, -%d) == %#016x; want %#016x", want64, k, got64, x64)
		}
		if UintSize == 64 {
			x := uint(m)
			got := RotateLeft(x, int(k))
			want := x<<(k&0x3f) | x>>(64-k&0x3f)
			if got != want {
				t.Fatalf("RotateLeft(%#016x, %d) == %#016x; want %#016x", x, k, got, want)
			}
			got = RotateLeft(want, -int(k))
			if got != x {
				t.Fatalf("RotateLeft(%#08x, -%d) == %#08x; want %#08x", want, k, got, x)
			}
		}
	}
}

func BenchmarkRotateLeft(b *testing.B) {
	var s uint
	for i := 0; i < b.N; i++ {
		s += RotateLeft(uint(Input), i)
	}
	Output = int(s)
}

func BenchmarkRotateLeft8(b *testing.B) {
	var s uint8
	for i := 0; i < b.N; i++ {
		s += RotateLeft8(uint8(Input), i)
	}
	Output = int(s)
}

func BenchmarkRotateLeft16(b *testing.B) {
	var s uint16
	for i := 0; i < b.N; i++ {
		s += RotateLeft16(uint16(Input), i)
	}
	Output = int(s)
}

func BenchmarkRotateLeft32(b *testing.B) {
	var s uint32
	for i := 0; i < b.N; i++ {
		s += RotateLeft32(uint32(Input), i)
	}
	Output = int(s)
}

func BenchmarkRotateLeft64(b *testing.B) {
	var s uint64
	for i := 0; i < b.N; i++ {
		s += RotateLeft64(uint64(Input), i)
	}
	Output = int(s)
}

func TestReverse(t *testing.T) {
	// test each bit
	for i := uint(0); i < 64; i++ {
		testReverse(t, uint64(1)<<i, uint64(1)<<(63-i))
	}

	// test a few patterns
	for _, test := range []struct {
		x, r uint64
	}{
		{0, 0},
		{0x1, 0x8 << 60},
		{0x2, 0x4 << 60},
		{0x3, 0xc << 60},
		{0x4, 0x2 << 60},
		{0x5, 0xa << 60},
		{0x6, 0x6 << 60},
		{0x7, 0xe << 60},
		{0x8, 0x1 << 60},
		{0x9, 0x9 << 60},
		{0xa, 0x5 << 60},
		{0xb, 0xd << 60},
		{0xc, 0x3 << 60},
		{0xd, 0xb << 60},
		{0xe, 0x7 << 60},
		{0xf, 0xf << 60},
		{0x5686487, 0xe12616a000000000},
		{0x0123456789abcdef, 0xf7b3d591e6a2c480},
	} {
		testReverse(t, test.x, test.r)
		testReverse(t, test.r, test.x)
	}
}

func testReverse(t *testing.T, x64, want64 uint64) {
	x8 := uint8(x64)
	got8 := Reverse8(x8)
	want8 := uint8(want64 >> (64 - 8))
	if got8 != want8 {
		t.Fatalf("Reverse8(%#02x) == %#02x; want %#02x", x8, got8, want8)
	}

	x16 := uint16(x64)
	got16 := Reverse16(x16)
	want16 := uint16(want64 >> (64 - 16))
	if got16 != want16 {
		t.Fatalf("Reverse16(%#04x) == %#04x; want %#04x", x16, got16, want16)
	}

	x32 := uint32(x64)
	got32 := Reverse32(x32)
	want32 := uint32(want64 >> (64 - 32))
	if got32 != want32 {
		t.Fatalf("Reverse32(%#08x) == %#08x; want %#08x", x32, got32, want32)
	}
	if UintSize == 32 {
		x := uint(x32)
		got := Reverse(x)
		want := uint(want32)
		if got != want {
			t.Fatalf("Reverse(%#08x) == %#08x; want %#08x", x, got, want)
		}
	}

	got64 := Reverse64(x64)
	if got64 != want64 {
		t.Fatalf("Reverse64(%#016x) == %#016x; want %#016x", x64, got64, want64)
	}
	if UintSize == 64 {
		x := uint(x64)
		got := Reverse(x)
		want := uint(want64)
		if got != want {
			t.Fatalf("Reverse(%#08x) == %#016x; want %#016x", x, got, want)
		}
	}
}

func BenchmarkReverse(b *testing.B) {
	var s uint
	for i := 0; i < b.N; i++ {
		s += Reverse(uint(i))
	}
	Output = int(s)
}

func BenchmarkReverse8(b *testing.B) {
	var s uint8
	for i := 0; i < b.N; i++ {
		s += Reverse8(uint8(i))
	}
	Output = int(s)
}

func BenchmarkReverse16(b *testing.B) {
	var s uint16
	for i := 0; i < b.N; i++ {
		s += Reverse16(uint16(i))
	}
	Output = int(s)
}

func BenchmarkReverse32(b *testing.B) {
	var s uint32
	for i := 0; i < b.N; i++ {
		s += Reverse32(uint32(i))
	}
	Output = int(s)
}

func BenchmarkReverse64(b *testing.B) {
	var s uint64
	for i := 0; i < b.N; i++ {
		s += Reverse64(uint64(i))
	}
	Output = int(s)
}

func TestReverseBytes(t *testing.T) {
	for _, test := range []struct {
		x, r uint64
	}{
		{0, 0},
		{0x01, 0x01 << 56},
		{0x0123, 0x2301 << 48},
		{0x012345, 0x452301 << 40},
		{0x01234567, 0x67452301 << 32},
		{0x0123456789, 0x8967452301 << 24},
		{0x0123456789ab, 0xab8967452301 << 16},
		{0x0123456789abcd, 0xcdab8967452301 << 8},
		{0x0123456789abcdef, 0xefcdab8967452301 << 0},
	} {
		testReverseBytes(t, test.x, test.r)
		testReverseBytes(t, test.r, test.x)
	}
}

func testReverseBytes(t *testing.T, x64, want64 uint64) {
	x16 := uint16(x64)
	got16 := ReverseBytes16(x16)
	want16 := uint16(want64 >> (64 - 16))
	if got16 != want16 {
		t.Fatalf("ReverseBytes16(%#04x) == %#04x; want %#04x", x16, got16, want16)
	}

	x32 := uint32(x64)
	got32 := ReverseBytes32(x32)
	want32 := uint32(want64 >> (64 - 32))
	if got32 != want32 {
		t.Fatalf("ReverseBytes32(%#08x) == %#08x; want %#08x", x32, got32, want32)
	}
	if UintSize == 32 {
		x := uint(x32)
		got := ReverseBytes(x)
		want := uint(want32)
		if got != want {
			t.Fatalf("ReverseBytes(%#08x) == %#08x; want %#08x", x, got, want)
		}
	}

	got64 := ReverseBytes64(x64)
	if got64 != want64 {
		t.Fatalf("ReverseBytes64(%#016x) == %#016x; want %#016x", x64, got64, want64)
	}
	if UintSize == 64 {
		x := uint(x64)
		got := ReverseBytes(x)
		want := uint(want64)
		if got != want {
			t.Fatalf("ReverseBytes(%#016x) == %#016x; want %#016x", x, got, want)
		}
	}
}

func BenchmarkReverseBytes(b *testing.B) {
	var s uint
	for i := 0; i < b.N; i++ {
		s += ReverseBytes(uint(i))
	}
	Output = int(s)
}

func BenchmarkReverseBytes16(b *testing.B) {
	var s uint16
	for i := 0; i < b.N; i++ {
		s += ReverseBytes16(uint16(i))
	}
	Output = int(s)
}

func BenchmarkReverseBytes32(b *testing.B) {
	var s uint32
	for i := 0; i < b.N; i++ {
		s += ReverseBytes32(uint32(i))
	}
	Output = int(s)
}

func BenchmarkReverseBytes64(b *testing.B) {
	var s uint64
	for i := 0; i < b.N; i++ {
		s += ReverseBytes64(uint64(i))
	}
	Output = int(s)
}

func TestLen(t *testing.T) {
	for i := 0; i < 256; i++ {
		len := 8 - tab[i].nlz
		for k := 0; k < 64-8; k++ {
			x := uint64(i) << uint(k)
			want := 0
			if x != 0 {
				want = len + k
			}
			if x <= 1<<8-1 {
				got := Len8(uint8(x))
				if got != want {
					t.Fatalf("Len8(%#02x) == %d; want %d", x, got, want)
				}
			}

			if x <= 1<<16-1 {
				got := Len16(uint16(x))
				if got != want {
					t.Fatalf("Len16(%#04x) == %d; want %d", x, got, want)
				}
			}

			if x <= 1<<32-1 {
				got := Len32(uint32(x))
				if got != want {
					t.Fatalf("Len32(%#08x) == %d; want %d", x, got, want)
				}
				if UintSize == 32 {
					got := Len(uint(x))
					if got != want {
						t.Fatalf("Len(%#08x) == %d; want %d", x, got, want)
					}
				}
			}

			if x <= 1<<64-1 {
				got := Len64(uint64(x))
				if got != want {
					t.Fatalf("Len64(%#016x) == %d; want %d", x, got, want)
				}
				if UintSize == 64 {
					got := Len(uint(x))
					if got != want {
						t.Fatalf("Len(%#016x) == %d; want %d", x, got, want)
					}
				}
			}
		}
	}
}

const (
	_M   = 1<<UintSize - 1
	_M32 = 1<<32 - 1
	_M64 = 1<<64 - 1
)

func TestAddSubUint(t *testing.T) {
	test := func(msg string, f func(x, y, c uint) (z, cout uint), x, y, c, z, cout uint) {
		z1, cout1 := f(x, y, c)
		if z1 != z || cout1 != cout {
			t.Errorf("%s: got z:cout = %#x:%#x; want %#x:%#x", msg, z1, cout1, z, cout)
		}
	}
	for _, a := range []struct{ x, y, c, z, cout uint }{
		{0, 0, 0, 0, 0},
		{0, 1, 0, 1, 0},
		{0, 0, 1, 1, 0},
		{0, 1, 1, 2, 0},
		{12345, 67890, 0, 80235, 0},
		{12345, 67890, 1, 80236, 0},
		{_M, 1, 0, 0, 1},
		{_M, 0, 1, 0, 1},
		{_M, 1, 1, 1, 1},
		{_M, _M, 0, _M - 1, 1},
		{_M, _M, 1, _M, 1},
	} {
		test("Add", Add, a.x, a.y, a.c, a.z, a.cout)
		test("Add symmetric", Add, a.y, a.x, a.c, a.z, a.cout)
		test("Sub", Sub, a.z, a.x, a.c, a.y, a.cout)
		test("Sub symmetric", Sub, a.z, a.y, a.c, a.x, a.cout)
		// The above code can't test intrinsic implementation, because the passed function is not called directly.
		// The following code uses a closure to test the intrinsic version in case the function is intrinsified.
		test("Add intrinsic", func(x, y, c uint) (uint, uint) { return Add(x, y, c) }, a.x, a.y, a.c, a.z, a.cout)
		test("Add intrinsic symmetric", func(x, y, c uint) (uint, uint) { return Add(x, y, c) }, a.y, a.x, a.c, a.z, a.cout)
		test("Sub intrinsic", func(x, y, c uint) (uint, uint) { return Sub(x, y, c) }, a.z, a.x, a.c, a.y, a.cout)
		test("Sub intrinsic symmetric", func(x, y, c uint) (uint, uint) { return Sub(x, y, c) }, a.z, a.y, a.c, a.x, a.cout)

	}
}

func TestAddSubUint32(t *testing.T) {
	test := func(msg string, f func(x, y, c uint32) (z, cout uint32), x, y, c, z, cout uint32) {
		z1, cout1 := f(x, y, c)
		if z1 != z || cout1 != cout {
			t.Errorf("%s: got z:cout = %#x:%#x; want %#x:%#x", msg, z1, cout1, z, cout)
		}
	}
	for _, a := range []struct{ x, y, c, z, cout uint32 }{
		{0, 0, 0, 0, 0},
		{0, 1, 0, 1, 0},
		{0, 0, 1, 1, 0},
		{0, 1, 1, 2, 0},
		{12345, 67890, 0, 80235, 0},
		{12345, 67890, 1, 80236, 0},
		{_M32, 1, 0, 0, 1},
		{_M32, 0, 1, 0, 1},
		{_M32, 1, 1, 1, 1},
		{_M32, _M32, 0, _M32 - 1, 1},
		{_M32, _M32, 1, _M32, 1},
	} {
		test("Add32", Add32, a.x, a.y, a.c, a.z, a.cout)
		test("Add32 symmetric", Add32, a.y, a.x, a.c, a.z, a.cout)
		test("Sub32", Sub32, a.z, a.x, a.c, a.y, a.cout)
		test("Sub32 symmetric", Sub32, a.z, a.y, a.c, a.x, a.cout)
	}
}

func TestAddSubUint64(t *testing.T) {
	test := func(msg string, f func(x, y, c uint64) (z, cout uint64), x, y, c, z, cout uint64) {
		z1, cout1 := f(x, y, c)
		if z1 != z || cout1 != cout {
			t.Errorf("%s: got z:cout = %#x:%#x; want %#x:%#x", msg, z1, cout1, z, cout)
		}
	}
	for _, a := range []struct{ x, y, c, z, cout uint64 }{
		{0, 0, 0, 0, 0},
		{0, 1, 0, 1, 0},
		{0, 0, 1, 1, 0},
		{0, 1, 1, 2, 0},
		{12345, 67890, 0, 80235, 0},
		{12345, 67890, 1, 80236, 0},
		{_M64, 1, 0, 0, 1},
		{_M64, 0, 1, 0, 1},
		{_M64, 1, 1, 1, 1},
		{_M64, _M64, 0, _M64 - 1, 1},
		{_M64, _M64, 1, _M64, 1},
	} {
		test("Add64", Add64, a.x, a.y, a.c, a.z, a.cout)
		test("Add64 symmetric", Add64, a.y, a.x, a.c, a.z, a.cout)
		test("Sub64", Sub64, a.z, a.x, a.c, a.y, a.cout)
		test("Sub64 symmetric", Sub64, a.z, a.y, a.c, a.x, a.cout)
		// The above code can't test intrinsic implementation, because the passed function is not called directly.
		// The following code uses a closure to test the intrinsic version in case the function is intrinsified.
		test("Add64 intrinsic", func(x, y, c uint64) (uint64, uint64) { return Add64(x, y, c) }, a.x, a.y, a.c, a.z, a.cout)
		test("Add64 intrinsic symmetric", func(x, y, c uint64) (uint64, uint64) { return Add64(x, y, c) }, a.y, a.x, a.c, a.z, a.cout)
		test("Sub64 intrinsic", func(x, y, c uint64) (uint64, uint64) { return Sub64(x, y, c) }, a.z, a.x, a.c, a.y, a.cout)
		test("Sub64 intrinsic symmetric", func(x, y, c uint64) (uint64, uint64) { return Sub64(x, y, c) }, a.z, a.y, a.c, a.x, a.cout)
	}
}

func TestAdd64OverflowPanic(t *testing.T) {
	// Test that 64-bit overflow panics fire correctly.
	// These are designed to improve coverage of compiler intrinsics.
	tests := []func(uint64, uint64) uint64{
		func(a, b uint64) uint64 {
			x, c := Add64(a, b, 0)
			if c > 0 {
				panic("overflow")
			}
			return x
		},
		func(a, b uint64) uint64 {
			x, c := Add64(a, b, 0)
			if c != 0 {
				panic("overflow")
			}
			return x
		},
		func(a, b uint64) uint64 {
			x, c := Add64(a, b, 0)
			if c == 1 {
				panic("overflow")
			}
			return x
		},
		func(a, b uint64) uint64 {
			x, c := Add64(a, b, 0)
			if c != 1 {
				return x
			}
			panic("overflow")
		},
		func(a, b uint64) uint64 {
			x, c := Add64(a, b, 0)
			if c == 0 {
				return x
			}
			panic("overflow")
		},
	}
	for _, test := range tests {
		shouldPanic := func(f func()) {
			defer func() {
				if err := recover(); err == nil {
					t.Fatalf("expected panic")
				}
			}()
			f()
		}

		// overflow
		shouldPanic(func() { test(_M64, 1) })
		shouldPanic(func() { test(1, _M64) })
		shouldPanic(func() { test(_M64, _M64) })

		// no overflow
		test(_M64, 0)
		test(0, 0)
		test(1, 1)
	}
}

func TestSub64OverflowPanic(t *testing.T) {
	// Test that 64-bit overflow panics fire correctly.
	// These are designed to improve coverage of compiler intrinsics.
	tests := []func(uint64, uint64) uint64{
		func(a, b uint64) uint64 {
			x, c := Sub64(a, b, 0)
			if c > 0 {
				panic("overflow")
			}
			return x
		},
		func(a, b uint64) uint64 {
			x, c := Sub64(a, b, 0)
			if c != 0 {
				panic("overflow")
			}
			return x
		},
		func(a, b uint64) uint64 {
			x, c := Sub64(a, b, 0)
			if c == 1 {
				panic("overflow")
			}
			return x
		},
		func(a, b uint64) uint64 {
			x, c := Sub64(a, b, 0)
			if c != 1 {
				return x
			}
			panic("overflow")
		},
		func(a, b uint64) uint64 {
			x, c := Sub64(a, b, 0)
			if c == 0 {
				return x
			}
			panic("overflow")
		},
	}
	for _, test := range tests {
		shouldPanic := func(f func()) {
			defer func() {
				if err := recover(); err == nil {
					t.Fatalf("expected panic")
				}
			}()
			f()
		}

		// overflow
		shouldPanic(func() { test(0, 1) })
		shouldPanic(func() { test(1, _M64) })
		shouldPanic(func() { test(_M64-1, _M64) })

		// no overflow
		test(_M64, 0)
		test(0, 0)
		test(1, 1)
	}
}

func TestMulDiv(t *testing.T) {
	testMul := func(msg string, f func(x, y uint) (hi, lo uint), x, y, hi, lo uint) {
		hi1, lo1 := f(x, y)
		if hi1 != hi || lo1 != lo {
			t.Errorf("%s: got hi:lo = %#x:%#x; want %#x:%#x", msg, hi1, lo1, hi, lo)
		}
	}
	testDiv := func(msg string, f func(hi, lo, y uint) (q, r uint), hi, lo, y, q, r uint) {
		q1, r1 := f(hi, lo, y)
		if q1 != q || r1 != r {
			t.Errorf("%s: got q:r = %#x:%#x; want %#x:%#x", msg, q1, r1, q, r)
		}
	}
	for _, a := range []struct {
		x, y      uint
		hi, lo, r uint
	}{
		{1 << (UintSize - 1), 2, 1, 0, 1},
		{_M, _M, _M - 1, 1, 42},
	} {
		testMul("Mul", Mul, a.x, a.y, a.hi, a.lo)
		testMul("Mul symmetric", Mul, a.y, a.x, a.hi, a.lo)
		testDiv("Div", Div, a.hi, a.lo+a.r, a.y, a.x, a.r)
		testDiv("Div symmetric", Div, a.hi, a.lo+a.r, a.x, a.y, a.r)
		// The above code can't test intrinsic implementation, because the passed function is not called directly.
		// The following code uses a closure to test the intrinsic version in case the function is intrinsified.
		testMul("Mul intrinsic", func(x, y uint) (uint, uint) { return Mul(x, y) }, a.x, a.y, a.hi, a.lo)
		testMul("Mul intrinsic symmetric", func(x, y uint) (uint, uint) { return Mul(x, y) }, a.y, a.x, a.hi, a.lo)
		testDiv("Div intrinsic", func(hi, lo, y uint) (uint, uint) { return Div(hi, lo, y) }, a.hi, a.lo+a.r, a.y, a.x, a.r)
		testDiv("Div intrinsic symmetric", func(hi, lo, y uint) (uint, uint) { return Div(hi, lo, y) }, a.hi, a.lo+a.r, a.x, a.y, a.r)
	}
}

func TestMulDiv32(t *testing.T) {
	testMul := func(msg string, f func(x, y uint32) (hi, lo uint32), x, y, hi, lo uint32) {
		hi1, lo1 := f(x, y)
		if hi1 != hi || lo1 != lo {
			t.Errorf("%s: got hi:lo = %#x:%#x; want %#x:%#x", msg, hi1, lo1, hi, lo)
		}
	}
	testDiv := func(msg string, f func(hi, lo, y uint32) (q, r uint32), hi, lo, y, q, r uint32) {
		q1, r1 := f(hi, lo, y)
		if q1 != q || r1 != r {
			t.Errorf("%s: got q:r = %#x:%#x; want %#x:%#x", msg, q1, r1, q, r)
		}
	}
	for _, a := range []struct {
		x, y      uint32
		hi, lo, r uint32
	}{
		{1 << 31, 2, 1, 0, 1},
		{0xc47dfa8c, 50911, 0x98a4, 0x998587f4, 13},
		{_M32, _M32, _M32 - 1, 1, 42},
	} {
		testMul("Mul32", Mul32, a.x, a.y, a.hi, a.lo)
		testMul("Mul32 symmetric", Mul32, a.y, a.x, a.hi, a.lo)
		testDiv("Div32", Div32, a.hi, a.lo+a.r, a.y, a.x, a.r)
		testDiv("Div32 symmetric", Div32, a.hi, a.lo+a.r, a.x, a.y, a.r)
	}
}

func TestMulDiv64(t *testing.T) {
	testMul := func(msg string, f func(x, y uint64) (hi, lo uint64), x, y, hi, lo uint64) {
		hi1, lo1 := f(x, y)
		if hi1 != hi || lo1 != lo {
			t.Errorf("%s: got hi:lo = %#x:%#x; want %#x:%#x", msg, hi1, lo1, hi, lo)
		}
	}
	testDiv := func(msg string, f func(hi, lo, y uint64) (q, r uint64), hi, lo, y, q, r uint64) {
		q1, r1 := f(hi, lo, y)
		if q1 != q || r1 != r {
			t.Errorf("%s: got q:r = %#x:%#x; want %#x:%#x", msg, q1, r1, q, r)
		}
	}
	for _, a := range []struct {
		x, y      uint64
		hi, lo, r uint64
	}{
		{1 << 63, 2, 1, 0, 1},
		{0x3626229738a3b9, 0xd8988a9f1cc4a61, 0x2dd0712657fe8, 0x9dd6a3364c358319, 13},
		{_M64, _M64, _M64 - 1, 1, 42},
	} {
		testMul("Mul64", Mul64, a.x, a.y, a.hi, a.lo)
		testMul("Mul64 symmetric", Mul64, a.y, a.x, a.hi, a.lo)
		testDiv("Div64", Div64, a.hi, a.lo+a.r, a.y, a.x, a.r)
		testDiv("Div64 symmetric", Div64, a.hi, a.lo+a.r, a.x, a.y, a.r)
		// The above code can't test intrinsic implementation, because the passed function is not called directly.
		// The following code uses a closure to test the intrinsic version in case the function is intrinsified.
		testMul("Mul64 intrinsic", func(x, y uint64) (uint64, uint64) { return Mul64(x, y) }, a.x, a.y, a.hi, a.lo)
		testMul("Mul64 intrinsic symmetric", func(x, y uint64) (uint64, uint64) { return Mul64(x, y) }, a.y, a.x, a.hi, a.lo)
		testDiv("Div64 intrinsic", func(hi, lo, y uint64) (uint64, uint64) { return Div64(hi, lo, y) }, a.hi, a.lo+a.r, a.y, a.x, a.r)
		testDiv("Div64 intrinsic symmetric", func(hi, lo, y uint64) (uint64, uint64) { return Div64(hi, lo, y) }, a.hi, a.lo+a.r, a.x, a.y, a.r)
	}
}

const (
	divZeroError  = "runtime error: integer divide by zero"
	overflowError = "runtime error: integer overflow"
)

func TestDivPanicOverflow(t *testing.T) {
	// Expect a panic
	defer func() {
		if err := recover(); err == nil {
			t.Error("Div should have panicked when y<=hi")
		} else if e, ok := err.(runtime.Error); !ok || e.Error() != overflowError {
			t.Errorf("Div expected panic: %q, got: %q ", overflowError, e.Error())
		}
	}()
	q, r := Div(1, 0, 1)
	t.Errorf("undefined q, r = %v, %v calculated when Div should have panicked", q, r)
}

func TestDiv32PanicOverflow(t *testing.T) {
	// Expect a panic
	defer func() {
		if err := recover(); err == nil {
			t.Error("Div32 should have panicked when y<=hi")
		} else if e, ok := err.(runtime.Error); !ok || e.Error() != overflowError {
			t.Errorf("Div32 expected panic: %q, got: %q ", overflowError, e.Error())
		}
	}()
	q, r := Div32(1, 0, 1)
	t.Errorf("undefined q, r = %v, %v calculated when Div32 should have panicked", q, r)
}

func TestDiv64PanicOverflow(t *testing.T) {
	// Expect a panic
	defer func() {
		if err := recover(); err == nil {
			t.Error("Div64 should have panicked when y<=hi")
		} else if e, ok := err.(runtime.Error); !ok || e.Error() != overflowError {
			t.Errorf("Div64 expected panic: %q, got: %q ", overflowError, e.Error())
		}
	}()
	q, r := Div64(1, 0, 1)
	t.Errorf("undefined q, r = %v, %v calculated when Div64 should have panicked", q, r)
}

func TestDivPanicZero(t *testing.T) {
	// Expect a panic
	defer func() {
		if err := recover(); err == nil {
			t.Error("Div should have panicked when y==0")
		} else if e, ok := err.(runtime.Error); !ok || e.Error() != divZeroError {
			t.Errorf("Div expected panic: %q, got: %q ", divZeroError, e.Error())
		}
	}()
	q, r := Div(1, 1, 0)
	t.Errorf("undefined q, r = %v, %v calculated when Div should have panicked", q, r)
}

func TestDiv32PanicZero(t *testing.T) {
	// Expect a panic
	defer func() {
		if err := recover(); err == nil {
			t.Error("Div32 should have panicked when y==0")
		} else if e, ok := err.(runtime.Error); !ok || e.Error() != divZeroError {
			t.Errorf("Div32 expected panic: %q, got: %q ", divZeroError, e.Error())
		}
	}()
	q, r := Div32(1, 1, 0)
	t.Errorf("undefined q, r = %v, %v calculated when Div32 should have panicked", q, r)
}

func TestDiv64PanicZero(t *testing.T) {
	// Expect a panic
	defer func() {
		if err := recover(); err == nil {
			t.Error("Div64 should have panicked when y==0")
		} else if e, ok := err.(runtime.Error); !ok || e.Error() != divZeroError {
			t.Errorf("Div64 expected panic: %q, got: %q ", divZeroError, e.Error())
		}
	}()
	q, r := Div64(1, 1, 0)
	t.Errorf("undefined q, r = %v, %v calculated when Div64 should have panicked", q, r)
}

func TestRem32(t *testing.T) {
	// Sanity check: for non-oveflowing dividends, the result is the
	// same as the rem returned by Div32
	hi, lo, y := uint32(510510), uint32(9699690), uint32(510510+1) // ensure hi < y
	for i := 0; i < 1000; i++ {
		r := Rem32(hi, lo, y)
		_, r2 := Div32(hi, lo, y)
		if r != r2 {
			t.Errorf("Rem32(%v, %v, %v) returned %v, but Div32 returned rem %v", hi, lo, y, r, r2)
		}
		y += 13
	}
}

func TestRem32Overflow(t *testing.T) {
	// To trigger a quotient overflow, we need y <= hi
	hi, lo, y := uint32(510510), uint32(9699690), uint32(7)
	for i := 0; i < 1000; i++ {
		r := Rem32(hi, lo, y)
		_, r2 := Div64(0, uint64(hi)<<32|uint64(lo), uint64(y))
		if r != uint32(r2) {
			t.Errorf("Rem32(%v, %v, %v) returned %v, but Div64 returned rem %v", hi, lo, y, r, r2)
		}
		y += 13
	}
}

func TestRem64(t *testing.T) {
	// Sanity check: for non-oveflowing dividends, the result is the
	// same as the rem returned by Div64
	hi, lo, y := uint64(510510), uint64(9699690), uint64(510510+1) // ensure hi < y
	for i := 0; i < 1000; i++ {
		r := Rem64(hi, lo, y)
		_, r2 := Div64(hi, lo, y)
		if r != r2 {
			t.Errorf("Rem64(%v, %v, %v) returned %v, but Div64 returned rem %v", hi, lo, y, r, r2)
		}
		y += 13
	}
}

func TestRem64Overflow(t *testing.T) {
	Rem64Tests := []struct {
		hi, lo, y uint64
		rem       uint64
	}{
		// Testcases computed using Python 3, as:
		//   >>> hi = 42; lo = 1119; y = 42
		//   >>> ((hi<<64)+lo) % y
		{42, 1119, 42, 27},
		{42, 1119, 38, 9},
		{42, 1119, 26, 23},
		{469, 0, 467, 271},
		{469, 0, 113, 58},
		{111111, 111111, 1171, 803},
		{3968194946088682615, 3192705705065114702, 1000037, 56067},
	}

	for _, rt := range Rem64Tests {
		if rt.hi < rt.y {
			t.Fatalf("Rem64(%v, %v, %v) is not a test with quo overflow", rt.hi, rt.lo, rt.y)
		}
		rem := Rem64(rt.hi, rt.lo, rt.y)
		if rem != rt.rem {
			t.Errorf("Rem64(%v, %v, %v) returned %v, wanted %v",
				rt.hi, rt.lo, rt.y, rem, rt.rem)
		}
	}
}

func BenchmarkAdd(b *testing.B) {
	var z, c uint
	for i := 0; i < b.N; i++ {
		z, c = Add(uint(Input), uint(i), c)
	}
	Output = int(z + c)
}

func BenchmarkAdd32(b *testing.B) {
	var z, c uint32
	for i := 0; i < b.N; i++ {
		z, c = Add32(uint32(Input), uint32(i), c)
	}
	Output = int(z + c)
}

func BenchmarkAdd64(b *testing.B) {
	var z, c uint64
	for i := 0; i < b.N; i++ {
		z, c = Add64(uint64(Input), uint64(i), c)
	}
	Output = int(z + c)
}

func BenchmarkAdd64multiple(b *testing.B) {
	var z0 = uint64(Input)
	var z1 = uint64(Input)
	var z2 = uint64(Input)
	var z3 = uint64(Input)
	for i := 0; i < b.N; i++ {
		var c uint64
		z0, c = Add64(z0, uint64(i), c)
		z1, c = Add64(z1, uint64(i), c)
		z2, c = Add64(z2, uint64(i), c)
		z3, _ = Add64(z3, uint64(i), c)
	}
	Output = int(z0 + z1 + z2 + z3)
}

func BenchmarkSub(b *testing.B) {
	var z, c uint
	for i := 0; i < b.N; i++ {
		z, c = Sub(uint(Input), uint(i), c)
	}
	Output = int(z + c)
}

func BenchmarkSub32(b *testing.B) {
	var z, c uint32
	for i := 0; i < b.N; i++ {
		z, c = Sub32(uint32(Input), uint32(i), c)
	}
	Output = int(z + c)
}

func BenchmarkSub64(b *testing.B) {
	var z, c uint64
	for i := 0; i < b.N; i++ {
		z, c = Sub64(uint64(Input), uint64(i), c)
	}
	Output = int(z + c)
}

func BenchmarkSub64multiple(b *testing.B) {
	var z0 = uint64(Input)
	var z1 = uint64(Input)
	var z2 = uint64(Input)
	var z3 = uint64(Input)
	for i := 0; i < b.N; i++ {
		var c uint64
		z0, c = Sub64(z0, uint64(i), c)
		z1, c = Sub64(z1, uint64(i), c)
		z2, c = Sub64(z2, uint64(i), c)
		z3, _ = Sub64(z3, uint64(i), c)
	}
	Output = int(z0 + z1 + z2 + z3)
}

func BenchmarkMul(b *testing.B) {
	var hi, lo uint
	for i := 0; i < b.N; i++ {
		hi, lo = Mul(uint(Input), uint(i))
	}
	Output = int(hi + lo)
}

func BenchmarkMul32(b *testing.B) {
	var hi, lo uint32
	for i := 0; i < b.N; i++ {
		hi, lo = Mul32(uint32(Input), uint32(i))
	}
	Output = int(hi + lo)
}

func BenchmarkMul64(b *testing.B) {
	var hi, lo uint64
	for i := 0; i < b.N; i++ {
		hi, lo = Mul64(uint64(Input), uint64(i))
	}
	Output = int(hi + lo)
}

func BenchmarkDiv(b *testing.B) {
	var q, r uint
	for i := 0; i < b.N; i++ {
		q, r = Div(1, uint(i), uint(Input))
	}
	Output = int(q + r)
}

func BenchmarkDiv32(b *testing.B) {
	var q, r uint32
	for i := 0; i < b.N; i++ {
		q, r = Div32(1, uint32(i), uint32(Input))
	}
	Output = int(q + r)
}

func BenchmarkDiv64(b *testing.B) {
	var q, r uint64
	for i := 0; i < b.N; i++ {
		q, r = Div64(1, uint64(i), uint64(Input))
	}
	Output = int(q + r)
}

// ----------------------------------------------------------------------------
// Testing support

type entry = struct {
	nlz, ntz, pop int
}

// tab contains results for all uint8 values
var tab [256]entry

func init() {
	tab[0] = entry{8, 8, 0}
	for i := 1; i < len(tab); i++ {
		// nlz
		x := i // x != 0
		n := 0
		for x&0x80 == 0 {
			n++
			x <<= 1
		}
		tab[i].nlz = n

		// ntz
		x = i // x != 0
		n = 0
		for x&1 == 0 {
			n++
			x >>= 1
		}
		tab[i].ntz = n

		// pop
		x = i // x != 0
		n = 0
		for x != 0 {
			n += int(x & 1)
			x >>= 1
		}
		tab[i].pop = n
	}
}
