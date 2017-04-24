// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bits

import (
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
var Input uint64 = deBruijn64

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
	var m uint64 = deBruijn64

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
