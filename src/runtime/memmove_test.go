// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"internal/race"
	"internal/testenv"
	. "runtime"
	"testing"
)

func TestMemmove(t *testing.T) {
	t.Parallel()
	size := 256
	if testing.Short() {
		size = 128 + 16
	}
	src := make([]byte, size)
	dst := make([]byte, size)
	for i := 0; i < size; i++ {
		src[i] = byte(128 + (i & 127))
	}
	for i := 0; i < size; i++ {
		dst[i] = byte(i & 127)
	}
	for n := 0; n <= size; n++ {
		for x := 0; x <= size-n; x++ { // offset in src
			for y := 0; y <= size-n; y++ { // offset in dst
				copy(dst[y:y+n], src[x:x+n])
				for i := 0; i < y; i++ {
					if dst[i] != byte(i&127) {
						t.Fatalf("prefix dst[%d] = %d", i, dst[i])
					}
				}
				for i := y; i < y+n; i++ {
					if dst[i] != byte(128+((i-y+x)&127)) {
						t.Fatalf("copied dst[%d] = %d", i, dst[i])
					}
					dst[i] = byte(i & 127) // reset dst
				}
				for i := y + n; i < size; i++ {
					if dst[i] != byte(i&127) {
						t.Fatalf("suffix dst[%d] = %d", i, dst[i])
					}
				}
			}
		}
	}
}

func TestMemmoveAlias(t *testing.T) {
	t.Parallel()
	size := 256
	if testing.Short() {
		size = 128 + 16
	}
	buf := make([]byte, size)
	for i := 0; i < size; i++ {
		buf[i] = byte(i)
	}
	for n := 0; n <= size; n++ {
		for x := 0; x <= size-n; x++ { // src offset
			for y := 0; y <= size-n; y++ { // dst offset
				copy(buf[y:y+n], buf[x:x+n])
				for i := 0; i < y; i++ {
					if buf[i] != byte(i) {
						t.Fatalf("prefix buf[%d] = %d", i, buf[i])
					}
				}
				for i := y; i < y+n; i++ {
					if buf[i] != byte(i-y+x) {
						t.Fatalf("copied buf[%d] = %d", i, buf[i])
					}
					buf[i] = byte(i) // reset buf
				}
				for i := y + n; i < size; i++ {
					if buf[i] != byte(i) {
						t.Fatalf("suffix buf[%d] = %d", i, buf[i])
					}
				}
			}
		}
	}
}

func TestMemmoveLarge0x180000(t *testing.T) {
	if testing.Short() && testenv.Builder() == "" {
		t.Skip("-short")
	}

	t.Parallel()
	if race.Enabled {
		t.Skip("skipping large memmove test under race detector")
	}
	testSize(t, 0x180000)
}

func TestMemmoveOverlapLarge0x120000(t *testing.T) {
	if testing.Short() && testenv.Builder() == "" {
		t.Skip("-short")
	}

	t.Parallel()
	if race.Enabled {
		t.Skip("skipping large memmove test under race detector")
	}
	testOverlap(t, 0x120000)
}

func testSize(t *testing.T, size int) {
	src := make([]byte, size)
	dst := make([]byte, size)
	_, _ = rand.Read(src)
	_, _ = rand.Read(dst)

	ref := make([]byte, size)
	copyref(ref, dst)

	for n := size - 50; n > 1; n >>= 1 {
		for x := 0; x <= size-n; x = x*7 + 1 { // offset in src
			for y := 0; y <= size-n; y = y*9 + 1 { // offset in dst
				copy(dst[y:y+n], src[x:x+n])
				copyref(ref[y:y+n], src[x:x+n])
				p := cmpb(dst, ref)
				if p >= 0 {
					t.Fatalf("Copy failed, copying from src[%d:%d] to dst[%d:%d].\nOffset %d is different, %v != %v", x, x+n, y, y+n, p, dst[p], ref[p])
				}
			}
		}
	}
}

func testOverlap(t *testing.T, size int) {
	src := make([]byte, size)
	test := make([]byte, size)
	ref := make([]byte, size)
	_, _ = rand.Read(src)

	for n := size - 50; n > 1; n >>= 1 {
		for x := 0; x <= size-n; x = x*7 + 1 { // offset in src
			for y := 0; y <= size-n; y = y*9 + 1 { // offset in dst
				// Reset input
				copyref(test, src)
				copyref(ref, src)
				copy(test[y:y+n], test[x:x+n])
				if y <= x {
					copyref(ref[y:y+n], ref[x:x+n])
				} else {
					copybw(ref[y:y+n], ref[x:x+n])
				}
				p := cmpb(test, ref)
				if p >= 0 {
					t.Fatalf("Copy failed, copying from src[%d:%d] to dst[%d:%d].\nOffset %d is different, %v != %v", x, x+n, y, y+n, p, test[p], ref[p])
				}
			}
		}
	}

}

// Forward copy.
func copyref(dst, src []byte) {
	for i, v := range src {
		dst[i] = v
	}
}

// Backwards copy
func copybw(dst, src []byte) {
	if len(src) == 0 {
		return
	}
	for i := len(src) - 1; i >= 0; i-- {
		dst[i] = src[i]
	}
}

// Returns offset of difference
func matchLen(a, b []byte, max int) int {
	a = a[:max]
	b = b[:max]
	for i, av := range a {
		if b[i] != av {
			return i
		}
	}
	return max
}

func cmpb(a, b []byte) int {
	l := matchLen(a, b, len(a))
	if l == len(a) {
		return -1
	}
	return l
}

func benchmarkSizes(b *testing.B, sizes []int, fn func(b *testing.B, n int)) {
	for _, n := range sizes {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			b.SetBytes(int64(n))
			fn(b, n)
		})
	}
}

var bufSizes = []int{
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	32, 64, 128, 256, 512, 1024, 2048, 4096,
}

func BenchmarkMemmove(b *testing.B) {
	benchmarkSizes(b, bufSizes, func(b *testing.B, n int) {
		x := make([]byte, n)
		y := make([]byte, n)
		for i := 0; i < b.N; i++ {
			copy(x, y)
		}
	})
}

func BenchmarkMemmoveUnalignedDst(b *testing.B) {
	benchmarkSizes(b, bufSizes, func(b *testing.B, n int) {
		x := make([]byte, n+1)
		y := make([]byte, n)
		for i := 0; i < b.N; i++ {
			copy(x[1:], y)
		}
	})
}

func BenchmarkMemmoveUnalignedSrc(b *testing.B) {
	benchmarkSizes(b, bufSizes, func(b *testing.B, n int) {
		x := make([]byte, n)
		y := make([]byte, n+1)
		for i := 0; i < b.N; i++ {
			copy(x, y[1:])
		}
	})
}

func TestMemclr(t *testing.T) {
	size := 512
	if testing.Short() {
		size = 128 + 16
	}
	mem := make([]byte, size)
	for i := 0; i < size; i++ {
		mem[i] = 0xee
	}
	for n := 0; n < size; n++ {
		for x := 0; x <= size-n; x++ { // offset in mem
			MemclrBytes(mem[x : x+n])
			for i := 0; i < x; i++ {
				if mem[i] != 0xee {
					t.Fatalf("overwrite prefix mem[%d] = %d", i, mem[i])
				}
			}
			for i := x; i < x+n; i++ {
				if mem[i] != 0 {
					t.Fatalf("failed clear mem[%d] = %d", i, mem[i])
				}
				mem[i] = 0xee
			}
			for i := x + n; i < size; i++ {
				if mem[i] != 0xee {
					t.Fatalf("overwrite suffix mem[%d] = %d", i, mem[i])
				}
			}
		}
	}
}

func BenchmarkMemclr(b *testing.B) {
	for _, n := range []int{5, 16, 64, 256, 4096, 65536} {
		x := make([]byte, n)
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				MemclrBytes(x)
			}
		})
	}
	for _, m := range []int{1, 4, 8, 16, 64} {
		x := make([]byte, m<<20)
		b.Run(fmt.Sprint(m, "M"), func(b *testing.B) {
			b.SetBytes(int64(m << 20))
			for i := 0; i < b.N; i++ {
				MemclrBytes(x)
			}
		})
	}
}

func BenchmarkGoMemclr(b *testing.B) {
	benchmarkSizes(b, []int{5, 16, 64, 256}, func(b *testing.B, n int) {
		x := make([]byte, n)
		for i := 0; i < b.N; i++ {
			for j := range x {
				x[j] = 0
			}
		}
	})
}

func BenchmarkClearFat8(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var x [8 / 4]uint32
		_ = x
	}
}
func BenchmarkClearFat12(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var x [12 / 4]uint32
		_ = x
	}
}
func BenchmarkClearFat16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var x [16 / 4]uint32
		_ = x
	}
}
func BenchmarkClearFat24(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var x [24 / 4]uint32
		_ = x
	}
}
func BenchmarkClearFat32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var x [32 / 4]uint32
		_ = x
	}
}
func BenchmarkClearFat40(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var x [40 / 4]uint32
		_ = x
	}
}
func BenchmarkClearFat48(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var x [48 / 4]uint32
		_ = x
	}
}
func BenchmarkClearFat56(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var x [56 / 4]uint32
		_ = x
	}
}
func BenchmarkClearFat64(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var x [64 / 4]uint32
		_ = x
	}
}
func BenchmarkClearFat128(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var x [128 / 4]uint32
		_ = x
	}
}
func BenchmarkClearFat256(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var x [256 / 4]uint32
		_ = x
	}
}
func BenchmarkClearFat512(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var x [512 / 4]uint32
		_ = x
	}
}
func BenchmarkClearFat1024(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var x [1024 / 4]uint32
		_ = x
	}
}

func BenchmarkCopyFat8(b *testing.B) {
	var x [8 / 4]uint32
	for i := 0; i < b.N; i++ {
		y := x
		_ = y
	}
}
func BenchmarkCopyFat12(b *testing.B) {
	var x [12 / 4]uint32
	for i := 0; i < b.N; i++ {
		y := x
		_ = y
	}
}
func BenchmarkCopyFat16(b *testing.B) {
	var x [16 / 4]uint32
	for i := 0; i < b.N; i++ {
		y := x
		_ = y
	}
}
func BenchmarkCopyFat24(b *testing.B) {
	var x [24 / 4]uint32
	for i := 0; i < b.N; i++ {
		y := x
		_ = y
	}
}
func BenchmarkCopyFat32(b *testing.B) {
	var x [32 / 4]uint32
	for i := 0; i < b.N; i++ {
		y := x
		_ = y
	}
}
func BenchmarkCopyFat64(b *testing.B) {
	var x [64 / 4]uint32
	for i := 0; i < b.N; i++ {
		y := x
		_ = y
	}
}
func BenchmarkCopyFat128(b *testing.B) {
	var x [128 / 4]uint32
	for i := 0; i < b.N; i++ {
		y := x
		_ = y
	}
}
func BenchmarkCopyFat256(b *testing.B) {
	var x [256 / 4]uint32
	for i := 0; i < b.N; i++ {
		y := x
		_ = y
	}
}
func BenchmarkCopyFat512(b *testing.B) {
	var x [512 / 4]uint32
	for i := 0; i < b.N; i++ {
		y := x
		_ = y
	}
}
func BenchmarkCopyFat520(b *testing.B) {
	var x [520 / 4]uint32
	for i := 0; i < b.N; i++ {
		y := x
		_ = y
	}
}
func BenchmarkCopyFat1024(b *testing.B) {
	var x [1024 / 4]uint32
	for i := 0; i < b.N; i++ {
		y := x
		_ = y
	}
}

func BenchmarkIssue18740(b *testing.B) {
	// This tests that memmove uses one 4-byte load/store to move 4 bytes.
	// It used to do 2 2-byte load/stores, which leads to a pipeline stall
	// when we try to read the result with one 4-byte load.
	var buf [4]byte
	for j := 0; j < b.N; j++ {
		s := uint32(0)
		for i := 0; i < 4096; i += 4 {
			copy(buf[:], g[i:])
			s += binary.LittleEndian.Uint32(buf[:])
		}
		sink = uint64(s)
	}
}

// TODO: 2 byte and 8 byte benchmarks also.

var g [4096]byte
