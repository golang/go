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
	"sync/atomic"
	"testing"
	"unsafe"
)

func TestMemmove(t *testing.T) {
	if *flagQuick {
		t.Skip("-quick")
	}
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
	if *flagQuick {
		t.Skip("-quick")
	}
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

// Ensure that memmove writes pointers atomically, so the GC won't
// observe a partially updated pointer.
func TestMemmoveAtomicity(t *testing.T) {
	if race.Enabled {
		t.Skip("skip under the race detector -- this test is intentionally racy")
	}

	var x int

	for _, backward := range []bool{true, false} {
		for _, n := range []int{3, 4, 5, 6, 7, 8, 9, 10, 15, 25, 49} {
			n := n

			// test copying [N]*int.
			sz := uintptr(n * PtrSize)
			name := fmt.Sprint(sz)
			if backward {
				name += "-backward"
			} else {
				name += "-forward"
			}
			t.Run(name, func(t *testing.T) {
				// Use overlapping src and dst to force forward/backward copy.
				var s [100]*int
				src := s[n-1 : 2*n-1]
				dst := s[:n]
				if backward {
					src, dst = dst, src
				}
				for i := range src {
					src[i] = &x
				}
				for i := range dst {
					dst[i] = nil
				}

				var ready uint32
				go func() {
					sp := unsafe.Pointer(&src[0])
					dp := unsafe.Pointer(&dst[0])
					atomic.StoreUint32(&ready, 1)
					for i := 0; i < 10000; i++ {
						Memmove(dp, sp, sz)
						MemclrNoHeapPointers(dp, sz)
					}
					atomic.StoreUint32(&ready, 2)
				}()

				for atomic.LoadUint32(&ready) == 0 {
					Gosched()
				}

				for atomic.LoadUint32(&ready) != 2 {
					for i := range dst {
						p := dst[i]
						if p != nil && p != &x {
							t.Fatalf("got partially updated pointer %p at dst[%d], want either nil or %p", p, i, &x)
						}
					}
				}
			})
		}
	}
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
var bufSizesOverlap = []int{
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

func BenchmarkMemmoveOverlap(b *testing.B) {
	benchmarkSizes(b, bufSizesOverlap, func(b *testing.B, n int) {
		x := make([]byte, n+16)
		for i := 0; i < b.N; i++ {
			copy(x[16:n+16], x[:n])
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

func BenchmarkMemmoveUnalignedDstOverlap(b *testing.B) {
	benchmarkSizes(b, bufSizesOverlap, func(b *testing.B, n int) {
		x := make([]byte, n+16)
		for i := 0; i < b.N; i++ {
			copy(x[16:n+16], x[1:n+1])
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

func BenchmarkMemmoveUnalignedSrcOverlap(b *testing.B) {
	benchmarkSizes(b, bufSizesOverlap, func(b *testing.B, n int) {
		x := make([]byte, n+1)
		for i := 0; i < b.N; i++ {
			copy(x[1:n+1], x[:n])
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

// BenchmarkIssue18740 ensures that memmove uses 4 and 8 byte load/store to move 4 and 8 bytes.
// It used to do 2 2-byte load/stores, which leads to a pipeline stall
// when we try to read the result with one 4-byte load.
func BenchmarkIssue18740(b *testing.B) {
	benchmarks := []struct {
		name  string
		nbyte int
		f     func([]byte) uint64
	}{
		{"2byte", 2, func(buf []byte) uint64 { return uint64(binary.LittleEndian.Uint16(buf)) }},
		{"4byte", 4, func(buf []byte) uint64 { return uint64(binary.LittleEndian.Uint32(buf)) }},
		{"8byte", 8, func(buf []byte) uint64 { return binary.LittleEndian.Uint64(buf) }},
	}

	var g [4096]byte
	for _, bm := range benchmarks {
		buf := make([]byte, bm.nbyte)
		b.Run(bm.name, func(b *testing.B) {
			for j := 0; j < b.N; j++ {
				for i := 0; i < 4096; i += bm.nbyte {
					copy(buf[:], g[i:])
					sink += bm.f(buf[:])
				}
			}
		})
	}
}
