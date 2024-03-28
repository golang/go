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
				clear(dst)

				var ready atomic.Uint32
				go func() {
					sp := unsafe.Pointer(&src[0])
					dp := unsafe.Pointer(&dst[0])
					ready.Store(1)
					for i := 0; i < 10000; i++ {
						Memmove(dp, sp, sz)
						MemclrNoHeapPointers(dp, sz)
					}
					ready.Store(2)
				}()

				for ready.Load() == 0 {
					Gosched()
				}

				for ready.Load() != 2 {
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

func BenchmarkMemmoveUnalignedSrcDst(b *testing.B) {
	for _, n := range []int{16, 64, 256, 4096, 65536} {
		buf := make([]byte, (n+8)*2)
		x := buf[:len(buf)/2]
		y := buf[len(buf)/2:]
		for _, off := range []int{0, 1, 4, 7} {
			b.Run(fmt.Sprint("f_", n, off), func(b *testing.B) {
				b.SetBytes(int64(n))
				for i := 0; i < b.N; i++ {
					copy(x[off:n+off], y[off:n+off])
				}
			})

			b.Run(fmt.Sprint("b_", n, off), func(b *testing.B) {
				b.SetBytes(int64(n))
				for i := 0; i < b.N; i++ {
					copy(y[off:n+off], x[off:n+off])
				}
			})
		}
	}
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

func BenchmarkMemclrUnaligned(b *testing.B) {
	for _, off := range []int{0, 1, 4, 7} {
		for _, n := range []int{5, 16, 64, 256, 4096, 65536} {
			x := make([]byte, n+off)
			b.Run(fmt.Sprint(off, n), func(b *testing.B) {
				b.SetBytes(int64(n))
				for i := 0; i < b.N; i++ {
					MemclrBytes(x[off:])
				}
			})
		}
	}

	for _, off := range []int{0, 1, 4, 7} {
		for _, m := range []int{1, 4, 8, 16, 64} {
			x := make([]byte, (m<<20)+off)
			b.Run(fmt.Sprint(off, m, "M"), func(b *testing.B) {
				b.SetBytes(int64(m << 20))
				for i := 0; i < b.N; i++ {
					MemclrBytes(x[off:])
				}
			})
		}
	}
}

func BenchmarkGoMemclr(b *testing.B) {
	benchmarkSizes(b, []int{5, 16, 64, 256}, func(b *testing.B, n int) {
		x := make([]byte, n)
		for i := 0; i < b.N; i++ {
			clear(x)
		}
	})
}

func BenchmarkMemclrRange(b *testing.B) {
	type RunData struct {
		data []int
	}

	benchSizes := []RunData{
		{[]int{1043, 1078, 1894, 1582, 1044, 1165, 1467, 1100, 1919, 1562, 1932, 1645,
			1412, 1038, 1576, 1200, 1029, 1336, 1095, 1494, 1350, 1025, 1502, 1548, 1316, 1296,
			1868, 1639, 1546, 1626, 1642, 1308, 1726, 1665, 1678, 1187, 1515, 1598, 1353, 1237,
			1977, 1452, 2012, 1914, 1514, 1136, 1975, 1618, 1536, 1695, 1600, 1733, 1392, 1099,
			1358, 1996, 1224, 1783, 1197, 1838, 1460, 1556, 1554, 2020}}, // 1kb-2kb
		{[]int{3964, 5139, 6573, 7775, 6553, 2413, 3466, 5394, 2469, 7336, 7091, 6745,
			4028, 5643, 6164, 3475, 4138, 6908, 7559, 3335, 5660, 4122, 3945, 2082, 7564, 6584,
			5111, 2288, 6789, 2797, 4928, 7986, 5163, 5447, 2999, 4968, 3174, 3202, 7908, 8137,
			4735, 6161, 4646, 7592, 3083, 5329, 3687, 2754, 3599, 7231, 6455, 2549, 8063, 2189,
			7121, 5048, 4277, 6626, 6306, 2815, 7473, 3963, 7549, 7255}}, // 2kb-8kb
		{[]int{16304, 15936, 15760, 4736, 9136, 11184, 10160, 5952, 14560, 15744,
			6624, 5872, 13088, 14656, 14192, 10304, 4112, 10384, 9344, 4496, 11392, 7024,
			5200, 10064, 14784, 5808, 13504, 10480, 8512, 4896, 13264, 5600}}, // 4kb-16kb
		{[]int{164576, 233136, 220224, 183280, 214112, 217248, 228560, 201728}}, // 128kb-256kb
	}

	for _, t := range benchSizes {
		total := 0
		minLen := 0
		maxLen := 0

		for _, clrLen := range t.data {
			maxLen = max(maxLen, clrLen)
			if clrLen < minLen || minLen == 0 {
				minLen = clrLen
			}
			total += clrLen
		}
		buffer := make([]byte, maxLen)

		text := ""
		if minLen >= (1 << 20) {
			text = fmt.Sprint(minLen>>20, "M ", (maxLen+(1<<20-1))>>20, "M")
		} else if minLen >= (1 << 10) {
			text = fmt.Sprint(minLen>>10, "K ", (maxLen+(1<<10-1))>>10, "K")
		} else {
			text = fmt.Sprint(minLen, " ", maxLen)
		}
		b.Run(text, func(b *testing.B) {
			b.SetBytes(int64(total))
			for i := 0; i < b.N; i++ {
				for _, clrLen := range t.data {
					MemclrBytes(buffer[:clrLen])
				}
			}
		})
	}
}

func BenchmarkClearFat7(b *testing.B) {
	p := new([7]byte)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [7]byte{}
	}
}

func BenchmarkClearFat8(b *testing.B) {
	p := new([8 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [8 / 4]uint32{}
	}
}

func BenchmarkClearFat11(b *testing.B) {
	p := new([11]byte)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [11]byte{}
	}
}

func BenchmarkClearFat12(b *testing.B) {
	p := new([12 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [12 / 4]uint32{}
	}
}

func BenchmarkClearFat13(b *testing.B) {
	p := new([13]byte)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [13]byte{}
	}
}

func BenchmarkClearFat14(b *testing.B) {
	p := new([14]byte)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [14]byte{}
	}
}

func BenchmarkClearFat15(b *testing.B) {
	p := new([15]byte)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [15]byte{}
	}
}

func BenchmarkClearFat16(b *testing.B) {
	p := new([16 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [16 / 4]uint32{}
	}
}

func BenchmarkClearFat24(b *testing.B) {
	p := new([24 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [24 / 4]uint32{}
	}
}

func BenchmarkClearFat32(b *testing.B) {
	p := new([32 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [32 / 4]uint32{}
	}
}

func BenchmarkClearFat40(b *testing.B) {
	p := new([40 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [40 / 4]uint32{}
	}
}

func BenchmarkClearFat48(b *testing.B) {
	p := new([48 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [48 / 4]uint32{}
	}
}

func BenchmarkClearFat56(b *testing.B) {
	p := new([56 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [56 / 4]uint32{}
	}
}

func BenchmarkClearFat64(b *testing.B) {
	p := new([64 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [64 / 4]uint32{}
	}
}

func BenchmarkClearFat72(b *testing.B) {
	p := new([72 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [72 / 4]uint32{}
	}
}

func BenchmarkClearFat128(b *testing.B) {
	p := new([128 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [128 / 4]uint32{}
	}
}

func BenchmarkClearFat256(b *testing.B) {
	p := new([256 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [256 / 4]uint32{}
	}
}

func BenchmarkClearFat512(b *testing.B) {
	p := new([512 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [512 / 4]uint32{}
	}
}

func BenchmarkClearFat1024(b *testing.B) {
	p := new([1024 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [1024 / 4]uint32{}
	}
}

func BenchmarkClearFat1032(b *testing.B) {
	p := new([1032 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [1032 / 4]uint32{}
	}
}

func BenchmarkClearFat1040(b *testing.B) {
	p := new([1040 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = [1040 / 4]uint32{}
	}
}

func BenchmarkCopyFat7(b *testing.B) {
	var x [7]byte
	p := new([7]byte)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat8(b *testing.B) {
	var x [8 / 4]uint32
	p := new([8 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat11(b *testing.B) {
	var x [11]byte
	p := new([11]byte)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat12(b *testing.B) {
	var x [12 / 4]uint32
	p := new([12 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat13(b *testing.B) {
	var x [13]byte
	p := new([13]byte)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat14(b *testing.B) {
	var x [14]byte
	p := new([14]byte)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat15(b *testing.B) {
	var x [15]byte
	p := new([15]byte)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat16(b *testing.B) {
	var x [16 / 4]uint32
	p := new([16 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat24(b *testing.B) {
	var x [24 / 4]uint32
	p := new([24 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat32(b *testing.B) {
	var x [32 / 4]uint32
	p := new([32 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat64(b *testing.B) {
	var x [64 / 4]uint32
	p := new([64 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat72(b *testing.B) {
	var x [72 / 4]uint32
	p := new([72 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat128(b *testing.B) {
	var x [128 / 4]uint32
	p := new([128 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat256(b *testing.B) {
	var x [256 / 4]uint32
	p := new([256 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat512(b *testing.B) {
	var x [512 / 4]uint32
	p := new([512 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat520(b *testing.B) {
	var x [520 / 4]uint32
	p := new([520 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat1024(b *testing.B) {
	var x [1024 / 4]uint32
	p := new([1024 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat1032(b *testing.B) {
	var x [1032 / 4]uint32
	p := new([1032 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
	}
}

func BenchmarkCopyFat1040(b *testing.B) {
	var x [1040 / 4]uint32
	p := new([1040 / 4]uint32)
	Escape(p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		*p = x
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

var memclrSink []int8

func BenchmarkMemclrKnownSize1(b *testing.B) {
	var x [1]int8

	b.SetBytes(1)
	for i := 0; i < b.N; i++ {
		for a := range x {
			x[a] = 0
		}
	}

	memclrSink = x[:]
}
func BenchmarkMemclrKnownSize2(b *testing.B) {
	var x [2]int8

	b.SetBytes(2)
	for i := 0; i < b.N; i++ {
		for a := range x {
			x[a] = 0
		}
	}

	memclrSink = x[:]
}
func BenchmarkMemclrKnownSize4(b *testing.B) {
	var x [4]int8

	b.SetBytes(4)
	for i := 0; i < b.N; i++ {
		for a := range x {
			x[a] = 0
		}
	}

	memclrSink = x[:]
}
func BenchmarkMemclrKnownSize8(b *testing.B) {
	var x [8]int8

	b.SetBytes(8)
	for i := 0; i < b.N; i++ {
		for a := range x {
			x[a] = 0
		}
	}

	memclrSink = x[:]
}
func BenchmarkMemclrKnownSize16(b *testing.B) {
	var x [16]int8

	b.SetBytes(16)
	for i := 0; i < b.N; i++ {
		for a := range x {
			x[a] = 0
		}
	}

	memclrSink = x[:]
}
func BenchmarkMemclrKnownSize32(b *testing.B) {
	var x [32]int8

	b.SetBytes(32)
	for i := 0; i < b.N; i++ {
		for a := range x {
			x[a] = 0
		}
	}

	memclrSink = x[:]
}
func BenchmarkMemclrKnownSize64(b *testing.B) {
	var x [64]int8

	b.SetBytes(64)
	for i := 0; i < b.N; i++ {
		for a := range x {
			x[a] = 0
		}
	}

	memclrSink = x[:]
}
func BenchmarkMemclrKnownSize112(b *testing.B) {
	var x [112]int8

	b.SetBytes(112)
	for i := 0; i < b.N; i++ {
		for a := range x {
			x[a] = 0
		}
	}

	memclrSink = x[:]
}

func BenchmarkMemclrKnownSize128(b *testing.B) {
	var x [128]int8

	b.SetBytes(128)
	for i := 0; i < b.N; i++ {
		for a := range x {
			x[a] = 0
		}
	}

	memclrSink = x[:]
}

func BenchmarkMemclrKnownSize192(b *testing.B) {
	var x [192]int8

	b.SetBytes(192)
	for i := 0; i < b.N; i++ {
		for a := range x {
			x[a] = 0
		}
	}

	memclrSink = x[:]
}

func BenchmarkMemclrKnownSize248(b *testing.B) {
	var x [248]int8

	b.SetBytes(248)
	for i := 0; i < b.N; i++ {
		for a := range x {
			x[a] = 0
		}
	}

	memclrSink = x[:]
}

func BenchmarkMemclrKnownSize256(b *testing.B) {
	var x [256]int8

	b.SetBytes(256)
	for i := 0; i < b.N; i++ {
		for a := range x {
			x[a] = 0
		}
	}

	memclrSink = x[:]
}
func BenchmarkMemclrKnownSize512(b *testing.B) {
	var x [512]int8

	b.SetBytes(512)
	for i := 0; i < b.N; i++ {
		for a := range x {
			x[a] = 0
		}
	}

	memclrSink = x[:]
}
func BenchmarkMemclrKnownSize1024(b *testing.B) {
	var x [1024]int8

	b.SetBytes(1024)
	for i := 0; i < b.N; i++ {
		for a := range x {
			x[a] = 0
		}
	}

	memclrSink = x[:]
}
func BenchmarkMemclrKnownSize4096(b *testing.B) {
	var x [4096]int8

	b.SetBytes(4096)
	for i := 0; i < b.N; i++ {
		for a := range x {
			x[a] = 0
		}
	}

	memclrSink = x[:]
}
func BenchmarkMemclrKnownSize512KiB(b *testing.B) {
	var x [524288]int8

	b.SetBytes(524288)
	for i := 0; i < b.N; i++ {
		for a := range x {
			x[a] = 0
		}
	}

	memclrSink = x[:]
}
