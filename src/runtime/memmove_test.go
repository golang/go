// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	. "runtime"
	"testing"
)

func TestMemmove(t *testing.T) {
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
func BenchmarkCopyFat1024(b *testing.B) {
	var x [1024 / 4]uint32
	for i := 0; i < b.N; i++ {
		y := x
		_ = y
	}
}
