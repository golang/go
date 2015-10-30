// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
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

func bmMemmove(b *testing.B, n int) {
	x := make([]byte, n)
	y := make([]byte, n)
	b.SetBytes(int64(n))
	for i := 0; i < b.N; i++ {
		copy(x, y)
	}
}

func BenchmarkMemmove0(b *testing.B)    { bmMemmove(b, 0) }
func BenchmarkMemmove1(b *testing.B)    { bmMemmove(b, 1) }
func BenchmarkMemmove2(b *testing.B)    { bmMemmove(b, 2) }
func BenchmarkMemmove3(b *testing.B)    { bmMemmove(b, 3) }
func BenchmarkMemmove4(b *testing.B)    { bmMemmove(b, 4) }
func BenchmarkMemmove5(b *testing.B)    { bmMemmove(b, 5) }
func BenchmarkMemmove6(b *testing.B)    { bmMemmove(b, 6) }
func BenchmarkMemmove7(b *testing.B)    { bmMemmove(b, 7) }
func BenchmarkMemmove8(b *testing.B)    { bmMemmove(b, 8) }
func BenchmarkMemmove9(b *testing.B)    { bmMemmove(b, 9) }
func BenchmarkMemmove10(b *testing.B)   { bmMemmove(b, 10) }
func BenchmarkMemmove11(b *testing.B)   { bmMemmove(b, 11) }
func BenchmarkMemmove12(b *testing.B)   { bmMemmove(b, 12) }
func BenchmarkMemmove13(b *testing.B)   { bmMemmove(b, 13) }
func BenchmarkMemmove14(b *testing.B)   { bmMemmove(b, 14) }
func BenchmarkMemmove15(b *testing.B)   { bmMemmove(b, 15) }
func BenchmarkMemmove16(b *testing.B)   { bmMemmove(b, 16) }
func BenchmarkMemmove32(b *testing.B)   { bmMemmove(b, 32) }
func BenchmarkMemmove64(b *testing.B)   { bmMemmove(b, 64) }
func BenchmarkMemmove128(b *testing.B)  { bmMemmove(b, 128) }
func BenchmarkMemmove256(b *testing.B)  { bmMemmove(b, 256) }
func BenchmarkMemmove512(b *testing.B)  { bmMemmove(b, 512) }
func BenchmarkMemmove1024(b *testing.B) { bmMemmove(b, 1024) }
func BenchmarkMemmove2048(b *testing.B) { bmMemmove(b, 2048) }
func BenchmarkMemmove4096(b *testing.B) { bmMemmove(b, 4096) }

func bmMemmoveUnaligned(b *testing.B, n int) {
	x := make([]byte, n+1)
	y := make([]byte, n)
	b.SetBytes(int64(n))
	for i := 0; i < b.N; i++ {
		copy(x[1:], y)
	}
}

func BenchmarkMemmoveUnaligned0(b *testing.B)    { bmMemmoveUnaligned(b, 0) }
func BenchmarkMemmoveUnaligned1(b *testing.B)    { bmMemmoveUnaligned(b, 1) }
func BenchmarkMemmoveUnaligned2(b *testing.B)    { bmMemmoveUnaligned(b, 2) }
func BenchmarkMemmoveUnaligned3(b *testing.B)    { bmMemmoveUnaligned(b, 3) }
func BenchmarkMemmoveUnaligned4(b *testing.B)    { bmMemmoveUnaligned(b, 4) }
func BenchmarkMemmoveUnaligned5(b *testing.B)    { bmMemmoveUnaligned(b, 5) }
func BenchmarkMemmoveUnaligned6(b *testing.B)    { bmMemmoveUnaligned(b, 6) }
func BenchmarkMemmoveUnaligned7(b *testing.B)    { bmMemmoveUnaligned(b, 7) }
func BenchmarkMemmoveUnaligned8(b *testing.B)    { bmMemmoveUnaligned(b, 8) }
func BenchmarkMemmoveUnaligned9(b *testing.B)    { bmMemmoveUnaligned(b, 9) }
func BenchmarkMemmoveUnaligned10(b *testing.B)   { bmMemmoveUnaligned(b, 10) }
func BenchmarkMemmoveUnaligned11(b *testing.B)   { bmMemmoveUnaligned(b, 11) }
func BenchmarkMemmoveUnaligned12(b *testing.B)   { bmMemmoveUnaligned(b, 12) }
func BenchmarkMemmoveUnaligned13(b *testing.B)   { bmMemmoveUnaligned(b, 13) }
func BenchmarkMemmoveUnaligned14(b *testing.B)   { bmMemmoveUnaligned(b, 14) }
func BenchmarkMemmoveUnaligned15(b *testing.B)   { bmMemmoveUnaligned(b, 15) }
func BenchmarkMemmoveUnaligned16(b *testing.B)   { bmMemmoveUnaligned(b, 16) }
func BenchmarkMemmoveUnaligned32(b *testing.B)   { bmMemmoveUnaligned(b, 32) }
func BenchmarkMemmoveUnaligned64(b *testing.B)   { bmMemmoveUnaligned(b, 64) }
func BenchmarkMemmoveUnaligned128(b *testing.B)  { bmMemmoveUnaligned(b, 128) }
func BenchmarkMemmoveUnaligned256(b *testing.B)  { bmMemmoveUnaligned(b, 256) }
func BenchmarkMemmoveUnaligned512(b *testing.B)  { bmMemmoveUnaligned(b, 512) }
func BenchmarkMemmoveUnaligned1024(b *testing.B) { bmMemmoveUnaligned(b, 1024) }
func BenchmarkMemmoveUnaligned2048(b *testing.B) { bmMemmoveUnaligned(b, 2048) }
func BenchmarkMemmoveUnaligned4096(b *testing.B) { bmMemmoveUnaligned(b, 4096) }

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

func bmMemclr(b *testing.B, n int) {
	x := make([]byte, n)
	b.SetBytes(int64(n))
	for i := 0; i < b.N; i++ {
		MemclrBytes(x)
	}
}
func BenchmarkMemclr5(b *testing.B)     { bmMemclr(b, 5) }
func BenchmarkMemclr16(b *testing.B)    { bmMemclr(b, 16) }
func BenchmarkMemclr64(b *testing.B)    { bmMemclr(b, 64) }
func BenchmarkMemclr256(b *testing.B)   { bmMemclr(b, 256) }
func BenchmarkMemclr4096(b *testing.B)  { bmMemclr(b, 4096) }
func BenchmarkMemclr65536(b *testing.B) { bmMemclr(b, 65536) }
func BenchmarkMemclr1M(b *testing.B)    { bmMemclr(b, 1<<20) }
func BenchmarkMemclr4M(b *testing.B)    { bmMemclr(b, 4<<20) }
func BenchmarkMemclr8M(b *testing.B)    { bmMemclr(b, 8<<20) }
func BenchmarkMemclr16M(b *testing.B)   { bmMemclr(b, 16<<20) }
func BenchmarkMemclr64M(b *testing.B)   { bmMemclr(b, 64<<20) }

func bmGoMemclr(b *testing.B, n int) {
	x := make([]byte, n)
	b.SetBytes(int64(n))
	for i := 0; i < b.N; i++ {
		for j := range x {
			x[j] = 0
		}
	}
}
func BenchmarkGoMemclr5(b *testing.B)   { bmGoMemclr(b, 5) }
func BenchmarkGoMemclr16(b *testing.B)  { bmGoMemclr(b, 16) }
func BenchmarkGoMemclr64(b *testing.B)  { bmGoMemclr(b, 64) }
func BenchmarkGoMemclr256(b *testing.B) { bmGoMemclr(b, 256) }

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
