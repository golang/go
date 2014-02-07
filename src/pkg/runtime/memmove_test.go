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
