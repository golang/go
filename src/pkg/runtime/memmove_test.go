// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
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

func bmMemmove(n int, b *testing.B) {
	x := make([]byte, n)
	y := make([]byte, n)
	b.SetBytes(int64(n))
	for i := 0; i < b.N; i++ {
		copy(x, y)
	}
}

func BenchmarkMemmove0(b *testing.B)    { bmMemmove(0, b) }
func BenchmarkMemmove1(b *testing.B)    { bmMemmove(1, b) }
func BenchmarkMemmove2(b *testing.B)    { bmMemmove(2, b) }
func BenchmarkMemmove3(b *testing.B)    { bmMemmove(3, b) }
func BenchmarkMemmove4(b *testing.B)    { bmMemmove(4, b) }
func BenchmarkMemmove5(b *testing.B)    { bmMemmove(5, b) }
func BenchmarkMemmove6(b *testing.B)    { bmMemmove(6, b) }
func BenchmarkMemmove7(b *testing.B)    { bmMemmove(7, b) }
func BenchmarkMemmove8(b *testing.B)    { bmMemmove(8, b) }
func BenchmarkMemmove9(b *testing.B)    { bmMemmove(9, b) }
func BenchmarkMemmove10(b *testing.B)   { bmMemmove(10, b) }
func BenchmarkMemmove11(b *testing.B)   { bmMemmove(11, b) }
func BenchmarkMemmove12(b *testing.B)   { bmMemmove(12, b) }
func BenchmarkMemmove13(b *testing.B)   { bmMemmove(13, b) }
func BenchmarkMemmove14(b *testing.B)   { bmMemmove(14, b) }
func BenchmarkMemmove15(b *testing.B)   { bmMemmove(15, b) }
func BenchmarkMemmove16(b *testing.B)   { bmMemmove(16, b) }
func BenchmarkMemmove32(b *testing.B)   { bmMemmove(32, b) }
func BenchmarkMemmove64(b *testing.B)   { bmMemmove(64, b) }
func BenchmarkMemmove128(b *testing.B)  { bmMemmove(128, b) }
func BenchmarkMemmove256(b *testing.B)  { bmMemmove(256, b) }
func BenchmarkMemmove512(b *testing.B)  { bmMemmove(512, b) }
func BenchmarkMemmove1024(b *testing.B) { bmMemmove(1024, b) }
func BenchmarkMemmove2048(b *testing.B) { bmMemmove(2048, b) }
func BenchmarkMemmove4096(b *testing.B) { bmMemmove(4096, b) }
