// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package runtime_test

import "testing"

const N = 20

func BenchmarkAppend(b *testing.B) {
	b.StopTimer()
	x := make([]int, 0, N)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		x = x[0:0]
		for j := 0; j < N; j++ {
			x = append(x, j)
		}
	}
}

func benchmarkAppendBytes(b *testing.B, length int) {
	b.StopTimer()
	x := make([]byte, 0, N)
	y := make([]byte, length)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		x = x[0:0]
		x = append(x, y...)
	}
}

func BenchmarkAppend1Byte(b *testing.B) {
	benchmarkAppendBytes(b, 1)
}

func BenchmarkAppend4Bytes(b *testing.B) {
	benchmarkAppendBytes(b, 4)
}

func BenchmarkAppend7Bytes(b *testing.B) {
	benchmarkAppendBytes(b, 7)
}

func BenchmarkAppend8Bytes(b *testing.B) {
	benchmarkAppendBytes(b, 8)
}

func BenchmarkAppend15Bytes(b *testing.B) {
	benchmarkAppendBytes(b, 15)
}

func BenchmarkAppend16Bytes(b *testing.B) {
	benchmarkAppendBytes(b, 16)
}

func BenchmarkAppend32Bytes(b *testing.B) {
	benchmarkAppendBytes(b, 32)
}

func benchmarkAppendStr(b *testing.B, str string) {
	b.StopTimer()
	x := make([]byte, 0, N)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		x = x[0:0]
		x = append(x, str...)
	}
}

func BenchmarkAppendStr1Byte(b *testing.B) {
	benchmarkAppendStr(b, "1")
}

func BenchmarkAppendStr4Bytes(b *testing.B) {
	benchmarkAppendStr(b, "1234")
}

func BenchmarkAppendStr8Bytes(b *testing.B) {
	benchmarkAppendStr(b, "12345678")
}

func BenchmarkAppendStr16Bytes(b *testing.B) {
	benchmarkAppendStr(b, "1234567890123456")
}

func BenchmarkAppendStr32Bytes(b *testing.B) {
	benchmarkAppendStr(b, "12345678901234567890123456789012")
}

func BenchmarkAppendSpecialCase(b *testing.B) {
	b.StopTimer()
	x := make([]int, 0, N)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		x = x[0:0]
		for j := 0; j < N; j++ {
			if len(x) < cap(x) {
				x = x[:len(x)+1]
				x[len(x)-1] = j
			} else {
				x = append(x, j)
			}
		}
	}
}

var x []int

func f() int {
	x[:1][0] = 3
	return 2
}

func TestSideEffectOrder(t *testing.T) {
	x = make([]int, 0, 10)
	x = append(x, 1, f())
	if x[0] != 1 || x[1] != 2 {
		t.Error("append failed: ", x[0], x[1])
	}
}

func TestAppendOverlap(t *testing.T) {
	x := []byte("1234")
	x = append(x[1:], x...) // p > q in runtime·appendslice.
	got := string(x)
	want := "2341234"
	if got != want {
		t.Errorf("overlap failed: got %q want %q", got, want)
	}
}

func benchmarkCopySlice(b *testing.B, l int) {
	s := make([]byte, l)
	buf := make([]byte, 4096)
	var n int
	for i := 0; i < b.N; i++ {
		n = copy(buf, s)
	}
	b.SetBytes(int64(n))
}

func benchmarkCopyStr(b *testing.B, l int) {
	s := string(make([]byte, l))
	buf := make([]byte, 4096)
	var n int
	for i := 0; i < b.N; i++ {
		n = copy(buf, s)
	}
	b.SetBytes(int64(n))
}

func BenchmarkCopy1Byte(b *testing.B)    { benchmarkCopySlice(b, 1) }
func BenchmarkCopy2Byte(b *testing.B)    { benchmarkCopySlice(b, 2) }
func BenchmarkCopy4Byte(b *testing.B)    { benchmarkCopySlice(b, 4) }
func BenchmarkCopy8Byte(b *testing.B)    { benchmarkCopySlice(b, 8) }
func BenchmarkCopy12Byte(b *testing.B)   { benchmarkCopySlice(b, 12) }
func BenchmarkCopy16Byte(b *testing.B)   { benchmarkCopySlice(b, 16) }
func BenchmarkCopy32Byte(b *testing.B)   { benchmarkCopySlice(b, 32) }
func BenchmarkCopy128Byte(b *testing.B)  { benchmarkCopySlice(b, 128) }
func BenchmarkCopy1024Byte(b *testing.B) { benchmarkCopySlice(b, 1024) }

func BenchmarkCopy1String(b *testing.B)    { benchmarkCopyStr(b, 1) }
func BenchmarkCopy2String(b *testing.B)    { benchmarkCopyStr(b, 2) }
func BenchmarkCopy4String(b *testing.B)    { benchmarkCopyStr(b, 4) }
func BenchmarkCopy8String(b *testing.B)    { benchmarkCopyStr(b, 8) }
func BenchmarkCopy12String(b *testing.B)   { benchmarkCopyStr(b, 12) }
func BenchmarkCopy16String(b *testing.B)   { benchmarkCopyStr(b, 16) }
func BenchmarkCopy32String(b *testing.B)   { benchmarkCopyStr(b, 32) }
func BenchmarkCopy128String(b *testing.B)  { benchmarkCopyStr(b, 128) }
func BenchmarkCopy1024String(b *testing.B) { benchmarkCopyStr(b, 1024) }
