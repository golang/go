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

func BenchmarkAppend8Bytes(b *testing.B) {
	benchmarkAppendBytes(b, 8)
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
