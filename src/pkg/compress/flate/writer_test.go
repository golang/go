// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"io/ioutil"
	"runtime"
	"testing"
)

func benchmarkEncoder(b *testing.B, level, n int) {
	b.StopTimer()
	b.SetBytes(int64(n))
	buf0, err := ioutil.ReadFile("../testdata/e.txt")
	if err != nil {
		b.Fatal(err)
	}
	buf0 = buf0[:10000]
	buf1 := make([]byte, n)
	for i := 0; i < n; i += len(buf0) {
		copy(buf1[i:], buf0)
	}
	buf0 = nil
	runtime.GC()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		w, err := NewWriter(ioutil.Discard, level)
		if err != nil {
			b.Fatal(err)
		}
		w.Write(buf1)
		w.Close()
	}
}

func BenchmarkEncoderBestSpeed1K(b *testing.B) {
	benchmarkEncoder(b, BestSpeed, 1e4)
}

func BenchmarkEncoderBestSpeed10K(b *testing.B) {
	benchmarkEncoder(b, BestSpeed, 1e5)
}

func BenchmarkEncoderBestSpeed100K(b *testing.B) {
	benchmarkEncoder(b, BestSpeed, 1e6)
}

func BenchmarkEncoderDefaultCompression1K(b *testing.B) {
	benchmarkEncoder(b, DefaultCompression, 1e4)
}

func BenchmarkEncoderDefaultCompression10K(b *testing.B) {
	benchmarkEncoder(b, DefaultCompression, 1e5)
}

func BenchmarkEncoderDefaultCompression100K(b *testing.B) {
	benchmarkEncoder(b, DefaultCompression, 1e6)
}

func BenchmarkEncoderBestCompression1K(b *testing.B) {
	benchmarkEncoder(b, BestCompression, 1e4)
}

func BenchmarkEncoderBestCompression10K(b *testing.B) {
	benchmarkEncoder(b, BestCompression, 1e5)
}

func BenchmarkEncoderBestCompression100K(b *testing.B) {
	benchmarkEncoder(b, BestCompression, 1e6)
}
