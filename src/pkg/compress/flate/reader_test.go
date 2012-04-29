// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"bytes"
	"io"
	"io/ioutil"
	"runtime"
	"testing"
)

func benchmarkDecoder(b *testing.B, level, n int) {
	b.StopTimer()
	b.SetBytes(int64(n))
	buf0, err := ioutil.ReadFile("../testdata/e.txt")
	if err != nil {
		b.Fatal(err)
	}
	buf0 = buf0[:10000]
	compressed := new(bytes.Buffer)
	w, err := NewWriter(compressed, level)
	if err != nil {
		b.Fatal(err)
	}
	for i := 0; i < n; i += len(buf0) {
		io.Copy(w, bytes.NewBuffer(buf0))
	}
	w.Close()
	buf1 := compressed.Bytes()
	buf0, compressed, w = nil, nil, nil
	runtime.GC()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		io.Copy(ioutil.Discard, NewReader(bytes.NewBuffer(buf1)))
	}
}

func BenchmarkDecoderBestSpeed1K(b *testing.B) {
	benchmarkDecoder(b, BestSpeed, 1e4)
}

func BenchmarkDecoderBestSpeed10K(b *testing.B) {
	benchmarkDecoder(b, BestSpeed, 1e5)
}

func BenchmarkDecoderBestSpeed100K(b *testing.B) {
	benchmarkDecoder(b, BestSpeed, 1e6)
}

func BenchmarkDecoderDefaultCompression1K(b *testing.B) {
	benchmarkDecoder(b, DefaultCompression, 1e4)
}

func BenchmarkDecoderDefaultCompression10K(b *testing.B) {
	benchmarkDecoder(b, DefaultCompression, 1e5)
}

func BenchmarkDecoderDefaultCompression100K(b *testing.B) {
	benchmarkDecoder(b, DefaultCompression, 1e6)
}

func BenchmarkDecoderBestCompression1K(b *testing.B) {
	benchmarkDecoder(b, BestCompression, 1e4)
}

func BenchmarkDecoderBestCompression10K(b *testing.B) {
	benchmarkDecoder(b, BestCompression, 1e5)
}

func BenchmarkDecoderBestCompression100K(b *testing.B) {
	benchmarkDecoder(b, BestCompression, 1e6)
}
