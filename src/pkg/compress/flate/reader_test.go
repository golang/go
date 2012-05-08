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

const (
	digits = iota
	twain
)

var testfiles = []string{
	// Digits is the digits of the irrational number e. Its decimal representation
	// does not repeat, but there are only 10 posible digits, so it should be
	// reasonably compressible.
	//
	// TODO(nigeltao): e.txt is only 10K long, so when benchmarking 100K or 1000K
	// of input, the digits are just repeated from the beginning, and flate can
	// trivially compress this as a length/distance copy operation. Thus,
	// BenchmarkDecodeDigitsXxx1e6 is essentially just measuring the speed of the
	// forwardCopy implementation, but isn't particularly representative of real
	// usage. The TODO is to replace e.txt with 100K digits, not just 10K digits,
	// since that's larger than the windowSize 1<<15 (= 32768).
	digits: "../testdata/e.txt",
	// Twain is Project Gutenberg's edition of Mark Twain's classic English novel.
	twain: "../testdata/Mark.Twain-Tom.Sawyer.txt",
}

func benchmarkDecode(b *testing.B, testfile, level, n int) {
	b.StopTimer()
	b.SetBytes(int64(n))
	buf0, err := ioutil.ReadFile(testfiles[testfile])
	if err != nil {
		b.Fatal(err)
	}
	if len(buf0) == 0 {
		b.Fatalf("test file %q has no data", testfiles[testfile])
	}
	compressed := new(bytes.Buffer)
	w, err := NewWriter(compressed, level)
	if err != nil {
		b.Fatal(err)
	}
	for i := 0; i < n; i += len(buf0) {
		if len(buf0) > n-i {
			buf0 = buf0[:n-i]
		}
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

// These short names are so that gofmt doesn't break the BenchmarkXxx function
// bodies below over multiple lines.
const (
	speed    = BestSpeed
	default_ = DefaultCompression
	compress = BestCompression
)

func BenchmarkDecodeDigitsSpeed1e4(b *testing.B)    { benchmarkDecode(b, digits, speed, 1e4) }
func BenchmarkDecodeDigitsSpeed1e5(b *testing.B)    { benchmarkDecode(b, digits, speed, 1e5) }
func BenchmarkDecodeDigitsSpeed1e6(b *testing.B)    { benchmarkDecode(b, digits, speed, 1e6) }
func BenchmarkDecodeDigitsDefault1e4(b *testing.B)  { benchmarkDecode(b, digits, default_, 1e4) }
func BenchmarkDecodeDigitsDefault1e5(b *testing.B)  { benchmarkDecode(b, digits, default_, 1e5) }
func BenchmarkDecodeDigitsDefault1e6(b *testing.B)  { benchmarkDecode(b, digits, default_, 1e6) }
func BenchmarkDecodeDigitsCompress1e4(b *testing.B) { benchmarkDecode(b, digits, compress, 1e4) }
func BenchmarkDecodeDigitsCompress1e5(b *testing.B) { benchmarkDecode(b, digits, compress, 1e5) }
func BenchmarkDecodeDigitsCompress1e6(b *testing.B) { benchmarkDecode(b, digits, compress, 1e6) }
func BenchmarkDecodeTwainSpeed1e4(b *testing.B)     { benchmarkDecode(b, twain, speed, 1e4) }
func BenchmarkDecodeTwainSpeed1e5(b *testing.B)     { benchmarkDecode(b, twain, speed, 1e5) }
func BenchmarkDecodeTwainSpeed1e6(b *testing.B)     { benchmarkDecode(b, twain, speed, 1e6) }
func BenchmarkDecodeTwainDefault1e4(b *testing.B)   { benchmarkDecode(b, twain, default_, 1e4) }
func BenchmarkDecodeTwainDefault1e5(b *testing.B)   { benchmarkDecode(b, twain, default_, 1e5) }
func BenchmarkDecodeTwainDefault1e6(b *testing.B)   { benchmarkDecode(b, twain, default_, 1e6) }
func BenchmarkDecodeTwainCompress1e4(b *testing.B)  { benchmarkDecode(b, twain, compress, 1e4) }
func BenchmarkDecodeTwainCompress1e5(b *testing.B)  { benchmarkDecode(b, twain, compress, 1e5) }
func BenchmarkDecodeTwainCompress1e6(b *testing.B)  { benchmarkDecode(b, twain, compress, 1e6) }
