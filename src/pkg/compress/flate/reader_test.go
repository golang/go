// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"bytes"
	"io"
	"io/ioutil"
	"runtime"
	"strings"
	"testing"
)

func TestNlitOutOfRange(t *testing.T) {
	// Trying to decode this bogus flate data, which has a Huffman table
	// with nlit=288, should not panic.
	io.Copy(ioutil.Discard, NewReader(strings.NewReader(
		"\xfc\xfe\x36\xe7\x5e\x1c\xef\xb3\x55\x58\x77\xb6\x56\xb5\x43\xf4"+
			"\x6f\xf2\xd2\xe6\x3d\x99\xa0\x85\x8c\x48\xeb\xf8\xda\x83\x04\x2a"+
			"\x75\xc4\xf8\x0f\x12\x11\xb9\xb4\x4b\x09\xa0\xbe\x8b\x91\x4c")))
}

const (
	digits = iota
	twain
)

var testfiles = []string{
	// Digits is the digits of the irrational number e. Its decimal representation
	// does not repeat, but there are only 10 posible digits, so it should be
	// reasonably compressible.
	digits: "../testdata/e.txt",
	// Twain is Project Gutenberg's edition of Mark Twain's classic English novel.
	twain: "../testdata/Mark.Twain-Tom.Sawyer.txt",
}

func benchmarkDecode(b *testing.B, testfile, level, n int) {
	b.ReportAllocs()
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
		io.Copy(w, bytes.NewReader(buf0))
	}
	w.Close()
	buf1 := compressed.Bytes()
	buf0, compressed, w = nil, nil, nil
	runtime.GC()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		io.Copy(ioutil.Discard, NewReader(bytes.NewReader(buf1)))
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
