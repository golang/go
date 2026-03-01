// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"bytes"
	"io"
	"os"
	"runtime"
	"strings"
	"testing"
)

func TestNlitOutOfRange(t *testing.T) {
	// Trying to decode this bogus flate data, which has a Huffman table
	// with nlit=288, should not panic.
	io.Copy(io.Discard, NewReader(strings.NewReader(
		"\xfc\xfe\x36\xe7\x5e\x1c\xef\xb3\x55\x58\x77\xb6\x56\xb5\x43\xf4"+
			"\x6f\xf2\xd2\xe6\x3d\x99\xa0\x85\x8c\x48\xeb\xf8\xda\x83\x04\x2a"+
			"\x75\xc4\xf8\x0f\x12\x11\xb9\xb4\x4b\x09\xa0\xbe\x8b\x91\x4c")))
}

var suites = []struct{ name, file string }{
	// Digits is the digits of the irrational number e. Its decimal representation
	// does not repeat, but there are only 10 possible digits, so it should be
	// reasonably compressible.
	{"Digits", "../testdata/e.txt"},
	// Newton is Isaac Newtons's educational text on Opticks.
	{"Newton", "../../testdata/Isaac.Newton-Opticks.txt"},
}

func BenchmarkDecode(b *testing.B) {
	doBench(b, func(b *testing.B, buf0 []byte, level, n int) {
		b.ReportAllocs()
		b.StopTimer()
		b.SetBytes(int64(n))

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
		src := bytes.NewReader(buf1)
		dec := NewReader(src)
		runtime.GC()
		b.StartTimer()
		for i := 0; i < b.N; i++ {
			src.Reset(buf1)
			dec.(Resetter).Reset(src, nil)
			io.Copy(io.Discard, dec)
		}
	})
}

var levelTests = []struct {
	name  string
	level int
}{
	{"Huffman", HuffmanOnly},
	{"Speed", BestSpeed},
	{"Default", DefaultCompression},
	{"Compression", BestCompression},
}

var sizes = []struct {
	name string
	n    int
}{
	{"1e4", 1e4},
	{"1e5", 1e5},
	{"1e6", 1e6},
}

func doBench(b *testing.B, f func(b *testing.B, buf []byte, level, n int)) {
	for _, suite := range suites {
		buf, err := os.ReadFile(suite.file)
		if err != nil {
			b.Fatal(err)
		}
		if len(buf) == 0 {
			b.Fatalf("test file %q has no data", suite.file)
		}
		for _, l := range levelTests {
			for _, s := range sizes {
				b.Run(suite.name+"/"+l.name+"/"+s.name, func(b *testing.B) {
					f(b, buf, l.level, s.n)
				})
			}
		}
	}
}
