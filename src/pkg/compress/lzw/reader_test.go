// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzw

import (
	"bytes"
	"io"
	"io/ioutil"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

type lzwTest struct {
	desc       string
	raw        string
	compressed string
	err        error
}

var lzwTests = []lzwTest{
	{
		"empty;LSB;8",
		"",
		"\x01\x01",
		nil,
	},
	{
		"empty;MSB;8",
		"",
		"\x80\x80",
		nil,
	},
	{
		"tobe;LSB;7",
		"TOBEORNOTTOBEORTOBEORNOT",
		"\x54\x4f\x42\x45\x4f\x52\x4e\x4f\x54\x82\x84\x86\x8b\x85\x87\x89\x81",
		nil,
	},
	{
		"tobe;LSB;8",
		"TOBEORNOTTOBEORTOBEORNOT",
		"\x54\x9e\x08\x29\xf2\x44\x8a\x93\x27\x54\x04\x12\x34\xb8\xb0\xe0\xc1\x84\x01\x01",
		nil,
	},
	{
		"tobe;MSB;7",
		"TOBEORNOTTOBEORTOBEORNOT",
		"\x54\x4f\x42\x45\x4f\x52\x4e\x4f\x54\x82\x84\x86\x8b\x85\x87\x89\x81",
		nil,
	},
	{
		"tobe;MSB;8",
		"TOBEORNOTTOBEORTOBEORNOT",
		"\x2a\x13\xc8\x44\x52\x79\x48\x9c\x4f\x2a\x40\xa0\x90\x68\x5c\x16\x0f\x09\x80\x80",
		nil,
	},
	{
		"tobe-truncated;LSB;8",
		"TOBEORNOTTOBEORTOBEORNOT",
		"\x54\x9e\x08\x29\xf2\x44\x8a\x93\x27\x54\x04",
		io.ErrUnexpectedEOF,
	},
	// This example comes from http://en.wikipedia.org/wiki/Graphics_Interchange_Format.
	{
		"gif;LSB;8",
		"\x28\xff\xff\xff\x28\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff",
		"\x00\x51\xfc\x1b\x28\x70\xa0\xc1\x83\x01\x01",
		nil,
	},
	// This example comes from http://compgroups.net/comp.lang.ruby/Decompressing-LZW-compression-from-PDF-file
	{
		"pdf;MSB;8",
		"-----A---B",
		"\x80\x0b\x60\x50\x22\x0c\x0c\x85\x01",
		nil,
	},
}

func TestReader(t *testing.T) {
	var b bytes.Buffer
	for _, tt := range lzwTests {
		d := strings.Split(tt.desc, ";")
		var order Order
		switch d[1] {
		case "LSB":
			order = LSB
		case "MSB":
			order = MSB
		default:
			t.Errorf("%s: bad order %q", tt.desc, d[1])
		}
		litWidth, _ := strconv.Atoi(d[2])
		rc := NewReader(strings.NewReader(tt.compressed), order, litWidth)
		defer rc.Close()
		b.Reset()
		n, err := io.Copy(&b, rc)
		if err != nil {
			if err != tt.err {
				t.Errorf("%s: io.Copy: %v want %v", tt.desc, err, tt.err)
			}
			continue
		}
		s := b.String()
		if s != tt.raw {
			t.Errorf("%s: got %d-byte %q want %d-byte %q", tt.desc, n, s, len(tt.raw), tt.raw)
		}
	}
}

func benchmarkDecoder(b *testing.B, n int) {
	b.StopTimer()
	b.SetBytes(int64(n))
	buf0, _ := ioutil.ReadFile("../testdata/e.txt")
	buf0 = buf0[:10000]
	compressed := new(bytes.Buffer)
	w := NewWriter(compressed, LSB, 8)
	for i := 0; i < n; i += len(buf0) {
		io.Copy(w, bytes.NewBuffer(buf0))
	}
	w.Close()
	buf1 := compressed.Bytes()
	buf0, compressed, w = nil, nil, nil
	runtime.GC()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		io.Copy(ioutil.Discard, NewReader(bytes.NewBuffer(buf1), LSB, 8))
	}
}

func BenchmarkDecoder1e4(b *testing.B) {
	benchmarkDecoder(b, 1e4)
}

func BenchmarkDecoder1e5(b *testing.B) {
	benchmarkDecoder(b, 1e5)
}

func BenchmarkDecoder1e6(b *testing.B) {
	benchmarkDecoder(b, 1e6)
}
