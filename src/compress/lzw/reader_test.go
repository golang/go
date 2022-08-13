// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzw

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"os"
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
	// This example comes from https://en.wikipedia.org/wiki/Graphics_Interchange_Format.
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
		s := b.String()
		if err != nil {
			if err != tt.err {
				t.Errorf("%s: io.Copy: %v want %v", tt.desc, err, tt.err)
			}
			if err == io.ErrUnexpectedEOF {
				// Even if the input is truncated, we should still return the
				// partial decoded result.
				if n == 0 || !strings.HasPrefix(tt.raw, s) {
					t.Errorf("got %d bytes (%q), want a non-empty prefix of %q", n, s, tt.raw)
				}
			}
			continue
		}
		if s != tt.raw {
			t.Errorf("%s: got %d-byte %q want %d-byte %q", tt.desc, n, s, len(tt.raw), tt.raw)
		}
	}
}

func TestReaderReset(t *testing.T) {
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
		b1 := b.Bytes()
		if err != nil {
			if err != tt.err {
				t.Errorf("%s: io.Copy: %v want %v", tt.desc, err, tt.err)
			}
			if err == io.ErrUnexpectedEOF {
				// Even if the input is truncated, we should still return the
				// partial decoded result.
				if n == 0 || !strings.HasPrefix(tt.raw, b.String()) {
					t.Errorf("got %d bytes (%q), want a non-empty prefix of %q", n, b.String(), tt.raw)
				}
			}
			continue
		}

		b.Reset()
		rc.(*Reader).Reset(strings.NewReader(tt.compressed), order, litWidth)
		n, err = io.Copy(&b, rc)
		b2 := b.Bytes()
		if err != nil {
			t.Errorf("%s: io.Copy: %v want %v", tt.desc, err, nil)
			continue
		}
		if !bytes.Equal(b1, b2) {
			t.Errorf("bytes read were not the same")
		}
	}
}

type devZero struct{}

func (devZero) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = 0
	}
	return len(p), nil
}

func TestHiCodeDoesNotOverflow(t *testing.T) {
	r := NewReader(devZero{}, LSB, 8)
	d := r.(*Reader)
	buf := make([]byte, 1024)
	oldHi := uint16(0)
	for i := 0; i < 100; i++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			t.Fatalf("i=%d: %v", i, err)
		}
		// The hi code should never decrease.
		if d.hi < oldHi {
			t.Fatalf("i=%d: hi=%d decreased from previous value %d", i, d.hi, oldHi)
		}
		oldHi = d.hi
	}
}

// TestNoLongerSavingPriorExpansions tests the decoder state when codes other
// than clear codes continue to be seen after decoder.hi and decoder.width
// reach their maximum values (4095 and 12), i.e. after we no longer save prior
// expansions. In particular, it tests seeing the highest possible code, 4095.
func TestNoLongerSavingPriorExpansions(t *testing.T) {
	// Iterations is used to calculate how many input bits are needed to get
	// the decoder.hi and decoder.width values up to their maximum.
	iterations := []struct {
		width, n int
	}{
		// The final term is 257, not 256, as NewReader initializes d.hi to
		// d.clear+1 and the clear code is 256.
		{9, 512 - 257},
		{10, 1024 - 512},
		{11, 2048 - 1024},
		{12, 4096 - 2048},
	}
	nCodes, nBits := 0, 0
	for _, e := range iterations {
		nCodes += e.n
		nBits += e.n * e.width
	}
	if nCodes != 3839 {
		t.Fatalf("nCodes: got %v, want %v", nCodes, 3839)
	}
	if nBits != 43255 {
		t.Fatalf("nBits: got %v, want %v", nBits, 43255)
	}

	// Construct our input of 43255 zero bits (which gets d.hi and d.width up
	// to 4095 and 12), followed by 0xfff (4095) as 12 bits, followed by 0x101
	// (EOF) as 12 bits.
	//
	// 43255 = 5406*8 + 7, and codes are read in LSB order. The final bytes are
	// therefore:
	//
	// xwwwwwww xxxxxxxx yyyyyxxx zyyyyyyy
	// 10000000 11111111 00001111 00001000
	//
	// or split out:
	//
	// .0000000 ........ ........ ........   w = 0x000
	// 1....... 11111111 .....111 ........   x = 0xfff
	// ........ ........ 00001... .0001000   y = 0x101
	//
	// The 12 'w' bits (not all are shown) form the 3839'th code, with value
	// 0x000. Just after decoder.read returns that code, d.hi == 4095 and
	// d.last == 0.
	//
	// The 12 'x' bits form the 3840'th code, with value 0xfff or 4095. Just
	// after decoder.read returns that code, d.hi == 4095 and d.last ==
	// decoderInvalidCode.
	//
	// The 12 'y' bits form the 3841'st code, with value 0x101, the EOF code.
	//
	// The 'z' bit is unused.
	in := make([]byte, 5406)
	in = append(in, 0x80, 0xff, 0x0f, 0x08)

	r := NewReader(bytes.NewReader(in), LSB, 8)
	nDecoded, err := io.Copy(io.Discard, r)
	if err != nil {
		t.Fatalf("Copy: %v", err)
	}
	// nDecoded should be 3841: 3839 literal codes and then 2 decoded bytes
	// from 1 non-literal code. The EOF code contributes 0 decoded bytes.
	if nDecoded != int64(nCodes+2) {
		t.Fatalf("nDecoded: got %v, want %v", nDecoded, nCodes+2)
	}
}

func BenchmarkDecoder(b *testing.B) {
	buf, err := os.ReadFile("../testdata/e.txt")
	if err != nil {
		b.Fatal(err)
	}
	if len(buf) == 0 {
		b.Fatalf("test file has no data")
	}

	getInputBuf := func(buf []byte, n int) []byte {
		compressed := new(bytes.Buffer)
		w := NewWriter(compressed, LSB, 8)
		for i := 0; i < n; i += len(buf) {
			if len(buf) > n-i {
				buf = buf[:n-i]
			}
			w.Write(buf)
		}
		w.Close()
		return compressed.Bytes()
	}

	for e := 4; e <= 6; e++ {
		n := int(math.Pow10(e))
		b.Run(fmt.Sprint("1e", e), func(b *testing.B) {
			b.StopTimer()
			b.SetBytes(int64(n))
			buf1 := getInputBuf(buf, n)
			runtime.GC()
			b.StartTimer()
			for i := 0; i < b.N; i++ {
				io.Copy(io.Discard, NewReader(bytes.NewReader(buf1), LSB, 8))
			}
		})
		b.Run(fmt.Sprint("1e-Reuse", e), func(b *testing.B) {
			b.StopTimer()
			b.SetBytes(int64(n))
			buf1 := getInputBuf(buf, n)
			runtime.GC()
			b.StartTimer()
			r := NewReader(bytes.NewReader(buf1), LSB, 8)
			for i := 0; i < b.N; i++ {
				io.Copy(io.Discard, r)
				r.Close()
				r.(*Reader).Reset(bytes.NewReader(buf1), LSB, 8)
			}
		})
	}
}
