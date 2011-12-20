// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"image"
	"io/ioutil"
	"os"
	"strings"
	"testing"
)

// Read makes *buffer implements io.Reader, so that we can pass one to Decode.
func (*buffer) Read([]byte) (int, error) {
	panic("unimplemented")
}

// TestNoRPS tries to decode an image that has no RowsPerStrip tag.
// The tag is mandatory according to the spec but some software omits
// it in the case of a single strip.
func TestNoRPS(t *testing.T) {
	f, err := os.Open("testdata/no_rps.tiff")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	_, err = Decode(f)
	if err != nil {
		t.Fatal(err)
	}
}

// TestUnpackBits tests the decoding of PackBits-encoded data.
func TestUnpackBits(t *testing.T) {
	var unpackBitsTests = []struct {
		compressed   string
		uncompressed string
	}{{
		// Example data from Wikipedia.
		"\xfe\xaa\x02\x80\x00\x2a\xfd\xaa\x03\x80\x00\x2a\x22\xf7\xaa",
		"\xaa\xaa\xaa\x80\x00\x2a\xaa\xaa\xaa\xaa\x80\x00\x2a\x22\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa",
	}}
	for _, u := range unpackBitsTests {
		buf, err := unpackBits(strings.NewReader(u.compressed))
		if err != nil {
			t.Fatal(err)
		}
		if string(buf) != u.uncompressed {
			t.Fatalf("unpackBits: want %x, got %x", u.uncompressed, buf)
		}
	}
}

// TestDecompress tests that decoding some TIFF images that use different
// compression formats result in the same pixel data.
func TestDecompress(t *testing.T) {
	var decompressTests = []string{
		"bw-uncompressed.tiff",
		"bw-deflate.tiff",
		"bw-packbits.tiff",
	}
	var img0 image.Image
	for _, name := range decompressTests {
		f, err := os.Open("testdata/" + name)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		if img0 == nil {
			img0, err = Decode(f)
			if err != nil {
				t.Fatalf("decoding %s: %v", name, err)
			}
			continue
		}

		img1, err := Decode(f)
		if err != nil {
			t.Fatalf("decoding %s: %v", name, err)
		}
		b := img1.Bounds()
		// Compare images.
		if !b.Eq(img0.Bounds()) {
			t.Fatalf("wrong image size: want %s, got %s", img0.Bounds(), b)
		}
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				c0 := img0.At(x, y)
				c1 := img1.At(x, y)
				r0, g0, b0, a0 := c0.RGBA()
				r1, g1, b1, a1 := c1.RGBA()
				if r0 != r1 || g0 != g1 || b0 != b1 || a0 != a1 {
					t.Fatalf("pixel at (%d, %d) has wrong color: want %v, got %v", x, y, c0, c1)
				}
			}
		}
	}
}

const filename = "testdata/video-001-uncompressed.tiff"

// BenchmarkDecode benchmarks the decoding of an image.
func BenchmarkDecode(b *testing.B) {
	b.StopTimer()
	contents, err := ioutil.ReadFile(filename)
	if err != nil {
		panic(err)
	}
	r := &buffer{buf: contents}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		_, err := Decode(r)
		if err != nil {
			b.Fatal("Decode:", err)
		}
	}
}
