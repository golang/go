// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package image_test

import (
	"bufio"
	"image"
	"os"
	"testing"

	// TODO(nigeltao): implement bmp decoder.
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	_ "image/tiff"
)

const goldenFile = "testdata/video-001.png"

type imageTest struct {
	filename  string
	tolerance int
}

var imageTests = []imageTest{
	//{"testdata/video-001.bmp", 0},
	// GIF images are restricted to a 256-color palette and the conversion
	// to GIF loses significant image quality.
	{"testdata/video-001.gif", 64 << 8},
	// JPEG is a lossy format and hence needs a non-zero tolerance.
	{"testdata/video-001.jpeg", 8 << 8},
	{"testdata/video-001.png", 0},
	{"testdata/video-001.tiff", 0},
}

func decode(filename string) (image.Image, string, os.Error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()
	return image.Decode(bufio.NewReader(f))
}

func delta(u0, u1 uint32) int {
	d := int(u0) - int(u1)
	if d < 0 {
		return -d
	}
	return d
}

func withinTolerance(c0, c1 image.Color, tolerance int) bool {
	r0, g0, b0, a0 := c0.RGBA()
	r1, g1, b1, a1 := c1.RGBA()
	r := delta(r0, r1)
	g := delta(g0, g1)
	b := delta(b0, b1)
	a := delta(a0, a1)
	return r <= tolerance && g <= tolerance && b <= tolerance && a <= tolerance
}

func TestDecode(t *testing.T) {
	golden, _, err := decode(goldenFile)
	if err != nil {
		t.Errorf("%s: %v", goldenFile, err)
	}
loop:
	for _, it := range imageTests {
		m, _, err := decode(it.filename)
		if err != nil {
			t.Errorf("%s: %v", it.filename, err)
			continue loop
		}
		b := golden.Bounds()
		if !b.Eq(m.Bounds()) {
			t.Errorf("%s: want bounds %v got %v", it.filename, b, m.Bounds())
			continue loop
		}
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				if !withinTolerance(golden.At(x, y), m.At(x, y), it.tolerance) {
					t.Errorf("%s: at (%d, %d), want %v got %v", it.filename, x, y, golden.At(x, y), m.At(x, y))
					continue loop
				}
			}
		}
	}
}
