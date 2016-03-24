// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package image_test

import (
	"bufio"
	"fmt"
	"image"
	"image/color"
	"os"
	"testing"

	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
)

type imageTest struct {
	goldenFilename string
	filename       string
	tolerance      int
}

var imageTests = []imageTest{
	{"testdata/video-001.png", "testdata/video-001.png", 0},
	// GIF images are restricted to a 256-color palette and the conversion
	// to GIF loses significant image quality.
	{"testdata/video-001.png", "testdata/video-001.gif", 64 << 8},
	{"testdata/video-001.png", "testdata/video-001.interlaced.gif", 64 << 8},
	{"testdata/video-001.png", "testdata/video-001.5bpp.gif", 128 << 8},
	// JPEG is a lossy format and hence needs a non-zero tolerance.
	{"testdata/video-001.png", "testdata/video-001.jpeg", 8 << 8},
	{"testdata/video-001.png", "testdata/video-001.progressive.jpeg", 8 << 8},
	{"testdata/video-001.221212.png", "testdata/video-001.221212.jpeg", 8 << 8},
	{"testdata/video-001.cmyk.png", "testdata/video-001.cmyk.jpeg", 8 << 8},
	{"testdata/video-001.rgb.png", "testdata/video-001.rgb.jpeg", 8 << 8},
	{"testdata/video-001.progressive.truncated.png", "testdata/video-001.progressive.truncated.jpeg", 8 << 8},
	// Grayscale images.
	{"testdata/video-005.gray.png", "testdata/video-005.gray.jpeg", 8 << 8},
	{"testdata/video-005.gray.png", "testdata/video-005.gray.png", 0},
}

func decode(filename string) (image.Image, string, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()
	return image.Decode(bufio.NewReader(f))
}

func decodeConfig(filename string) (image.Config, string, error) {
	f, err := os.Open(filename)
	if err != nil {
		return image.Config{}, "", err
	}
	defer f.Close()
	return image.DecodeConfig(bufio.NewReader(f))
}

func delta(u0, u1 uint32) int {
	d := int(u0) - int(u1)
	if d < 0 {
		return -d
	}
	return d
}

func withinTolerance(c0, c1 color.Color, tolerance int) bool {
	r0, g0, b0, a0 := c0.RGBA()
	r1, g1, b1, a1 := c1.RGBA()
	r := delta(r0, r1)
	g := delta(g0, g1)
	b := delta(b0, b1)
	a := delta(a0, a1)
	return r <= tolerance && g <= tolerance && b <= tolerance && a <= tolerance
}

func TestDecode(t *testing.T) {
	rgba := func(c color.Color) string {
		r, g, b, a := c.RGBA()
		return fmt.Sprintf("rgba = 0x%04x, 0x%04x, 0x%04x, 0x%04x for %T%v", r, g, b, a, c, c)
	}

	golden := make(map[string]image.Image)
loop:
	for _, it := range imageTests {
		g := golden[it.goldenFilename]
		if g == nil {
			var err error
			g, _, err = decode(it.goldenFilename)
			if err != nil {
				t.Errorf("%s: %v", it.goldenFilename, err)
				continue loop
			}
			golden[it.goldenFilename] = g
		}
		m, imageFormat, err := decode(it.filename)
		if err != nil {
			t.Errorf("%s: %v", it.filename, err)
			continue loop
		}
		b := g.Bounds()
		if !b.Eq(m.Bounds()) {
			t.Errorf("%s: got bounds %v want %v", it.filename, m.Bounds(), b)
			continue loop
		}
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				if !withinTolerance(g.At(x, y), m.At(x, y), it.tolerance) {
					t.Errorf("%s: at (%d, %d):\ngot  %v\nwant %v",
						it.filename, x, y, rgba(m.At(x, y)), rgba(g.At(x, y)))
					continue loop
				}
			}
		}
		if imageFormat == "gif" {
			// Each frame of a GIF can have a frame-local palette override the
			// GIF-global palette. Thus, image.Decode can yield a different ColorModel
			// than image.DecodeConfig.
			continue
		}
		c, _, err := decodeConfig(it.filename)
		if m.ColorModel() != c.ColorModel {
			t.Errorf("%s: color models differ", it.filename)
			continue loop
		}
	}
}
