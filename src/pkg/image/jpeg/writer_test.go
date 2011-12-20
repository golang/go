// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jpeg

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"io/ioutil"
	"math/rand"
	"os"
	"testing"
)

var testCase = []struct {
	filename  string
	quality   int
	tolerance int64
}{
	{"../testdata/video-001.png", 1, 24 << 8},
	{"../testdata/video-001.png", 20, 12 << 8},
	{"../testdata/video-001.png", 60, 8 << 8},
	{"../testdata/video-001.png", 80, 6 << 8},
	{"../testdata/video-001.png", 90, 4 << 8},
	{"../testdata/video-001.png", 100, 2 << 8},
}

func delta(u0, u1 uint32) int64 {
	d := int64(u0) - int64(u1)
	if d < 0 {
		return -d
	}
	return d
}

func readPng(filename string) (image.Image, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return png.Decode(f)
}

func TestWriter(t *testing.T) {
	for _, tc := range testCase {
		// Read the image.
		m0, err := readPng(tc.filename)
		if err != nil {
			t.Error(tc.filename, err)
			continue
		}
		// Encode that image as JPEG.
		buf := bytes.NewBuffer(nil)
		err = Encode(buf, m0, &Options{Quality: tc.quality})
		if err != nil {
			t.Error(tc.filename, err)
			continue
		}
		// Decode that JPEG.
		m1, err := Decode(buf)
		if err != nil {
			t.Error(tc.filename, err)
			continue
		}
		// Compute the average delta in RGB space.
		b := m0.Bounds()
		var sum, n int64
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				c0 := m0.At(x, y)
				c1 := m1.At(x, y)
				r0, g0, b0, _ := c0.RGBA()
				r1, g1, b1, _ := c1.RGBA()
				sum += delta(r0, r1)
				sum += delta(g0, g1)
				sum += delta(b0, b1)
				n += 3
			}
		}
		// Compare the average delta to the tolerance level.
		if sum/n > tc.tolerance {
			t.Errorf("%s, quality=%d: average delta is too high", tc.filename, tc.quality)
			continue
		}
	}
}

func BenchmarkEncodeRGBOpaque(b *testing.B) {
	b.StopTimer()
	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	// Set all pixels to 0xFF alpha to force opaque mode.
	bo := img.Bounds()
	rnd := rand.New(rand.NewSource(123))
	for y := bo.Min.Y; y < bo.Max.Y; y++ {
		for x := bo.Min.X; x < bo.Max.X; x++ {
			img.Set(x, y, color.RGBA{
				uint8(rnd.Intn(256)),
				uint8(rnd.Intn(256)),
				uint8(rnd.Intn(256)),
				255})
		}
	}
	if !img.Opaque() {
		b.Fatal("expected image to be opaque")
	}
	b.SetBytes(640 * 480 * 4)
	b.StartTimer()
	options := &Options{Quality: 90}
	for i := 0; i < b.N; i++ {
		Encode(ioutil.Discard, img, options)
	}
}
