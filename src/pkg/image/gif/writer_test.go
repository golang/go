// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gif

import (
	"bytes"
	"image"
	"image/color"
	_ "image/png"
	"io/ioutil"
	"math/rand"
	"os"
	"testing"
)

func readImg(filename string) (image.Image, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	m, _, err := image.Decode(f)
	return m, err
}

func readGIF(filename string) (*GIF, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return DecodeAll(f)
}

func delta(u0, u1 uint32) int64 {
	d := int64(u0) - int64(u1)
	if d < 0 {
		return -d
	}
	return d
}

// averageDelta returns the average delta in RGB space. The two images must
// have the same bounds.
func averageDelta(m0, m1 image.Image) int64 {
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
	return sum / n
}

var testCase = []struct {
	filename  string
	tolerance int64
}{
	{"../testdata/video-001.png", 1 << 12},
	{"../testdata/video-001.gif", 0},
	{"../testdata/video-001.interlaced.gif", 0},
}

func TestWriter(t *testing.T) {
	for _, tc := range testCase {
		m0, err := readImg(tc.filename)
		if err != nil {
			t.Error(tc.filename, err)
			continue
		}
		var buf bytes.Buffer
		err = Encode(&buf, m0, nil)
		if err != nil {
			t.Error(tc.filename, err)
			continue
		}
		m1, err := Decode(&buf)
		if err != nil {
			t.Error(tc.filename, err)
			continue
		}
		if m0.Bounds() != m1.Bounds() {
			t.Errorf("%s, bounds differ: %v and %v", tc.filename, m0.Bounds(), m1.Bounds())
			continue
		}
		// Compare the average delta to the tolerance level.
		avgDelta := averageDelta(m0, m1)
		if avgDelta > tc.tolerance {
			t.Errorf("%s: average delta is too high. expected: %d, got %d", tc.filename, tc.tolerance, avgDelta)
			continue
		}
	}
}

var frames = []string{
	"../testdata/video-001.gif",
	"../testdata/video-005.gray.gif",
}

func TestEncodeAll(t *testing.T) {
	g0 := &GIF{
		Image:     make([]*image.Paletted, len(frames)),
		Delay:     make([]int, len(frames)),
		LoopCount: 5,
	}
	for i, f := range frames {
		m, err := readGIF(f)
		if err != nil {
			t.Error(f, err)
		}
		g0.Image[i] = m.Image[0]
	}
	var buf bytes.Buffer
	if err := EncodeAll(&buf, g0); err != nil {
		t.Fatal("EncodeAll:", err)
	}
	g1, err := DecodeAll(&buf)
	if err != nil {
		t.Fatal("DecodeAll:", err)
	}
	if g0.LoopCount != g1.LoopCount {
		t.Errorf("loop counts differ: %d and %d", g0.LoopCount, g1.LoopCount)
	}
	for i := range g0.Image {
		m0, m1 := g0.Image[i], g1.Image[i]
		if m0.Bounds() != m1.Bounds() {
			t.Errorf("%s, bounds differ: %v and %v", frames[i], m0.Bounds(), m1.Bounds())
		}
		d0, d1 := g0.Delay[i], g1.Delay[i]
		if d0 != d1 {
			t.Errorf("%s: delay values differ: %d and %d", frames[i], d0, d1)
		}
	}

	g1.Delay = make([]int, 1)
	if err := EncodeAll(ioutil.Discard, g1); err == nil {
		t.Error("expected error from mismatched delay and image slice lengths")
	}
	if err := EncodeAll(ioutil.Discard, &GIF{}); err == nil {
		t.Error("expected error from providing empty gif")
	}
}

func BenchmarkEncode(b *testing.B) {
	b.StopTimer()

	bo := image.Rect(0, 0, 640, 480)
	rnd := rand.New(rand.NewSource(123))

	// Restrict to a 256-color paletted image to avoid quantization path.
	palette := make(color.Palette, 256)
	for i := range palette {
		palette[i] = color.RGBA{
			uint8(rnd.Intn(256)),
			uint8(rnd.Intn(256)),
			uint8(rnd.Intn(256)),
			255,
		}
	}
	img := image.NewPaletted(image.Rect(0, 0, 640, 480), palette)
	for y := bo.Min.Y; y < bo.Max.Y; y++ {
		for x := bo.Min.X; x < bo.Max.X; x++ {
			img.Set(x, y, palette[rnd.Intn(256)])
		}
	}

	b.SetBytes(640 * 480 * 4)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Encode(ioutil.Discard, img, nil)
	}
}

func BenchmarkQuantizedEncode(b *testing.B) {
	b.StopTimer()
	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	bo := img.Bounds()
	rnd := rand.New(rand.NewSource(123))
	for y := bo.Min.Y; y < bo.Max.Y; y++ {
		for x := bo.Min.X; x < bo.Max.X; x++ {
			img.SetRGBA(x, y, color.RGBA{
				uint8(rnd.Intn(256)),
				uint8(rnd.Intn(256)),
				uint8(rnd.Intn(256)),
				255,
			})
		}
	}
	b.SetBytes(640 * 480 * 4)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Encode(ioutil.Discard, img, nil)
	}
}
