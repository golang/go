// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package png

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"testing"
)

func diff(m0, m1 image.Image) error {
	b0, b1 := m0.Bounds(), m1.Bounds()
	if !b0.Size().Eq(b1.Size()) {
		return fmt.Errorf("dimensions differ: %v vs %v", b0, b1)
	}
	dx := b1.Min.X - b0.Min.X
	dy := b1.Min.Y - b0.Min.Y
	for y := b0.Min.Y; y < b0.Max.Y; y++ {
		for x := b0.Min.X; x < b0.Max.X; x++ {
			c0 := m0.At(x, y)
			c1 := m1.At(x+dx, y+dy)
			r0, g0, b0, a0 := c0.RGBA()
			r1, g1, b1, a1 := c1.RGBA()
			if r0 != r1 || g0 != g1 || b0 != b1 || a0 != a1 {
				return fmt.Errorf("colors differ at (%d, %d): %v vs %v", x, y, c0, c1)
			}
		}
	}
	return nil
}

func encodeDecode(m image.Image) (image.Image, error) {
	var b bytes.Buffer
	err := Encode(&b, m)
	if err != nil {
		return nil, err
	}
	return Decode(&b)
}

func TestWriter(t *testing.T) {
	// The filenames variable is declared in reader_test.go.
	names := filenames
	if testing.Short() {
		names = filenamesShort
	}
	for _, fn := range names {
		qfn := "testdata/pngsuite/" + fn + ".png"
		// Read the image.
		m0, err := readPNG(qfn)
		if err != nil {
			t.Error(fn, err)
			continue
		}
		// Read the image again, encode it, and decode it.
		m1, err := readPNG(qfn)
		if err != nil {
			t.Error(fn, err)
			return
		}
		m2, err := encodeDecode(m1)
		if err != nil {
			t.Error(fn, err)
			return
		}
		// Compare the two.
		err = diff(m0, m2)
		if err != nil {
			t.Error(fn, err)
			continue
		}
	}
}

func TestWriterLevels(t *testing.T) {
	m := image.NewNRGBA(image.Rect(0, 0, 100, 100))

	var b1, b2 bytes.Buffer
	if err := (&Encoder{}).Encode(&b1, m); err != nil {
		t.Fatal(err)
	}
	noenc := &Encoder{CompressionLevel: NoCompression}
	if err := noenc.Encode(&b2, m); err != nil {
		t.Fatal(err)
	}

	if b2.Len() <= b1.Len() {
		t.Error("DefaultCompression encoding was larger than NoCompression encoding")
	}
	if _, err := Decode(&b1); err != nil {
		t.Error("cannot decode DefaultCompression")
	}
	if _, err := Decode(&b2); err != nil {
		t.Error("cannot decode NoCompression")
	}
}

func TestSubImage(t *testing.T) {
	m0 := image.NewRGBA(image.Rect(0, 0, 256, 256))
	for y := 0; y < 256; y++ {
		for x := 0; x < 256; x++ {
			m0.Set(x, y, color.RGBA{uint8(x), uint8(y), 0, 255})
		}
	}
	m0 = m0.SubImage(image.Rect(50, 30, 250, 130)).(*image.RGBA)
	m1, err := encodeDecode(m0)
	if err != nil {
		t.Error(err)
		return
	}
	err = diff(m0, m1)
	if err != nil {
		t.Error(err)
		return
	}
}

func BenchmarkEncodeGray(b *testing.B) {
	b.StopTimer()
	img := image.NewGray(image.Rect(0, 0, 640, 480))
	b.SetBytes(640 * 480 * 1)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Encode(ioutil.Discard, img)
	}
}

type pool struct {
	b *EncoderBuffer
}

func (p *pool) Get() *EncoderBuffer {
	return p.b
}

func (p *pool) Put(b *EncoderBuffer) {
	p.b = b
}

func BenchmarkEncodeGrayWithBufferPool(b *testing.B) {
	b.StopTimer()
	img := image.NewGray(image.Rect(0, 0, 640, 480))
	e := Encoder{
		BufferPool: &pool{},
	}
	b.SetBytes(640 * 480 * 1)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		e.Encode(ioutil.Discard, img)
	}
}

func BenchmarkEncodeNRGBOpaque(b *testing.B) {
	b.StopTimer()
	img := image.NewNRGBA(image.Rect(0, 0, 640, 480))
	// Set all pixels to 0xFF alpha to force opaque mode.
	bo := img.Bounds()
	for y := bo.Min.Y; y < bo.Max.Y; y++ {
		for x := bo.Min.X; x < bo.Max.X; x++ {
			img.Set(x, y, color.NRGBA{0, 0, 0, 255})
		}
	}
	if !img.Opaque() {
		b.Fatal("expected image to be opaque")
	}
	b.SetBytes(640 * 480 * 4)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Encode(ioutil.Discard, img)
	}
}

func BenchmarkEncodeNRGBA(b *testing.B) {
	b.StopTimer()
	img := image.NewNRGBA(image.Rect(0, 0, 640, 480))
	if img.Opaque() {
		b.Fatal("expected image not to be opaque")
	}
	b.SetBytes(640 * 480 * 4)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Encode(ioutil.Discard, img)
	}
}

func BenchmarkEncodePaletted(b *testing.B) {
	b.StopTimer()
	img := image.NewPaletted(image.Rect(0, 0, 640, 480), color.Palette{
		color.RGBA{0, 0, 0, 255},
		color.RGBA{255, 255, 255, 255},
	})
	b.SetBytes(640 * 480 * 1)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Encode(ioutil.Discard, img)
	}
}

func BenchmarkEncodeRGBOpaque(b *testing.B) {
	b.StopTimer()
	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	// Set all pixels to 0xFF alpha to force opaque mode.
	bo := img.Bounds()
	for y := bo.Min.Y; y < bo.Max.Y; y++ {
		for x := bo.Min.X; x < bo.Max.X; x++ {
			img.Set(x, y, color.RGBA{0, 0, 0, 255})
		}
	}
	if !img.Opaque() {
		b.Fatal("expected image to be opaque")
	}
	b.SetBytes(640 * 480 * 4)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Encode(ioutil.Discard, img)
	}
}

func BenchmarkEncodeRGBA(b *testing.B) {
	b.StopTimer()
	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	if img.Opaque() {
		b.Fatal("expected image not to be opaque")
	}
	b.SetBytes(640 * 480 * 4)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Encode(ioutil.Discard, img)
	}
}
