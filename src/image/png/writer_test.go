// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package png

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"io"
	"testing"
)

// totalLoss returns the total difference between two colors, in the range [0, 257*4]
// 0 means the colors are identical
// Difference between opaque black and white is 257*3
// Difference between NRGBA{10, 20, 30, 255}, NRGBA{11, 22, 33, 255} ≈ 6
func totalLoss(c0, c1 color.Color) float64 {
	var loss uint32
	r0, g0, b0, a0 := c0.RGBA()
	r1, g1, b1, a1 := c1.RGBA()
	s0 := []uint32{r0, g0, b0, a0}
	s1 := []uint32{r1, g1, b1, a1}
	for i := 0; i < 4; i++ {
		if s0[i] > s1[i] {
			loss += s0[i] - s1[i]
		} else {
			loss += s1[i] - s0[i]
		}
	}
	return float64(loss) / 255.0
}

func assertAlmostEqual(c0, c1 color.Color, expectedLoss float64) error {
	r0, g0, b0, a0 := c0.RGBA()
	r1, g1, b1, a1 := c1.RGBA()
	loss := totalLoss(c0, c1)
	if loss > expectedLoss {
		return fmt.Errorf("%T%v vs %T%v ({%d %d %d %d} vs {%d %d %d %d}), expected loss: %.2f, actual: %.2f",
			c0, c0, c1, c1, r0, g0, b0, a0, r1, g1, b1, a1, expectedLoss, loss)
	}
	return nil
}

func diff(m0, m1 image.Image, expectedLoss float64) error {
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
			if err := assertAlmostEqual(c0, c1, expectedLoss); err != nil {
				//fmt.Printf("colors differ at (%d, %d): %v\n", x, y, err)
				return fmt.Errorf("colors differ at (%d, %d): %w", x, y, err)
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

// Test for test
func TestLossCalculation(t *testing.T) {
	const uncertainty = 0.05
	testCases := []struct {
		c0, c1       color.Color
		expectedLoss float64
	}{
		{color.Black, color.White, 257 * 3},
		{color.Transparent, color.White, 257 * 4},
		{color.NRGBA{11, 222, 33, 255}, color.RGBA{11, 222, 33, 255}, 0},
		{color.NRGBA{2, 4, 8, 127}, color.RGBA{1, 2, 4, 127}, uncertainty},
		{color.NRGBA{10, 20, 30, 255}, color.NRGBA{11, 22, 33, 255}, 6},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%T%v vs %T%v", tc.c0, tc.c0, tc.c1, tc.c1), func(t *testing.T) {
			loss := totalLoss(tc.c0, tc.c1)
			if loss < tc.expectedLoss-uncertainty || loss > tc.expectedLoss+uncertainty {
				t.Errorf("expected loss to be %.2f±%.2f, got %.2f",
					tc.expectedLoss, uncertainty, loss)
			}
		})
	}
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
			continue
		}
		m2, err := encodeDecode(m1)
		if err != nil {
			t.Error(fn, err)
			continue
		}
		// Compare the two.
		err = diff(m0, m2, 0)
		if err != nil {
			t.Error(fn, err)
			continue
		}
	}
}

func TestWriterPaletted(t *testing.T) {
	const width, height = 32, 16

	testCases := []struct {
		plen     int
		bitdepth uint8
		datalen  int
	}{

		{
			plen:     256,
			bitdepth: 8,
			datalen:  (1 + width) * height,
		},

		{
			plen:     128,
			bitdepth: 8,
			datalen:  (1 + width) * height,
		},

		{
			plen:     16,
			bitdepth: 4,
			datalen:  (1 + width/2) * height,
		},

		{
			plen:     4,
			bitdepth: 2,
			datalen:  (1 + width/4) * height,
		},

		{
			plen:     2,
			bitdepth: 1,
			datalen:  (1 + width/8) * height,
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("plen-%d", tc.plen), func(t *testing.T) {
			// Create a paletted image with the correct palette length
			palette := make(color.Palette, tc.plen)
			for i := range palette {
				palette[i] = color.NRGBA{
					R: uint8(i),
					G: uint8(i),
					B: uint8(i),
					A: 255,
				}
			}
			m0 := image.NewPaletted(image.Rect(0, 0, width, height), palette)

			i := 0
			for y := 0; y < height; y++ {
				for x := 0; x < width; x++ {
					m0.SetColorIndex(x, y, uint8(i%tc.plen))
					i++
				}
			}

			// Encode the image
			var b bytes.Buffer
			if err := Encode(&b, m0); err != nil {
				t.Error(err)
				return
			}
			const chunkFieldsLength = 12 // 4 bytes for length, name and crc
			data := b.Bytes()
			i = len(pngHeader)

			for i < len(data)-chunkFieldsLength {
				length := binary.BigEndian.Uint32(data[i : i+4])
				name := string(data[i+4 : i+8])

				switch name {
				case "IHDR":
					bitdepth := data[i+8+8]
					if bitdepth != tc.bitdepth {
						t.Errorf("got bitdepth %d, want %d", bitdepth, tc.bitdepth)
					}
				case "IDAT":
					// Uncompress the image data
					r, err := zlib.NewReader(bytes.NewReader(data[i+8 : i+8+int(length)]))
					if err != nil {
						t.Error(err)
						return
					}
					n, err := io.Copy(io.Discard, r)
					if err != nil {
						t.Errorf("got error while reading image data: %v", err)
					}
					if n != int64(tc.datalen) {
						t.Errorf("got uncompressed data length %d, want %d", n, tc.datalen)
					}
				}

				i += chunkFieldsLength + int(length)
			}
		})

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
	err = diff(m0, m1, 0)
	if err != nil {
		t.Error(err)
		return
	}
}

func BenchmarkEncodeGray(b *testing.B) {
	img := image.NewGray(image.Rect(0, 0, 640, 480))
	b.SetBytes(640 * 480 * 1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Encode(io.Discard, img)
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
	img := image.NewGray(image.Rect(0, 0, 640, 480))
	e := Encoder{
		BufferPool: &pool{},
	}
	b.SetBytes(640 * 480 * 1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.Encode(io.Discard, img)
	}
}

func BenchmarkEncodeNRGBOpaque(b *testing.B) {
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
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Encode(io.Discard, img)
	}
}

func BenchmarkEncodeNRGBA(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 640, 480))
	if img.Opaque() {
		b.Fatal("expected image not to be opaque")
	}
	b.SetBytes(640 * 480 * 4)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Encode(io.Discard, img)
	}
}

func BenchmarkEncodePaletted(b *testing.B) {
	img := image.NewPaletted(image.Rect(0, 0, 640, 480), color.Palette{
		color.RGBA{0, 0, 0, 255},
		color.RGBA{255, 255, 255, 255},
	})
	b.SetBytes(640 * 480 * 1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Encode(io.Discard, img)
	}
}

func BenchmarkEncodeRGBOpaque(b *testing.B) {
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
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Encode(io.Discard, img)
	}
}

func BenchmarkEncodeRGBA(b *testing.B) {
	const width, height = 640, 480
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			percent := (x + y) % 100
			switch {
			case percent < 10: // 10% of pixels are translucent (have alpha >0 and <255)
				img.Set(x, y, color.NRGBA{uint8(x), uint8(y), uint8(x * y), uint8(percent)})
			case percent < 40: // 30% of pixels are transparent (have alpha == 0)
				img.Set(x, y, color.NRGBA{uint8(x), uint8(y), uint8(x * y), 0})
			default: // 60% of pixels are opaque (have alpha == 255)
				img.Set(x, y, color.NRGBA{uint8(x), uint8(y), uint8(x * y), 255})
			}
		}
	}
	if img.Opaque() {
		b.Fatal("expected image not to be opaque")
	}
	b.SetBytes(width * height * 4)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Encode(io.Discard, img)
	}
}

func TestWriteRGBA(t *testing.T) {
	const width, height = 640, 480
	transparentImg := image.NewRGBA(image.Rect(0, 0, width, height))
	opaqueImg := image.NewRGBA(image.Rect(0, 0, width, height))
	mixedImg := image.NewRGBA(image.Rect(0, 0, width, height))
	translucentImg := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			opaqueColor := color.RGBA{uint8(x), uint8(y), uint8(y + x), 255}
			transparentColor := color.RGBA{uint8(x) % 127, uint8(y) % 127, uint8(y+x) % 127, 128}
			opaqueImg.Set(x, y, opaqueColor)
			translucentImg.Set(x, y, transparentColor)
			if y%2 == 0 {
				mixedImg.Set(x, y, opaqueColor)
			}
		}
	}

	testCases := []struct {
		name         string
		img          image.Image
		expectedLoss float64
	}{
		{"Transparent RGBA", transparentImg, 0},
		{"Opaque RGBA", opaqueImg, 0},
		{"50/50 Transparent/Opaque RGBA", mixedImg, 0},
		{"RGBA with variable alpha", translucentImg, 1.51},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			m0 := tc.img
			m1, err := encodeDecode(m0)
			if err != nil {
				t.Fatal(err)
			}
			err = diff(m0, m1, tc.expectedLoss)
			if err != nil {
				t.Error(err)
			}
		})
	}
}
