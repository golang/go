// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gif

import (
	"bytes"
	"compress/lzw"
	"image"
	"image/color"
	"image/color/palette"
	"io"
	"io/ioutil"
	"reflect"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
)

// header, palette and trailer are parts of a valid 2x1 GIF image.
const (
	headerStr = "GIF89a" +
		"\x02\x00\x01\x00" + // width=2, height=1
		"\x80\x00\x00" // headerFields=(a color table of 2 pixels), backgroundIndex, aspect
	paletteStr = "\x10\x20\x30\x40\x50\x60" // the color table, also known as a palette
	trailerStr = "\x3b"
)

// lzw.NewReader wants a io.ByteReader, this ensures we're compatible.
var _ io.ByteReader = (*blockReader)(nil)

// lzwEncode returns an LZW encoding (with 2-bit literals) of in.
func lzwEncode(in []byte) []byte {
	b := &bytes.Buffer{}
	w := lzw.NewWriter(b, lzw.LSB, 2)
	if _, err := w.Write(in); err != nil {
		panic(err)
	}
	if err := w.Close(); err != nil {
		panic(err)
	}
	return b.Bytes()
}

func TestDecode(t *testing.T) {
	// extra contains superfluous bytes to inject into the GIF, either at the end
	// of an existing data sub-block (past the LZW End of Information code) or in
	// a separate data sub-block. The 0x02 values are arbitrary.
	const extra = "\x02\x02\x02\x02"

	testCases := []struct {
		nPix int // The number of pixels in the image data.
		// If non-zero, write this many extra bytes inside the data sub-block
		// containing the LZW end code.
		extraExisting int
		// If non-zero, write an extra block of this many bytes.
		extraSeparate int
		wantErr       error
	}{
		{0, 0, 0, errNotEnough},
		{1, 0, 0, errNotEnough},
		{2, 0, 0, nil},
		// An extra data sub-block after the compressed section with 1 byte which we
		// silently skip.
		{2, 0, 1, nil},
		// An extra data sub-block after the compressed section with 2 bytes. In
		// this case we complain that there is too much data.
		{2, 0, 2, errTooMuch},
		// Too much pixel data.
		{3, 0, 0, errTooMuch},
		// An extra byte after LZW data, but inside the same data sub-block.
		{2, 1, 0, nil},
		// Two extra bytes after LZW data, but inside the same data sub-block.
		{2, 2, 0, nil},
		// Extra data exists in the final sub-block with LZW data, AND there is
		// a bogus sub-block following.
		{2, 1, 1, errTooMuch},
	}
	for _, tc := range testCases {
		b := &bytes.Buffer{}
		b.WriteString(headerStr)
		b.WriteString(paletteStr)
		// Write an image with bounds 2x1 but tc.nPix pixels. If tc.nPix != 2
		// then this should result in an invalid GIF image. First, write a
		// magic 0x2c (image descriptor) byte, bounds=(0,0)-(2,1), a flags
		// byte, and 2-bit LZW literals.
		b.WriteString("\x2c\x00\x00\x00\x00\x02\x00\x01\x00\x00\x02")
		if tc.nPix > 0 {
			enc := lzwEncode(make([]byte, tc.nPix))
			if len(enc)+tc.extraExisting > 0xff {
				t.Errorf("nPix=%d, extraExisting=%d, extraSeparate=%d: compressed length %d is too large",
					tc.nPix, tc.extraExisting, tc.extraSeparate, len(enc))
				continue
			}

			// Write the size of the data sub-block containing the LZW data.
			b.WriteByte(byte(len(enc) + tc.extraExisting))

			// Write the LZW data.
			b.Write(enc)

			// Write extra bytes inside the same data sub-block where LZW data
			// ended. Each arbitrarily 0x02.
			b.WriteString(extra[:tc.extraExisting])
		}

		if tc.extraSeparate > 0 {
			// Data sub-block size. This indicates how many extra bytes follow.
			b.WriteByte(byte(tc.extraSeparate))
			b.WriteString(extra[:tc.extraSeparate])
		}
		b.WriteByte(0x00) // An empty block signifies the end of the image data.
		b.WriteString(trailerStr)

		got, err := Decode(b)
		if err != tc.wantErr {
			t.Errorf("nPix=%d, extraExisting=%d, extraSeparate=%d\ngot  %v\nwant %v",
				tc.nPix, tc.extraExisting, tc.extraSeparate, err, tc.wantErr)
		}

		if tc.wantErr != nil {
			continue
		}
		want := &image.Paletted{
			Pix:    []uint8{0, 0},
			Stride: 2,
			Rect:   image.Rect(0, 0, 2, 1),
			Palette: color.Palette{
				color.RGBA{0x10, 0x20, 0x30, 0xff},
				color.RGBA{0x40, 0x50, 0x60, 0xff},
			},
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("nPix=%d, extraExisting=%d, extraSeparate=%d\ngot  %v\nwant %v",
				tc.nPix, tc.extraExisting, tc.extraSeparate, got, want)
		}
	}
}

func TestTransparentIndex(t *testing.T) {
	b := &bytes.Buffer{}
	b.WriteString(headerStr)
	b.WriteString(paletteStr)
	for transparentIndex := 0; transparentIndex < 3; transparentIndex++ {
		if transparentIndex < 2 {
			// Write the graphic control for the transparent index.
			b.WriteString("\x21\xf9\x04\x01\x00\x00")
			b.WriteByte(byte(transparentIndex))
			b.WriteByte(0)
		}
		// Write an image with bounds 2x1, as per TestDecode.
		b.WriteString("\x2c\x00\x00\x00\x00\x02\x00\x01\x00\x00\x02")
		enc := lzwEncode([]byte{0x00, 0x00})
		if len(enc) > 0xff {
			t.Fatalf("compressed length %d is too large", len(enc))
		}
		b.WriteByte(byte(len(enc)))
		b.Write(enc)
		b.WriteByte(0x00)
	}
	b.WriteString(trailerStr)

	g, err := DecodeAll(b)
	if err != nil {
		t.Fatalf("DecodeAll: %v", err)
	}
	c0 := color.RGBA{paletteStr[0], paletteStr[1], paletteStr[2], 0xff}
	c1 := color.RGBA{paletteStr[3], paletteStr[4], paletteStr[5], 0xff}
	cz := color.RGBA{}
	wants := []color.Palette{
		{cz, c1},
		{c0, cz},
		{c0, c1},
	}
	if len(g.Image) != len(wants) {
		t.Fatalf("got %d images, want %d", len(g.Image), len(wants))
	}
	for i, want := range wants {
		got := g.Image[i].Palette
		if !reflect.DeepEqual(got, want) {
			t.Errorf("palette #%d:\ngot  %v\nwant %v", i, got, want)
		}
	}
}

// testGIF is a simple GIF that we can modify to test different scenarios.
var testGIF = []byte{
	'G', 'I', 'F', '8', '9', 'a',
	1, 0, 1, 0, // w=1, h=1 (6)
	128, 0, 0, // headerFields, bg, aspect (10)
	0, 0, 0, 1, 1, 1, // color table and graphics control (13)
	0x21, 0xf9, 0x04, 0x00, 0x00, 0x00, 0xff, 0x00, // (19)
	// frame 1 (0,0 - 1,1)
	0x2c,
	0x00, 0x00, 0x00, 0x00,
	0x01, 0x00, 0x01, 0x00, // (32)
	0x00,
	0x02, 0x02, 0x4c, 0x01, 0x00, // lzw pixels
	// trailer
	0x3b,
}

func try(t *testing.T, b []byte, want string) {
	_, err := DecodeAll(bytes.NewReader(b))
	var got string
	if err != nil {
		got = err.Error()
	}
	if got != want {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestBounds(t *testing.T) {
	// Make a local copy of testGIF.
	gif := make([]byte, len(testGIF))
	copy(gif, testGIF)
	// Make the bounds too big, just by one.
	gif[32] = 2
	want := "gif: frame bounds larger than image bounds"
	try(t, gif, want)

	// Make the bounds too small; does not trigger bounds
	// check, but now there's too much data.
	gif[32] = 0
	want = "gif: too much image data"
	try(t, gif, want)
	gif[32] = 1

	// Make the bounds really big, expect an error.
	want = "gif: frame bounds larger than image bounds"
	for i := 0; i < 4; i++ {
		gif[32+i] = 0xff
	}
	try(t, gif, want)
}

func TestNoPalette(t *testing.T) {
	b := &bytes.Buffer{}

	// Manufacture a GIF with no palette, so any pixel at all
	// will be invalid.
	b.WriteString(headerStr[:len(headerStr)-3])
	b.WriteString("\x00\x00\x00") // No global palette.

	// Image descriptor: 2x1, no local palette, and 2-bit LZW literals.
	b.WriteString("\x2c\x00\x00\x00\x00\x02\x00\x01\x00\x00\x02")

	// Encode the pixels: neither is in range, because there is no palette.
	enc := lzwEncode([]byte{0x00, 0x03})
	b.WriteByte(byte(len(enc)))
	b.Write(enc)
	b.WriteByte(0x00) // An empty block signifies the end of the image data.

	b.WriteString(trailerStr)

	try(t, b.Bytes(), "gif: no color table")
}

func TestPixelOutsidePaletteRange(t *testing.T) {
	for _, pval := range []byte{0, 1, 2, 3} {
		b := &bytes.Buffer{}

		// Manufacture a GIF with a 2 color palette.
		b.WriteString(headerStr)
		b.WriteString(paletteStr)

		// Image descriptor: 2x1, no local palette, and 2-bit LZW literals.
		b.WriteString("\x2c\x00\x00\x00\x00\x02\x00\x01\x00\x00\x02")

		// Encode the pixels; some pvals trigger the expected error.
		enc := lzwEncode([]byte{pval, pval})
		b.WriteByte(byte(len(enc)))
		b.Write(enc)
		b.WriteByte(0x00) // An empty block signifies the end of the image data.

		b.WriteString(trailerStr)

		// No error expected, unless the pixels are beyond the 2 color palette.
		want := ""
		if pval >= 2 {
			want = "gif: invalid pixel value"
		}
		try(t, b.Bytes(), want)
	}
}

func TestTransparentPixelOutsidePaletteRange(t *testing.T) {
	b := &bytes.Buffer{}

	// Manufacture a GIF with a 2 color palette.
	b.WriteString(headerStr)
	b.WriteString(paletteStr)

	// Graphic Control Extension: transparency, transparent color index = 3.
	//
	// This index, 3, is out of range of the global palette and there is no
	// local palette in the subsequent image descriptor. This is an error
	// according to the spec, but Firefox and Google Chrome seem OK with this.
	//
	// See golang.org/issue/15059.
	b.WriteString("\x21\xf9\x04\x01\x00\x00\x03\x00")

	// Image descriptor: 2x1, no local palette, and 2-bit LZW literals.
	b.WriteString("\x2c\x00\x00\x00\x00\x02\x00\x01\x00\x00\x02")

	// Encode the pixels.
	enc := lzwEncode([]byte{0x03, 0x03})
	b.WriteByte(byte(len(enc)))
	b.Write(enc)
	b.WriteByte(0x00) // An empty block signifies the end of the image data.

	b.WriteString(trailerStr)

	try(t, b.Bytes(), "")
}

func TestLoopCount(t *testing.T) {
	testCases := []struct {
		name      string
		data      []byte
		loopCount int
	}{
		{
			"loopcount-missing",
			[]byte("GIF89a000\x00000" +
				",0\x00\x00\x00\n\x00\n\x00\x80000000" + // image 0 descriptor & color table
				"\x02\b\xf01u\xb9\xfdal\x05\x00;"), // image 0 image data & trailer
			-1,
		},
		{
			"loopcount-0",
			[]byte("GIF89a000\x00000" +
				"!\xff\vNETSCAPE2.0\x03\x01\x00\x00\x00" + // loop count = 0
				",0\x00\x00\x00\n\x00\n\x00\x80000000" + // image 0 descriptor & color table
				"\x02\b\xf01u\xb9\xfdal\x05\x00" + // image 0 image data
				",0\x00\x00\x00\n\x00\n\x00\x80000000" + // image 1 descriptor & color table
				"\x02\b\xf01u\xb9\xfdal\x05\x00;"), // image 1 image data & trailer
			0,
		},
		{
			"loopcount-1",
			[]byte("GIF89a000\x00000" +
				"!\xff\vNETSCAPE2.0\x03\x01\x01\x00\x00" + // loop count = 1
				",0\x00\x00\x00\n\x00\n\x00\x80000000" + // image 0 descriptor & color table
				"\x02\b\xf01u\xb9\xfdal\x05\x00" + // image 0 image data
				",0\x00\x00\x00\n\x00\n\x00\x80000000" + // image 1 descriptor & color table
				"\x02\b\xf01u\xb9\xfdal\x05\x00;"), // image 1 image data & trailer
			1,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			img, err := DecodeAll(bytes.NewReader(tc.data))
			if err != nil {
				t.Fatal("DecodeAll:", err)
			}
			w := new(bytes.Buffer)
			err = EncodeAll(w, img)
			if err != nil {
				t.Fatal("EncodeAll:", err)
			}
			img1, err := DecodeAll(w)
			if err != nil {
				t.Fatal("DecodeAll:", err)
			}
			if img.LoopCount != tc.loopCount {
				t.Errorf("loop count mismatch: %d vs %d", img.LoopCount, tc.loopCount)
			}
			if img.LoopCount != img1.LoopCount {
				t.Errorf("loop count failed round-trip: %d vs %d", img.LoopCount, img1.LoopCount)
			}
		})
	}
}

func TestUnexpectedEOF(t *testing.T) {
	for i := len(testGIF) - 1; i >= 0; i-- {
		_, err := Decode(bytes.NewReader(testGIF[:i]))
		if err == errNotEnough {
			continue
		}
		text := ""
		if err != nil {
			text = err.Error()
		}
		if !strings.HasPrefix(text, "gif:") || !strings.HasSuffix(text, ": unexpected EOF") {
			t.Errorf("Decode(testGIF[:%d]) = %v, want gif: ...: unexpected EOF", i, err)
		}
	}
}

// See golang.org/issue/22237
func TestDecodeMemoryConsumption(t *testing.T) {
	const frames = 3000
	img := image.NewPaletted(image.Rectangle{Max: image.Point{1, 1}}, palette.WebSafe)
	hugeGIF := &GIF{
		Image:    make([]*image.Paletted, frames),
		Delay:    make([]int, frames),
		Disposal: make([]byte, frames),
	}
	for i := 0; i < frames; i++ {
		hugeGIF.Image[i] = img
		hugeGIF.Delay[i] = 60
	}
	buf := new(bytes.Buffer)
	if err := EncodeAll(buf, hugeGIF); err != nil {
		t.Fatal("EncodeAll:", err)
	}
	s0, s1 := new(runtime.MemStats), new(runtime.MemStats)
	runtime.GC()
	defer debug.SetGCPercent(debug.SetGCPercent(5))
	runtime.ReadMemStats(s0)
	if _, err := Decode(buf); err != nil {
		t.Fatal("Decode:", err)
	}
	runtime.ReadMemStats(s1)
	if heapDiff := int64(s1.HeapAlloc - s0.HeapAlloc); heapDiff > 30<<20 {
		t.Fatalf("Decode of %d frames increased heap by %dMB", frames, heapDiff>>20)
	}
}

func BenchmarkDecode(b *testing.B) {
	data, err := ioutil.ReadFile("../testdata/video-001.gif")
	if err != nil {
		b.Fatal(err)
	}
	cfg, err := DecodeConfig(bytes.NewReader(data))
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(cfg.Width * cfg.Height))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Decode(bytes.NewReader(data))
	}
}
