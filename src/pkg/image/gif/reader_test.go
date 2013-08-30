package gif

import (
	"bytes"
	"compress/lzw"
	"image"
	"image/color"
	"reflect"
	"testing"
)

// header, palette and trailer are parts of a valid 2x1 GIF image.
const (
	headerStr = "GIF89a" +
		"\x02\x00\x01\x00" + // width=2, height=1
		"\x80\x00\x00" // headerFields=(a color map of 2 pixels), backgroundIndex, aspect
	paletteStr = "\x10\x20\x30\x40\x50\x60" // the color map, also known as a palette
	trailerStr = "\x3b"
)

func TestDecode(t *testing.T) {
	// lzwEncode returns an LZW encoding (with 2-bit literals) of n zeroes.
	lzwEncode := func(n int) []byte {
		b := &bytes.Buffer{}
		w := lzw.NewWriter(b, lzw.LSB, 2)
		w.Write(make([]byte, n))
		w.Close()
		return b.Bytes()
	}

	testCases := []struct {
		nPix    int  // The number of pixels in the image data.
		extra   bool // Whether to write an extra block after the LZW-encoded data.
		wantErr error
	}{
		{0, false, errNotEnough},
		{1, false, errNotEnough},
		{2, false, nil},
		{2, true, errTooMuch},
		{3, false, errTooMuch},
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
			enc := lzwEncode(tc.nPix)
			if len(enc) > 0xff {
				t.Errorf("nPix=%d, extra=%t: compressed length %d is too large", tc.nPix, tc.extra, len(enc))
				continue
			}
			b.WriteByte(byte(len(enc)))
			b.Write(enc)
		}
		if tc.extra {
			b.WriteString("\x01\x02") // A 1-byte payload with an 0x02 byte.
		}
		b.WriteByte(0x00) // An empty block signifies the end of the image data.
		b.WriteString(trailerStr)

		got, err := Decode(b)
		if err != tc.wantErr {
			t.Errorf("nPix=%d, extra=%t\ngot  %v\nwant %v", tc.nPix, tc.extra, err, tc.wantErr)
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
			t.Errorf("nPix=%d, extra=%t\ngot  %v\nwant %v", tc.nPix, tc.extra, got, want)
		}
	}
}

// testGIF is a simple GIF that we can modify to test different scenarios.
var testGIF = []byte{
	'G', 'I', 'F', '8', '9', 'a',
	1, 0, 1, 0, // w=1, h=1 (6)
	128, 0, 0, // headerFields, bg, aspect (10)
	0, 0, 0, 1, 1, 1, // color map and graphics control (13)
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

	// Image descriptor: 2x1, no local palette.
	b.WriteString("\x2c\x00\x00\x00\x00\x02\x00\x01\x00\x00\x02")

	// Encode the pixels: neither is in range, because there is no palette.
	pix := []byte{0, 128}
	enc := &bytes.Buffer{}
	w := lzw.NewWriter(enc, lzw.LSB, 2)
	w.Write(pix)
	w.Close()
	b.WriteByte(byte(len(enc.Bytes())))
	b.Write(enc.Bytes())
	b.WriteByte(0x00) // An empty block signifies the end of the image data.

	b.WriteString(trailerStr)

	try(t, b.Bytes(), "gif: invalid pixel value")
}

func TestPixelOutsidePaletteRange(t *testing.T) {
	for _, pval := range []byte{0, 1, 2, 3, 255} {
		b := &bytes.Buffer{}

		// Manufacture a GIF with a 2 color palette.
		b.WriteString(headerStr)
		b.WriteString(paletteStr)

		// Image descriptor: 2x1, no local palette.
		b.WriteString("\x2c\x00\x00\x00\x00\x02\x00\x01\x00\x00\x02")

		// Encode the pixels; some pvals trigger the expected error.
		pix := []byte{pval, pval}
		enc := &bytes.Buffer{}
		w := lzw.NewWriter(enc, lzw.LSB, 2)
		w.Write(pix)
		w.Close()
		b.WriteByte(byte(len(enc.Bytes())))
		b.Write(enc.Bytes())
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
