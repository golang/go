package gif

import (
	"bytes"
	"compress/lzw"
	"image"
	"image/color"
	"reflect"
	"testing"
)

func TestDecode(t *testing.T) {
	// header and trailer are parts of a valid 2x1 GIF image.
	const (
		header = "GIF89a" +
			"\x02\x00\x01\x00" + // width=2, height=1
			"\x80\x00\x00" + // headerFields=(a color map of 2 pixels), backgroundIndex, aspect
			"\x10\x20\x30\x40\x50\x60" // the color map, also known as a palette
		trailer = "\x3b"
	)

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
		b.WriteString(header)
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
		b.WriteString(trailer)

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
