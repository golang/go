// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package png

import (
	"bufio"
	"bytes"
	"fmt"
	"image"
	"image/color"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"
)

var filenames = []string{
	"basn0g01",
	"basn0g01-30",
	"basn0g02",
	"basn0g02-29",
	"basn0g04",
	"basn0g04-31",
	"basn0g08",
	"basn0g16",
	"basn2c08",
	"basn2c16",
	"basn3p01",
	"basn3p02",
	"basn3p04",
	"basn3p04-31i",
	"basn3p08",
	"basn3p08-trns",
	"basn4a08",
	"basn4a16",
	"basn6a08",
	"basn6a16",
	"ftbbn0g01",
	"ftbbn0g02",
	"ftbbn0g04",
	"ftbbn2c16",
	"ftbbn3p08",
	"ftbgn2c16",
	"ftbgn3p08",
	"ftbrn2c08",
	"ftbwn0g16",
	"ftbwn3p08",
	"ftbyn3p08",
	"ftp0n0g08",
	"ftp0n2c08",
	"ftp0n3p08",
	"ftp1n3p08",
}

var filenamesPaletted = []string{
	"basn3p01",
	"basn3p02",
	"basn3p04",
	"basn3p08",
	"basn3p08-trns",
}

var filenamesShort = []string{
	"basn0g01",
	"basn0g04-31",
	"basn6a16",
}

func readPNG(filename string) (image.Image, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Decode(f)
}

// fakebKGDs maps from filenames to fake bKGD chunks for our approximation to
// the sng command-line tool. Package png doesn't keep that metadata when
// png.Decode returns an image.Image.
var fakebKGDs = map[string]string{
	"ftbbn0g01": "bKGD {gray: 0;}\n",
	"ftbbn0g02": "bKGD {gray: 0;}\n",
	"ftbbn0g04": "bKGD {gray: 0;}\n",
	"ftbbn2c16": "bKGD {red: 0;  green: 0;  blue: 65535;}\n",
	"ftbbn3p08": "bKGD {index: 245}\n",
	"ftbgn2c16": "bKGD {red: 0;  green: 65535;  blue: 0;}\n",
	"ftbgn3p08": "bKGD {index: 245}\n",
	"ftbrn2c08": "bKGD {red: 255;  green: 0;  blue: 0;}\n",
	"ftbwn0g16": "bKGD {gray: 65535;}\n",
	"ftbwn3p08": "bKGD {index: 0}\n",
	"ftbyn3p08": "bKGD {index: 245}\n",
}

// fakegAMAs maps from filenames to fake gAMA chunks for our approximation to
// the sng command-line tool. Package png doesn't keep that metadata when
// png.Decode returns an image.Image.
var fakegAMAs = map[string]string{
	"ftbbn0g01": "",
	"ftbbn0g02": "gAMA {0.45455}\n",
}

// fakeIHDRUsings maps from filenames to fake IHDR "using" lines for our
// approximation to the sng command-line tool. The PNG model is that
// transparency (in the tRNS chunk) is separate to the color/grayscale/palette
// color model (in the IHDR chunk). The Go model is that the concrete
// image.Image type returned by png.Decode, such as image.RGBA (with all pixels
// having 100% alpha) or image.NRGBA, encapsulates whether or not the image has
// transparency. This map is a hack to work around the fact that the Go model
// can't otherwise discriminate PNG's "IHDR says color (with no alpha) but tRNS
// says alpha" and "IHDR says color with alpha".
var fakeIHDRUsings = map[string]string{
	"ftbbn0g01": "    using grayscale;\n",
	"ftbbn0g02": "    using grayscale;\n",
	"ftbbn0g04": "    using grayscale;\n",
	"ftbbn2c16": "    using color;\n",
	"ftbgn2c16": "    using color;\n",
	"ftbrn2c08": "    using color;\n",
	"ftbwn0g16": "    using grayscale;\n",
}

// An approximation of the sng command-line tool.
func sng(w io.WriteCloser, filename string, png image.Image) {
	defer w.Close()
	bounds := png.Bounds()
	cm := png.ColorModel()
	var bitdepth int
	switch cm {
	case color.RGBAModel, color.NRGBAModel, color.AlphaModel, color.GrayModel:
		bitdepth = 8
	default:
		bitdepth = 16
	}
	cpm, _ := cm.(color.Palette)
	var paletted *image.Paletted
	if cpm != nil {
		switch {
		case len(cpm) <= 2:
			bitdepth = 1
		case len(cpm) <= 4:
			bitdepth = 2
		case len(cpm) <= 16:
			bitdepth = 4
		default:
			bitdepth = 8
		}
		paletted = png.(*image.Paletted)
	}

	// Write the filename and IHDR.
	io.WriteString(w, "#SNG: from "+filename+".png\nIHDR {\n")
	fmt.Fprintf(w, "    width: %d; height: %d; bitdepth: %d;\n", bounds.Dx(), bounds.Dy(), bitdepth)
	if s, ok := fakeIHDRUsings[filename]; ok {
		io.WriteString(w, s)
	} else {
		switch {
		case cm == color.RGBAModel, cm == color.RGBA64Model:
			io.WriteString(w, "    using color;\n")
		case cm == color.NRGBAModel, cm == color.NRGBA64Model:
			io.WriteString(w, "    using color alpha;\n")
		case cm == color.GrayModel, cm == color.Gray16Model:
			io.WriteString(w, "    using grayscale;\n")
		case cpm != nil:
			io.WriteString(w, "    using color palette;\n")
		default:
			io.WriteString(w, "unknown PNG decoder color model\n")
		}
	}
	io.WriteString(w, "}\n")

	// We fake a gAMA chunk. The test files have a gAMA chunk but the go PNG
	// parser ignores it (the PNG spec section 11.3 says "Ancillary chunks may
	// be ignored by a decoder").
	if s, ok := fakegAMAs[filename]; ok {
		io.WriteString(w, s)
	} else {
		io.WriteString(w, "gAMA {1.0000}\n")
	}

	// Write the PLTE and tRNS (if applicable).
	useTransparent := false
	if cpm != nil {
		lastAlpha := -1
		io.WriteString(w, "PLTE {\n")
		for i, c := range cpm {
			var r, g, b, a uint8
			switch c := c.(type) {
			case color.RGBA:
				r, g, b, a = c.R, c.G, c.B, 0xff
			case color.NRGBA:
				r, g, b, a = c.R, c.G, c.B, c.A
			default:
				panic("unknown palette color type")
			}
			if a != 0xff {
				lastAlpha = i
			}
			fmt.Fprintf(w, "    (%3d,%3d,%3d)     # rgb = (0x%02x,0x%02x,0x%02x)\n", r, g, b, r, g, b)
		}
		io.WriteString(w, "}\n")
		if s, ok := fakebKGDs[filename]; ok {
			io.WriteString(w, s)
		}
		if lastAlpha != -1 {
			io.WriteString(w, "tRNS {\n")
			for i := 0; i <= lastAlpha; i++ {
				_, _, _, a := cpm[i].RGBA()
				a >>= 8
				fmt.Fprintf(w, " %d", a)
			}
			io.WriteString(w, "}\n")
		}
	} else if strings.HasPrefix(filename, "ft") {
		if s, ok := fakebKGDs[filename]; ok {
			io.WriteString(w, s)
		}
		// We fake a tRNS chunk. The test files' grayscale and truecolor
		// transparent images all have their top left corner transparent.
		switch c := png.At(0, 0).(type) {
		case color.NRGBA:
			if c.A == 0 {
				useTransparent = true
				io.WriteString(w, "tRNS {\n")
				switch filename {
				case "ftbbn0g01", "ftbbn0g02", "ftbbn0g04":
					// The standard image package doesn't have a "gray with
					// alpha" type. Instead, we use an image.NRGBA.
					fmt.Fprintf(w, "    gray: %d;\n", c.R)
				default:
					fmt.Fprintf(w, "    red: %d; green: %d; blue: %d;\n", c.R, c.G, c.B)
				}
				io.WriteString(w, "}\n")
			}
		case color.NRGBA64:
			if c.A == 0 {
				useTransparent = true
				io.WriteString(w, "tRNS {\n")
				switch filename {
				case "ftbwn0g16":
					// The standard image package doesn't have a "gray16 with
					// alpha" type. Instead, we use an image.NRGBA64.
					fmt.Fprintf(w, "    gray: %d;\n", c.R)
				default:
					fmt.Fprintf(w, "    red: %d; green: %d; blue: %d;\n", c.R, c.G, c.B)
				}
				io.WriteString(w, "}\n")
			}
		}
	}

	// Write the IMAGE.
	io.WriteString(w, "IMAGE {\n    pixels hex\n")
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		switch {
		case cm == color.GrayModel:
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				gray := png.At(x, y).(color.Gray)
				fmt.Fprintf(w, "%02x", gray.Y)
			}
		case cm == color.Gray16Model:
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				gray16 := png.At(x, y).(color.Gray16)
				fmt.Fprintf(w, "%04x ", gray16.Y)
			}
		case cm == color.RGBAModel:
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				rgba := png.At(x, y).(color.RGBA)
				fmt.Fprintf(w, "%02x%02x%02x ", rgba.R, rgba.G, rgba.B)
			}
		case cm == color.RGBA64Model:
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				rgba64 := png.At(x, y).(color.RGBA64)
				fmt.Fprintf(w, "%04x%04x%04x ", rgba64.R, rgba64.G, rgba64.B)
			}
		case cm == color.NRGBAModel:
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				nrgba := png.At(x, y).(color.NRGBA)
				switch filename {
				case "ftbbn0g01", "ftbbn0g02", "ftbbn0g04":
					fmt.Fprintf(w, "%02x", nrgba.R)
				default:
					if useTransparent {
						fmt.Fprintf(w, "%02x%02x%02x ", nrgba.R, nrgba.G, nrgba.B)
					} else {
						fmt.Fprintf(w, "%02x%02x%02x%02x ", nrgba.R, nrgba.G, nrgba.B, nrgba.A)
					}
				}
			}
		case cm == color.NRGBA64Model:
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				nrgba64 := png.At(x, y).(color.NRGBA64)
				switch filename {
				case "ftbwn0g16":
					fmt.Fprintf(w, "%04x ", nrgba64.R)
				default:
					if useTransparent {
						fmt.Fprintf(w, "%04x%04x%04x ", nrgba64.R, nrgba64.G, nrgba64.B)
					} else {
						fmt.Fprintf(w, "%04x%04x%04x%04x ", nrgba64.R, nrgba64.G, nrgba64.B, nrgba64.A)
					}
				}
			}
		case cpm != nil:
			var b, c int
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				b = b<<uint(bitdepth) | int(paletted.ColorIndexAt(x, y))
				c++
				if c == 8/bitdepth {
					fmt.Fprintf(w, "%02x", b)
					b = 0
					c = 0
				}
			}
			if c != 0 {
				for c != 8/bitdepth {
					b = b << uint(bitdepth)
					c++
				}
				fmt.Fprintf(w, "%02x", b)
			}
		}
		io.WriteString(w, "\n")
	}
	io.WriteString(w, "}\n")
}

func TestReader(t *testing.T) {
	names := filenames
	if testing.Short() {
		names = filenamesShort
	}
	for _, fn := range names {
		// Read the .png file.
		img, err := readPNG("testdata/pngsuite/" + fn + ".png")
		if err != nil {
			t.Error(fn, err)
			continue
		}

		if fn == "basn4a16" {
			// basn4a16.sng is gray + alpha but sng() will produce true color + alpha
			// so we just check a single random pixel.
			c := img.At(2, 1).(color.NRGBA64)
			if c.R != 0x11a7 || c.G != 0x11a7 || c.B != 0x11a7 || c.A != 0x1085 {
				t.Error(fn, fmt.Errorf("wrong pixel value at (2, 1): %x", c))
			}
			continue
		}

		piper, pipew := io.Pipe()
		pb := bufio.NewScanner(piper)
		go sng(pipew, fn, img)
		defer piper.Close()

		// Read the .sng file.
		sf, err := os.Open("testdata/pngsuite/" + fn + ".sng")
		if err != nil {
			t.Error(fn, err)
			continue
		}
		defer sf.Close()
		sb := bufio.NewScanner(sf)

		// Compare the two, in SNG format, line by line.
		for {
			pdone := !pb.Scan()
			sdone := !sb.Scan()
			if pdone && sdone {
				break
			}
			if pdone || sdone {
				t.Errorf("%s: Different sizes", fn)
				break
			}
			ps := pb.Text()
			ss := sb.Text()

			// Newer versions of the sng command line tool append an optional
			// color name to the RGB tuple. For example:
			//	# rgb = (0xff,0xff,0xff) grey100
			//	# rgb = (0x00,0x00,0xff) blue1
			// instead of the older version's plainer:
			//	# rgb = (0xff,0xff,0xff)
			//	# rgb = (0x00,0x00,0xff)
			// We strip any such name.
			if strings.Contains(ss, "# rgb = (") && !strings.HasSuffix(ss, ")") {
				if i := strings.LastIndex(ss, ") "); i >= 0 {
					ss = ss[:i+1]
				}
			}

			if ps != ss {
				t.Errorf("%s: Mismatch\n%s\nversus\n%s\n", fn, ps, ss)
				break
			}
		}
		if pb.Err() != nil {
			t.Error(fn, pb.Err())
		}
		if sb.Err() != nil {
			t.Error(fn, sb.Err())
		}
	}
}

var readerErrors = []struct {
	file string
	err  string
}{
	{"invalid-zlib.png", "zlib: invalid checksum"},
	{"invalid-crc32.png", "invalid checksum"},
	{"invalid-noend.png", "unexpected EOF"},
	{"invalid-trunc.png", "unexpected EOF"},
}

func TestReaderError(t *testing.T) {
	for _, tt := range readerErrors {
		img, err := readPNG("testdata/" + tt.file)
		if err == nil {
			t.Errorf("decoding %s: missing error", tt.file)
			continue
		}
		if !strings.Contains(err.Error(), tt.err) {
			t.Errorf("decoding %s: %s, want %s", tt.file, err, tt.err)
		}
		if img != nil {
			t.Errorf("decoding %s: have image + error", tt.file)
		}
	}
}

func TestPalettedDecodeConfig(t *testing.T) {
	for _, fn := range filenamesPaletted {
		f, err := os.Open("testdata/pngsuite/" + fn + ".png")
		if err != nil {
			t.Errorf("%s: open failed: %v", fn, err)
			continue
		}
		defer f.Close()
		cfg, err := DecodeConfig(f)
		if err != nil {
			t.Errorf("%s: %v", fn, err)
			continue
		}
		pal, ok := cfg.ColorModel.(color.Palette)
		if !ok {
			t.Errorf("%s: expected paletted color model", fn)
			continue
		}
		if pal == nil {
			t.Errorf("%s: palette not initialized", fn)
			continue
		}
	}
}

func TestInterlaced(t *testing.T) {
	a, err := readPNG("testdata/gray-gradient.png")
	if err != nil {
		t.Fatal(err)
	}
	b, err := readPNG("testdata/gray-gradient.interlaced.png")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("decodings differ:\nnon-interlaced:\n%#v\ninterlaced:\n%#v", a, b)
	}
}

func TestIncompleteIDATOnRowBoundary(t *testing.T) {
	// The following is an invalid 1x2 grayscale PNG image. The header is OK,
	// but the zlib-compressed IDAT payload contains two bytes "\x02\x00",
	// which is only one row of data (the leading "\x02" is a row filter).
	const (
		ihdr = "\x00\x00\x00\x0dIHDR\x00\x00\x00\x01\x00\x00\x00\x02\x08\x00\x00\x00\x00\xbc\xea\xe9\xfb"
		idat = "\x00\x00\x00\x0eIDAT\x78\x9c\x62\x62\x00\x04\x00\x00\xff\xff\x00\x06\x00\x03\xfa\xd0\x59\xae"
		iend = "\x00\x00\x00\x00IEND\xae\x42\x60\x82"
	)
	_, err := Decode(strings.NewReader(pngHeader + ihdr + idat + iend))
	if err == nil {
		t.Fatal("got nil error, want non-nil")
	}
}

func TestTrailingIDATChunks(t *testing.T) {
	// The following is a valid 1x1 PNG image containing color.Gray{255} and
	// a trailing zero-length IDAT chunk (see PNG specification section 12.9):
	const (
		ihdr      = "\x00\x00\x00\x0dIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00\x3a\x7e\x9b\x55"
		idatWhite = "\x00\x00\x00\x0eIDAT\x78\x9c\x62\xfa\x0f\x08\x00\x00\xff\xff\x01\x05\x01\x02\x5a\xdd\x39\xcd"
		idatZero  = "\x00\x00\x00\x00IDAT\x35\xaf\x06\x1e"
		iend      = "\x00\x00\x00\x00IEND\xae\x42\x60\x82"
	)
	_, err := Decode(strings.NewReader(pngHeader + ihdr + idatWhite + idatZero + iend))
	if err != nil {
		t.Fatalf("decoding valid image: %v", err)
	}

	// Non-zero-length trailing IDAT chunks should be ignored (recoverable error).
	// The following chunk contains a single pixel with color.Gray{0}.
	const idatBlack = "\x00\x00\x00\x0eIDAT\x78\x9c\x62\x62\x00\x04\x00\x00\xff\xff\x00\x06\x00\x03\xfa\xd0\x59\xae"

	img, err := Decode(strings.NewReader(pngHeader + ihdr + idatWhite + idatBlack + iend))
	if err != nil {
		t.Fatalf("trailing IDAT not ignored: %v", err)
	}
	if img.At(0, 0) == (color.Gray{0}) {
		t.Fatal("decoded image from trailing IDAT chunk")
	}
}

func TestMultipletRNSChunks(t *testing.T) {
	/*
		The following is a valid 1x1 paletted PNG image with a 1-element palette
		containing color.NRGBA{0xff, 0x00, 0x00, 0x7f}:
			0000000: 8950 4e47 0d0a 1a0a 0000 000d 4948 4452  .PNG........IHDR
			0000010: 0000 0001 0000 0001 0803 0000 0028 cb34  .............(.4
			0000020: bb00 0000 0350 4c54 45ff 0000 19e2 0937  .....PLTE......7
			0000030: 0000 0001 7452 4e53 7f80 5cb4 cb00 0000  ....tRNS..\.....
			0000040: 0e49 4441 5478 9c62 6200 0400 00ff ff00  .IDATx.bb.......
			0000050: 0600 03fa d059 ae00 0000 0049 454e 44ae  .....Y.....IEND.
			0000060: 4260 82                                  B`.
		Dropping the tRNS chunk makes that color's alpha 0xff instead of 0x7f.
	*/
	const (
		ihdr = "\x00\x00\x00\x0dIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x03\x00\x00\x00\x28\xcb\x34\xbb"
		plte = "\x00\x00\x00\x03PLTE\xff\x00\x00\x19\xe2\x09\x37"
		trns = "\x00\x00\x00\x01tRNS\x7f\x80\x5c\xb4\xcb"
		idat = "\x00\x00\x00\x0eIDAT\x78\x9c\x62\x62\x00\x04\x00\x00\xff\xff\x00\x06\x00\x03\xfa\xd0\x59\xae"
		iend = "\x00\x00\x00\x00IEND\xae\x42\x60\x82"
	)
	for i := 0; i < 4; i++ {
		var b []byte
		b = append(b, pngHeader...)
		b = append(b, ihdr...)
		b = append(b, plte...)
		for j := 0; j < i; j++ {
			b = append(b, trns...)
		}
		b = append(b, idat...)
		b = append(b, iend...)

		var want color.Color
		m, err := Decode(bytes.NewReader(b))
		switch i {
		case 0:
			if err != nil {
				t.Errorf("%d tRNS chunks: %v", i, err)
				continue
			}
			want = color.RGBA{0xff, 0x00, 0x00, 0xff}
		case 1:
			if err != nil {
				t.Errorf("%d tRNS chunks: %v", i, err)
				continue
			}
			want = color.NRGBA{0xff, 0x00, 0x00, 0x7f}
		default:
			if err == nil {
				t.Errorf("%d tRNS chunks: got nil error, want non-nil", i)
			}
			continue
		}
		if got := m.At(0, 0); got != want {
			t.Errorf("%d tRNS chunks: got %T %v, want %T %v", i, got, got, want, want)
		}
	}
}

func TestUnknownChunkLengthUnderflow(t *testing.T) {
	data := []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x06, 0xf4, 0x7c, 0x55, 0x04, 0x1a,
		0xd3, 0x11, 0x9a, 0x73, 0x00, 0x00, 0xf8, 0x1e, 0xf3, 0x2e, 0x00, 0x00,
		0x01, 0x00, 0xff, 0xff, 0xff, 0xff, 0x07, 0xf4, 0x7c, 0x55, 0x04, 0x1a,
		0xd3}
	_, err := Decode(bytes.NewReader(data))
	if err == nil {
		t.Errorf("Didn't fail reading an unknown chunk with length 0xffffffff")
	}
}

func TestPaletted8OutOfRangePixel(t *testing.T) {
	// IDAT contains a reference to a palette index that does not exist in the file.
	img, err := readPNG("testdata/invalid-palette.png")
	if err != nil {
		t.Errorf("decoding invalid-palette.png: unexpected error %v", err)
		return
	}

	// Expect that the palette is extended with opaque black.
	want := color.RGBA{0x00, 0x00, 0x00, 0xff}
	if got := img.At(15, 15); got != want {
		t.Errorf("got %F %v, expected %T %v", got, got, want, want)
	}
}

func TestGray8Transparent(t *testing.T) {
	// These bytes come from https://golang.org/issues/19553
	m, err := Decode(bytes.NewReader([]byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
		0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x0b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x85, 0x2c, 0x88,
		0x80, 0x00, 0x00, 0x00, 0x02, 0x74, 0x52, 0x4e, 0x53, 0x00, 0xff, 0x5b, 0x91, 0x22, 0xb5, 0x00,
		0x00, 0x00, 0x02, 0x62, 0x4b, 0x47, 0x44, 0x00, 0xff, 0x87, 0x8f, 0xcc, 0xbf, 0x00, 0x00, 0x00,
		0x09, 0x70, 0x48, 0x59, 0x73, 0x00, 0x00, 0x0a, 0xf0, 0x00, 0x00, 0x0a, 0xf0, 0x01, 0x42, 0xac,
		0x34, 0x98, 0x00, 0x00, 0x00, 0x07, 0x74, 0x49, 0x4d, 0x45, 0x07, 0xd5, 0x04, 0x02, 0x12, 0x11,
		0x11, 0xf7, 0x65, 0x3d, 0x8b, 0x00, 0x00, 0x00, 0x4f, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63,
		0xf8, 0xff, 0xff, 0xff, 0xb9, 0xbd, 0x70, 0xf0, 0x8c, 0x01, 0xc8, 0xaf, 0x6e, 0x99, 0x02, 0x05,
		0xd9, 0x7b, 0xc1, 0xfc, 0x6b, 0xff, 0xa1, 0xa0, 0x87, 0x30, 0xff, 0xd9, 0xde, 0xbd, 0xd5, 0x4b,
		0xf7, 0xee, 0xfd, 0x0e, 0xe3, 0xef, 0xcd, 0x06, 0x19, 0x14, 0xf5, 0x1e, 0xce, 0xef, 0x01, 0x31,
		0x92, 0xd7, 0x82, 0x41, 0x31, 0x9c, 0x3f, 0x07, 0x02, 0xee, 0xa1, 0xaa, 0xff, 0xff, 0x9f, 0xe1,
		0xd9, 0x56, 0x30, 0xf8, 0x0e, 0xe5, 0x03, 0x00, 0xa9, 0x42, 0x84, 0x3d, 0xdf, 0x8f, 0xa6, 0x8f,
		0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
	}))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}

	const hex = "0123456789abcdef"
	var got []byte
	bounds := m.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			if r, _, _, a := m.At(x, y).RGBA(); a != 0 {
				got = append(got,
					hex[0x0f&(r>>12)],
					hex[0x0f&(r>>8)],
					' ',
				)
			} else {
				got = append(got,
					'.',
					'.',
					' ',
				)
			}
		}
		got = append(got, '\n')
	}

	const want = "" +
		".. .. .. ce bd bd bd bd bd bd bd bd bd bd e6 \n" +
		".. .. .. 7b 84 94 94 94 94 94 94 94 94 6b bd \n" +
		".. .. .. 7b d6 .. .. .. .. .. .. .. .. 8c bd \n" +
		".. .. .. 7b d6 .. .. .. .. .. .. .. .. 8c bd \n" +
		".. .. .. 7b d6 .. .. .. .. .. .. .. .. 8c bd \n" +
		"e6 bd bd 7b a5 bd bd f7 .. .. .. .. .. 8c bd \n" +
		"bd 6b 94 94 94 94 5a ef .. .. .. .. .. 8c bd \n" +
		"bd 8c .. .. .. .. 63 ad ad ad ad ad ad 73 bd \n" +
		"bd 8c .. .. .. .. 63 9c 9c 9c 9c 9c 9c 9c de \n" +
		"bd 6b 94 94 94 94 5a ef .. .. .. .. .. .. .. \n" +
		"e6 b5 b5 b5 b5 b5 b5 f7 .. .. .. .. .. .. .. \n"

	if string(got) != want {
		t.Errorf("got:\n%swant:\n%s", got, want)
	}
}

func TestDimensionOverflow(t *testing.T) {
	maxInt32AsInt := int((1 << 31) - 1)
	have32BitInts := 0 > (1 + maxInt32AsInt)

	testCases := []struct {
		src               []byte
		unsupportedConfig bool
		width             int
		height            int
	}{
		// These bytes come from https://golang.org/issues/22304
		//
		// It encodes a 2147483646 × 2147483646 (i.e. 0x7ffffffe × 0x7ffffffe)
		// NRGBA image. The (width × height) per se doesn't overflow an int64, but
		// (width × height × bytesPerPixel) will.
		{
			src: []byte{
				0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
				0x7f, 0xff, 0xff, 0xfe, 0x7f, 0xff, 0xff, 0xfe, 0x08, 0x06, 0x00, 0x00, 0x00, 0x30, 0x57, 0xb3,
				0xfd, 0x00, 0x00, 0x00, 0x15, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x62, 0x62, 0x20, 0x12, 0x8c,
				0x2a, 0xa4, 0xb3, 0x42, 0x40, 0x00, 0x00, 0x00, 0xff, 0xff, 0x13, 0x38, 0x00, 0x15, 0x2d, 0xef,
				0x5f, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
			},
			// It's debatable whether DecodeConfig (which does not allocate a
			// pixel buffer, unlike Decode) should fail in this case. The Go
			// standard library has made its choice, and the standard library
			// has compatibility constraints.
			unsupportedConfig: true,
			width:             0x7ffffffe,
			height:            0x7ffffffe,
		},

		// The next three cases come from https://golang.org/issues/38435

		{
			src: []byte{
				0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
				0x00, 0x00, 0xb5, 0x04, 0x00, 0x00, 0xb5, 0x04, 0x08, 0x06, 0x00, 0x00, 0x00, 0xf5, 0x60, 0x2c,
				0xb8, 0x00, 0x00, 0x00, 0x15, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x62, 0x62, 0x20, 0x12, 0x8c,
				0x2a, 0xa4, 0xb3, 0x42, 0x40, 0x00, 0x00, 0x00, 0xff, 0xff, 0x13, 0x38, 0x00, 0x15, 0x2d, 0xef,
				0x5f, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
			},
			// Here, width * height = 0x7ffea810, just under MaxInt32, but at 4
			// bytes per pixel, the number of pixels overflows an int32.
			unsupportedConfig: have32BitInts,
			width:             0x0000b504,
			height:            0x0000b504,
		},

		{
			src: []byte{
				0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
				0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x30, 0x6e, 0xc5,
				0x21, 0x00, 0x00, 0x00, 0x15, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x62, 0x62, 0x20, 0x12, 0x8c,
				0x2a, 0xa4, 0xb3, 0x42, 0x40, 0x00, 0x00, 0x00, 0xff, 0xff, 0x13, 0x38, 0x00, 0x15, 0x2d, 0xef,
				0x5f, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
			},
			unsupportedConfig: false,
			width:             0x04000000,
			height:            0x00000001,
		},

		{
			src: []byte{
				0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
				0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0xaa, 0xd4, 0x7c,
				0xda, 0x00, 0x00, 0x00, 0x15, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x62, 0x66, 0x20, 0x12, 0x30,
				0x8d, 0x2a, 0xa4, 0xaf, 0x42, 0x40, 0x00, 0x00, 0x00, 0xff, 0xff, 0x14, 0xd2, 0x00, 0x16, 0x00,
				0x00, 0x00,
			},
			unsupportedConfig: false,
			width:             0x08000000,
			height:            0x00000001,
		},
	}

	for i, tc := range testCases {
		cfg, err := DecodeConfig(bytes.NewReader(tc.src))
		if tc.unsupportedConfig {
			if err == nil {
				t.Errorf("i=%d: DecodeConfig: got nil error, want non-nil", i)
			} else if _, ok := err.(UnsupportedError); !ok {
				t.Fatalf("Decode: got %v (of type %T), want non-nil error (of type png.UnsupportedError)", err, err)
			}
			continue
		} else if err != nil {
			t.Errorf("i=%d: DecodeConfig: %v", i, err)
			continue
		} else if cfg.Width != tc.width {
			t.Errorf("i=%d: width: got %d, want %d", i, cfg.Width, tc.width)
			continue
		} else if cfg.Height != tc.height {
			t.Errorf("i=%d: height: got %d, want %d", i, cfg.Height, tc.height)
			continue
		}

		if nPixels := int64(cfg.Width) * int64(cfg.Height); nPixels > 0x7f000000 {
			// In theory, calling Decode would succeed, given several gigabytes
			// of memory. In practice, trying to make a []uint8 big enough to
			// hold all of the pixels can often result in OOM (out of memory).
			// OOM is unrecoverable; we can't write a test that passes when OOM
			// happens. Instead we skip the Decode call (and its tests).
			continue
		} else if testing.Short() {
			// Even for smaller image dimensions, calling Decode might allocate
			// 1 GiB or more of memory. This is usually feasible, and we want
			// to check that calling Decode doesn't panic if there's enough
			// memory, but we provide a runtime switch (testing.Short) to skip
			// these if it would OOM. See also http://golang.org/issue/5050
			// "decoding... images can cause huge memory allocations".
			continue
		}

		// Even if we don't panic, these aren't valid PNG images.
		if _, err := Decode(bytes.NewReader(tc.src)); err == nil {
			t.Errorf("i=%d: Decode: got nil error, want non-nil", i)
		}
	}

	if testing.Short() {
		t.Skip("skipping tests which allocate large pixel buffers")
	}
}

func TestDecodePalettedWithTransparency(t *testing.T) {
	// These bytes come from https://go.dev/issue/54325
	//
	// Per the PNG spec, a PLTE chunk contains 3 (not 4) bytes per palette
	// entry: RGB (not RGBA). The alpha value comes from the optional tRNS
	// chunk. Here, the PLTE chunk (0x50, 0x4c, 0x54, 0x45, etc) has 16 entries
	// (0x30 = 48 bytes) and the tRNS chunk (0x74, 0x52, 0x4e, 0x53, etc) has 1
	// entry (0x01 = 1 byte) that sets the first palette entry's alpha to zero.
	//
	// Both Decode and DecodeConfig should pick up that the first palette
	// entry's alpha is zero.
	src := []byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
		0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x20, 0x04, 0x03, 0x00, 0x00, 0x00, 0x81, 0x54, 0x67,
		0xc7, 0x00, 0x00, 0x00, 0x30, 0x50, 0x4c, 0x54, 0x45, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x0e,
		0x00, 0x23, 0x27, 0x7b, 0xb1, 0x2d, 0x0a, 0x49, 0x3f, 0x19, 0x78, 0x5f, 0xcd, 0xe4, 0x69, 0x69,
		0xe4, 0x71, 0x59, 0x53, 0x80, 0x11, 0x14, 0x8b, 0x00, 0xa9, 0x8d, 0x95, 0xcb, 0x99, 0x2f, 0x6b,
		0xd7, 0x29, 0x91, 0xd7, 0x7b, 0xba, 0xff, 0xe3, 0xd7, 0x13, 0xc6, 0xd3, 0x58, 0x00, 0x00, 0x00,
		0x01, 0x74, 0x52, 0x4e, 0x53, 0x00, 0x40, 0xe6, 0xd8, 0x66, 0x00, 0x00, 0x00, 0xfd, 0x49, 0x44,
		0x41, 0x54, 0x28, 0xcf, 0x63, 0x60, 0x00, 0x83, 0x55, 0x0c, 0x68, 0x60, 0x9d, 0x02, 0x9a, 0x80,
		0xde, 0x23, 0x74, 0x15, 0xef, 0x50, 0x94, 0x70, 0x2d, 0xd2, 0x7b, 0x87, 0xa2, 0x84, 0xeb, 0xee,
		0xbb, 0x77, 0x6f, 0x51, 0x94, 0xe8, 0xbd, 0x7d, 0xf7, 0xee, 0x12, 0xb2, 0x80, 0xd2, 0x3d, 0x54,
		0x01, 0x26, 0x10, 0x1f, 0x59, 0x40, 0x0f, 0xc8, 0xd7, 0x7e, 0x84, 0x70, 0x1c, 0xd7, 0xba, 0xb7,
		0x4a, 0xda, 0xda, 0x77, 0x11, 0xf6, 0xac, 0x5a, 0xa5, 0xf4, 0xf9, 0xbf, 0xfd, 0x3d, 0x24, 0x6b,
		0x98, 0x94, 0xf4, 0xff, 0x7f, 0x52, 0x42, 0x16, 0x30, 0x0e, 0xd9, 0xed, 0x6a, 0x8c, 0xec, 0x10,
		0x65, 0x53, 0x97, 0x60, 0x23, 0x64, 0x1d, 0x8a, 0x2e, 0xc6, 0x2e, 0x42, 0x08, 0x3d, 0x4c, 0xca,
		0x81, 0xc1, 0x82, 0xa6, 0xa2, 0x46, 0x08, 0x3d, 0x4a, 0xa1, 0x82, 0xc6, 0x82, 0xa1, 0x4a, 0x08,
		0x3d, 0xfa, 0xa6, 0x81, 0xa1, 0xa2, 0xc1, 0x9f, 0x10, 0x66, 0xd4, 0x2b, 0x87, 0x0a, 0x86, 0x1a,
		0x7d, 0x57, 0x80, 0x9b, 0x99, 0xaf, 0x62, 0x1a, 0x1a, 0xec, 0xf0, 0x0d, 0x66, 0x2a, 0x7b, 0x5a,
		0xba, 0xd2, 0x64, 0x63, 0x4b, 0xa6, 0xb2, 0xb4, 0x02, 0xa8, 0x12, 0xb5, 0x24, 0xa5, 0x99, 0x2e,
		0x33, 0x95, 0xd4, 0x92, 0x10, 0xee, 0xd0, 0x59, 0xb9, 0x6a, 0xd6, 0x21, 0x24, 0xb7, 0x33, 0x9d,
		0x01, 0x01, 0x64, 0xbf, 0xac, 0x59, 0xb2, 0xca, 0xeb, 0x14, 0x92, 0x80, 0xd6, 0x9a, 0x53, 0x4a,
		0x6b, 0x4e, 0x2d, 0x42, 0x52, 0xa1, 0x73, 0x28, 0x54, 0xe7, 0x90, 0x6a, 0x00, 0x92, 0x92, 0x45,
		0xa1, 0x40, 0x84, 0x2c, 0xe0, 0xc4, 0xa0, 0xb2, 0x28, 0x14, 0xc1, 0x67, 0xe9, 0x50, 0x60, 0x60,
		0xea, 0x70, 0x40, 0x12, 0x00, 0x79, 0x54, 0x09, 0x22, 0x00, 0x00, 0x30, 0xf3, 0x52, 0x87, 0xc6,
		0xe4, 0xbd, 0x70, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
	}

	cfg, err := DecodeConfig(bytes.NewReader(src))
	if err != nil {
		t.Fatalf("DecodeConfig: %v", err)
	} else if _, _, _, alpha := cfg.ColorModel.(color.Palette)[0].RGBA(); alpha != 0 {
		t.Errorf("DecodeConfig: got %d, want 0", alpha)
	}

	img, err := Decode(bytes.NewReader(src))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	} else if _, _, _, alpha := img.ColorModel().(color.Palette)[0].RGBA(); alpha != 0 {
		t.Errorf("Decode: got %d, want 0", alpha)
	}
}

func benchmarkDecode(b *testing.B, filename string, bytesPerPixel int) {
	data, err := os.ReadFile(filename)
	if err != nil {
		b.Fatal(err)
	}
	cfg, err := DecodeConfig(bytes.NewReader(data))
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(cfg.Width * cfg.Height * bytesPerPixel))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Decode(bytes.NewReader(data))
	}
}

func BenchmarkDecodeGray(b *testing.B) {
	benchmarkDecode(b, "testdata/benchGray.png", 1)
}

func BenchmarkDecodeNRGBAGradient(b *testing.B) {
	benchmarkDecode(b, "testdata/benchNRGBA-gradient.png", 4)
}

func BenchmarkDecodeNRGBAOpaque(b *testing.B) {
	benchmarkDecode(b, "testdata/benchNRGBA-opaque.png", 4)
}

func BenchmarkDecodePaletted(b *testing.B) {
	benchmarkDecode(b, "testdata/benchPaletted.png", 1)
}

func BenchmarkDecodeRGB(b *testing.B) {
	benchmarkDecode(b, "testdata/benchRGB.png", 4)
}

func BenchmarkDecodeInterlacing(b *testing.B) {
	benchmarkDecode(b, "testdata/benchRGB-interlace.png", 4)
}
