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
	"io/ioutil"
	"os"
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
	io.WriteString(w, "}\n")

	// We fake a gAMA output. The test files have a gAMA chunk but the go PNG parser ignores it
	// (the PNG spec section 11.3 says "Ancillary chunks may be ignored by a decoder").
	io.WriteString(w, "gAMA {1.0000}\n")

	// Write the PLTE and tRNS (if applicable).
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
		if lastAlpha != -1 {
			io.WriteString(w, "tRNS {\n")
			for i := 0; i <= lastAlpha; i++ {
				_, _, _, a := cpm[i].RGBA()
				a >>= 8
				fmt.Fprintf(w, " %d", a)
			}
			io.WriteString(w, "}\n")
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
				fmt.Fprintf(w, "%02x%02x%02x%02x ", nrgba.R, nrgba.G, nrgba.B, nrgba.A)
			}
		case cm == color.NRGBA64Model:
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				nrgba64 := png.At(x, y).(color.NRGBA64)
				fmt.Fprintf(w, "%04x%04x%04x%04x ", nrgba64.R, nrgba64.G, nrgba64.B, nrgba64.A)
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
		if err != nil {
			t.Error(fn, err)
			continue
		}

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
			if ps != ss {
				t.Errorf("%s: Mismatch\n%sversus\n%s\n", fn, ps, ss)
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

func benchmarkDecode(b *testing.B, filename string, bytesPerPixel int) {
	b.StopTimer()
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		b.Fatal(err)
	}
	s := string(data)
	cfg, err := DecodeConfig(strings.NewReader(s))
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(cfg.Width * cfg.Height * bytesPerPixel))
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Decode(strings.NewReader(s))
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
