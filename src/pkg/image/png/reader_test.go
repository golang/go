// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package png

import (
	"bufio"
	"fmt"
	"image"
	"io"
	"os"
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
	"basn3p08",
	"basn3p08-trns",
	"basn4a08",
	"basn4a16",
	"basn6a08",
	"basn6a16",
}

var filenamesShort = []string{
	"basn0g01",
	"basn0g04-31",
	"basn6a16",
}

func readPng(filename string) (image.Image, os.Error) {
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
	case image.RGBAColorModel, image.NRGBAColorModel, image.AlphaColorModel, image.GrayColorModel:
		bitdepth = 8
	default:
		bitdepth = 16
	}
	cpm, _ := cm.(image.PalettedColorModel)
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
	case cm == image.RGBAColorModel, cm == image.RGBA64ColorModel:
		io.WriteString(w, "    using color;\n")
	case cm == image.NRGBAColorModel, cm == image.NRGBA64ColorModel:
		io.WriteString(w, "    using color alpha;\n")
	case cm == image.GrayColorModel, cm == image.Gray16ColorModel:
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
			r, g, b, a := c.RGBA()
			if a != 0xffff {
				lastAlpha = i
			}
			r >>= 8
			g >>= 8
			b >>= 8
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
		case cm == image.GrayColorModel:
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				gray := png.At(x, y).(image.GrayColor)
				fmt.Fprintf(w, "%02x", gray.Y)
			}
		case cm == image.Gray16ColorModel:
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				gray16 := png.At(x, y).(image.Gray16Color)
				fmt.Fprintf(w, "%04x ", gray16.Y)
			}
		case cm == image.RGBAColorModel:
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				rgba := png.At(x, y).(image.RGBAColor)
				fmt.Fprintf(w, "%02x%02x%02x ", rgba.R, rgba.G, rgba.B)
			}
		case cm == image.RGBA64ColorModel:
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				rgba64 := png.At(x, y).(image.RGBA64Color)
				fmt.Fprintf(w, "%04x%04x%04x ", rgba64.R, rgba64.G, rgba64.B)
			}
		case cm == image.NRGBAColorModel:
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				nrgba := png.At(x, y).(image.NRGBAColor)
				fmt.Fprintf(w, "%02x%02x%02x%02x ", nrgba.R, nrgba.G, nrgba.B, nrgba.A)
			}
		case cm == image.NRGBA64ColorModel:
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				nrgba64 := png.At(x, y).(image.NRGBA64Color)
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
		img, err := readPng("testdata/pngsuite/" + fn + ".png")
		if err != nil {
			t.Error(fn, err)
			continue
		}

		if fn == "basn4a16" {
			// basn4a16.sng is gray + alpha but sng() will produce true color + alpha
			// so we just check a single random pixel.
			c := img.At(2, 1).(image.NRGBA64Color)
			if c.R != 0x11a7 || c.G != 0x11a7 || c.B != 0x11a7 || c.A != 0x1085 {
				t.Error(fn, fmt.Errorf("wrong pixel value at (2, 1): %x", c))
			}
			continue
		}

		piper, pipew := io.Pipe()
		pb := bufio.NewReader(piper)
		go sng(pipew, fn, img)
		defer piper.Close()

		// Read the .sng file.
		sf, err := os.Open("testdata/pngsuite/" + fn + ".sng")
		if err != nil {
			t.Error(fn, err)
			continue
		}
		defer sf.Close()
		sb := bufio.NewReader(sf)
		if err != nil {
			t.Error(fn, err)
			continue
		}

		// Compare the two, in SNG format, line by line.
		for {
			ps, perr := pb.ReadString('\n')
			ss, serr := sb.ReadString('\n')
			if perr == os.EOF && serr == os.EOF {
				break
			}
			if perr != nil {
				t.Error(fn, perr)
				break
			}
			if serr != nil {
				t.Error(fn, serr)
				break
			}
			if ps != ss {
				t.Errorf("%s: Mismatch\n%sversus\n%s\n", fn, ps, ss)
				break
			}
		}
	}
}
