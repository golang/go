// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package png

import (
	"bufio";
	"fmt";
	"image";
	"io";
	"os";
	"testing";
)

// The go PNG library currently supports only a subset of the full PNG specification.
// In particular, bit depths other than 8 are not supported, and neither are grayscale images.
var filenames = []string{
	//"basn0g01",	// bit depth is not 8
	//"basn0g02",	// bit depth is not 8
	//"basn0g04",	// bit depth is not 8
	//"basn0g08",	// grayscale color model
	//"basn0g16",	// bit depth is not 8
	"basn2c08",
	//"basn2c16",	// bit depth is not 8
	//"basn3p01",	// bit depth is not 8
	//"basn3p02",	// bit depth is not 8
	//"basn3p04",	// bit depth is not 8
	"basn3p08",
	//"basn4a08",	// grayscale color model
	//"basn4a16",	// bit depth is not 8
	"basn6a08",
	//"basn6a16",	// bit depth is not 8
}

func readPng(filename string) (image.Image, os.Error) {
	f, err := os.Open(filename, os.O_RDONLY, 0444);
	if err != nil {
		return nil, err
	}
	defer f.Close();
	return Decode(f);
}

// An approximation of the sng command-line tool.
func sng(w io.WriteCloser, filename string, png image.Image) {
	defer w.Close();
	// For now, the go PNG parser only reads bitdepths of 8.
	bitdepth := 8;

	// Write the filename and IHDR.
	io.WriteString(w, "#SNG: from " + filename + ".png\nIHDR {\n");
	fmt.Fprintf(w, "    width: %d; height: %d; bitdepth: %d;\n", png.Width(), png.Height(), bitdepth);
	cm := png.ColorModel();
	var paletted *image.Paletted;
	cpm, _ := cm.(image.PalettedColorModel);
	switch {
	case cm == image.RGBAColorModel:
		io.WriteString(w, "    using color;\n")
	case cm == image.NRGBAColorModel:
		io.WriteString(w, "    using color alpha;\n")
	case cpm != nil:
		io.WriteString(w, "    using color palette;\n");
		paletted = png.(*image.Paletted);
	default:
		io.WriteString(w, "unknown PNG decoder color model\n")
	}
	io.WriteString(w, "}\n");

	// We fake a gAMA output. The test files have a gAMA chunk but the go PNG parser ignores it
	// (the PNG spec section 11.3 says "Ancillary chunks may be ignored by a decoder").
	io.WriteString(w, "gAMA {1.0000}\n");

	// Write the PLTE (if applicable).
	if cpm != nil {
		io.WriteString(w, "PLTE {\n");
		for i := 0; i < len(cpm); i++ {
			r, g, b, _ := cpm[i].RGBA();
			r >>= 24;
			g >>= 24;
			b >>= 24;
			fmt.Fprintf(w, "    (%3d,%3d,%3d)     # rgb = (0x%02x,0x%02x,0x%02x)\n", r, g, b, r, g, b);
		}
		io.WriteString(w, "}\n");
	}

	// Write the IMAGE.
	io.WriteString(w, "IMAGE {\n    pixels hex\n");
	for y := 0; y < png.Height(); y++ {
		switch {
		case cm == image.RGBAColorModel:
			for x := 0; x < png.Width(); x++ {
				rgba := png.At(x, y).(image.RGBAColor);
				fmt.Fprintf(w, "%02x%02x%02x ", rgba.R, rgba.G, rgba.B);
			}
		case cm == image.NRGBAColorModel:
			for x := 0; x < png.Width(); x++ {
				nrgba := png.At(x, y).(image.NRGBAColor);
				fmt.Fprintf(w, "%02x%02x%02x%02x ", nrgba.R, nrgba.G, nrgba.B, nrgba.A);
			}
		case cpm != nil:
			for x := 0; x < png.Width(); x++ {
				fmt.Fprintf(w, "%02x", paletted.ColorIndexAt(x, y))
			}
		}
		io.WriteString(w, "\n");
	}
	io.WriteString(w, "}\n");
}

func TestReader(t *testing.T) {
	for _, fn := range filenames {
		// Read the .png file.
		image, err := readPng("testdata/pngsuite/" + fn + ".png");
		if err != nil {
			t.Error(fn, err);
			continue;
		}
		piper, pipew := io.Pipe();
		pb := bufio.NewReader(piper);
		go sng(pipew, fn, image);
		defer piper.Close();

		// Read the .sng file.
		sf, err := os.Open("testdata/pngsuite/" + fn + ".sng", os.O_RDONLY, 0444);
		if err != nil {
			t.Error(fn, err);
			continue;
		}
		defer sf.Close();
		sb := bufio.NewReader(sf);
		if err != nil {
			t.Error(fn, err);
			continue;
		}

		// Compare the two, in SNG format, line by line.
		for {
			ps, perr := pb.ReadString('\n');
			ss, serr := sb.ReadString('\n');
			if perr == os.EOF && serr == os.EOF {
				break
			}
			if perr != nil {
				t.Error(fn, perr);
				break;
			}
			if serr != nil {
				t.Error(fn, serr);
				break;
			}
			if ps != ss {
				t.Errorf("%s: Mismatch\n%sversus\n%s\n", fn, ps, ss);
				break;
			}
		}
	}
}
