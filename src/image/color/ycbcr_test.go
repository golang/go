// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package color

import (
	"fmt"
	"testing"
)

func delta(x, y uint8) uint8 {
	if x >= y {
		return x - y
	}
	return y - x
}

func eq(c0, c1 Color) error {
	r0, g0, b0, a0 := c0.RGBA()
	r1, g1, b1, a1 := c1.RGBA()
	if r0 != r1 || g0 != g1 || b0 != b1 || a0 != a1 {
		return fmt.Errorf("got  0x%04x 0x%04x 0x%04x 0x%04x\nwant 0x%04x 0x%04x 0x%04x 0x%04x",
			r0, g0, b0, a0, r1, g1, b1, a1)
	}
	return nil
}

// TestYCbCrRoundtrip tests that a subset of RGB space can be converted to YCbCr
// and back to within 2/256 tolerance.
func TestYCbCrRoundtrip(t *testing.T) {
	for r := 0; r < 256; r += 7 {
		for g := 0; g < 256; g += 5 {
			for b := 0; b < 256; b += 3 {
				r0, g0, b0 := uint8(r), uint8(g), uint8(b)
				y, cb, cr := RGBToYCbCr(r0, g0, b0)
				r1, g1, b1 := YCbCrToRGB(y, cb, cr)
				if delta(r0, r1) > 2 || delta(g0, g1) > 2 || delta(b0, b1) > 2 {
					t.Fatalf("\nr0, g0, b0 = %d, %d, %d\ny,  cb, cr = %d, %d, %d\nr1, g1, b1 = %d, %d, %d",
						r0, g0, b0, y, cb, cr, r1, g1, b1)
				}
			}
		}
	}
}

// TestYCbCrToRGBConsistency tests that calling the RGBA method (16 bit color)
// then truncating to 8 bits is equivalent to calling the YCbCrToRGB function (8
// bit color).
func TestYCbCrToRGBConsistency(t *testing.T) {
	for y := 0; y < 256; y += 7 {
		for cb := 0; cb < 256; cb += 5 {
			for cr := 0; cr < 256; cr += 3 {
				x := YCbCr{uint8(y), uint8(cb), uint8(cr)}
				r0, g0, b0, _ := x.RGBA()
				r1, g1, b1 := uint8(r0>>8), uint8(g0>>8), uint8(b0>>8)
				r2, g2, b2 := YCbCrToRGB(x.Y, x.Cb, x.Cr)
				if r1 != r2 || g1 != g2 || b1 != b2 {
					t.Fatalf("y, cb, cr = %d, %d, %d\nr1, g1, b1 = %d, %d, %d\nr2, g2, b2 = %d, %d, %d",
						y, cb, cr, r1, g1, b1, r2, g2, b2)
				}
			}
		}
	}
}

// TestYCbCrGray tests that YCbCr colors are a superset of Gray colors.
func TestYCbCrGray(t *testing.T) {
	for i := 0; i < 256; i++ {
		c0 := YCbCr{uint8(i), 0x80, 0x80}
		c1 := Gray{uint8(i)}
		if err := eq(c0, c1); err != nil {
			t.Errorf("i=0x%02x:\n%v", i, err)
		}
	}
}

// TestNYCbCrAAlpha tests that NYCbCrA colors are a superset of Alpha colors.
func TestNYCbCrAAlpha(t *testing.T) {
	for i := 0; i < 256; i++ {
		c0 := NYCbCrA{YCbCr{0xff, 0x80, 0x80}, uint8(i)}
		c1 := Alpha{uint8(i)}
		if err := eq(c0, c1); err != nil {
			t.Errorf("i=0x%02x:\n%v", i, err)
		}
	}
}

// TestNYCbCrAYCbCr tests that NYCbCrA colors are a superset of YCbCr colors.
func TestNYCbCrAYCbCr(t *testing.T) {
	for i := 0; i < 256; i++ {
		c0 := NYCbCrA{YCbCr{uint8(i), 0x40, 0xc0}, 0xff}
		c1 := YCbCr{uint8(i), 0x40, 0xc0}
		if err := eq(c0, c1); err != nil {
			t.Errorf("i=0x%02x:\n%v", i, err)
		}
	}
}

// TestCMYKRoundtrip tests that a subset of RGB space can be converted to CMYK
// and back to within 1/256 tolerance.
func TestCMYKRoundtrip(t *testing.T) {
	for r := 0; r < 256; r += 7 {
		for g := 0; g < 256; g += 5 {
			for b := 0; b < 256; b += 3 {
				r0, g0, b0 := uint8(r), uint8(g), uint8(b)
				c, m, y, k := RGBToCMYK(r0, g0, b0)
				r1, g1, b1 := CMYKToRGB(c, m, y, k)
				if delta(r0, r1) > 1 || delta(g0, g1) > 1 || delta(b0, b1) > 1 {
					t.Fatalf("\nr0, g0, b0 = %d, %d, %d\nc, m, y, k = %d, %d, %d, %d\nr1, g1, b1 = %d, %d, %d",
						r0, g0, b0, c, m, y, k, r1, g1, b1)
				}
			}
		}
	}
}

// TestCMYKToRGBConsistency tests that calling the RGBA method (16 bit color)
// then truncating to 8 bits is equivalent to calling the CMYKToRGB function (8
// bit color).
func TestCMYKToRGBConsistency(t *testing.T) {
	for c := 0; c < 256; c += 7 {
		for m := 0; m < 256; m += 5 {
			for y := 0; y < 256; y += 3 {
				for k := 0; k < 256; k += 11 {
					x := CMYK{uint8(c), uint8(m), uint8(y), uint8(k)}
					r0, g0, b0, _ := x.RGBA()
					r1, g1, b1 := uint8(r0>>8), uint8(g0>>8), uint8(b0>>8)
					r2, g2, b2 := CMYKToRGB(x.C, x.M, x.Y, x.K)
					if r1 != r2 || g1 != g2 || b1 != b2 {
						t.Fatalf("c, m, y, k = %d, %d, %d, %d\nr1, g1, b1 = %d, %d, %d\nr2, g2, b2 = %d, %d, %d",
							c, m, y, k, r1, g1, b1, r2, g2, b2)
					}
				}
			}
		}
	}
}

// TestCMYKGray tests that CMYK colors are a superset of Gray colors.
func TestCMYKGray(t *testing.T) {
	for i := 0; i < 256; i++ {
		if err := eq(CMYK{0x00, 0x00, 0x00, uint8(255 - i)}, Gray{uint8(i)}); err != nil {
			t.Errorf("i=0x%02x:\n%v", i, err)
		}
	}
}

func TestPalette(t *testing.T) {
	p := Palette{
		RGBA{0xff, 0xff, 0xff, 0xff},
		RGBA{0x80, 0x00, 0x00, 0xff},
		RGBA{0x7f, 0x00, 0x00, 0x7f},
		RGBA{0x00, 0x00, 0x00, 0x7f},
		RGBA{0x00, 0x00, 0x00, 0x00},
		RGBA{0x40, 0x40, 0x40, 0x40},
	}
	// Check that, for a Palette with no repeated colors, the closest color to
	// each element is itself.
	for i, c := range p {
		j := p.Index(c)
		if i != j {
			t.Errorf("Index(%v): got %d (color = %v), want %d", c, j, p[j], i)
		}
	}
	// Check that finding the closest color considers alpha, not just red,
	// green and blue.
	got := p.Convert(RGBA{0x80, 0x00, 0x00, 0x80})
	want := RGBA{0x7f, 0x00, 0x00, 0x7f}
	if got != want {
		t.Errorf("got %v, want %v", got, want)
	}
}

var sink uint8

func BenchmarkYCbCrToRGB(b *testing.B) {
	// YCbCrToRGB does saturating arithmetic.
	// Low, middle, and high values can take
	// different paths through the generated code.
	b.Run("0", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sink, sink, sink = YCbCrToRGB(0, 0, 0)
		}
	})
	b.Run("128", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sink, sink, sink = YCbCrToRGB(128, 128, 128)
		}
	})
	b.Run("255", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sink, sink, sink = YCbCrToRGB(255, 255, 255)
		}
	})
}

func BenchmarkRGBToYCbCr(b *testing.B) {
	// RGBToYCbCr does saturating arithmetic.
	// Different values can take different paths
	// through the generated code.
	b.Run("0", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sink, sink, sink = RGBToYCbCr(0, 0, 0)
		}
	})
	b.Run("Cb", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sink, sink, sink = RGBToYCbCr(0, 0, 255)
		}
	})
	b.Run("Cr", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sink, sink, sink = RGBToYCbCr(255, 0, 0)
		}
	})
}
