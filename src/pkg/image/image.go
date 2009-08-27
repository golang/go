// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The image package implements a basic 2-D image library.
package image

// An Image is a rectangular grid of Colors drawn from a ColorModel.
type Image interface {
	ColorModel() ColorModel;
	Width() int;
	Height() int;
	// At(0, 0) returns the upper-left pixel of the grid.
	// At(Width()-1, Height()-1) returns the lower-right pixel.
	At(x, y int) Color;
}

// An RGBA is an in-memory image backed by a 2-D slice of RGBAColor values.
type RGBA struct {
	// The Pixel field's indices are y first, then x, so that At(x, y) == Pixel[y][x].
	Pixel [][]RGBAColor;
}

func (p *RGBA) ColorModel() ColorModel {
	return RGBAColorModel;
}

func (p *RGBA) Width() int {
	if len(p.Pixel) == 0 {
		return 0;
	}
	return len(p.Pixel[0]);
}

func (p *RGBA) Height() int {
	return len(p.Pixel);
}

func (p *RGBA) At(x, y int) Color {
	return p.Pixel[y][x];
}

func (p *RGBA) Set(x, y int, c Color) {
	p.Pixel[y][x] = toRGBAColor(c).(RGBAColor);
}

// An RGBA64 is an in-memory image backed by a 2-D slice of RGBA64Color values.
type RGBA64 struct {
	// The Pixel field's indices are y first, then x, so that At(x, y) == Pixel[y][x].
	Pixel [][]RGBA64Color;
}

func (p *RGBA64) ColorModel() ColorModel {
	return RGBA64ColorModel;
}

func (p *RGBA64) Width() int {
	if len(p.Pixel) == 0 {
		return 0;
	}
	return len(p.Pixel[0]);
}

func (p *RGBA64) Height() int {
	return len(p.Pixel);
}

func (p *RGBA64) At(x, y int) Color {
	return p.Pixel[y][x];
}

func (p *RGBA64) Set(x, y int, c Color) {
	p.Pixel[y][x] = toRGBA64Color(c).(RGBA64Color);
}

// A PalettedColorModel represents a fixed palette of colors.
type PalettedColorModel []Color;

func diff(a, b uint32) uint32 {
	if a > b {
		return a - b;
	}
	return b - a;
}

// Convert returns the palette color closest to c in Euclidean R,G,B space.
func (p PalettedColorModel) Convert(c Color) Color {
	if len(p) == 0 {
		return nil;
	}
	// TODO(nigeltao): Revisit the "pick the palette color which minimizes sum-squared-difference"
	// algorithm when the premultiplied vs unpremultiplied issue is resolved.
	// Currently, we only compare the R, G and B values, and ignore A.
	cr, cg, cb, ca := c.RGBA();
	// Shift by 17 bits to avoid potential uint32 overflow in sum-squared-difference.
	cr >>= 17;
	cg >>= 17;
	cb >>= 17;
	result := Color(nil);
	bestSSD := uint32(1<<32 - 1);
	for _, v := range p {
		vr, vg, vb, va := v.RGBA();
		vr >>= 17;
		vg >>= 17;
		vb >>= 17;
		dr, dg, db := diff(cr, vr), diff(cg, vg), diff(cb, vb);
		ssd := (dr * dr) + (dg * dg) + (db * db);
		if ssd < bestSSD {
			bestSSD = ssd;
			result = v;
		}
	}
	return result;
}

// A Paletted is an in-memory image backed by a 2-D slice of byte values and a PalettedColorModel.
type Paletted struct {
	// The Pixel field's indices are y first, then x, so that At(x, y) == Palette[Pixel[y][x]].
	Pixel [][]byte;
	Palette PalettedColorModel;
}

func (p *Paletted) ColorModel() ColorModel {
	return p.Palette;
}

func (p *Paletted) Width() int {
	if len(p.Pixel) == 0 {
		return 0;
	}
	return len(p.Pixel[0]);
}

func (p *Paletted) Height() int {
	return len(p.Pixel);
}

func (p *Paletted) At(x, y int) Color {
	return p.Palette[p.Pixel[y][x]];
}
