// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The image package implements a basic 2-D image library.
package image

// An Image is a rectangular grid of Colors drawn from a ColorModel.
type Image interface {
	ColorModel() ColorModel
	Width() int
	Height() int
	// At(0, 0) returns the upper-left pixel of the grid.
	// At(Width()-1, Height()-1) returns the lower-right pixel.
	At(x, y int) Color
}

// An RGBA is an in-memory image backed by a 2-D slice of RGBAColor values.
type RGBA struct {
	// The Pixel field's indices are y first, then x, so that At(x, y) == Pixel[y][x].
	Pixel [][]RGBAColor
}

func (p *RGBA) ColorModel() ColorModel { return RGBAColorModel }

func (p *RGBA) Width() int {
	if len(p.Pixel) == 0 {
		return 0
	}
	return len(p.Pixel[0])
}

func (p *RGBA) Height() int { return len(p.Pixel) }

func (p *RGBA) At(x, y int) Color { return p.Pixel[y][x] }

func (p *RGBA) Set(x, y int, c Color) { p.Pixel[y][x] = toRGBAColor(c).(RGBAColor) }

// NewRGBA returns a new RGBA with the given width and height.
func NewRGBA(w, h int) *RGBA {
	pixel := make([][]RGBAColor, h)
	for y := 0; y < h; y++ {
		pixel[y] = make([]RGBAColor, w)
	}
	return &RGBA{pixel}
}

// An RGBA64 is an in-memory image backed by a 2-D slice of RGBA64Color values.
type RGBA64 struct {
	// The Pixel field's indices are y first, then x, so that At(x, y) == Pixel[y][x].
	Pixel [][]RGBA64Color
}

func (p *RGBA64) ColorModel() ColorModel { return RGBA64ColorModel }

func (p *RGBA64) Width() int {
	if len(p.Pixel) == 0 {
		return 0
	}
	return len(p.Pixel[0])
}

func (p *RGBA64) Height() int { return len(p.Pixel) }

func (p *RGBA64) At(x, y int) Color { return p.Pixel[y][x] }

func (p *RGBA64) Set(x, y int, c Color) { p.Pixel[y][x] = toRGBA64Color(c).(RGBA64Color) }

// NewRGBA64 returns a new RGBA64 with the given width and height.
func NewRGBA64(w, h int) *RGBA64 {
	pixel := make([][]RGBA64Color, h)
	for y := 0; y < h; y++ {
		pixel[y] = make([]RGBA64Color, w)
	}
	return &RGBA64{pixel}
}

// A NRGBA is an in-memory image backed by a 2-D slice of NRGBAColor values.
type NRGBA struct {
	// The Pixel field's indices are y first, then x, so that At(x, y) == Pixel[y][x].
	Pixel [][]NRGBAColor
}

func (p *NRGBA) ColorModel() ColorModel { return NRGBAColorModel }

func (p *NRGBA) Width() int {
	if len(p.Pixel) == 0 {
		return 0
	}
	return len(p.Pixel[0])
}

func (p *NRGBA) Height() int { return len(p.Pixel) }

func (p *NRGBA) At(x, y int) Color { return p.Pixel[y][x] }

func (p *NRGBA) Set(x, y int, c Color) { p.Pixel[y][x] = toNRGBAColor(c).(NRGBAColor) }

// NewNRGBA returns a new NRGBA with the given width and height.
func NewNRGBA(w, h int) *NRGBA {
	pixel := make([][]NRGBAColor, h)
	for y := 0; y < h; y++ {
		pixel[y] = make([]NRGBAColor, w)
	}
	return &NRGBA{pixel}
}

// A NRGBA64 is an in-memory image backed by a 2-D slice of NRGBA64Color values.
type NRGBA64 struct {
	// The Pixel field's indices are y first, then x, so that At(x, y) == Pixel[y][x].
	Pixel [][]NRGBA64Color
}

func (p *NRGBA64) ColorModel() ColorModel { return NRGBA64ColorModel }

func (p *NRGBA64) Width() int {
	if len(p.Pixel) == 0 {
		return 0
	}
	return len(p.Pixel[0])
}

func (p *NRGBA64) Height() int { return len(p.Pixel) }

func (p *NRGBA64) At(x, y int) Color { return p.Pixel[y][x] }

func (p *NRGBA64) Set(x, y int, c Color) { p.Pixel[y][x] = toNRGBA64Color(c).(NRGBA64Color) }

// NewNRGBA64 returns a new NRGBA64 with the given width and height.
func NewNRGBA64(w, h int) *NRGBA64 {
	pixel := make([][]NRGBA64Color, h)
	for y := 0; y < h; y++ {
		pixel[y] = make([]NRGBA64Color, w)
	}
	return &NRGBA64{pixel}
}

// An Alpha is an in-memory image backed by a 2-D slice of AlphaColor values.
type Alpha struct {
	// The Pixel field's indices are y first, then x, so that At(x, y) == Pixel[y][x].
	Pixel [][]AlphaColor
}

func (p *Alpha) ColorModel() ColorModel { return AlphaColorModel }

func (p *Alpha) Width() int {
	if len(p.Pixel) == 0 {
		return 0
	}
	return len(p.Pixel[0])
}

func (p *Alpha) Height() int { return len(p.Pixel) }

func (p *Alpha) At(x, y int) Color { return p.Pixel[y][x] }

func (p *Alpha) Set(x, y int, c Color) { p.Pixel[y][x] = toAlphaColor(c).(AlphaColor) }

// NewAlpha returns a new Alpha with the given width and height.
func NewAlpha(w, h int) *Alpha {
	pixel := make([][]AlphaColor, h)
	for y := 0; y < h; y++ {
		pixel[y] = make([]AlphaColor, w)
	}
	return &Alpha{pixel}
}

// A PalettedColorModel represents a fixed palette of colors.
type PalettedColorModel []Color

func diff(a, b uint32) uint32 {
	if a > b {
		return a - b
	}
	return b - a
}

// Convert returns the palette color closest to c in Euclidean R,G,B space.
func (p PalettedColorModel) Convert(c Color) Color {
	if len(p) == 0 {
		return nil
	}
	// TODO(nigeltao): Revisit the "pick the palette color which minimizes sum-squared-difference"
	// algorithm when the premultiplied vs unpremultiplied issue is resolved.
	// Currently, we only compare the R, G and B values, and ignore A.
	cr, cg, cb, _ := c.RGBA()
	// Shift by 17 bits to avoid potential uint32 overflow in sum-squared-difference.
	cr >>= 17
	cg >>= 17
	cb >>= 17
	result := Color(nil)
	bestSSD := uint32(1<<32 - 1)
	for _, v := range p {
		vr, vg, vb, _ := v.RGBA()
		vr >>= 17
		vg >>= 17
		vb >>= 17
		dr, dg, db := diff(cr, vr), diff(cg, vg), diff(cb, vb)
		ssd := (dr * dr) + (dg * dg) + (db * db)
		if ssd < bestSSD {
			bestSSD = ssd
			result = v
		}
	}
	return result
}

// A Paletted is an in-memory image backed by a 2-D slice of uint8 values and a PalettedColorModel.
type Paletted struct {
	// The Pixel field's indices are y first, then x, so that At(x, y) == Palette[Pixel[y][x]].
	Pixel   [][]uint8
	Palette PalettedColorModel
}

func (p *Paletted) ColorModel() ColorModel { return p.Palette }

func (p *Paletted) Width() int {
	if len(p.Pixel) == 0 {
		return 0
	}
	return len(p.Pixel[0])
}

func (p *Paletted) Height() int { return len(p.Pixel) }

func (p *Paletted) At(x, y int) Color { return p.Palette[p.Pixel[y][x]] }

func (p *Paletted) ColorIndexAt(x, y int) uint8 {
	return p.Pixel[y][x]
}

func (p *Paletted) SetColorIndex(x, y int, index uint8) {
	p.Pixel[y][x] = index
}

// NewPaletted returns a new Paletted with the given width, height and palette.
func NewPaletted(w, h int, m PalettedColorModel) *Paletted {
	pixel := make([][]uint8, h)
	for y := 0; y < h; y++ {
		pixel[y] = make([]uint8, w)
	}
	return &Paletted{pixel, m}
}
