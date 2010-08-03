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

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *RGBA) Opaque() bool {
	h := len(p.Pixel)
	if h > 0 {
		w := len(p.Pixel[0])
		for y := 0; y < h; y++ {
			pix := p.Pixel[y]
			for x := 0; x < w; x++ {
				if pix[x].A != 0xff {
					return false
				}
			}
		}
	}
	return true
}

// NewRGBA returns a new RGBA with the given width and height.
func NewRGBA(w, h int) *RGBA {
	buf := make([]RGBAColor, w*h)
	pix := make([][]RGBAColor, h)
	for y := range pix {
		pix[y] = buf[w*y : w*(y+1)]
	}
	return &RGBA{pix}
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

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *RGBA64) Opaque() bool {
	h := len(p.Pixel)
	if h > 0 {
		w := len(p.Pixel[0])
		for y := 0; y < h; y++ {
			pix := p.Pixel[y]
			for x := 0; x < w; x++ {
				if pix[x].A != 0xffff {
					return false
				}
			}
		}
	}
	return true
}

// NewRGBA64 returns a new RGBA64 with the given width and height.
func NewRGBA64(w, h int) *RGBA64 {
	buf := make([]RGBA64Color, w*h)
	pix := make([][]RGBA64Color, h)
	for y := range pix {
		pix[y] = buf[w*y : w*(y+1)]
	}
	return &RGBA64{pix}
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

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *NRGBA) Opaque() bool {
	h := len(p.Pixel)
	if h > 0 {
		w := len(p.Pixel[0])
		for y := 0; y < h; y++ {
			pix := p.Pixel[y]
			for x := 0; x < w; x++ {
				if pix[x].A != 0xff {
					return false
				}
			}
		}
	}
	return true
}

// NewNRGBA returns a new NRGBA with the given width and height.
func NewNRGBA(w, h int) *NRGBA {
	buf := make([]NRGBAColor, w*h)
	pix := make([][]NRGBAColor, h)
	for y := range pix {
		pix[y] = buf[w*y : w*(y+1)]
	}
	return &NRGBA{pix}
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

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *NRGBA64) Opaque() bool {
	h := len(p.Pixel)
	if h > 0 {
		w := len(p.Pixel[0])
		for y := 0; y < h; y++ {
			pix := p.Pixel[y]
			for x := 0; x < w; x++ {
				if pix[x].A != 0xffff {
					return false
				}
			}
		}
	}
	return true
}

// NewNRGBA64 returns a new NRGBA64 with the given width and height.
func NewNRGBA64(w, h int) *NRGBA64 {
	buf := make([]NRGBA64Color, w*h)
	pix := make([][]NRGBA64Color, h)
	for y := range pix {
		pix[y] = buf[w*y : w*(y+1)]
	}
	return &NRGBA64{pix}
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

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *Alpha) Opaque() bool {
	h := len(p.Pixel)
	if h > 0 {
		w := len(p.Pixel[0])
		for y := 0; y < h; y++ {
			pix := p.Pixel[y]
			for x := 0; x < w; x++ {
				if pix[x].A != 0xff {
					return false
				}
			}
		}
	}
	return true
}

// NewAlpha returns a new Alpha with the given width and height.
func NewAlpha(w, h int) *Alpha {
	buf := make([]AlphaColor, w*h)
	pix := make([][]AlphaColor, h)
	for y := range pix {
		pix[y] = buf[w*y : w*(y+1)]
	}
	return &Alpha{pix}
}

// An Alpha16 is an in-memory image backed by a 2-D slice of Alpha16Color values.
type Alpha16 struct {
	// The Pixel field's indices are y first, then x, so that At(x, y) == Pixel[y][x].
	Pixel [][]Alpha16Color
}

func (p *Alpha16) ColorModel() ColorModel { return Alpha16ColorModel }

func (p *Alpha16) Width() int {
	if len(p.Pixel) == 0 {
		return 0
	}
	return len(p.Pixel[0])
}

func (p *Alpha16) Height() int { return len(p.Pixel) }

func (p *Alpha16) At(x, y int) Color { return p.Pixel[y][x] }

func (p *Alpha16) Set(x, y int, c Color) { p.Pixel[y][x] = toAlpha16Color(c).(Alpha16Color) }

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *Alpha16) Opaque() bool {
	h := len(p.Pixel)
	if h > 0 {
		w := len(p.Pixel[0])
		for y := 0; y < h; y++ {
			pix := p.Pixel[y]
			for x := 0; x < w; x++ {
				if pix[x].A != 0xffff {
					return false
				}
			}
		}
	}
	return true
}

// NewAlpha16 returns a new Alpha16 with the given width and height.
func NewAlpha16(w, h int) *Alpha16 {
	buf := make([]Alpha16Color, w*h)
	pix := make([][]Alpha16Color, h)
	for y := range pix {
		pix[y] = buf[w*y : w*(y+1)]
	}
	return &Alpha16{pix}
}

// A Gray is an in-memory image backed by a 2-D slice of GrayColor values.
type Gray struct {
	// The Pixel field's indices are y first, then x, so that At(x, y) == Pixel[y][x].
	Pixel [][]GrayColor
}

func (p *Gray) ColorModel() ColorModel { return GrayColorModel }

func (p *Gray) Width() int {
	if len(p.Pixel) == 0 {
		return 0
	}
	return len(p.Pixel[0])
}

func (p *Gray) Height() int { return len(p.Pixel) }

func (p *Gray) At(x, y int) Color { return p.Pixel[y][x] }

func (p *Gray) Set(x, y int, c Color) { p.Pixel[y][x] = toGrayColor(c).(GrayColor) }

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *Gray) Opaque() bool {
	return true
}

// NewGray returns a new Gray with the given width and height.
func NewGray(w, h int) *Gray {
	buf := make([]GrayColor, w*h)
	pix := make([][]GrayColor, h)
	for y := range pix {
		pix[y] = buf[w*y : w*(y+1)]
	}
	return &Gray{pix}
}

// A Gray16 is an in-memory image backed by a 2-D slice of Gray16Color values.
type Gray16 struct {
	// The Pixel field's indices are y first, then x, so that At(x, y) == Pixel[y][x].
	Pixel [][]Gray16Color
}

func (p *Gray16) ColorModel() ColorModel { return Gray16ColorModel }

func (p *Gray16) Width() int {
	if len(p.Pixel) == 0 {
		return 0
	}
	return len(p.Pixel[0])
}

func (p *Gray16) Height() int { return len(p.Pixel) }

func (p *Gray16) At(x, y int) Color { return p.Pixel[y][x] }

func (p *Gray16) Set(x, y int, c Color) { p.Pixel[y][x] = toGray16Color(c).(Gray16Color) }

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *Gray16) Opaque() bool {
	return true
}

// NewGray16 returns a new Gray16 with the given width and height.
func NewGray16(w, h int) *Gray16 {
	buf := make([]Gray16Color, w*h)
	pix := make([][]Gray16Color, h)
	for y := range pix {
		pix[y] = buf[w*y : w*(y+1)]
	}
	return &Gray16{pix}
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
	cr, cg, cb, _ := c.RGBA()
	// Shift by 1 bit to avoid potential uint32 overflow in sum-squared-difference.
	cr >>= 1
	cg >>= 1
	cb >>= 1
	result := Color(nil)
	bestSSD := uint32(1<<32 - 1)
	for _, v := range p {
		vr, vg, vb, _ := v.RGBA()
		vr >>= 1
		vg >>= 1
		vb >>= 1
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

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *Paletted) Opaque() bool {
	for _, c := range p.Palette {
		_, _, _, a := c.RGBA()
		if a != 0xffff {
			return false
		}
	}
	return true
}

// NewPaletted returns a new Paletted with the given width, height and palette.
func NewPaletted(w, h int, m PalettedColorModel) *Paletted {
	buf := make([]uint8, w*h)
	pix := make([][]uint8, h)
	for y := range pix {
		pix[y] = buf[w*y : w*(y+1)]
	}
	return &Paletted{pix, m}
}
