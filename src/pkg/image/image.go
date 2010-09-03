// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The image package implements a basic 2-D image library.
package image

// A Config consists of an image's color model and dimensions.
type Config struct {
	ColorModel    ColorModel
	Width, Height int
}

// An Image is a finite rectangular grid of Colors drawn from a ColorModel.
type Image interface {
	// ColorModel returns the Image's ColorModel.
	ColorModel() ColorModel
	// Bounds returns the domain for which At can return non-zero color.
	// The bounds do not necessarily contain the point (0, 0).
	Bounds() Rectangle
	// At returns the color of the pixel at (x, y).
	// At(Bounds().Min.X, Bounds().Min.Y) returns the upper-left pixel of the grid.
	// At(Bounds().Max.X-1, Bounds().Max.Y-1) returns the lower-right one.
	At(x, y int) Color
}

// An RGBA is an in-memory image of RGBAColor values.
type RGBA struct {
	// Pix holds the image's pixels. The pixel at (x, y) is Pix[y*Stride+x].
	Pix    []RGBAColor
	Stride int
	// Rect is the image's bounds.
	Rect Rectangle
}

func (p *RGBA) ColorModel() ColorModel { return RGBAColorModel }

func (p *RGBA) Bounds() Rectangle { return p.Rect }

func (p *RGBA) At(x, y int) Color {
	if !p.Rect.Contains(Point{x, y}) {
		return RGBAColor{}
	}
	return p.Pix[y*p.Stride+x]
}

func (p *RGBA) Set(x, y int, c Color) {
	if !p.Rect.Contains(Point{x, y}) {
		return
	}
	p.Pix[y*p.Stride+x] = toRGBAColor(c).(RGBAColor)
}

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *RGBA) Opaque() bool {
	if p.Rect.Empty() {
		return true
	}
	base := p.Rect.Min.Y * p.Stride
	i0, i1 := base+p.Rect.Min.X, base+p.Rect.Max.X
	for y := p.Rect.Min.Y; y < p.Rect.Max.Y; y++ {
		for _, c := range p.Pix[i0:i1] {
			if c.A != 0xff {
				return false
			}
		}
		i0 += p.Stride
		i1 += p.Stride
	}
	return true
}

// NewRGBA returns a new RGBA with the given width and height.
func NewRGBA(w, h int) *RGBA {
	buf := make([]RGBAColor, w*h)
	return &RGBA{buf, w, Rectangle{ZP, Point{w, h}}}
}

// An RGBA64 is an in-memory image of RGBA64Color values.
type RGBA64 struct {
	// Pix holds the image's pixels. The pixel at (x, y) is Pix[y*Stride+x].
	Pix    []RGBA64Color
	Stride int
	// Rect is the image's bounds.
	Rect Rectangle
}

func (p *RGBA64) ColorModel() ColorModel { return RGBA64ColorModel }

func (p *RGBA64) Bounds() Rectangle { return p.Rect }

func (p *RGBA64) At(x, y int) Color {
	if !p.Rect.Contains(Point{x, y}) {
		return RGBA64Color{}
	}
	return p.Pix[y*p.Stride+x]
}

func (p *RGBA64) Set(x, y int, c Color) {
	if !p.Rect.Contains(Point{x, y}) {
		return
	}
	p.Pix[y*p.Stride+x] = toRGBA64Color(c).(RGBA64Color)
}

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *RGBA64) Opaque() bool {
	if p.Rect.Empty() {
		return true
	}
	base := p.Rect.Min.Y * p.Stride
	i0, i1 := base+p.Rect.Min.X, base+p.Rect.Max.X
	for y := p.Rect.Min.Y; y < p.Rect.Max.Y; y++ {
		for _, c := range p.Pix[i0:i1] {
			if c.A != 0xffff {
				return false
			}
		}
		i0 += p.Stride
		i1 += p.Stride
	}
	return true
}

// NewRGBA64 returns a new RGBA64 with the given width and height.
func NewRGBA64(w, h int) *RGBA64 {
	pix := make([]RGBA64Color, w*h)
	return &RGBA64{pix, w, Rectangle{ZP, Point{w, h}}}
}

// An NRGBA is an in-memory image of NRGBAColor values.
type NRGBA struct {
	// Pix holds the image's pixels. The pixel at (x, y) is Pix[y*Stride+x].
	Pix    []NRGBAColor
	Stride int
	// Rect is the image's bounds.
	Rect Rectangle
}

func (p *NRGBA) ColorModel() ColorModel { return NRGBAColorModel }

func (p *NRGBA) Bounds() Rectangle { return p.Rect }

func (p *NRGBA) At(x, y int) Color {
	if !p.Rect.Contains(Point{x, y}) {
		return NRGBAColor{}
	}
	return p.Pix[y*p.Stride+x]
}

func (p *NRGBA) Set(x, y int, c Color) {
	if !p.Rect.Contains(Point{x, y}) {
		return
	}
	p.Pix[y*p.Stride+x] = toNRGBAColor(c).(NRGBAColor)
}

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *NRGBA) Opaque() bool {
	if p.Rect.Empty() {
		return true
	}
	base := p.Rect.Min.Y * p.Stride
	i0, i1 := base+p.Rect.Min.X, base+p.Rect.Max.X
	for y := p.Rect.Min.Y; y < p.Rect.Max.Y; y++ {
		for _, c := range p.Pix[i0:i1] {
			if c.A != 0xff {
				return false
			}
		}
		i0 += p.Stride
		i1 += p.Stride
	}
	return true
}

// NewNRGBA returns a new NRGBA with the given width and height.
func NewNRGBA(w, h int) *NRGBA {
	pix := make([]NRGBAColor, w*h)
	return &NRGBA{pix, w, Rectangle{ZP, Point{w, h}}}
}

// An NRGBA64 is an in-memory image of NRGBA64Color values.
type NRGBA64 struct {
	// Pix holds the image's pixels. The pixel at (x, y) is Pix[y*Stride+x].
	Pix    []NRGBA64Color
	Stride int
	// Rect is the image's bounds.
	Rect Rectangle
}

func (p *NRGBA64) ColorModel() ColorModel { return NRGBA64ColorModel }

func (p *NRGBA64) Bounds() Rectangle { return p.Rect }

func (p *NRGBA64) At(x, y int) Color {
	if !p.Rect.Contains(Point{x, y}) {
		return NRGBA64Color{}
	}
	return p.Pix[y*p.Stride+x]
}

func (p *NRGBA64) Set(x, y int, c Color) {
	if !p.Rect.Contains(Point{x, y}) {
		return
	}
	p.Pix[y*p.Stride+x] = toNRGBA64Color(c).(NRGBA64Color)
}

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *NRGBA64) Opaque() bool {
	if p.Rect.Empty() {
		return true
	}
	base := p.Rect.Min.Y * p.Stride
	i0, i1 := base+p.Rect.Min.X, base+p.Rect.Max.X
	for y := p.Rect.Min.Y; y < p.Rect.Max.Y; y++ {
		for _, c := range p.Pix[i0:i1] {
			if c.A != 0xffff {
				return false
			}
		}
		i0 += p.Stride
		i1 += p.Stride
	}
	return true
}

// NewNRGBA64 returns a new NRGBA64 with the given width and height.
func NewNRGBA64(w, h int) *NRGBA64 {
	pix := make([]NRGBA64Color, w*h)
	return &NRGBA64{pix, w, Rectangle{ZP, Point{w, h}}}
}

// An Alpha is an in-memory image of AlphaColor values.
type Alpha struct {
	// Pix holds the image's pixels. The pixel at (x, y) is Pix[y*Stride+x].
	Pix    []AlphaColor
	Stride int
	// Rect is the image's bounds.
	Rect Rectangle
}

func (p *Alpha) ColorModel() ColorModel { return AlphaColorModel }

func (p *Alpha) Bounds() Rectangle { return p.Rect }

func (p *Alpha) At(x, y int) Color {
	if !p.Rect.Contains(Point{x, y}) {
		return AlphaColor{}
	}
	return p.Pix[y*p.Stride+x]
}

func (p *Alpha) Set(x, y int, c Color) {
	if !p.Rect.Contains(Point{x, y}) {
		return
	}
	p.Pix[y*p.Stride+x] = toAlphaColor(c).(AlphaColor)
}

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *Alpha) Opaque() bool {
	if p.Rect.Empty() {
		return true
	}
	base := p.Rect.Min.Y * p.Stride
	i0, i1 := base+p.Rect.Min.X, base+p.Rect.Max.X
	for y := p.Rect.Min.Y; y < p.Rect.Max.Y; y++ {
		for _, c := range p.Pix[i0:i1] {
			if c.A != 0xff {
				return false
			}
		}
		i0 += p.Stride
		i1 += p.Stride
	}
	return true
}

// NewAlpha returns a new Alpha with the given width and height.
func NewAlpha(w, h int) *Alpha {
	pix := make([]AlphaColor, w*h)
	return &Alpha{pix, w, Rectangle{ZP, Point{w, h}}}
}

// An Alpha16 is an in-memory image of Alpha16Color values.
type Alpha16 struct {
	// Pix holds the image's pixels. The pixel at (x, y) is Pix[y*Stride+x].
	Pix    []Alpha16Color
	Stride int
	// Rect is the image's bounds.
	Rect Rectangle
}

func (p *Alpha16) ColorModel() ColorModel { return Alpha16ColorModel }

func (p *Alpha16) Bounds() Rectangle { return p.Rect }

func (p *Alpha16) At(x, y int) Color {
	if !p.Rect.Contains(Point{x, y}) {
		return Alpha16Color{}
	}
	return p.Pix[y*p.Stride+x]
}

func (p *Alpha16) Set(x, y int, c Color) {
	if !p.Rect.Contains(Point{x, y}) {
		return
	}
	p.Pix[y*p.Stride+x] = toAlpha16Color(c).(Alpha16Color)
}

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *Alpha16) Opaque() bool {
	if p.Rect.Empty() {
		return true
	}
	base := p.Rect.Min.Y * p.Stride
	i0, i1 := base+p.Rect.Min.X, base+p.Rect.Max.X
	for y := p.Rect.Min.Y; y < p.Rect.Max.Y; y++ {
		for _, c := range p.Pix[i0:i1] {
			if c.A != 0xffff {
				return false
			}
		}
		i0 += p.Stride
		i1 += p.Stride
	}
	return true
}

// NewAlpha16 returns a new Alpha16 with the given width and height.
func NewAlpha16(w, h int) *Alpha16 {
	pix := make([]Alpha16Color, w*h)
	return &Alpha16{pix, w, Rectangle{ZP, Point{w, h}}}
}

// A Gray is an in-memory image of GrayColor values.
type Gray struct {
	// Pix holds the image's pixels. The pixel at (x, y) is Pix[y*Stride+x].
	Pix    []GrayColor
	Stride int
	// Rect is the image's bounds.
	Rect Rectangle
}

func (p *Gray) ColorModel() ColorModel { return GrayColorModel }

func (p *Gray) Bounds() Rectangle { return p.Rect }

func (p *Gray) At(x, y int) Color {
	if !p.Rect.Contains(Point{x, y}) {
		return GrayColor{}
	}
	return p.Pix[y*p.Stride+x]
}

func (p *Gray) Set(x, y int, c Color) {
	if !p.Rect.Contains(Point{x, y}) {
		return
	}
	p.Pix[y*p.Stride+x] = toGrayColor(c).(GrayColor)
}

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *Gray) Opaque() bool {
	return true
}

// NewGray returns a new Gray with the given width and height.
func NewGray(w, h int) *Gray {
	pix := make([]GrayColor, w*h)
	return &Gray{pix, w, Rectangle{ZP, Point{w, h}}}
}

// A Gray16 is an in-memory image of Gray16Color values.
type Gray16 struct {
	// Pix holds the image's pixels. The pixel at (x, y) is Pix[y*Stride+x].
	Pix    []Gray16Color
	Stride int
	// Rect is the image's bounds.
	Rect Rectangle
}

func (p *Gray16) ColorModel() ColorModel { return Gray16ColorModel }

func (p *Gray16) Bounds() Rectangle { return p.Rect }

func (p *Gray16) At(x, y int) Color {
	if !p.Rect.Contains(Point{x, y}) {
		return Gray16Color{}
	}
	return p.Pix[y*p.Stride+x]
}

func (p *Gray16) Set(x, y int, c Color) {
	if !p.Rect.Contains(Point{x, y}) {
		return
	}
	p.Pix[y*p.Stride+x] = toGray16Color(c).(Gray16Color)
}

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *Gray16) Opaque() bool {
	return true
}

// NewGray16 returns a new Gray16 with the given width and height.
func NewGray16(w, h int) *Gray16 {
	pix := make([]Gray16Color, w*h)
	return &Gray16{pix, w, Rectangle{ZP, Point{w, h}}}
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
	// Pix holds the image's pixels. The pixel at (x, y) is Pix[y*Stride+x].
	Pix    []uint8
	Stride int
	// Rect is the image's bounds.
	Rect Rectangle
	// Palette is the image's palette.
	Palette PalettedColorModel
}

func (p *Paletted) ColorModel() ColorModel { return p.Palette }

func (p *Paletted) Bounds() Rectangle { return p.Rect }

func (p *Paletted) At(x, y int) Color {
	if len(p.Palette) == 0 {
		return nil
	}
	if !p.Rect.Contains(Point{x, y}) {
		return p.Palette[0]
	}
	return p.Palette[p.Pix[y*p.Stride+x]]
}

func (p *Paletted) ColorIndexAt(x, y int) uint8 {
	if !p.Rect.Contains(Point{x, y}) {
		return 0
	}
	return p.Pix[y*p.Stride+x]
}

func (p *Paletted) SetColorIndex(x, y int, index uint8) {
	if !p.Rect.Contains(Point{x, y}) {
		return
	}
	p.Pix[y*p.Stride+x] = index
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
	pix := make([]uint8, w*h)
	return &Paletted{pix, w, Rectangle{ZP, Point{w, h}}, m}
}
