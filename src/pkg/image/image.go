// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package image implements a basic 2-D image library.
package image

// Config holds an image's color model and dimensions.
type Config struct {
	ColorModel    ColorModel
	Width, Height int
}

// Image is a finite rectangular grid of Colors drawn from a ColorModel.
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

// RGBA is an in-memory image of RGBAColor values.
type RGBA struct {
	// Pix holds the image's pixels, in R, G, B, A order. The pixel at
	// (x, y) starts at Pix[(y-Rect.Min.Y)*Stride + (x-Rect.Min.X)*4].
	Pix []uint8
	// Stride is the Pix stride (in bytes) between vertically adjacent pixels.
	Stride int
	// Rect is the image's bounds.
	Rect Rectangle
}

func (p *RGBA) ColorModel() ColorModel { return RGBAColorModel }

func (p *RGBA) Bounds() Rectangle { return p.Rect }

func (p *RGBA) At(x, y int) Color {
	if !(Point{x, y}.In(p.Rect)) {
		return RGBAColor{}
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*4
	return RGBAColor{p.Pix[i+0], p.Pix[i+1], p.Pix[i+2], p.Pix[i+3]}
}

func (p *RGBA) Set(x, y int, c Color) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*4
	c1 := toRGBAColor(c).(RGBAColor)
	p.Pix[i+0] = c1.R
	p.Pix[i+1] = c1.G
	p.Pix[i+2] = c1.B
	p.Pix[i+3] = c1.A
}

func (p *RGBA) SetRGBA(x, y int, c RGBAColor) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*4
	p.Pix[i+0] = c.R
	p.Pix[i+1] = c.G
	p.Pix[i+2] = c.B
	p.Pix[i+3] = c.A
}

// SubImage returns an image representing the portion of the image p visible
// through r. The returned value shares pixels with the original image.
func (p *RGBA) SubImage(r Rectangle) Image {
	r = r.Intersect(p.Rect)
	// If r1 and r2 are Rectangles, r1.Intersect(r2) is not guaranteed to be inside
	// either r1 or r2 if the intersection is empty. Without explicitly checking for
	// this, the Pix[i:] expression below can panic.
	if r.Empty() {
		return &RGBA{}
	}
	i := (r.Min.Y-p.Rect.Min.Y)*p.Stride + (r.Min.X-p.Rect.Min.X)*4
	return &RGBA{
		Pix:    p.Pix[i:],
		Stride: p.Stride,
		Rect:   r,
	}
}

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *RGBA) Opaque() bool {
	if p.Rect.Empty() {
		return true
	}
	i0, i1 := 3, p.Rect.Dx()*4
	for y := p.Rect.Min.Y; y < p.Rect.Max.Y; y++ {
		for i := i0; i < i1; i += 4 {
			if p.Pix[i] != 0xff {
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
	buf := make([]uint8, 4*w*h)
	return &RGBA{buf, 4 * w, Rectangle{ZP, Point{w, h}}}
}

// RGBA64 is an in-memory image of RGBA64Color values.
type RGBA64 struct {
	// Pix holds the image's pixels, in R, G, B, A order and big-endian format. The pixel at
	// (x, y) starts at Pix[(y-Rect.Min.Y)*Stride + (x-Rect.Min.X)*8].
	Pix []uint8
	// Stride is the Pix stride (in bytes) between vertically adjacent pixels.
	Stride int
	// Rect is the image's bounds.
	Rect Rectangle
}

func (p *RGBA64) ColorModel() ColorModel { return RGBA64ColorModel }

func (p *RGBA64) Bounds() Rectangle { return p.Rect }

func (p *RGBA64) At(x, y int) Color {
	if !(Point{x, y}.In(p.Rect)) {
		return RGBA64Color{}
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*8
	return RGBA64Color{
		uint16(p.Pix[i+0])<<8 | uint16(p.Pix[i+1]),
		uint16(p.Pix[i+2])<<8 | uint16(p.Pix[i+3]),
		uint16(p.Pix[i+4])<<8 | uint16(p.Pix[i+5]),
		uint16(p.Pix[i+6])<<8 | uint16(p.Pix[i+7]),
	}
}

func (p *RGBA64) Set(x, y int, c Color) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*8
	c1 := toRGBA64Color(c).(RGBA64Color)
	p.Pix[i+0] = uint8(c1.R >> 8)
	p.Pix[i+1] = uint8(c1.R)
	p.Pix[i+2] = uint8(c1.G >> 8)
	p.Pix[i+3] = uint8(c1.G)
	p.Pix[i+4] = uint8(c1.B >> 8)
	p.Pix[i+5] = uint8(c1.B)
	p.Pix[i+6] = uint8(c1.A >> 8)
	p.Pix[i+7] = uint8(c1.A)
}

func (p *RGBA64) SetRGBA64(x, y int, c RGBA64Color) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*8
	p.Pix[i+0] = uint8(c.R >> 8)
	p.Pix[i+1] = uint8(c.R)
	p.Pix[i+2] = uint8(c.G >> 8)
	p.Pix[i+3] = uint8(c.G)
	p.Pix[i+4] = uint8(c.B >> 8)
	p.Pix[i+5] = uint8(c.B)
	p.Pix[i+6] = uint8(c.A >> 8)
	p.Pix[i+7] = uint8(c.A)
}

// SubImage returns an image representing the portion of the image p visible
// through r. The returned value shares pixels with the original image.
func (p *RGBA64) SubImage(r Rectangle) Image {
	r = r.Intersect(p.Rect)
	// If r1 and r2 are Rectangles, r1.Intersect(r2) is not guaranteed to be inside
	// either r1 or r2 if the intersection is empty. Without explicitly checking for
	// this, the Pix[i:] expression below can panic.
	if r.Empty() {
		return &RGBA64{}
	}
	i := (r.Min.Y-p.Rect.Min.Y)*p.Stride + (r.Min.X-p.Rect.Min.X)*8
	return &RGBA64{
		Pix:    p.Pix[i:],
		Stride: p.Stride,
		Rect:   r,
	}
}

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *RGBA64) Opaque() bool {
	if p.Rect.Empty() {
		return true
	}
	i0, i1 := 6, p.Rect.Dx()*8
	for y := p.Rect.Min.Y; y < p.Rect.Max.Y; y++ {
		for i := i0; i < i1; i += 8 {
			if p.Pix[i+0] != 0xff || p.Pix[i+1] != 0xff {
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
	pix := make([]uint8, 8*w*h)
	return &RGBA64{pix, 8 * w, Rectangle{ZP, Point{w, h}}}
}

// NRGBA is an in-memory image of NRGBAColor values.
type NRGBA struct {
	// Pix holds the image's pixels, in R, G, B, A order. The pixel at
	// (x, y) starts at Pix[(y-Rect.Min.Y)*Stride + (x-Rect.Min.X)*4].
	Pix []uint8
	// Stride is the Pix stride (in bytes) between vertically adjacent pixels.
	Stride int
	// Rect is the image's bounds.
	Rect Rectangle
}

func (p *NRGBA) ColorModel() ColorModel { return NRGBAColorModel }

func (p *NRGBA) Bounds() Rectangle { return p.Rect }

func (p *NRGBA) At(x, y int) Color {
	if !(Point{x, y}.In(p.Rect)) {
		return NRGBAColor{}
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*4
	return NRGBAColor{p.Pix[i+0], p.Pix[i+1], p.Pix[i+2], p.Pix[i+3]}
}

func (p *NRGBA) Set(x, y int, c Color) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*4
	c1 := toNRGBAColor(c).(NRGBAColor)
	p.Pix[i+0] = c1.R
	p.Pix[i+1] = c1.G
	p.Pix[i+2] = c1.B
	p.Pix[i+3] = c1.A
}

func (p *NRGBA) SetNRGBA(x, y int, c NRGBAColor) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*4
	p.Pix[i+0] = c.R
	p.Pix[i+1] = c.G
	p.Pix[i+2] = c.B
	p.Pix[i+3] = c.A
}

// SubImage returns an image representing the portion of the image p visible
// through r. The returned value shares pixels with the original image.
func (p *NRGBA) SubImage(r Rectangle) Image {
	r = r.Intersect(p.Rect)
	// If r1 and r2 are Rectangles, r1.Intersect(r2) is not guaranteed to be inside
	// either r1 or r2 if the intersection is empty. Without explicitly checking for
	// this, the Pix[i:] expression below can panic.
	if r.Empty() {
		return &NRGBA{}
	}
	i := (r.Min.Y-p.Rect.Min.Y)*p.Stride + (r.Min.X-p.Rect.Min.X)*4
	return &NRGBA{
		Pix:    p.Pix[i:],
		Stride: p.Stride,
		Rect:   r,
	}
}

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *NRGBA) Opaque() bool {
	if p.Rect.Empty() {
		return true
	}
	i0, i1 := 3, p.Rect.Dx()*4
	for y := p.Rect.Min.Y; y < p.Rect.Max.Y; y++ {
		for i := i0; i < i1; i += 4 {
			if p.Pix[i] != 0xff {
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
	pix := make([]uint8, 4*w*h)
	return &NRGBA{pix, 4 * w, Rectangle{ZP, Point{w, h}}}
}

// NRGBA64 is an in-memory image of NRGBA64Color values.
type NRGBA64 struct {
	// Pix holds the image's pixels, in R, G, B, A order and big-endian format. The pixel at
	// (x, y) starts at Pix[(y-Rect.Min.Y)*Stride + (x-Rect.Min.X)*8].
	Pix []uint8
	// Stride is the Pix stride (in bytes) between vertically adjacent pixels.
	Stride int
	// Rect is the image's bounds.
	Rect Rectangle
}

func (p *NRGBA64) ColorModel() ColorModel { return NRGBA64ColorModel }

func (p *NRGBA64) Bounds() Rectangle { return p.Rect }

func (p *NRGBA64) At(x, y int) Color {
	if !(Point{x, y}.In(p.Rect)) {
		return NRGBA64Color{}
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*8
	return NRGBA64Color{
		uint16(p.Pix[i+0])<<8 | uint16(p.Pix[i+1]),
		uint16(p.Pix[i+2])<<8 | uint16(p.Pix[i+3]),
		uint16(p.Pix[i+4])<<8 | uint16(p.Pix[i+5]),
		uint16(p.Pix[i+6])<<8 | uint16(p.Pix[i+7]),
	}
}

func (p *NRGBA64) Set(x, y int, c Color) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*8
	c1 := toNRGBA64Color(c).(NRGBA64Color)
	p.Pix[i+0] = uint8(c1.R >> 8)
	p.Pix[i+1] = uint8(c1.R)
	p.Pix[i+2] = uint8(c1.G >> 8)
	p.Pix[i+3] = uint8(c1.G)
	p.Pix[i+4] = uint8(c1.B >> 8)
	p.Pix[i+5] = uint8(c1.B)
	p.Pix[i+6] = uint8(c1.A >> 8)
	p.Pix[i+7] = uint8(c1.A)
}

func (p *NRGBA64) SetNRGBA64(x, y int, c NRGBA64Color) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*8
	p.Pix[i+0] = uint8(c.R >> 8)
	p.Pix[i+1] = uint8(c.R)
	p.Pix[i+2] = uint8(c.G >> 8)
	p.Pix[i+3] = uint8(c.G)
	p.Pix[i+4] = uint8(c.B >> 8)
	p.Pix[i+5] = uint8(c.B)
	p.Pix[i+6] = uint8(c.A >> 8)
	p.Pix[i+7] = uint8(c.A)
}

// SubImage returns an image representing the portion of the image p visible
// through r. The returned value shares pixels with the original image.
func (p *NRGBA64) SubImage(r Rectangle) Image {
	r = r.Intersect(p.Rect)
	// If r1 and r2 are Rectangles, r1.Intersect(r2) is not guaranteed to be inside
	// either r1 or r2 if the intersection is empty. Without explicitly checking for
	// this, the Pix[i:] expression below can panic.
	if r.Empty() {
		return &NRGBA64{}
	}
	i := (r.Min.Y-p.Rect.Min.Y)*p.Stride + (r.Min.X-p.Rect.Min.X)*8
	return &NRGBA64{
		Pix:    p.Pix[i:],
		Stride: p.Stride,
		Rect:   r,
	}
}

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *NRGBA64) Opaque() bool {
	if p.Rect.Empty() {
		return true
	}
	i0, i1 := 6, p.Rect.Dx()*8
	for y := p.Rect.Min.Y; y < p.Rect.Max.Y; y++ {
		for i := i0; i < i1; i += 8 {
			if p.Pix[i+0] != 0xff || p.Pix[i+1] != 0xff {
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
	pix := make([]uint8, 8*w*h)
	return &NRGBA64{pix, 8 * w, Rectangle{ZP, Point{w, h}}}
}

// Alpha is an in-memory image of AlphaColor values.
type Alpha struct {
	// Pix holds the image's pixels, as alpha values. The pixel at
	// (x, y) starts at Pix[(y-Rect.Min.Y)*Stride + (x-Rect.Min.X)*1].
	Pix []uint8
	// Stride is the Pix stride (in bytes) between vertically adjacent pixels.
	Stride int
	// Rect is the image's bounds.
	Rect Rectangle
}

func (p *Alpha) ColorModel() ColorModel { return AlphaColorModel }

func (p *Alpha) Bounds() Rectangle { return p.Rect }

func (p *Alpha) At(x, y int) Color {
	if !(Point{x, y}.In(p.Rect)) {
		return AlphaColor{}
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x - p.Rect.Min.X)
	return AlphaColor{p.Pix[i]}
}

func (p *Alpha) Set(x, y int, c Color) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x - p.Rect.Min.X)
	p.Pix[i] = toAlphaColor(c).(AlphaColor).A
}

func (p *Alpha) SetAlpha(x, y int, c AlphaColor) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x - p.Rect.Min.X)
	p.Pix[i] = c.A
}

// SubImage returns an image representing the portion of the image p visible
// through r. The returned value shares pixels with the original image.
func (p *Alpha) SubImage(r Rectangle) Image {
	r = r.Intersect(p.Rect)
	// If r1 and r2 are Rectangles, r1.Intersect(r2) is not guaranteed to be inside
	// either r1 or r2 if the intersection is empty. Without explicitly checking for
	// this, the Pix[i:] expression below can panic.
	if r.Empty() {
		return &Alpha{}
	}
	i := (r.Min.Y-p.Rect.Min.Y)*p.Stride + (r.Min.X-p.Rect.Min.X)*1
	return &Alpha{
		Pix:    p.Pix[i:],
		Stride: p.Stride,
		Rect:   r,
	}
}

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *Alpha) Opaque() bool {
	if p.Rect.Empty() {
		return true
	}
	i0, i1 := 0, p.Rect.Dx()
	for y := p.Rect.Min.Y; y < p.Rect.Max.Y; y++ {
		for i := i0; i < i1; i++ {
			if p.Pix[i] != 0xff {
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
	pix := make([]uint8, 1*w*h)
	return &Alpha{pix, 1 * w, Rectangle{ZP, Point{w, h}}}
}

// Alpha16 is an in-memory image of Alpha16Color values.
type Alpha16 struct {
	// Pix holds the image's pixels, as alpha values in big-endian format. The pixel at
	// (x, y) starts at Pix[(y-Rect.Min.Y)*Stride + (x-Rect.Min.X)*2].
	Pix []uint8
	// Stride is the Pix stride (in bytes) between vertically adjacent pixels.
	Stride int
	// Rect is the image's bounds.
	Rect Rectangle
}

func (p *Alpha16) ColorModel() ColorModel { return Alpha16ColorModel }

func (p *Alpha16) Bounds() Rectangle { return p.Rect }

func (p *Alpha16) At(x, y int) Color {
	if !(Point{x, y}.In(p.Rect)) {
		return Alpha16Color{}
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*2
	return Alpha16Color{uint16(p.Pix[i+0])<<8 | uint16(p.Pix[i+1])}
}

func (p *Alpha16) Set(x, y int, c Color) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*2
	c1 := toAlpha16Color(c).(Alpha16Color)
	p.Pix[i+0] = uint8(c1.A >> 8)
	p.Pix[i+1] = uint8(c1.A)
}

func (p *Alpha16) SetAlpha16(x, y int, c Alpha16Color) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*2
	p.Pix[i+0] = uint8(c.A >> 8)
	p.Pix[i+1] = uint8(c.A)
}

// SubImage returns an image representing the portion of the image p visible
// through r. The returned value shares pixels with the original image.
func (p *Alpha16) SubImage(r Rectangle) Image {
	r = r.Intersect(p.Rect)
	// If r1 and r2 are Rectangles, r1.Intersect(r2) is not guaranteed to be inside
	// either r1 or r2 if the intersection is empty. Without explicitly checking for
	// this, the Pix[i:] expression below can panic.
	if r.Empty() {
		return &Alpha16{}
	}
	i := (r.Min.Y-p.Rect.Min.Y)*p.Stride + (r.Min.X-p.Rect.Min.X)*2
	return &Alpha16{
		Pix:    p.Pix[i:],
		Stride: p.Stride,
		Rect:   r,
	}
}

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *Alpha16) Opaque() bool {
	if p.Rect.Empty() {
		return true
	}
	i0, i1 := 0, p.Rect.Dx()*2
	for y := p.Rect.Min.Y; y < p.Rect.Max.Y; y++ {
		for i := i0; i < i1; i += 2 {
			if p.Pix[i+0] != 0xff || p.Pix[i+1] != 0xff {
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
	pix := make([]uint8, 2*w*h)
	return &Alpha16{pix, 2 * w, Rectangle{ZP, Point{w, h}}}
}

// Gray is an in-memory image of GrayColor values.
type Gray struct {
	// Pix holds the image's pixels, as gray values. The pixel at
	// (x, y) starts at Pix[(y-Rect.Min.Y)*Stride + (x-Rect.Min.X)*1].
	Pix []uint8
	// Stride is the Pix stride (in bytes) between vertically adjacent pixels.
	Stride int
	// Rect is the image's bounds.
	Rect Rectangle
}

func (p *Gray) ColorModel() ColorModel { return GrayColorModel }

func (p *Gray) Bounds() Rectangle { return p.Rect }

func (p *Gray) At(x, y int) Color {
	if !(Point{x, y}.In(p.Rect)) {
		return GrayColor{}
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x - p.Rect.Min.X)
	return GrayColor{p.Pix[i]}
}

func (p *Gray) Set(x, y int, c Color) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x - p.Rect.Min.X)
	p.Pix[i] = toGrayColor(c).(GrayColor).Y
}

func (p *Gray) SetGray(x, y int, c GrayColor) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x - p.Rect.Min.X)
	p.Pix[i] = c.Y
}

// SubImage returns an image representing the portion of the image p visible
// through r. The returned value shares pixels with the original image.
func (p *Gray) SubImage(r Rectangle) Image {
	r = r.Intersect(p.Rect)
	// If r1 and r2 are Rectangles, r1.Intersect(r2) is not guaranteed to be inside
	// either r1 or r2 if the intersection is empty. Without explicitly checking for
	// this, the Pix[i:] expression below can panic.
	if r.Empty() {
		return &Gray{}
	}
	i := (r.Min.Y-p.Rect.Min.Y)*p.Stride + (r.Min.X-p.Rect.Min.X)*1
	return &Gray{
		Pix:    p.Pix[i:],
		Stride: p.Stride,
		Rect:   r,
	}
}

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *Gray) Opaque() bool {
	return true
}

// NewGray returns a new Gray with the given width and height.
func NewGray(w, h int) *Gray {
	pix := make([]uint8, 1*w*h)
	return &Gray{pix, 1 * w, Rectangle{ZP, Point{w, h}}}
}

// Gray16 is an in-memory image of Gray16Color values.
type Gray16 struct {
	// Pix holds the image's pixels, as gray values in big-endian format. The pixel at
	// (x, y) starts at Pix[(y-Rect.Min.Y)*Stride + (x-Rect.Min.X)*2].
	Pix []uint8
	// Stride is the Pix stride (in bytes) between vertically adjacent pixels.
	Stride int
	// Rect is the image's bounds.
	Rect Rectangle
}

func (p *Gray16) ColorModel() ColorModel { return Gray16ColorModel }

func (p *Gray16) Bounds() Rectangle { return p.Rect }

func (p *Gray16) At(x, y int) Color {
	if !(Point{x, y}.In(p.Rect)) {
		return Gray16Color{}
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*2
	return Gray16Color{uint16(p.Pix[i+0])<<8 | uint16(p.Pix[i+1])}
}

func (p *Gray16) Set(x, y int, c Color) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*2
	c1 := toGray16Color(c).(Gray16Color)
	p.Pix[i+0] = uint8(c1.Y >> 8)
	p.Pix[i+1] = uint8(c1.Y)
}

func (p *Gray16) SetGray16(x, y int, c Gray16Color) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*2
	p.Pix[i+0] = uint8(c.Y >> 8)
	p.Pix[i+1] = uint8(c.Y)
}

// SubImage returns an image representing the portion of the image p visible
// through r. The returned value shares pixels with the original image.
func (p *Gray16) SubImage(r Rectangle) Image {
	r = r.Intersect(p.Rect)
	// If r1 and r2 are Rectangles, r1.Intersect(r2) is not guaranteed to be inside
	// either r1 or r2 if the intersection is empty. Without explicitly checking for
	// this, the Pix[i:] expression below can panic.
	if r.Empty() {
		return &Gray16{}
	}
	i := (r.Min.Y-p.Rect.Min.Y)*p.Stride + (r.Min.X-p.Rect.Min.X)*2
	return &Gray16{
		Pix:    p.Pix[i:],
		Stride: p.Stride,
		Rect:   r,
	}
}

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *Gray16) Opaque() bool {
	return true
}

// NewGray16 returns a new Gray16 with the given width and height.
func NewGray16(w, h int) *Gray16 {
	pix := make([]uint8, 2*w*h)
	return &Gray16{pix, 2 * w, Rectangle{ZP, Point{w, h}}}
}

// A PalettedColorModel represents a fixed palette of at most 256 colors.
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
	return p[p.Index(c)]
}

// Index returns the index of the palette color closest to c in Euclidean
// R,G,B space.
func (p PalettedColorModel) Index(c Color) int {
	cr, cg, cb, _ := c.RGBA()
	// Shift by 1 bit to avoid potential uint32 overflow in sum-squared-difference.
	cr >>= 1
	cg >>= 1
	cb >>= 1
	ret, bestSSD := 0, uint32(1<<32-1)
	for i, v := range p {
		vr, vg, vb, _ := v.RGBA()
		vr >>= 1
		vg >>= 1
		vb >>= 1
		dr, dg, db := diff(cr, vr), diff(cg, vg), diff(cb, vb)
		ssd := (dr * dr) + (dg * dg) + (db * db)
		if ssd < bestSSD {
			ret, bestSSD = i, ssd
		}
	}
	return ret
}

// Paletted is an in-memory image of uint8 indices into a given palette.
type Paletted struct {
	// Pix holds the image's pixels, as palette indices. The pixel at
	// (x, y) starts at Pix[(y-Rect.Min.Y)*Stride + (x-Rect.Min.X)*1].
	Pix []uint8
	// Stride is the Pix stride (in bytes) between vertically adjacent pixels.
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
	if !(Point{x, y}.In(p.Rect)) {
		return p.Palette[0]
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x - p.Rect.Min.X)
	return p.Palette[p.Pix[i]]
}

func (p *Paletted) Set(x, y int, c Color) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x - p.Rect.Min.X)
	p.Pix[i] = uint8(p.Palette.Index(c))
}

func (p *Paletted) ColorIndexAt(x, y int) uint8 {
	if !(Point{x, y}.In(p.Rect)) {
		return 0
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x - p.Rect.Min.X)
	return p.Pix[i]
}

func (p *Paletted) SetColorIndex(x, y int, index uint8) {
	if !(Point{x, y}.In(p.Rect)) {
		return
	}
	i := (y-p.Rect.Min.Y)*p.Stride + (x - p.Rect.Min.X)
	p.Pix[i] = index
}

// SubImage returns an image representing the portion of the image p visible
// through r. The returned value shares pixels with the original image.
func (p *Paletted) SubImage(r Rectangle) Image {
	r = r.Intersect(p.Rect)
	// If r1 and r2 are Rectangles, r1.Intersect(r2) is not guaranteed to be inside
	// either r1 or r2 if the intersection is empty. Without explicitly checking for
	// this, the Pix[i:] expression below can panic.
	if r.Empty() {
		return &Paletted{
			Palette: p.Palette,
		}
	}
	i := (r.Min.Y-p.Rect.Min.Y)*p.Stride + (r.Min.X-p.Rect.Min.X)*1
	return &Paletted{
		Pix:     p.Pix[i:],
		Stride:  p.Stride,
		Rect:    p.Rect.Intersect(r),
		Palette: p.Palette,
	}
}

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (p *Paletted) Opaque() bool {
	var present [256]bool
	i0, i1 := 0, p.Rect.Dx()
	for y := p.Rect.Min.Y; y < p.Rect.Max.Y; y++ {
		for _, c := range p.Pix[i0:i1] {
			present[c] = true
		}
		i0 += p.Stride
		i1 += p.Stride
	}
	for i, c := range p.Palette {
		if !present[i] {
			continue
		}
		_, _, _, a := c.RGBA()
		if a != 0xffff {
			return false
		}
	}
	return true
}

// NewPaletted returns a new Paletted with the given width, height and palette.
func NewPaletted(w, h int, m PalettedColorModel) *Paletted {
	pix := make([]uint8, 1*w*h)
	return &Paletted{pix, 1 * w, Rectangle{ZP, Point{w, h}}, m}
}
