// compile

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the code snippets included in "The Go image/draw package."

package main

import (
	"image"
	"image/color"
	"image/draw"
)

func main() {
	Color()
	Rect()
	RectAndScroll()
	ConvAndCircle()
	Glyph()
}

func Color() {
	c := color.RGBA{255, 0, 255, 255}
	r := image.Rect(0, 0, 640, 480)
	dst := image.NewRGBA(r)

	// ZERO OMIT
	// image.ZP is the zero point -- the origin.
	draw.Draw(dst, r, &image.Uniform{c}, image.ZP, draw.Src)
	// STOP OMIT

	// BLUE OMIT
	m := image.NewRGBA(image.Rect(0, 0, 640, 480))
	blue := color.RGBA{0, 0, 255, 255}
	draw.Draw(m, m.Bounds(), &image.Uniform{blue}, image.ZP, draw.Src)
	// STOP OMIT

	// RESET OMIT
	draw.Draw(m, m.Bounds(), image.Transparent, image.ZP, draw.Src)
	// STOP OMIT
}

func Rect() {
	dst := image.NewRGBA(image.Rect(0, 0, 640, 480))
	sr := image.Rect(0, 0, 200, 200)
	src := image.Black
	dp := image.Point{100, 100}

	// RECT OMIT
	r := image.Rectangle{dp, dp.Add(sr.Size())}
	draw.Draw(dst, r, src, sr.Min, draw.Src)
	// STOP OMIT
}

func RectAndScroll() {
	dst := image.NewRGBA(image.Rect(0, 0, 640, 480))
	sr := image.Rect(0, 0, 200, 200)
	src := image.Black
	dp := image.Point{100, 100}

	// RECT2 OMIT
	r := sr.Sub(sr.Min).Add(dp)
	draw.Draw(dst, r, src, sr.Min, draw.Src)
	// STOP OMIT

	m := dst

	// SCROLL OMIT
	b := m.Bounds()
	p := image.Pt(0, 20)
	// Note that even though the second argument is b,
	// the effective rectangle is smaller due to clipping.
	draw.Draw(m, b, m, b.Min.Add(p), draw.Src)
	dirtyRect := b.Intersect(image.Rect(b.Min.X, b.Max.Y-20, b.Max.X, b.Max.Y))
	// STOP OMIT

	_ = dirtyRect // noop
}

func ConvAndCircle() {
	src := image.NewRGBA(image.Rect(0, 0, 640, 480))
	dst := image.NewRGBA(image.Rect(0, 0, 640, 480))

	// CONV OMIT
	b := src.Bounds()
	m := image.NewRGBA(b)
	draw.Draw(m, b, src, b.Min, draw.Src)
	// STOP OMIT

	p := image.Point{100, 100}
	r := 50

	// CIRCLE2 OMIT
	draw.DrawMask(dst, dst.Bounds(), src, image.ZP, &circle{p, r}, image.ZP, draw.Over)
	// STOP OMIT
}

func theGlyphImageForAFont() image.Image {
	return image.NewRGBA(image.Rect(0, 0, 640, 480))
}

func theBoundsFor(index int) image.Rectangle {
	return image.Rect(0, 0, 32, 32)
}

func Glyph() {
	p := image.Point{100, 100}
	dst := image.NewRGBA(image.Rect(0, 0, 640, 480))
	glyphIndex := 42

	// GLYPH OMIT
	src := &image.Uniform{color.RGBA{0, 0, 255, 255}}
	mask := theGlyphImageForAFont()
	mr := theBoundsFor(glyphIndex)
	draw.DrawMask(dst, mr.Sub(mr.Min).Add(p), src, image.ZP, mask, mr.Min, draw.Over)
	// STOP OMIT
}

//CIRCLESTRUCT OMIT
type circle struct {
	p image.Point
	r int
}

func (c *circle) ColorModel() color.Model {
	return color.AlphaModel
}

func (c *circle) Bounds() image.Rectangle {
	return image.Rect(c.p.X-c.r, c.p.Y-c.r, c.p.X+c.r, c.p.Y+c.r)
}

func (c *circle) At(x, y int) color.Color {
	xx, yy, rr := float64(x-c.p.X)+0.5, float64(y-c.p.Y)+0.5, float64(c.r)
	if xx*xx+yy*yy < rr*rr {
		return color.Alpha{255}
	}
	return color.Alpha{0}
}

//STOP OMIT
