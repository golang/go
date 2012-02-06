// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package image

import (
	"image/color"
)

var (
	// Black is an opaque black uniform image.
	Black = NewUniform(color.Black)
	// White is an opaque white uniform image.
	White = NewUniform(color.White)
	// Transparent is a fully transparent uniform image.
	Transparent = NewUniform(color.Transparent)
	// Opaque is a fully opaque uniform image.
	Opaque = NewUniform(color.Opaque)
)

// Uniform is an infinite-sized Image of uniform color.
// It implements the color.Color, color.ColorModel, and Image interfaces.
type Uniform struct {
	C color.Color
}

func (c *Uniform) RGBA() (r, g, b, a uint32) {
	return c.C.RGBA()
}

func (c *Uniform) ColorModel() color.Model {
	return c
}

func (c *Uniform) Convert(color.Color) color.Color {
	return c.C
}

func (c *Uniform) Bounds() Rectangle { return Rectangle{Point{-1e9, -1e9}, Point{1e9, 1e9}} }

func (c *Uniform) At(x, y int) color.Color { return c.C }

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (c *Uniform) Opaque() bool {
	_, _, _, a := c.C.RGBA()
	return a == 0xffff
}

func NewUniform(c color.Color) *Uniform {
	return &Uniform{c}
}

// Repeated is an Image that is a source Image translated by -Offset and then
// repeated in all four directions to infinity.
//
// Repeated{src, off}.At(x, y) will equal src.At(x+off.X, y+off.Y) for all
// points {x+off.X, y+off.Y} within src's Bounds.
type Repeated struct {
	// I is the source image.
	I Image
	// Offset is the translation vector from result pixel to source pixel.
	Offset Point
}

func (r *Repeated) ColorModel() color.Model {
	return r.I.ColorModel()
}

func (r *Repeated) Bounds() Rectangle { return Rectangle{Point{-1e9, -1e9}, Point{1e9, 1e9}} }

func (r *Repeated) At(x, y int) color.Color {
	p := Point{x, y}.Add(r.Offset).Mod(r.I.Bounds())
	return r.I.At(p.X, p.Y)
}

func NewRepeated(i Image, offset Point) *Repeated {
	return &Repeated{i, offset}
}
