// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package image

var (
	// Black is an opaque black ColorImage.
	Black = NewColorImage(Gray16Color{0})
	// White is an opaque white ColorImage.
	White = NewColorImage(Gray16Color{0xffff})
	// Transparent is a fully transparent ColorImage.
	Transparent = NewColorImage(Alpha16Color{0})
	// Opaque is a fully opaque ColorImage.
	Opaque = NewColorImage(Alpha16Color{0xffff})
)

// A ColorImage is a practically infinite-sized Image of uniform Color.
// It implements both the Color and Image interfaces.
type ColorImage struct {
	C Color
}

func (c *ColorImage) RGBA() (r, g, b, a uint32) {
	return c.C.RGBA()
}

func (c *ColorImage) ColorModel() ColorModel {
	return ColorModelFunc(func(Color) Color { return c.C })
}

func (c *ColorImage) Bounds() Rectangle { return Rectangle{Point{-1e9, -1e9}, Point{1e9, 1e9}} }

func (c *ColorImage) At(x, y int) Color { return c.C }

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (c *ColorImage) Opaque() bool {
	_, _, _, a := c.C.RGBA()
	return a == 0xffff
}

func NewColorImage(c Color) *ColorImage {
	return &ColorImage{c}
}

// A Tiled is a practically infinite-sized Image that repeats another Image in
// both directions. Tiled{i}.At(x, y) will equal i.At(x, y) for all points
// within i's Bounds.
type Tiled struct {
	I Image
}

func (t *Tiled) ColorModel() ColorModel {
	return t.I.ColorModel()
}

func (t *Tiled) Bounds() Rectangle { return Rectangle{Point{-1e9, -1e9}, Point{1e9, 1e9}} }

func (t *Tiled) At(x, y int) Color {
	p := Point{x, y}.Mod(t.I.Bounds())
	return t.I.At(p.X, p.Y)
}

func NewTiled(i Image) *Tiled {
	return &Tiled{i}
}
