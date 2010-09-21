// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package av

import (
	"image"
)

// Native Client image format:
// a single linear array of 32-bit ARGB as packed uint32s.

// An Image represents a Native Client frame buffer.
// The pixels in the image can be accessed as a single
// linear slice or as a two-dimensional slice of slices.
// Image implements image.Image.
type Image struct {
	Linear []Color
	Pixel  [][]Color
}

var _ image.Image = (*Image)(nil)

func (m *Image) ColorModel() image.ColorModel { return ColorModel }

func (m *Image) Bounds() image.Rectangle {
	if len(m.Pixel) == 0 {
		return image.ZR
	}
	return image.Rectangle{image.ZP, image.Point{len(m.Pixel[0]), len(m.Pixel)}}
}

func (m *Image) At(x, y int) image.Color { return m.Pixel[y][x] }

func (m *Image) Set(x, y int, color image.Color) {
	if c, ok := color.(Color); ok {
		m.Pixel[y][x] = c
		return
	}
	m.Pixel[y][x] = makeColor(color.RGBA())
}

func newImage(dx, dy int, linear []Color) *Image {
	if linear == nil {
		linear = make([]Color, dx*dy)
	}
	pix := make([][]Color, dy)
	for i := range pix {
		pix[i] = linear[dx*i : dx*(i+1)]
	}
	return &Image{linear, pix}
}

// A Color represents a Native Client color value,
// a 32-bit R, G, B, A value packed as 0xAARRGGBB.
type Color uint32

func (p Color) RGBA() (r, g, b, a uint32) {
	x := uint32(p)
	a = x >> 24
	a |= a << 8
	r = (x >> 16) & 0xFF
	r |= r << 8
	g = (x >> 8) & 0xFF
	g |= g << 8
	b = x & 0xFF
	b |= b << 8
	return
}

func makeColor(r, g, b, a uint32) Color {
	return Color(a>>8<<24 | r>>8<<16 | g>>8<<8 | b>>8)
}

func toColor(color image.Color) image.Color {
	if c, ok := color.(Color); ok {
		return c
	}
	return makeColor(color.RGBA())
}

// ColorModel is the color model corresponding to the Native Client Color.
var ColorModel = image.ColorModelFunc(toColor)
