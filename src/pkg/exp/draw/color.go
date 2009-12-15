// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package draw

import "image"

// A Color represents a color with 8-bit R, G, B, and A values,
// packed into a uint32—0xRRGGBBAA—so that comparison
// is defined on colors.
// Color implements image.Color.
// Color also implements image.Image: it is a
// 10⁹x10⁹-pixel image of uniform color.
type Color uint32

// Check that Color implements image.Color and image.Image
var _ image.Color = Black
var _ image.Image = Black

var (
	Opaque        Color = 0xFFFFFFFF
	Transparent   Color = 0x00000000
	Black         Color = 0x000000FF
	White         Color = 0xFFFFFFFF
	Red           Color = 0xFF0000FF
	Green         Color = 0x00FF00FF
	Blue          Color = 0x0000FFFF
	Cyan          Color = 0x00FFFFFF
	Magenta       Color = 0xFF00FFFF
	Yellow        Color = 0xFFFF00FF
	PaleYellow    Color = 0xFFFFAAFF
	DarkYellow    Color = 0xEEEE9EFF
	DarkGreen     Color = 0x448844FF
	PaleGreen     Color = 0xAAFFAAFF
	MedGreen      Color = 0x88CC88FF
	DarkBlue      Color = 0x000055FF
	PaleBlueGreen Color = 0xAAFFFFFF
	PaleBlue      Color = 0x0000BBFF
	BlueGreen     Color = 0x008888FF
	GreyGreen     Color = 0x55AAAAFF
	PaleGreyGreen Color = 0x9EEEEEFF
	YellowGreen   Color = 0x99994CFF
	MedBlue       Color = 0x000099FF
	GreyBlue      Color = 0x005DBBFF
	PaleGreyBlue  Color = 0x4993DDFF
	PurpleBlue    Color = 0x8888CCFF
)

func (c Color) RGBA() (r, g, b, a uint32) {
	x := uint32(c)
	r, g, b, a = x>>24, (x>>16)&0xFF, (x>>8)&0xFF, x&0xFF
	r |= r << 8
	r |= r << 16
	g |= g << 8
	g |= g << 16
	b |= b << 8
	b |= b << 16
	a |= a << 8
	a |= a << 16
	return
}

// SetAlpha returns the color obtained by changing
// c's alpha value to a and scaling r, g, and b appropriately.
func (c Color) SetAlpha(a uint8) Color {
	r, g, b, oa := c>>24, (c>>16)&0xFF, (c>>8)&0xFF, c&0xFF
	if oa == 0 {
		return 0
	}
	r = r * Color(a) / oa
	if r < 0 {
		r = 0
	}
	if r > 0xFF {
		r = 0xFF
	}
	g = g * Color(a) / oa
	if g < 0 {
		g = 0
	}
	if g > 0xFF {
		g = 0xFF
	}
	b = b * Color(a) / oa
	if b < 0 {
		b = 0
	}
	if b > 0xFF {
		b = 0xFF
	}
	return r<<24 | g<<16 | b<<8 | Color(a)
}

func (c Color) Width() int { return 1e9 }

func (c Color) Height() int { return 1e9 }

func (c Color) At(x, y int) image.Color { return c }

func toColor(color image.Color) image.Color {
	if c, ok := color.(Color); ok {
		return c
	}
	r, g, b, a := color.RGBA()
	return Color(r>>24<<24 | g>>24<<16 | b>>24<<8 | a>>24)
}

func (c Color) ColorModel() image.ColorModel { return image.ColorModelFunc(toColor) }
