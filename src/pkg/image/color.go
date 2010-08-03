// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package image

// All Colors can convert themselves, with a possible loss of precision,
// to 64-bit alpha-premultiplied RGBA. Each channel value ranges within
// [0, 0xFFFF].
type Color interface {
	RGBA() (r, g, b, a uint32)
}

// An RGBAColor represents a traditional 32-bit alpha-premultiplied color,
// having 8 bits for each of red, green, blue and alpha.
type RGBAColor struct {
	R, G, B, A uint8
}

func (c RGBAColor) RGBA() (r, g, b, a uint32) {
	r = uint32(c.R)
	r |= r << 8
	g = uint32(c.G)
	g |= g << 8
	b = uint32(c.B)
	b |= b << 8
	a = uint32(c.A)
	a |= a << 8
	return
}

// An RGBA64Color represents a 64-bit alpha-premultiplied color,
// having 16 bits for each of red, green, blue and alpha.
type RGBA64Color struct {
	R, G, B, A uint16
}

func (c RGBA64Color) RGBA() (r, g, b, a uint32) {
	return uint32(c.R), uint32(c.G), uint32(c.B), uint32(c.A)
}

// An NRGBAColor represents a non-alpha-premultiplied 32-bit color.
type NRGBAColor struct {
	R, G, B, A uint8
}

func (c NRGBAColor) RGBA() (r, g, b, a uint32) {
	r = uint32(c.R)
	r |= r << 8
	r *= uint32(c.A)
	r /= 0xff
	g = uint32(c.G)
	g |= g << 8
	g *= uint32(c.A)
	g /= 0xff
	b = uint32(c.B)
	b |= b << 8
	b *= uint32(c.A)
	b /= 0xff
	a = uint32(c.A)
	a |= a << 8
	return
}

// An NRGBA64Color represents a non-alpha-premultiplied 64-bit color,
// having 16 bits for each of red, green, blue and alpha.
type NRGBA64Color struct {
	R, G, B, A uint16
}

func (c NRGBA64Color) RGBA() (r, g, b, a uint32) {
	r = uint32(c.R)
	r *= uint32(c.A)
	r /= 0xffff
	g = uint32(c.G)
	g *= uint32(c.A)
	g /= 0xffff
	b = uint32(c.B)
	b *= uint32(c.A)
	b /= 0xffff
	a = uint32(c.A)
	return
}

// An AlphaColor represents an 8-bit alpha.
type AlphaColor struct {
	A uint8
}

func (c AlphaColor) RGBA() (r, g, b, a uint32) {
	a = uint32(c.A)
	a |= a << 8
	return a, a, a, a
}

// An Alpha16Color represents a 16-bit alpha.
type Alpha16Color struct {
	A uint16
}

func (c Alpha16Color) RGBA() (r, g, b, a uint32) {
	a = uint32(c.A)
	return a, a, a, a
}

// A GrayColor represents an 8-bit grayscale color.
type GrayColor struct {
	Y uint8
}

func (c GrayColor) RGBA() (r, g, b, a uint32) {
	y := uint32(c.Y)
	y |= y << 8
	return y, y, y, 0xffff
}

// A Gray16Color represents a 16-bit grayscale color.
type Gray16Color struct {
	Y uint16
}

func (c Gray16Color) RGBA() (r, g, b, a uint32) {
	y := uint32(c.Y)
	return y, y, y, 0xffff
}

// A ColorModel can convert foreign Colors, with a possible loss of precision,
// to a Color from its own color model.
type ColorModel interface {
	Convert(c Color) Color
}

// The ColorModelFunc type is an adapter to allow the use of an ordinary
// color conversion function as a ColorModel.  If f is such a function,
// ColorModelFunc(f) is a ColorModel object that invokes f to implement
// the conversion.
type ColorModelFunc func(Color) Color

func (f ColorModelFunc) Convert(c Color) Color {
	return f(c)
}

func toRGBAColor(c Color) Color {
	if _, ok := c.(RGBAColor); ok {
		return c
	}
	r, g, b, a := c.RGBA()
	return RGBAColor{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), uint8(a >> 8)}
}

func toRGBA64Color(c Color) Color {
	if _, ok := c.(RGBA64Color); ok {
		return c
	}
	r, g, b, a := c.RGBA()
	return RGBA64Color{uint16(r), uint16(g), uint16(b), uint16(a)}
}

func toNRGBAColor(c Color) Color {
	if _, ok := c.(NRGBAColor); ok {
		return c
	}
	r, g, b, a := c.RGBA()
	if a == 0xffff {
		return NRGBAColor{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), 0xff}
	}
	if a == 0 {
		return NRGBAColor{0, 0, 0, 0}
	}
	// Since Color.RGBA returns a alpha-premultiplied color, we should have r <= a && g <= a && b <= a.
	r = (r * 0xffff) / a
	g = (g * 0xffff) / a
	b = (b * 0xffff) / a
	return NRGBAColor{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), uint8(a >> 8)}
}

func toNRGBA64Color(c Color) Color {
	if _, ok := c.(NRGBA64Color); ok {
		return c
	}
	r, g, b, a := c.RGBA()
	if a == 0xffff {
		return NRGBA64Color{uint16(r), uint16(g), uint16(b), 0xffff}
	}
	if a == 0 {
		return NRGBA64Color{0, 0, 0, 0}
	}
	// Since Color.RGBA returns a alpha-premultiplied color, we should have r <= a && g <= a && b <= a.
	r = (r * 0xffff) / a
	g = (g * 0xffff) / a
	b = (b * 0xffff) / a
	return NRGBA64Color{uint16(r), uint16(g), uint16(b), uint16(a)}
}

func toAlphaColor(c Color) Color {
	if _, ok := c.(AlphaColor); ok {
		return c
	}
	_, _, _, a := c.RGBA()
	return AlphaColor{uint8(a >> 8)}
}

func toAlpha16Color(c Color) Color {
	if _, ok := c.(Alpha16Color); ok {
		return c
	}
	_, _, _, a := c.RGBA()
	return Alpha16Color{uint16(a)}
}

func toGrayColor(c Color) Color {
	if _, ok := c.(GrayColor); ok {
		return c
	}
	r, g, b, _ := c.RGBA()
	y := (299*r + 587*g + 114*b + 500) / 1000
	return GrayColor{uint8(y >> 8)}
}

func toGray16Color(c Color) Color {
	if _, ok := c.(Gray16Color); ok {
		return c
	}
	r, g, b, _ := c.RGBA()
	y := (299*r + 587*g + 114*b + 500) / 1000
	return Gray16Color{uint16(y)}
}

// The ColorModel associated with RGBAColor.
var RGBAColorModel ColorModel = ColorModelFunc(toRGBAColor)

// The ColorModel associated with RGBA64Color.
var RGBA64ColorModel ColorModel = ColorModelFunc(toRGBA64Color)

// The ColorModel associated with NRGBAColor.
var NRGBAColorModel ColorModel = ColorModelFunc(toNRGBAColor)

// The ColorModel associated with NRGBA64Color.
var NRGBA64ColorModel ColorModel = ColorModelFunc(toNRGBA64Color)

// The ColorModel associated with AlphaColor.
var AlphaColorModel ColorModel = ColorModelFunc(toAlphaColor)

// The ColorModel associated with Alpha16Color.
var Alpha16ColorModel ColorModel = ColorModelFunc(toAlpha16Color)

// The ColorModel associated with GrayColor.
var GrayColorModel ColorModel = ColorModelFunc(toGrayColor)

// The ColorModel associated with Gray16Color.
var Gray16ColorModel ColorModel = ColorModelFunc(toGray16Color)
