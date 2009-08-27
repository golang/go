// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package image

// TODO(nigeltao): Clarify semantics wrt premultiplied vs unpremultiplied colors.
// It's probably also worth thinking about floating-point color models.

// All Colors can convert themselves, with a possible loss of precision, to 128-bit RGBA.
type Color interface {
	RGBA() (r, g, b, a uint32);
}

// An RGBAColor represents a traditional 32-bit color, having 8 bits for each of red, green, blue and alpha.
type RGBAColor struct {
	R, G, B, A uint8;
}

func (c RGBAColor) RGBA() (r, g, b, a uint32) {
	r = uint32(c.R);
	r |= r<<8;
	r |= r<<16;
	g = uint32(c.G);
	g |= g<<8;
	g |= g<<16;
	b = uint32(c.B);
	b |= b<<8;
	b |= b<<16;
	a = uint32(c.A);
	a |= a<<8;
	a |= a<<16;
	return;
}

// An RGBA64Color represents a 64-bit color, having 16 bits for each of red, green, blue and alpha.
type RGBA64Color struct {
	R, G, B, A uint16;
}

func (c RGBA64Color) RGBA() (r, g, b, a uint32) {
	r = uint32(c.R);
	r |= r<<16;
	g = uint32(c.G);
	g |= g<<16;
	b = uint32(c.B);
	b |= b<<16;
	a = uint32(c.A);
	a |= a<<16;
	return;
}

// A ColorModel can convert foreign Colors, with a possible loss of precision, to a Color
// from its own color model.
type ColorModel interface {
	Convert(c Color) Color;
}

// The ColorModelFunc type is an adapter to allow the use of an ordinary
// color conversion function as a ColorModel.  If f is such a function,
// ColorModelFunc(f) is a ColorModel object that invokes f to implement
// the conversion.
type ColorModelFunc func(Color) Color

func (f ColorModelFunc) Convert(c Color) Color {
	return f(c);
}

func toRGBAColor(c Color) Color {
	if _, ok := c.(RGBAColor); ok {	// no-op conversion
		return c;
	}
	r, g, b, a := c.RGBA();
	return RGBAColor{ uint8(r>>24), uint8(g>>24), uint8(b>>24), uint8(a>>24) };
}

func toRGBA64Color(c Color) Color {
	if _, ok := c.(RGBA64Color); ok {	// no-op conversion
		return c;
	}
	r, g, b, a := c.RGBA();
	return RGBA64Color{ uint16(r>>16), uint16(g>>16), uint16(b>>16), uint16(a>>16) };
}

// The ColorModel associated with RGBAColor.
var RGBAColorModel ColorModel = ColorModelFunc(toRGBAColor);

// The ColorModel associated with RGBA64Color.
var RGBA64ColorModel ColorModel = ColorModelFunc(toRGBA64Color);

