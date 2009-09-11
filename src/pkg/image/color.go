// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package image

// TODO(nigeltao): Think about how floating-point color models work.

// All Colors can convert themselves, with a possible loss of precision, to 128-bit alpha-premultiplied RGBA.
type Color interface {
	RGBA() (r, g, b, a uint32);
}

// An RGBAColor represents a traditional 32-bit alpha-premultiplied color, having 8 bits for each of red, green, blue and alpha.
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

// An RGBA64Color represents a 64-bit alpha-premultiplied color, having 16 bits for each of red, green, blue and alpha.
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

// An NRGBAColor represents a non-alpha-premultiplied 32-bit color.
type NRGBAColor struct {
	R, G, B, A uint8;
}

func (c NRGBAColor) RGBA() (r, g, b, a uint32) {
	r = uint32(c.R);
	r |= r<<8;
	r *= uint32(c.A);
	r /= 0xff;
	r |= r<<16;
	g = uint32(c.G);
	g |= g<<8;
	g *= uint32(c.A);
	g /= 0xff;
	g |= g<<16;
	b = uint32(c.B);
	b |= b<<8;
	b *= uint32(c.A);
	b /= 0xff;
	b |= b<<16;
	a = uint32(c.A);
	a |= a<<8;
	a |= a<<16;
	return;
}

// An NRGBA64Color represents a non-alpha-premultiplied 64-bit color, having 16 bits for each of red, green, blue and alpha.
type NRGBA64Color struct {
	R, G, B, A uint16;
}

func (c NRGBA64Color) RGBA() (r, g, b, a uint32) {
	r = uint32(c.R);
	r *= uint32(c.A);
	r /= 0xffff;
	r |= r<<16;
	g = uint32(c.G);
	g *= uint32(c.A);
	g /= 0xffff;
	g |= g<<16;
	b = uint32(c.B);
	b *= uint32(c.A);
	b /= 0xffff;
	b |= b<<16;
	a = uint32(c.A);
	a |= a<<8;
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

func toNRGBAColor(c Color) Color {
	if _, ok := c.(NRGBAColor); ok {	// no-op conversion
		return c;
	}
	r, g, b, a := c.RGBA();
	a >>= 16;
	if a == 0xffff {
		return NRGBAColor{ uint8(r>>24), uint8(g>>24), uint8(b>>24), 0xff };
	}
	if a == 0 {
		return NRGBAColor{ 0, 0, 0, 0 };
	}
	r >>= 16;
	g >>= 16;
	b >>= 16;
	// Since Color.RGBA returns a alpha-premultiplied color, we should have r <= a && g <= a && b <= a.
	r = (r * 0xffff) / a;
	g = (g * 0xffff) / a;
	b = (b * 0xffff) / a;
	return NRGBAColor{ uint8(r>>8), uint8(g>>8), uint8(b>>8), uint8(a>>8) };
}

func toNRGBA64Color(c Color) Color {
	if _, ok := c.(NRGBA64Color); ok {	// no-op conversion
		return c;
	}
	r, g, b, a := c.RGBA();
	a >>= 16;
	r >>= 16;
	g >>= 16;
	b >>= 16;
	if a == 0xffff {
		return NRGBA64Color{ uint16(r), uint16(g), uint16(b), 0xffff };
	}
	if a == 0 {
		return NRGBA64Color{ 0, 0, 0, 0 };
	}
	// Since Color.RGBA returns a alpha-premultiplied color, we should have r <= a && g <= a && b <= a.
	r = (r * 0xffff) / a;
	g = (g * 0xffff) / a;
	b = (b * 0xffff) / a;
	return NRGBA64Color{ uint16(r), uint16(g), uint16(b), uint16(a) };
}

// The ColorModel associated with RGBAColor.
var RGBAColorModel ColorModel = ColorModelFunc(toRGBAColor);

// The ColorModel associated with RGBA64Color.
var RGBA64ColorModel ColorModel = ColorModelFunc(toRGBA64Color);

// The ColorModel associated with NRGBAColor.
var NRGBAColorModel ColorModel = ColorModelFunc(toNRGBAColor);

// The ColorModel associated with NRGBA64Color.
var NRGBA64ColorModel ColorModel = ColorModelFunc(toNRGBA64Color);

