// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package color implements a basic color library.
package color

// Color can convert itself to alpha-premultiplied 16-bits per channel RGBA.
// The conversion may be lossy.
type Color interface {
	// RGBA returns the alpha-premultiplied red, green, blue and alpha values
	// for the color. Each value ranges within [0, 0xffff], but is represented
	// by a uint32 so that multiplying by a blend factor up to 0xffff will not
	// overflow.
	//
	// An alpha-premultiplied color component c has been scaled by alpha (a),
	// so has valid values 0 <= c <= a.
	RGBA() (r, g, b, a uint32)
}

// RGBA represents a traditional 32-bit alpha-premultiplied color, having 8
// bits for each of red, green, blue and alpha.
//
// An alpha-premultiplied color component C has been scaled by alpha (A), so
// has valid values 0 <= C <= A.
type RGBA struct {
	R, G, B, A uint8
}

func (c RGBA) RGBA() (r, g, b, a uint32) {
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

// RGBA64 represents a 64-bit alpha-premultiplied color, having 16 bits for
// each of red, green, blue and alpha.
//
// An alpha-premultiplied color component C has been scaled by alpha (A), so
// has valid values 0 <= C <= A.
type RGBA64 struct {
	R, G, B, A uint16
}

func (c RGBA64) RGBA() (r, g, b, a uint32) {
	return uint32(c.R), uint32(c.G), uint32(c.B), uint32(c.A)
}

// NRGBA represents a non-alpha-premultiplied 32-bit color.
type NRGBA struct {
	R, G, B, A uint8
}

func (c NRGBA) RGBA() (r, g, b, a uint32) {
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

// NRGBA64 represents a non-alpha-premultiplied 64-bit color,
// having 16 bits for each of red, green, blue and alpha.
type NRGBA64 struct {
	R, G, B, A uint16
}

func (c NRGBA64) RGBA() (r, g, b, a uint32) {
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

// Alpha represents an 8-bit alpha color.
type Alpha struct {
	A uint8
}

func (c Alpha) RGBA() (r, g, b, a uint32) {
	a = uint32(c.A)
	a |= a << 8
	return a, a, a, a
}

// Alpha16 represents a 16-bit alpha color.
type Alpha16 struct {
	A uint16
}

func (c Alpha16) RGBA() (r, g, b, a uint32) {
	a = uint32(c.A)
	return a, a, a, a
}

// Gray represents an 8-bit grayscale color.
type Gray struct {
	Y uint8
}

func (c Gray) RGBA() (r, g, b, a uint32) {
	y := uint32(c.Y)
	y |= y << 8
	return y, y, y, 0xffff
}

// Gray16 represents a 16-bit grayscale color.
type Gray16 struct {
	Y uint16
}

func (c Gray16) RGBA() (r, g, b, a uint32) {
	y := uint32(c.Y)
	return y, y, y, 0xffff
}

// Model can convert any Color to one from its own color model. The conversion
// may be lossy.
type Model interface {
	Convert(c Color) Color
}

// ModelFunc returns a Model that invokes f to implement the conversion.
func ModelFunc(f func(Color) Color) Model {
	// Note: using *modelFunc as the implementation
	// means that callers can still use comparisons
	// like m == RGBAModel. This is not possible if
	// we use the func value directly, because funcs
	// are no longer comparable.
	return &modelFunc{f}
}

type modelFunc struct {
	f func(Color) Color
}

func (m *modelFunc) Convert(c Color) Color {
	return m.f(c)
}

// Models for the standard color types.
var (
	RGBAModel    Model = ModelFunc(rgbaModel)
	RGBA64Model  Model = ModelFunc(rgba64Model)
	NRGBAModel   Model = ModelFunc(nrgbaModel)
	NRGBA64Model Model = ModelFunc(nrgba64Model)
	AlphaModel   Model = ModelFunc(alphaModel)
	Alpha16Model Model = ModelFunc(alpha16Model)
	GrayModel    Model = ModelFunc(grayModel)
	Gray16Model  Model = ModelFunc(gray16Model)
)

func rgbaModel(c Color) Color {
	if _, ok := c.(RGBA); ok {
		return c
	}
	r, g, b, a := c.RGBA()
	return RGBA{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), uint8(a >> 8)}
}

func rgba64Model(c Color) Color {
	if _, ok := c.(RGBA64); ok {
		return c
	}
	r, g, b, a := c.RGBA()
	return RGBA64{uint16(r), uint16(g), uint16(b), uint16(a)}
}

func nrgbaModel(c Color) Color {
	if _, ok := c.(NRGBA); ok {
		return c
	}
	r, g, b, a := c.RGBA()
	if a == 0xffff {
		return NRGBA{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), 0xff}
	}
	if a == 0 {
		return NRGBA{0, 0, 0, 0}
	}
	// Since Color.RGBA returns an alpha-premultiplied color, we should have r <= a && g <= a && b <= a.
	r = (r * 0xffff) / a
	g = (g * 0xffff) / a
	b = (b * 0xffff) / a
	return NRGBA{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), uint8(a >> 8)}
}

func nrgba64Model(c Color) Color {
	if _, ok := c.(NRGBA64); ok {
		return c
	}
	r, g, b, a := c.RGBA()
	if a == 0xffff {
		return NRGBA64{uint16(r), uint16(g), uint16(b), 0xffff}
	}
	if a == 0 {
		return NRGBA64{0, 0, 0, 0}
	}
	// Since Color.RGBA returns an alpha-premultiplied color, we should have r <= a && g <= a && b <= a.
	r = (r * 0xffff) / a
	g = (g * 0xffff) / a
	b = (b * 0xffff) / a
	return NRGBA64{uint16(r), uint16(g), uint16(b), uint16(a)}
}

func alphaModel(c Color) Color {
	if _, ok := c.(Alpha); ok {
		return c
	}
	_, _, _, a := c.RGBA()
	return Alpha{uint8(a >> 8)}
}

func alpha16Model(c Color) Color {
	if _, ok := c.(Alpha16); ok {
		return c
	}
	_, _, _, a := c.RGBA()
	return Alpha16{uint16(a)}
}

func grayModel(c Color) Color {
	if _, ok := c.(Gray); ok {
		return c
	}
	r, g, b, _ := c.RGBA()

	// These coefficients (the fractions 0.299, 0.587 and 0.114) are the same
	// as those given by the JFIF specification and used by func RGBToYCbCr in
	// ycbcr.go.
	//
	// Note that 19595 + 38470 + 7471 equals 65536.
	//
	// The 24 is 16 + 8. The 16 is the same as used in RGBToYCbCr. The 8 is
	// because the return value is 8 bit color, not 16 bit color.
	y := (19595*r + 38470*g + 7471*b + 1<<15) >> 24

	return Gray{uint8(y)}
}

func gray16Model(c Color) Color {
	if _, ok := c.(Gray16); ok {
		return c
	}
	r, g, b, _ := c.RGBA()

	// These coefficients (the fractions 0.299, 0.587 and 0.114) are the same
	// as those given by the JFIF specification and used by func RGBToYCbCr in
	// ycbcr.go.
	//
	// Note that 19595 + 38470 + 7471 equals 65536.
	y := (19595*r + 38470*g + 7471*b + 1<<15) >> 16

	return Gray16{uint16(y)}
}

// Palette is a palette of colors.
type Palette []Color

// Convert returns the palette color closest to c in Euclidean R,G,B space.
func (p Palette) Convert(c Color) Color {
	if len(p) == 0 {
		return nil
	}
	return p[p.Index(c)]
}

// Index returns the index of the palette color closest to c in Euclidean
// R,G,B,A space.
func (p Palette) Index(c Color) int {
	// A batch version of this computation is in image/draw/draw.go.

	cr, cg, cb, ca := c.RGBA()
	ret, bestSum := 0, uint32(1<<32-1)
	for i, v := range p {
		vr, vg, vb, va := v.RGBA()
		sum := sqDiff(cr, vr) + sqDiff(cg, vg) + sqDiff(cb, vb) + sqDiff(ca, va)
		if sum < bestSum {
			if sum == 0 {
				return i
			}
			ret, bestSum = i, sum
		}
	}
	return ret
}

// sqDiff returns the squared-difference of x and y, shifted by 2 so that
// adding four of those won't overflow a uint32.
//
// x and y are both assumed to be in the range [0, 0xffff].
func sqDiff(x, y uint32) uint32 {
	// The canonical code of this function looks as follows:
	//
	//	var d uint32
	//	if x > y {
	//		d = x - y
	//	} else {
	//		d = y - x
	//	}
	//	return (d * d) >> 2
	//
	// Language spec guarantees the following properties of unsigned integer
	// values operations with respect to overflow/wrap around:
	//
	// > For unsigned integer values, the operations +, -, *, and << are
	// > computed modulo 2n, where n is the bit width of the unsigned
	// > integer's type. Loosely speaking, these unsigned integer operations
	// > discard high bits upon overflow, and programs may rely on ``wrap
	// > around''.
	//
	// Considering these properties and the fact that this function is
	// called in the hot paths (x,y loops), it is reduced to the below code
	// which is slightly faster. See TestSqDiff for correctness check.
	d := x - y
	return (d * d) >> 2
}

// Standard colors.
var (
	Black       = Gray16{0}
	White       = Gray16{0xffff}
	Transparent = Alpha16{0}
	Opaque      = Alpha16{0xffff}
)
