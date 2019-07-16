// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package color

// RGBToYCbCr converts an RGB triple to a Y'CbCr triple.
func RGBToYCbCr(r, g, b uint8) (uint8, uint8, uint8) {
	// The JFIF specification says:
	//	Y' =  0.2990*R + 0.5870*G + 0.1140*B
	//	Cb = -0.1687*R - 0.3313*G + 0.5000*B + 128
	//	Cr =  0.5000*R - 0.4187*G - 0.0813*B + 128
	// https://www.w3.org/Graphics/JPEG/jfif3.pdf says Y but means Y'.

	r1 := int32(r)
	g1 := int32(g)
	b1 := int32(b)

	// yy is in range [0,0xff].
	//
	// Note that 19595 + 38470 + 7471 equals 65536.
	yy := (19595*r1 + 38470*g1 + 7471*b1 + 1<<15) >> 16

	// The bit twiddling below is equivalent to
	//
	// cb := (-11056*r1 - 21712*g1 + 32768*b1 + 257<<15) >> 16
	// if cb < 0 {
	//     cb = 0
	// } else if cb > 0xff {
	//     cb = ^int32(0)
	// }
	//
	// but uses fewer branches and is faster.
	// Note that the uint8 type conversion in the return
	// statement will convert ^int32(0) to 0xff.
	// The code below to compute cr uses a similar pattern.
	//
	// Note that -11056 - 21712 + 32768 equals 0.
	cb := -11056*r1 - 21712*g1 + 32768*b1 + 257<<15
	if uint32(cb)&0xff000000 == 0 {
		cb >>= 16
	} else {
		cb = ^(cb >> 31)
	}

	// Note that 32768 - 27440 - 5328 equals 0.
	cr := 32768*r1 - 27440*g1 - 5328*b1 + 257<<15
	if uint32(cr)&0xff000000 == 0 {
		cr >>= 16
	} else {
		cr = ^(cr >> 31)
	}

	return uint8(yy), uint8(cb), uint8(cr)
}

// YCbCrToRGB converts a Y'CbCr triple to an RGB triple.
func YCbCrToRGB(y, cb, cr uint8) (uint8, uint8, uint8) {
	// The JFIF specification says:
	//	R = Y' + 1.40200*(Cr-128)
	//	G = Y' - 0.34414*(Cb-128) - 0.71414*(Cr-128)
	//	B = Y' + 1.77200*(Cb-128)
	// https://www.w3.org/Graphics/JPEG/jfif3.pdf says Y but means Y'.
	//
	// Those formulae use non-integer multiplication factors. When computing,
	// integer math is generally faster than floating point math. We multiply
	// all of those factors by 1<<16 and round to the nearest integer:
	//	 91881 = roundToNearestInteger(1.40200 * 65536).
	//	 22554 = roundToNearestInteger(0.34414 * 65536).
	//	 46802 = roundToNearestInteger(0.71414 * 65536).
	//	116130 = roundToNearestInteger(1.77200 * 65536).
	//
	// Adding a rounding adjustment in the range [0, 1<<16-1] and then shifting
	// right by 16 gives us an integer math version of the original formulae.
	//	R = (65536*Y' +  91881 *(Cr-128)                  + adjustment) >> 16
	//	G = (65536*Y' -  22554 *(Cb-128) - 46802*(Cr-128) + adjustment) >> 16
	//	B = (65536*Y' + 116130 *(Cb-128)                  + adjustment) >> 16
	// A constant rounding adjustment of 1<<15, one half of 1<<16, would mean
	// round-to-nearest when dividing by 65536 (shifting right by 16).
	// Similarly, a constant rounding adjustment of 0 would mean round-down.
	//
	// Defining YY1 = 65536*Y' + adjustment simplifies the formulae and
	// requires fewer CPU operations:
	//	R = (YY1 +  91881 *(Cr-128)                 ) >> 16
	//	G = (YY1 -  22554 *(Cb-128) - 46802*(Cr-128)) >> 16
	//	B = (YY1 + 116130 *(Cb-128)                 ) >> 16
	//
	// The inputs (y, cb, cr) are 8 bit color, ranging in [0x00, 0xff]. In this
	// function, the output is also 8 bit color, but in the related YCbCr.RGBA
	// method, below, the output is 16 bit color, ranging in [0x0000, 0xffff].
	// Outputting 16 bit color simply requires changing the 16 to 8 in the "R =
	// etc >> 16" equation, and likewise for G and B.
	//
	// As mentioned above, a constant rounding adjustment of 1<<15 is a natural
	// choice, but there is an additional constraint: if c0 := YCbCr{Y: y, Cb:
	// 0x80, Cr: 0x80} and c1 := Gray{Y: y} then c0.RGBA() should equal
	// c1.RGBA(). Specifically, if y == 0 then "R = etc >> 8" should yield
	// 0x0000 and if y == 0xff then "R = etc >> 8" should yield 0xffff. If we
	// used a constant rounding adjustment of 1<<15, then it would yield 0x0080
	// and 0xff80 respectively.
	//
	// Note that when cb == 0x80 and cr == 0x80 then the formulae collapse to:
	//	R = YY1 >> n
	//	G = YY1 >> n
	//	B = YY1 >> n
	// where n is 16 for this function (8 bit color output) and 8 for the
	// YCbCr.RGBA method (16 bit color output).
	//
	// The solution is to make the rounding adjustment non-constant, and equal
	// to 257*Y', which ranges over [0, 1<<16-1] as Y' ranges over [0, 255].
	// YY1 is then defined as:
	//	YY1 = 65536*Y' + 257*Y'
	// or equivalently:
	//	YY1 = Y' * 0x10101
	yy1 := int32(y) * 0x10101
	cb1 := int32(cb) - 128
	cr1 := int32(cr) - 128

	// The bit twiddling below is equivalent to
	//
	// r := (yy1 + 91881*cr1) >> 16
	// if r < 0 {
	//     r = 0
	// } else if r > 0xff {
	//     r = ^int32(0)
	// }
	//
	// but uses fewer branches and is faster.
	// Note that the uint8 type conversion in the return
	// statement will convert ^int32(0) to 0xff.
	// The code below to compute g and b uses a similar pattern.
	r := yy1 + 91881*cr1
	if uint32(r)&0xff000000 == 0 {
		r >>= 16
	} else {
		r = ^(r >> 31)
	}

	g := yy1 - 22554*cb1 - 46802*cr1
	if uint32(g)&0xff000000 == 0 {
		g >>= 16
	} else {
		g = ^(g >> 31)
	}

	b := yy1 + 116130*cb1
	if uint32(b)&0xff000000 == 0 {
		b >>= 16
	} else {
		b = ^(b >> 31)
	}

	return uint8(r), uint8(g), uint8(b)
}

// YCbCr represents a fully opaque 24-bit Y'CbCr color, having 8 bits each for
// one luma and two chroma components.
//
// JPEG, VP8, the MPEG family and other codecs use this color model. Such
// codecs often use the terms YUV and Y'CbCr interchangeably, but strictly
// speaking, the term YUV applies only to analog video signals, and Y' (luma)
// is Y (luminance) after applying gamma correction.
//
// Conversion between RGB and Y'CbCr is lossy and there are multiple, slightly
// different formulae for converting between the two. This package follows
// the JFIF specification at https://www.w3.org/Graphics/JPEG/jfif3.pdf.
type YCbCr struct {
	Y, Cb, Cr uint8
}

func (c YCbCr) RGBA() (uint32, uint32, uint32, uint32) {
	// This code is a copy of the YCbCrToRGB function above, except that it
	// returns values in the range [0, 0xffff] instead of [0, 0xff]. There is a
	// subtle difference between doing this and having YCbCr satisfy the Color
	// interface by first converting to an RGBA. The latter loses some
	// information by going to and from 8 bits per channel.
	//
	// For example, this code:
	//	const y, cb, cr = 0x7f, 0x7f, 0x7f
	//	r, g, b := color.YCbCrToRGB(y, cb, cr)
	//	r0, g0, b0, _ := color.YCbCr{y, cb, cr}.RGBA()
	//	r1, g1, b1, _ := color.RGBA{r, g, b, 0xff}.RGBA()
	//	fmt.Printf("0x%04x 0x%04x 0x%04x\n", r0, g0, b0)
	//	fmt.Printf("0x%04x 0x%04x 0x%04x\n", r1, g1, b1)
	// prints:
	//	0x7e18 0x808d 0x7db9
	//	0x7e7e 0x8080 0x7d7d

	yy1 := int32(c.Y) * 0x10101
	cb1 := int32(c.Cb) - 128
	cr1 := int32(c.Cr) - 128

	// The bit twiddling below is equivalent to
	//
	// r := (yy1 + 91881*cr1) >> 8
	// if r < 0 {
	//     r = 0
	// } else if r > 0xff {
	//     r = 0xffff
	// }
	//
	// but uses fewer branches and is faster.
	// The code below to compute g and b uses a similar pattern.
	r := yy1 + 91881*cr1
	if uint32(r)&0xff000000 == 0 {
		r >>= 8
	} else {
		r = ^(r >> 31) & 0xffff
	}

	g := yy1 - 22554*cb1 - 46802*cr1
	if uint32(g)&0xff000000 == 0 {
		g >>= 8
	} else {
		g = ^(g >> 31) & 0xffff
	}

	b := yy1 + 116130*cb1
	if uint32(b)&0xff000000 == 0 {
		b >>= 8
	} else {
		b = ^(b >> 31) & 0xffff
	}

	return uint32(r), uint32(g), uint32(b), 0xffff
}

// YCbCrModel is the Model for Y'CbCr colors.
var YCbCrModel Model = ModelFunc(yCbCrModel)

func yCbCrModel(c Color) Color {
	if _, ok := c.(YCbCr); ok {
		return c
	}
	r, g, b, _ := c.RGBA()
	y, u, v := RGBToYCbCr(uint8(r>>8), uint8(g>>8), uint8(b>>8))
	return YCbCr{y, u, v}
}

// NYCbCrA represents a non-alpha-premultiplied Y'CbCr-with-alpha color, having
// 8 bits each for one luma, two chroma and one alpha component.
type NYCbCrA struct {
	YCbCr
	A uint8
}

func (c NYCbCrA) RGBA() (uint32, uint32, uint32, uint32) {
	// The first part of this method is the same as YCbCr.RGBA.
	yy1 := int32(c.Y) * 0x10101
	cb1 := int32(c.Cb) - 128
	cr1 := int32(c.Cr) - 128

	// The bit twiddling below is equivalent to
	//
	// r := (yy1 + 91881*cr1) >> 8
	// if r < 0 {
	//     r = 0
	// } else if r > 0xff {
	//     r = 0xffff
	// }
	//
	// but uses fewer branches and is faster.
	// The code below to compute g and b uses a similar pattern.
	r := yy1 + 91881*cr1
	if uint32(r)&0xff000000 == 0 {
		r >>= 8
	} else {
		r = ^(r >> 31) & 0xffff
	}

	g := yy1 - 22554*cb1 - 46802*cr1
	if uint32(g)&0xff000000 == 0 {
		g >>= 8
	} else {
		g = ^(g >> 31) & 0xffff
	}

	b := yy1 + 116130*cb1
	if uint32(b)&0xff000000 == 0 {
		b >>= 8
	} else {
		b = ^(b >> 31) & 0xffff
	}

	// The second part of this method applies the alpha.
	a := uint32(c.A) * 0x101
	return uint32(r) * a / 0xffff, uint32(g) * a / 0xffff, uint32(b) * a / 0xffff, a
}

// NYCbCrAModel is the Model for non-alpha-premultiplied Y'CbCr-with-alpha
// colors.
var NYCbCrAModel Model = ModelFunc(nYCbCrAModel)

func nYCbCrAModel(c Color) Color {
	switch c := c.(type) {
	case NYCbCrA:
		return c
	case YCbCr:
		return NYCbCrA{c, 0xff}
	}
	r, g, b, a := c.RGBA()

	// Convert from alpha-premultiplied to non-alpha-premultiplied.
	if a != 0 {
		r = (r * 0xffff) / a
		g = (g * 0xffff) / a
		b = (b * 0xffff) / a
	}

	y, u, v := RGBToYCbCr(uint8(r>>8), uint8(g>>8), uint8(b>>8))
	return NYCbCrA{YCbCr{Y: y, Cb: u, Cr: v}, uint8(a >> 8)}
}

// RGBToCMYK converts an RGB triple to a CMYK quadruple.
func RGBToCMYK(r, g, b uint8) (uint8, uint8, uint8, uint8) {
	rr := uint32(r)
	gg := uint32(g)
	bb := uint32(b)
	w := rr
	if w < gg {
		w = gg
	}
	if w < bb {
		w = bb
	}
	if w == 0 {
		return 0, 0, 0, 0xff
	}
	c := (w - rr) * 0xff / w
	m := (w - gg) * 0xff / w
	y := (w - bb) * 0xff / w
	return uint8(c), uint8(m), uint8(y), uint8(0xff - w)
}

// CMYKToRGB converts a CMYK quadruple to an RGB triple.
func CMYKToRGB(c, m, y, k uint8) (uint8, uint8, uint8) {
	w := 0xffff - uint32(k)*0x101
	r := (0xffff - uint32(c)*0x101) * w / 0xffff
	g := (0xffff - uint32(m)*0x101) * w / 0xffff
	b := (0xffff - uint32(y)*0x101) * w / 0xffff
	return uint8(r >> 8), uint8(g >> 8), uint8(b >> 8)
}

// CMYK represents a fully opaque CMYK color, having 8 bits for each of cyan,
// magenta, yellow and black.
//
// It is not associated with any particular color profile.
type CMYK struct {
	C, M, Y, K uint8
}

func (c CMYK) RGBA() (uint32, uint32, uint32, uint32) {
	// This code is a copy of the CMYKToRGB function above, except that it
	// returns values in the range [0, 0xffff] instead of [0, 0xff].

	w := 0xffff - uint32(c.K)*0x101
	r := (0xffff - uint32(c.C)*0x101) * w / 0xffff
	g := (0xffff - uint32(c.M)*0x101) * w / 0xffff
	b := (0xffff - uint32(c.Y)*0x101) * w / 0xffff
	return r, g, b, 0xffff
}

// CMYKModel is the Model for CMYK colors.
var CMYKModel Model = ModelFunc(cmykModel)

func cmykModel(c Color) Color {
	if _, ok := c.(CMYK); ok {
		return c
	}
	r, g, b, _ := c.RGBA()
	cc, mm, yy, kk := RGBToCMYK(uint8(r>>8), uint8(g>>8), uint8(b>>8))
	return CMYK{cc, mm, yy, kk}
}
