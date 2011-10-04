// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ycbcr provides images from the Y'CbCr color model.
//
// JPEG, VP8, the MPEG family and other codecs use this color model. Such
// codecs often use the terms YUV and Y'CbCr interchangeably, but strictly
// speaking, the term YUV applies only to analog video signals.
//
// Conversion between RGB and Y'CbCr is lossy and there are multiple, slightly
// different formulae for converting between the two. This package follows
// the JFIF specification at http://www.w3.org/Graphics/JPEG/jfif3.pdf.
package ycbcr

import (
	"image"
	"image/color"
)

// RGBToYCbCr converts an RGB triple to a YCbCr triple. All components lie
// within the range [0, 255].
func RGBToYCbCr(r, g, b uint8) (uint8, uint8, uint8) {
	// The JFIF specification says:
	//	Y' =  0.2990*R + 0.5870*G + 0.1140*B
	//	Cb = -0.1687*R - 0.3313*G + 0.5000*B + 128
	//	Cr =  0.5000*R - 0.4187*G - 0.0813*B + 128
	// http://www.w3.org/Graphics/JPEG/jfif3.pdf says Y but means Y'.
	r1 := int(r)
	g1 := int(g)
	b1 := int(b)
	yy := (19595*r1 + 38470*g1 + 7471*b1 + 1<<15) >> 16
	cb := (-11056*r1 - 21712*g1 + 32768*b1 + 257<<15) >> 16
	cr := (32768*r1 - 27440*g1 - 5328*b1 + 257<<15) >> 16
	if yy < 0 {
		yy = 0
	} else if yy > 255 {
		yy = 255
	}
	if cb < 0 {
		cb = 0
	} else if cb > 255 {
		cb = 255
	}
	if cr < 0 {
		cr = 0
	} else if cr > 255 {
		cr = 255
	}
	return uint8(yy), uint8(cb), uint8(cr)
}

// YCbCrToRGB converts a YCbCr triple to an RGB triple. All components lie
// within the range [0, 255].
func YCbCrToRGB(y, cb, cr uint8) (uint8, uint8, uint8) {
	// The JFIF specification says:
	//	R = Y' + 1.40200*(Cr-128)
	//	G = Y' - 0.34414*(Cb-128) - 0.71414*(Cr-128)
	//	B = Y' + 1.77200*(Cb-128)
	// http://www.w3.org/Graphics/JPEG/jfif3.pdf says Y but means Y'.
	yy1 := int(y)<<16 + 1<<15
	cb1 := int(cb) - 128
	cr1 := int(cr) - 128
	r := (yy1 + 91881*cr1) >> 16
	g := (yy1 - 22554*cb1 - 46802*cr1) >> 16
	b := (yy1 + 116130*cb1) >> 16
	if r < 0 {
		r = 0
	} else if r > 255 {
		r = 255
	}
	if g < 0 {
		g = 0
	} else if g > 255 {
		g = 255
	}
	if b < 0 {
		b = 0
	} else if b > 255 {
		b = 255
	}
	return uint8(r), uint8(g), uint8(b)
}

// YCbCrColor represents a fully opaque 24-bit Y'CbCr color, having 8 bits for
// each of one luma and two chroma components.
type YCbCrColor struct {
	Y, Cb, Cr uint8
}

func (c YCbCrColor) RGBA() (uint32, uint32, uint32, uint32) {
	r, g, b := YCbCrToRGB(c.Y, c.Cb, c.Cr)
	return uint32(r) * 0x101, uint32(g) * 0x101, uint32(b) * 0x101, 0xffff
}

func toYCbCrColor(c color.Color) color.Color {
	if _, ok := c.(YCbCrColor); ok {
		return c
	}
	r, g, b, _ := c.RGBA()
	y, u, v := RGBToYCbCr(uint8(r>>8), uint8(g>>8), uint8(b>>8))
	return YCbCrColor{y, u, v}
}

// YCbCrColorModel is the color model for YCbCrColor.
var YCbCrColorModel color.Model = color.ModelFunc(toYCbCrColor)

// SubsampleRatio is the chroma subsample ratio used in a YCbCr image.
type SubsampleRatio int

const (
	SubsampleRatio444 SubsampleRatio = iota
	SubsampleRatio422
	SubsampleRatio420
)

// YCbCr is an in-memory image of YCbCr colors. There is one Y sample per pixel,
// but each Cb and Cr sample can span one or more pixels.
// YStride is the Y slice index delta between vertically adjacent pixels.
// CStride is the Cb and Cr slice index delta between vertically adjacent pixels
// that map to separate chroma samples.
// It is not an absolute requirement, but YStride and len(Y) are typically
// multiples of 8, and:
//	For 4:4:4, CStride == YStride/1 && len(Cb) == len(Cr) == len(Y)/1.
//	For 4:2:2, CStride == YStride/2 && len(Cb) == len(Cr) == len(Y)/2.
//	For 4:2:0, CStride == YStride/2 && len(Cb) == len(Cr) == len(Y)/4.
type YCbCr struct {
	Y              []uint8
	Cb             []uint8
	Cr             []uint8
	YStride        int
	CStride        int
	SubsampleRatio SubsampleRatio
	Rect           image.Rectangle
}

func (p *YCbCr) ColorModel() color.Model {
	return YCbCrColorModel
}

func (p *YCbCr) Bounds() image.Rectangle {
	return p.Rect
}

func (p *YCbCr) At(x, y int) color.Color {
	if !(image.Point{x, y}.In(p.Rect)) {
		return YCbCrColor{}
	}
	switch p.SubsampleRatio {
	case SubsampleRatio422:
		i := x / 2
		return YCbCrColor{
			p.Y[y*p.YStride+x],
			p.Cb[y*p.CStride+i],
			p.Cr[y*p.CStride+i],
		}
	case SubsampleRatio420:
		i, j := x/2, y/2
		return YCbCrColor{
			p.Y[y*p.YStride+x],
			p.Cb[j*p.CStride+i],
			p.Cr[j*p.CStride+i],
		}
	}
	// Default to 4:4:4 subsampling.
	return YCbCrColor{
		p.Y[y*p.YStride+x],
		p.Cb[y*p.CStride+x],
		p.Cr[y*p.CStride+x],
	}
}

// SubImage returns an image representing the portion of the image p visible
// through r. The returned value shares pixels with the original image.
func (p *YCbCr) SubImage(r image.Rectangle) image.Image {
	q := new(YCbCr)
	*q = *p
	q.Rect = q.Rect.Intersect(r)
	return q
}

func (p *YCbCr) Opaque() bool {
	return true
}
