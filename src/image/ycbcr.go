// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package image

import (
	"image/color"
)

// YCbCrSubsampleRatio is the chroma subsample ratio used in a YCbCr image.
type YCbCrSubsampleRatio int

const (
	YCbCrSubsampleRatio444 YCbCrSubsampleRatio = iota
	YCbCrSubsampleRatio422
	YCbCrSubsampleRatio420
	YCbCrSubsampleRatio440
	YCbCrSubsampleRatio411
	YCbCrSubsampleRatio410
)

func (s YCbCrSubsampleRatio) String() string {
	switch s {
	case YCbCrSubsampleRatio444:
		return "YCbCrSubsampleRatio444"
	case YCbCrSubsampleRatio422:
		return "YCbCrSubsampleRatio422"
	case YCbCrSubsampleRatio420:
		return "YCbCrSubsampleRatio420"
	case YCbCrSubsampleRatio440:
		return "YCbCrSubsampleRatio440"
	case YCbCrSubsampleRatio411:
		return "YCbCrSubsampleRatio411"
	case YCbCrSubsampleRatio410:
		return "YCbCrSubsampleRatio410"
	}
	return "YCbCrSubsampleRatioUnknown"
}

// YCbCr is an in-memory image of Y'CbCr colors. There is one Y sample per
// pixel, but each Cb and Cr sample can span one or more pixels.
// YStride is the Y slice index delta between vertically adjacent pixels.
// CStride is the Cb and Cr slice index delta between vertically adjacent pixels
// that map to separate chroma samples.
// It is not an absolute requirement, but YStride and len(Y) are typically
// multiples of 8, and:
//
//	For 4:4:4, CStride == YStride/1 && len(Cb) == len(Cr) == len(Y)/1.
//	For 4:2:2, CStride == YStride/2 && len(Cb) == len(Cr) == len(Y)/2.
//	For 4:2:0, CStride == YStride/2 && len(Cb) == len(Cr) == len(Y)/4.
//	For 4:4:0, CStride == YStride/1 && len(Cb) == len(Cr) == len(Y)/2.
//	For 4:1:1, CStride == YStride/4 && len(Cb) == len(Cr) == len(Y)/4.
//	For 4:1:0, CStride == YStride/4 && len(Cb) == len(Cr) == len(Y)/8.
type YCbCr struct {
	Y, Cb, Cr      []uint8
	YStride        int
	CStride        int
	SubsampleRatio YCbCrSubsampleRatio
	Rect           Rectangle
}

func (p *YCbCr) ColorModel() color.Model {
	return color.YCbCrModel
}

func (p *YCbCr) Bounds() Rectangle {
	return p.Rect
}

func (p *YCbCr) At(x, y int) color.Color {
	return p.YCbCrAt(x, y)
}

func (p *YCbCr) RGBA64At(x, y int) color.RGBA64 {
	r, g, b, a := p.YCbCrAt(x, y).RGBA()
	return color.RGBA64{uint16(r), uint16(g), uint16(b), uint16(a)}
}

func (p *YCbCr) YCbCrAt(x, y int) color.YCbCr {
	if !(Point{x, y}.In(p.Rect)) {
		return color.YCbCr{}
	}
	yi := p.YOffset(x, y)
	ci := p.COffset(x, y)
	return color.YCbCr{
		p.Y[yi],
		p.Cb[ci],
		p.Cr[ci],
	}
}

// YOffset returns the index of the first element of Y that corresponds to
// the pixel at (x, y).
func (p *YCbCr) YOffset(x, y int) int {
	return (y-p.Rect.Min.Y)*p.YStride + (x - p.Rect.Min.X)
}

// COffset returns the index of the first element of Cb or Cr that corresponds
// to the pixel at (x, y).
func (p *YCbCr) COffset(x, y int) int {
	switch p.SubsampleRatio {
	case YCbCrSubsampleRatio422:
		return (y-p.Rect.Min.Y)*p.CStride + (x/2 - p.Rect.Min.X/2)
	case YCbCrSubsampleRatio420:
		return (y/2-p.Rect.Min.Y/2)*p.CStride + (x/2 - p.Rect.Min.X/2)
	case YCbCrSubsampleRatio440:
		return (y/2-p.Rect.Min.Y/2)*p.CStride + (x - p.Rect.Min.X)
	case YCbCrSubsampleRatio411:
		return (y-p.Rect.Min.Y)*p.CStride + (x/4 - p.Rect.Min.X/4)
	case YCbCrSubsampleRatio410:
		return (y/2-p.Rect.Min.Y/2)*p.CStride + (x/4 - p.Rect.Min.X/4)
	}
	// Default to 4:4:4 subsampling.
	return (y-p.Rect.Min.Y)*p.CStride + (x - p.Rect.Min.X)
}

// SubImage returns an image representing the portion of the image p visible
// through r. The returned value shares pixels with the original image.
func (p *YCbCr) SubImage(r Rectangle) Image {
	r = r.Intersect(p.Rect)
	// If r1 and r2 are Rectangles, r1.Intersect(r2) is not guaranteed to be inside
	// either r1 or r2 if the intersection is empty. Without explicitly checking for
	// this, the Pix[i:] expression below can panic.
	if r.Empty() {
		return &YCbCr{
			SubsampleRatio: p.SubsampleRatio,
		}
	}
	yi := p.YOffset(r.Min.X, r.Min.Y)
	ci := p.COffset(r.Min.X, r.Min.Y)
	return &YCbCr{
		Y:              p.Y[yi:],
		Cb:             p.Cb[ci:],
		Cr:             p.Cr[ci:],
		SubsampleRatio: p.SubsampleRatio,
		YStride:        p.YStride,
		CStride:        p.CStride,
		Rect:           r,
	}
}

func (p *YCbCr) Opaque() bool {
	return true
}

func yCbCrSize(r Rectangle, subsampleRatio YCbCrSubsampleRatio) (w, h, cw, ch int) {
	w, h = r.Dx(), r.Dy()
	switch subsampleRatio {
	case YCbCrSubsampleRatio422:
		cw = (r.Max.X+1)/2 - r.Min.X/2
		ch = h
	case YCbCrSubsampleRatio420:
		cw = (r.Max.X+1)/2 - r.Min.X/2
		ch = (r.Max.Y+1)/2 - r.Min.Y/2
	case YCbCrSubsampleRatio440:
		cw = w
		ch = (r.Max.Y+1)/2 - r.Min.Y/2
	case YCbCrSubsampleRatio411:
		cw = (r.Max.X+3)/4 - r.Min.X/4
		ch = h
	case YCbCrSubsampleRatio410:
		cw = (r.Max.X+3)/4 - r.Min.X/4
		ch = (r.Max.Y+1)/2 - r.Min.Y/2
	default:
		// Default to 4:4:4 subsampling.
		cw = w
		ch = h
	}
	return
}

// NewYCbCr returns a new YCbCr image with the given bounds and subsample
// ratio.
func NewYCbCr(r Rectangle, subsampleRatio YCbCrSubsampleRatio) *YCbCr {
	w, h, cw, ch := yCbCrSize(r, subsampleRatio)

	// totalLength should be the same as i2, below, for a valid Rectangle r.
	totalLength := add2NonNeg(
		mul3NonNeg(1, w, h),
		mul3NonNeg(2, cw, ch),
	)
	if totalLength < 0 {
		panic("image: NewYCbCr Rectangle has huge or negative dimensions")
	}

	i0 := w*h + 0*cw*ch
	i1 := w*h + 1*cw*ch
	i2 := w*h + 2*cw*ch
	b := make([]byte, i2)
	return &YCbCr{
		Y:              b[:i0:i0],
		Cb:             b[i0:i1:i1],
		Cr:             b[i1:i2:i2],
		SubsampleRatio: subsampleRatio,
		YStride:        w,
		CStride:        cw,
		Rect:           r,
	}
}

// NYCbCrA is an in-memory image of non-alpha-premultiplied Y'CbCr-with-alpha
// colors. A and AStride are analogous to the Y and YStride fields of the
// embedded YCbCr.
type NYCbCrA struct {
	YCbCr
	A       []uint8
	AStride int
}

func (p *NYCbCrA) ColorModel() color.Model {
	return color.NYCbCrAModel
}

func (p *NYCbCrA) At(x, y int) color.Color {
	return p.NYCbCrAAt(x, y)
}

func (p *NYCbCrA) RGBA64At(x, y int) color.RGBA64 {
	r, g, b, a := p.NYCbCrAAt(x, y).RGBA()
	return color.RGBA64{uint16(r), uint16(g), uint16(b), uint16(a)}
}

func (p *NYCbCrA) NYCbCrAAt(x, y int) color.NYCbCrA {
	if !(Point{X: x, Y: y}.In(p.Rect)) {
		return color.NYCbCrA{}
	}
	yi := p.YOffset(x, y)
	ci := p.COffset(x, y)
	ai := p.AOffset(x, y)
	return color.NYCbCrA{
		color.YCbCr{
			Y:  p.Y[yi],
			Cb: p.Cb[ci],
			Cr: p.Cr[ci],
		},
		p.A[ai],
	}
}

// AOffset returns the index of the first element of A that corresponds to the
// pixel at (x, y).
func (p *NYCbCrA) AOffset(x, y int) int {
	return (y-p.Rect.Min.Y)*p.AStride + (x - p.Rect.Min.X)
}

// SubImage returns an image representing the portion of the image p visible
// through r. The returned value shares pixels with the original image.
func (p *NYCbCrA) SubImage(r Rectangle) Image {
	r = r.Intersect(p.Rect)
	// If r1 and r2 are Rectangles, r1.Intersect(r2) is not guaranteed to be inside
	// either r1 or r2 if the intersection is empty. Without explicitly checking for
	// this, the Pix[i:] expression below can panic.
	if r.Empty() {
		return &NYCbCrA{
			YCbCr: YCbCr{
				SubsampleRatio: p.SubsampleRatio,
			},
		}
	}
	yi := p.YOffset(r.Min.X, r.Min.Y)
	ci := p.COffset(r.Min.X, r.Min.Y)
	ai := p.AOffset(r.Min.X, r.Min.Y)
	return &NYCbCrA{
		YCbCr: YCbCr{
			Y:              p.Y[yi:],
			Cb:             p.Cb[ci:],
			Cr:             p.Cr[ci:],
			SubsampleRatio: p.SubsampleRatio,
			YStride:        p.YStride,
			CStride:        p.CStride,
			Rect:           r,
		},
		A:       p.A[ai:],
		AStride: p.AStride,
	}
}

// Opaque scans the entire image and reports whether it is fully opaque.
func (p *NYCbCrA) Opaque() bool {
	if p.Rect.Empty() {
		return true
	}
	i0, i1 := 0, p.Rect.Dx()
	for y := p.Rect.Min.Y; y < p.Rect.Max.Y; y++ {
		for _, a := range p.A[i0:i1] {
			if a != 0xff {
				return false
			}
		}
		i0 += p.AStride
		i1 += p.AStride
	}
	return true
}

// NewNYCbCrA returns a new [NYCbCrA] image with the given bounds and subsample
// ratio.
func NewNYCbCrA(r Rectangle, subsampleRatio YCbCrSubsampleRatio) *NYCbCrA {
	w, h, cw, ch := yCbCrSize(r, subsampleRatio)

	// totalLength should be the same as i3, below, for a valid Rectangle r.
	totalLength := add2NonNeg(
		mul3NonNeg(2, w, h),
		mul3NonNeg(2, cw, ch),
	)
	if totalLength < 0 {
		panic("image: NewNYCbCrA Rectangle has huge or negative dimension")
	}

	i0 := 1*w*h + 0*cw*ch
	i1 := 1*w*h + 1*cw*ch
	i2 := 1*w*h + 2*cw*ch
	i3 := 2*w*h + 2*cw*ch
	b := make([]byte, i3)
	return &NYCbCrA{
		YCbCr: YCbCr{
			Y:              b[:i0:i0],
			Cb:             b[i0:i1:i1],
			Cr:             b[i1:i2:i2],
			SubsampleRatio: subsampleRatio,
			YStride:        w,
			CStride:        cw,
			Rect:           r,
		},
		A:       b[i2:],
		AStride: w,
	}
}
