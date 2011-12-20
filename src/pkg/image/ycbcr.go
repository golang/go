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
)

// YCbCr is an in-memory image of Y'CbCr colors. There is one Y sample per
// pixel, but each Cb and Cr sample can span one or more pixels.
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
	if !(Point{x, y}.In(p.Rect)) {
		return color.YCbCr{}
	}
	switch p.SubsampleRatio {
	case YCbCrSubsampleRatio422:
		i := x / 2
		return color.YCbCr{
			p.Y[y*p.YStride+x],
			p.Cb[y*p.CStride+i],
			p.Cr[y*p.CStride+i],
		}
	case YCbCrSubsampleRatio420:
		i, j := x/2, y/2
		return color.YCbCr{
			p.Y[y*p.YStride+x],
			p.Cb[j*p.CStride+i],
			p.Cr[j*p.CStride+i],
		}
	}
	// Default to 4:4:4 subsampling.
	return color.YCbCr{
		p.Y[y*p.YStride+x],
		p.Cb[y*p.CStride+x],
		p.Cr[y*p.CStride+x],
	}
}

// SubImage returns an image representing the portion of the image p visible
// through r. The returned value shares pixels with the original image.
func (p *YCbCr) SubImage(r Rectangle) Image {
	q := new(YCbCr)
	*q = *p
	q.Rect = q.Rect.Intersect(r)
	return q
}

func (p *YCbCr) Opaque() bool {
	return true
}
