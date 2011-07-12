// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package draw

import (
	"image"
	"image/ycbcr"
	"testing"
)

const (
	dstw, dsth = 640, 480
	srcw, srch = 400, 300
)

// bench benchmarks drawing src and mask images onto a dst image with the
// given op and the color models to create those images from.
// The created images' pixels are initialized to non-zero values.
func bench(b *testing.B, dcm, scm, mcm image.ColorModel, op Op) {
	b.StopTimer()

	var dst Image
	switch dcm {
	case image.RGBAColorModel:
		dst1 := image.NewRGBA(dstw, dsth)
		for y := 0; y < dsth; y++ {
			for x := 0; x < dstw; x++ {
				dst1.SetRGBA(x, y, image.RGBAColor{
					uint8(5 * x % 0x100),
					uint8(7 * y % 0x100),
					uint8((7*x + 5*y) % 0x100),
					0xff,
				})
			}
		}
		dst = dst1
	case image.RGBA64ColorModel:
		dst1 := image.NewRGBA64(dstw, dsth)
		for y := 0; y < dsth; y++ {
			for x := 0; x < dstw; x++ {
				dst1.SetRGBA64(x, y, image.RGBA64Color{
					uint16(53 * x % 0x10000),
					uint16(59 * y % 0x10000),
					uint16((59*x + 53*y) % 0x10000),
					0xffff,
				})
			}
		}
		dst = dst1
	default:
		panic("unreachable")
	}

	var src image.Image
	switch scm {
	case nil:
		src = &image.ColorImage{image.RGBAColor{0x11, 0x22, 0x33, 0xff}}
	case image.RGBAColorModel:
		src1 := image.NewRGBA(srcw, srch)
		for y := 0; y < srch; y++ {
			for x := 0; x < srcw; x++ {
				src1.SetRGBA(x, y, image.RGBAColor{
					uint8(13 * x % 0x80),
					uint8(11 * y % 0x80),
					uint8((11*x + 13*y) % 0x80),
					0x7f,
				})
			}
		}
		src = src1
	case image.RGBA64ColorModel:
		src1 := image.NewRGBA64(srcw, srch)
		for y := 0; y < srch; y++ {
			for x := 0; x < srcw; x++ {
				src1.SetRGBA64(x, y, image.RGBA64Color{
					uint16(103 * x % 0x8000),
					uint16(101 * y % 0x8000),
					uint16((101*x + 103*y) % 0x8000),
					0x7fff,
				})
			}
		}
		src = src1
	case image.NRGBAColorModel:
		src1 := image.NewNRGBA(srcw, srch)
		for y := 0; y < srch; y++ {
			for x := 0; x < srcw; x++ {
				src1.SetNRGBA(x, y, image.NRGBAColor{
					uint8(13 * x % 0x100),
					uint8(11 * y % 0x100),
					uint8((11*x + 13*y) % 0x100),
					0x7f,
				})
			}
		}
		src = src1
	case ycbcr.YCbCrColorModel:
		yy := make([]uint8, srcw*srch)
		cb := make([]uint8, srcw*srch)
		cr := make([]uint8, srcw*srch)
		for i := range yy {
			yy[i] = uint8(3 * i % 0x100)
			cb[i] = uint8(5 * i % 0x100)
			cr[i] = uint8(7 * i % 0x100)
		}
		src = &ycbcr.YCbCr{
			Y:              yy,
			Cb:             cb,
			Cr:             cr,
			YStride:        srcw,
			CStride:        srcw,
			SubsampleRatio: ycbcr.SubsampleRatio444,
			Rect:           image.Rect(0, 0, srcw, srch),
		}
	default:
		panic("unreachable")
	}

	var mask image.Image
	switch mcm {
	case nil:
		// No-op.
	case image.AlphaColorModel:
		mask1 := image.NewAlpha(srcw, srch)
		for y := 0; y < srch; y++ {
			for x := 0; x < srcw; x++ {
				a := uint8((23*x + 29*y) % 0x100)
				// Glyph masks are typically mostly zero,
				// so we only set a quarter of mask1's pixels.
				if a >= 0xc0 {
					mask1.SetAlpha(x, y, image.AlphaColor{a})
				}
			}
		}
		mask = mask1
	default:
		panic("unreachable")
	}

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		// Scatter the destination rectangle to draw into.
		x := 3 * i % (dstw - srcw)
		y := 7 * i % (dsth - srch)

		DrawMask(dst, dst.Bounds().Add(image.Point{x, y}), src, image.ZP, mask, image.ZP, op)
	}
}

// The BenchmarkFoo functions exercise a drawFoo fast-path function in draw.go.

func BenchmarkFillOver(b *testing.B) {
	bench(b, image.RGBAColorModel, nil, nil, Over)
}

func BenchmarkFillSrc(b *testing.B) {
	bench(b, image.RGBAColorModel, nil, nil, Src)
}

func BenchmarkCopyOver(b *testing.B) {
	bench(b, image.RGBAColorModel, image.RGBAColorModel, nil, Over)
}

func BenchmarkCopySrc(b *testing.B) {
	bench(b, image.RGBAColorModel, image.RGBAColorModel, nil, Src)
}

func BenchmarkNRGBAOver(b *testing.B) {
	bench(b, image.RGBAColorModel, image.NRGBAColorModel, nil, Over)
}

func BenchmarkNRGBASrc(b *testing.B) {
	bench(b, image.RGBAColorModel, image.NRGBAColorModel, nil, Src)
}

func BenchmarkYCbCr(b *testing.B) {
	bench(b, image.RGBAColorModel, ycbcr.YCbCrColorModel, nil, Over)
}

func BenchmarkGlyphOver(b *testing.B) {
	bench(b, image.RGBAColorModel, nil, image.AlphaColorModel, Over)
}

func BenchmarkRGBA(b *testing.B) {
	bench(b, image.RGBAColorModel, image.RGBA64ColorModel, nil, Src)
}

// The BenchmarkGenericFoo functions exercise the generic, slow-path code.

func BenchmarkGenericOver(b *testing.B) {
	bench(b, image.RGBA64ColorModel, image.RGBA64ColorModel, nil, Over)
}

func BenchmarkGenericMaskOver(b *testing.B) {
	bench(b, image.RGBA64ColorModel, image.RGBA64ColorModel, image.AlphaColorModel, Over)
}

func BenchmarkGenericSrc(b *testing.B) {
	bench(b, image.RGBA64ColorModel, image.RGBA64ColorModel, nil, Src)
}

func BenchmarkGenericMaskSrc(b *testing.B) {
	bench(b, image.RGBA64ColorModel, image.RGBA64ColorModel, image.AlphaColorModel, Src)
}
