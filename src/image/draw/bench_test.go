// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package draw

import (
	"image"
	"image/color"
	"reflect"
	"testing"
)

const (
	dstw, dsth = 640, 480
	srcw, srch = 400, 300
)

var palette = color.Palette{
	color.Black,
	color.White,
}

// bench benchmarks drawing src and mask images onto a dst image with the
// given op and the color models to create those images from.
// The created images' pixels are initialized to non-zero values.
func bench(b *testing.B, dcm, scm, mcm color.Model, op Op) {
	b.StopTimer()

	var dst Image
	switch dcm {
	case color.RGBAModel:
		dst1 := image.NewRGBA(image.Rect(0, 0, dstw, dsth))
		for y := 0; y < dsth; y++ {
			for x := 0; x < dstw; x++ {
				dst1.SetRGBA(x, y, color.RGBA{
					uint8(5 * x % 0x100),
					uint8(7 * y % 0x100),
					uint8((7*x + 5*y) % 0x100),
					0xff,
				})
			}
		}
		dst = dst1
	case color.RGBA64Model:
		dst1 := image.NewRGBA64(image.Rect(0, 0, dstw, dsth))
		for y := 0; y < dsth; y++ {
			for x := 0; x < dstw; x++ {
				dst1.SetRGBA64(x, y, color.RGBA64{
					uint16(53 * x % 0x10000),
					uint16(59 * y % 0x10000),
					uint16((59*x + 53*y) % 0x10000),
					0xffff,
				})
			}
		}
		dst = dst1
	default:
		// The == operator isn't defined on a color.Palette (a slice), so we
		// use reflection.
		if reflect.DeepEqual(dcm, palette) {
			dst1 := image.NewPaletted(image.Rect(0, 0, dstw, dsth), palette)
			for y := 0; y < dsth; y++ {
				for x := 0; x < dstw; x++ {
					dst1.SetColorIndex(x, y, uint8(x^y)&1)
				}
			}
			dst = dst1
		} else {
			b.Fatal("unknown destination color model", dcm)
		}
	}

	var src image.Image
	switch scm {
	case nil:
		src = &image.Uniform{C: color.RGBA{0x11, 0x22, 0x33, 0x44}}
	case color.CMYKModel:
		src1 := image.NewCMYK(image.Rect(0, 0, srcw, srch))
		for y := 0; y < srch; y++ {
			for x := 0; x < srcw; x++ {
				src1.SetCMYK(x, y, color.CMYK{
					uint8(13 * x % 0x100),
					uint8(11 * y % 0x100),
					uint8((11*x + 13*y) % 0x100),
					uint8((31*x + 37*y) % 0x100),
				})
			}
		}
		src = src1
	case color.GrayModel:
		src1 := image.NewGray(image.Rect(0, 0, srcw, srch))
		for y := 0; y < srch; y++ {
			for x := 0; x < srcw; x++ {
				src1.SetGray(x, y, color.Gray{
					uint8((11*x + 13*y) % 0x100),
				})
			}
		}
		src = src1
	case color.RGBAModel:
		src1 := image.NewRGBA(image.Rect(0, 0, srcw, srch))
		for y := 0; y < srch; y++ {
			for x := 0; x < srcw; x++ {
				src1.SetRGBA(x, y, color.RGBA{
					uint8(13 * x % 0x80),
					uint8(11 * y % 0x80),
					uint8((11*x + 13*y) % 0x80),
					0x7f,
				})
			}
		}
		src = src1
	case color.RGBA64Model:
		src1 := image.NewRGBA64(image.Rect(0, 0, srcw, srch))
		for y := 0; y < srch; y++ {
			for x := 0; x < srcw; x++ {
				src1.SetRGBA64(x, y, color.RGBA64{
					uint16(103 * x % 0x8000),
					uint16(101 * y % 0x8000),
					uint16((101*x + 103*y) % 0x8000),
					0x7fff,
				})
			}
		}
		src = src1
	case color.NRGBAModel:
		src1 := image.NewNRGBA(image.Rect(0, 0, srcw, srch))
		for y := 0; y < srch; y++ {
			for x := 0; x < srcw; x++ {
				src1.SetNRGBA(x, y, color.NRGBA{
					uint8(13 * x % 0x100),
					uint8(11 * y % 0x100),
					uint8((11*x + 13*y) % 0x100),
					0x7f,
				})
			}
		}
		src = src1
	case color.YCbCrModel:
		yy := make([]uint8, srcw*srch)
		cb := make([]uint8, srcw*srch)
		cr := make([]uint8, srcw*srch)
		for i := range yy {
			yy[i] = uint8(3 * i % 0x100)
			cb[i] = uint8(5 * i % 0x100)
			cr[i] = uint8(7 * i % 0x100)
		}
		src = &image.YCbCr{
			Y:              yy,
			Cb:             cb,
			Cr:             cr,
			YStride:        srcw,
			CStride:        srcw,
			SubsampleRatio: image.YCbCrSubsampleRatio444,
			Rect:           image.Rect(0, 0, srcw, srch),
		}
	default:
		b.Fatal("unknown source color model", scm)
	}

	var mask image.Image
	switch mcm {
	case nil:
		// No-op.
	case color.AlphaModel:
		mask1 := image.NewAlpha(image.Rect(0, 0, srcw, srch))
		for y := 0; y < srch; y++ {
			for x := 0; x < srcw; x++ {
				a := uint8((23*x + 29*y) % 0x100)
				// Glyph masks are typically mostly zero,
				// so we only set a quarter of mask1's pixels.
				if a >= 0xc0 {
					mask1.SetAlpha(x, y, color.Alpha{a})
				}
			}
		}
		mask = mask1
	default:
		b.Fatal("unknown mask color model", mcm)
	}

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		// Scatter the destination rectangle to draw into.
		x := 3 * i % (dstw - srcw)
		y := 7 * i % (dsth - srch)

		DrawMask(dst, dst.Bounds().Add(image.Pt(x, y)), src, image.ZP, mask, image.ZP, op)
	}
}

// The BenchmarkFoo functions exercise a drawFoo fast-path function in draw.go.

func BenchmarkFillOver(b *testing.B) {
	bench(b, color.RGBAModel, nil, nil, Over)
}

func BenchmarkFillSrc(b *testing.B) {
	bench(b, color.RGBAModel, nil, nil, Src)
}

func BenchmarkCopyOver(b *testing.B) {
	bench(b, color.RGBAModel, color.RGBAModel, nil, Over)
}

func BenchmarkCopySrc(b *testing.B) {
	bench(b, color.RGBAModel, color.RGBAModel, nil, Src)
}

func BenchmarkNRGBAOver(b *testing.B) {
	bench(b, color.RGBAModel, color.NRGBAModel, nil, Over)
}

func BenchmarkNRGBASrc(b *testing.B) {
	bench(b, color.RGBAModel, color.NRGBAModel, nil, Src)
}

func BenchmarkYCbCr(b *testing.B) {
	bench(b, color.RGBAModel, color.YCbCrModel, nil, Over)
}

func BenchmarkGray(b *testing.B) {
	bench(b, color.RGBAModel, color.GrayModel, nil, Over)
}

func BenchmarkCMYK(b *testing.B) {
	bench(b, color.RGBAModel, color.CMYKModel, nil, Over)
}

func BenchmarkGlyphOver(b *testing.B) {
	bench(b, color.RGBAModel, nil, color.AlphaModel, Over)
}

func BenchmarkRGBAMaskOver(b *testing.B) {
	bench(b, color.RGBAModel, color.RGBAModel, color.AlphaModel, Over)
}

func BenchmarkGrayMaskOver(b *testing.B) {
	bench(b, color.RGBAModel, color.GrayModel, color.AlphaModel, Over)
}

func BenchmarkRGBA64ImageMaskOver(b *testing.B) {
	bench(b, color.RGBAModel, color.RGBA64Model, color.AlphaModel, Over)
}

func BenchmarkRGBA(b *testing.B) {
	bench(b, color.RGBAModel, color.RGBA64Model, nil, Src)
}

func BenchmarkPalettedFill(b *testing.B) {
	bench(b, palette, nil, nil, Src)
}

func BenchmarkPalettedRGBA(b *testing.B) {
	bench(b, palette, color.RGBAModel, nil, Src)
}

// The BenchmarkGenericFoo functions exercise the generic, slow-path code.

func BenchmarkGenericOver(b *testing.B) {
	bench(b, color.RGBA64Model, color.RGBA64Model, nil, Over)
}

func BenchmarkGenericMaskOver(b *testing.B) {
	bench(b, color.RGBA64Model, color.RGBA64Model, color.AlphaModel, Over)
}

func BenchmarkGenericSrc(b *testing.B) {
	bench(b, color.RGBA64Model, color.RGBA64Model, nil, Src)
}

func BenchmarkGenericMaskSrc(b *testing.B) {
	bench(b, color.RGBA64Model, color.RGBA64Model, color.AlphaModel, Src)
}
