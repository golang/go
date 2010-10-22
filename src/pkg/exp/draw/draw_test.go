// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package draw

import (
	"image"
	"testing"
)

func eq(c0, c1 image.Color) bool {
	r0, g0, b0, a0 := c0.RGBA()
	r1, g1, b1, a1 := c1.RGBA()
	return r0 == r1 && g0 == g1 && b0 == b1 && a0 == a1
}

func fillBlue(alpha int) image.Image {
	return image.NewColorImage(image.RGBAColor{0, 0, uint8(alpha), uint8(alpha)})
}

func fillAlpha(alpha int) image.Image {
	return image.NewColorImage(image.AlphaColor{uint8(alpha)})
}

func vgradGreen(alpha int) image.Image {
	m := image.NewRGBA(16, 16)
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			m.Set(x, y, image.RGBAColor{0, uint8(y * alpha / 15), 0, uint8(alpha)})
		}
	}
	return m
}

func vgradAlpha(alpha int) image.Image {
	m := image.NewAlpha(16, 16)
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			m.Set(x, y, image.AlphaColor{uint8(y * alpha / 15)})
		}
	}
	return m
}

func hgradRed(alpha int) Image {
	m := image.NewRGBA(16, 16)
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			m.Set(x, y, image.RGBAColor{uint8(x * alpha / 15), 0, 0, uint8(alpha)})
		}
	}
	return m
}

func gradYellow(alpha int) Image {
	m := image.NewRGBA(16, 16)
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			m.Set(x, y, image.RGBAColor{uint8(x * alpha / 15), uint8(y * alpha / 15), 0, uint8(alpha)})
		}
	}
	return m
}

type drawTest struct {
	desc     string
	src      image.Image
	mask     image.Image
	op       Op
	expected image.Color
}

var drawTests = []drawTest{
	// Uniform mask (0% opaque).
	{"nop", vgradGreen(255), fillAlpha(0), Over, image.RGBAColor{136, 0, 0, 255}},
	{"clear", vgradGreen(255), fillAlpha(0), Src, image.RGBAColor{0, 0, 0, 0}},
	// Uniform mask (100%, 75%, nil) and uniform source.
	// At (x, y) == (8, 8):
	// The destination pixel is {136, 0, 0, 255}.
	// The source pixel is {0, 0, 90, 90}.
	{"fill", fillBlue(90), fillAlpha(255), Over, image.RGBAColor{88, 0, 90, 255}},
	{"fillSrc", fillBlue(90), fillAlpha(255), Src, image.RGBAColor{0, 0, 90, 90}},
	{"fillAlpha", fillBlue(90), fillAlpha(192), Over, image.RGBAColor{100, 0, 68, 255}},
	{"fillAlphaSrc", fillBlue(90), fillAlpha(192), Src, image.RGBAColor{0, 0, 68, 68}},
	{"fillNil", fillBlue(90), nil, Over, image.RGBAColor{88, 0, 90, 255}},
	{"fillNilSrc", fillBlue(90), nil, Src, image.RGBAColor{0, 0, 90, 90}},
	// Uniform mask (100%, 75%, nil) and variable source.
	// At (x, y) == (8, 8):
	// The destination pixel is {136, 0, 0, 255}.
	// The source pixel is {0, 48, 0, 90}.
	{"copy", vgradGreen(90), fillAlpha(255), Over, image.RGBAColor{88, 48, 0, 255}},
	{"copySrc", vgradGreen(90), fillAlpha(255), Src, image.RGBAColor{0, 48, 0, 90}},
	{"copyAlpha", vgradGreen(90), fillAlpha(192), Over, image.RGBAColor{100, 36, 0, 255}},
	{"copyAlphaSrc", vgradGreen(90), fillAlpha(192), Src, image.RGBAColor{0, 36, 0, 68}},
	{"copyNil", vgradGreen(90), nil, Over, image.RGBAColor{88, 48, 0, 255}},
	{"copyNilSrc", vgradGreen(90), nil, Src, image.RGBAColor{0, 48, 0, 90}},
	// Variable mask and variable source.
	// At (x, y) == (8, 8):
	// The destination pixel is {136, 0, 0, 255}.
	// The source pixel is {0, 0, 255, 255}.
	// The mask pixel's alpha is 102, or 40%.
	{"generic", fillBlue(255), vgradAlpha(192), Over, image.RGBAColor{81, 0, 102, 255}},
	{"genericSrc", fillBlue(255), vgradAlpha(192), Src, image.RGBAColor{0, 0, 102, 102}},
}

func makeGolden(dst, src, mask image.Image, op Op) image.Image {
	// Since golden is a newly allocated image, we don't have to check if the
	// input source and mask images and the output golden image overlap.
	b := dst.Bounds()
	sx0 := src.Bounds().Min.X - b.Min.X
	sy0 := src.Bounds().Min.Y - b.Min.Y
	var mx0, my0 int
	if mask != nil {
		mx0 = mask.Bounds().Min.X - b.Min.X
		my0 = mask.Bounds().Min.Y - b.Min.Y
	}
	golden := image.NewRGBA(b.Max.X, b.Max.Y)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		my, sy := my0+y, sy0+y
		for x := b.Min.X; x < b.Max.X; x++ {
			mx, sx := mx0+x, sx0+x
			const M = 1<<16 - 1
			var dr, dg, db, da uint32
			if op == Over {
				dr, dg, db, da = dst.At(x, y).RGBA()
			}
			sr, sg, sb, sa := src.At(sx, sy).RGBA()
			ma := uint32(M)
			if mask != nil {
				_, _, _, ma = mask.At(mx, my).RGBA()
			}
			a := M - (sa * ma / M)
			golden.Set(x, y, image.RGBA64Color{
				uint16((dr*a + sr*ma) / M),
				uint16((dg*a + sg*ma) / M),
				uint16((db*a + sb*ma) / M),
				uint16((da*a + sa*ma) / M),
			})
		}
	}
	golden.Rect = b
	return golden
}

func TestDraw(t *testing.T) {
loop:
	for _, test := range drawTests {
		dst := hgradRed(255)
		// Draw the (src, mask, op) onto a copy of dst using a slow but obviously correct implementation.
		golden := makeGolden(dst, test.src, test.mask, test.op)
		b := dst.Bounds()
		if !b.Eq(golden.Bounds()) {
			t.Errorf("draw %s: bounds %v versus %v", test.desc, dst.Bounds(), golden.Bounds())
			continue
		}
		// Draw the same combination onto the actual dst using the optimized DrawMask implementation.
		DrawMask(dst, b, test.src, image.ZP, test.mask, image.ZP, test.op)
		// Check that the resultant pixel at (8, 8) matches what we expect
		// (the expected value can be verified by hand).
		if !eq(dst.At(8, 8), test.expected) {
			t.Errorf("draw %s: at (8, 8) %v versus %v", test.desc, dst.At(8, 8), test.expected)
			continue
		}
		// Check that the resultant dst image matches the golden output.
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				if !eq(dst.At(x, y), golden.At(x, y)) {
					t.Errorf("draw %s: at (%d, %d), %v versus golden %v", test.desc, x, y, dst.At(x, y), golden.At(x, y))
					continue loop
				}
			}
		}
	}
}

func TestDrawOverlap(t *testing.T) {
	for _, op := range []Op{Over, Src} {
		for yoff := -2; yoff <= 2; yoff++ {
		loop:
			for xoff := -2; xoff <= 2; xoff++ {
				m := gradYellow(127).(*image.RGBA)
				dst := &image.RGBA{
					Pix:    m.Pix,
					Stride: m.Stride,
					Rect:   image.Rect(5, 5, 10, 10),
				}
				src := &image.RGBA{
					Pix:    m.Pix,
					Stride: m.Stride,
					Rect:   image.Rect(5+xoff, 5+yoff, 10+xoff, 10+yoff),
				}
				// Draw the (src, mask, op) onto a copy of dst using a slow but obviously correct implementation.
				golden := makeGolden(dst, src, nil, op)
				b := dst.Bounds()
				if !b.Eq(golden.Bounds()) {
					t.Errorf("drawOverlap xoff=%d,yoff=%d: bounds %v versus %v", xoff, yoff, dst.Bounds(), golden.Bounds())
					continue
				}
				// Draw the same combination onto the actual dst using the optimized DrawMask implementation.
				DrawMask(dst, b, src, src.Bounds().Min, nil, image.ZP, op)
				// Check that the resultant dst image matches the golden output.
				for y := b.Min.Y; y < b.Max.Y; y++ {
					for x := b.Min.X; x < b.Max.X; x++ {
						if !eq(dst.At(x, y), golden.At(x, y)) {
							t.Errorf("drawOverlap xoff=%d,yoff=%d: at (%d, %d), %v versus golden %v", xoff, yoff, x, y, dst.At(x, y), golden.At(x, y))
							continue loop
						}
					}
				}
			}
		}
	}
}

// TestIssue836 verifies http://code.google.com/p/go/issues/detail?id=836.
func TestIssue836(t *testing.T) {
	a := image.NewRGBA(1, 1)
	b := image.NewRGBA(2, 2)
	b.Set(0, 0, image.RGBAColor{0, 0, 0, 5})
	b.Set(1, 0, image.RGBAColor{0, 0, 5, 5})
	b.Set(0, 1, image.RGBAColor{0, 5, 0, 5})
	b.Set(1, 1, image.RGBAColor{5, 0, 0, 5})
	Draw(a, image.Rect(0, 0, 1, 1), b, image.Pt(1, 1))
	if !eq(image.RGBAColor{5, 0, 0, 5}, a.At(0, 0)) {
		t.Errorf("Issue 836: want %v got %v", image.RGBAColor{5, 0, 0, 5}, a.At(0, 0))
	}
}
