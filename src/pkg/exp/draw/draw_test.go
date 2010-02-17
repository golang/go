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
	return image.ColorImage{image.RGBAColor{0, 0, uint8(alpha), uint8(alpha)}}
}

func fillAlpha(alpha int) image.Image {
	return image.ColorImage{image.AlphaColor{uint8(alpha)}}
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

type drawTest struct {
	desc     string
	src      image.Image
	mask     image.Image
	op       Op
	expected image.Color
}

var drawTests = []drawTest{
	// Uniform mask (0% opaque).
	drawTest{"nop", vgradGreen(255), fillAlpha(0), Over, image.RGBAColor{136, 0, 0, 255}},
	drawTest{"clear", vgradGreen(255), fillAlpha(0), Src, image.RGBAColor{0, 0, 0, 0}},
	// Uniform mask (100%, 75%, nil) and uniform source.
	// At (x, y) == (8, 8):
	// The destination pixel is {136, 0, 0, 255}.
	// The source pixel is {0, 0, 90, 90}.
	drawTest{"fill", fillBlue(90), fillAlpha(255), Over, image.RGBAColor{88, 0, 90, 255}},
	drawTest{"fillSrc", fillBlue(90), fillAlpha(255), Src, image.RGBAColor{0, 0, 90, 90}},
	drawTest{"fillAlpha", fillBlue(90), fillAlpha(192), Over, image.RGBAColor{100, 0, 68, 255}},
	drawTest{"fillAlphaSrc", fillBlue(90), fillAlpha(192), Src, image.RGBAColor{0, 0, 68, 68}},
	drawTest{"fillNil", fillBlue(90), nil, Over, image.RGBAColor{88, 0, 90, 255}},
	drawTest{"fillNilSrc", fillBlue(90), nil, Src, image.RGBAColor{0, 0, 90, 90}},
	// Uniform mask (100%, 75%, nil) and variable source.
	// At (x, y) == (8, 8):
	// The destination pixel is {136, 0, 0, 255}.
	// The source pixel is {0, 48, 0, 90}.
	drawTest{"copy", vgradGreen(90), fillAlpha(255), Over, image.RGBAColor{88, 48, 0, 255}},
	drawTest{"copySrc", vgradGreen(90), fillAlpha(255), Src, image.RGBAColor{0, 48, 0, 90}},
	drawTest{"copyAlpha", vgradGreen(90), fillAlpha(192), Over, image.RGBAColor{100, 36, 0, 255}},
	drawTest{"copyAlphaSrc", vgradGreen(90), fillAlpha(192), Src, image.RGBAColor{0, 36, 0, 68}},
	drawTest{"copyNil", vgradGreen(90), nil, Over, image.RGBAColor{88, 48, 0, 255}},
	drawTest{"copyNilSrc", vgradGreen(90), nil, Src, image.RGBAColor{0, 48, 0, 90}},
	// Variable mask and variable source.
	// At (x, y) == (8, 8):
	// The destination pixel is {136, 0, 0, 255}.
	// The source pixel is {0, 0, 255, 255}.
	// The mask pixel's alpha is 102, or 40%.
	drawTest{"generic", fillBlue(255), vgradAlpha(192), Over, image.RGBAColor{81, 0, 102, 255}},
	drawTest{"genericSrc", fillBlue(255), vgradAlpha(192), Src, image.RGBAColor{0, 0, 102, 102}},
}

func TestDraw(t *testing.T) {
	for _, test := range drawTests {
		dst := hgradRed(255)
		DrawMask(dst, Rect(0, 0, 16, 16), test.src, ZP, test.mask, ZP, test.op)
		if !eq(dst.At(8, 8), test.expected) {
			t.Errorf("draw %s: %v versus %v", test.desc, dst.At(8, 8), test.expected)
		}
	}
}
