// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package image

import (
	"image/color"
	"testing"
)

type image interface {
	Image
	Opaque() bool
	Set(int, int, color.Color)
	SubImage(Rectangle) Image
}

func cmp(t *testing.T, cm color.Model, c0, c1 color.Color) bool {
	r0, g0, b0, a0 := cm.Convert(c0).RGBA()
	r1, g1, b1, a1 := cm.Convert(c1).RGBA()
	return r0 == r1 && g0 == g1 && b0 == b1 && a0 == a1
}

func TestImage(t *testing.T) {
	testImage := []image{
		NewRGBA(Rect(0, 0, 10, 10)),
		NewRGBA64(Rect(0, 0, 10, 10)),
		NewNRGBA(Rect(0, 0, 10, 10)),
		NewNRGBA64(Rect(0, 0, 10, 10)),
		NewAlpha(Rect(0, 0, 10, 10)),
		NewAlpha16(Rect(0, 0, 10, 10)),
		NewGray(Rect(0, 0, 10, 10)),
		NewGray16(Rect(0, 0, 10, 10)),
		NewPaletted(Rect(0, 0, 10, 10), color.Palette{
			Transparent,
			Opaque,
		}),
	}
	for _, m := range testImage {
		if !Rect(0, 0, 10, 10).Eq(m.Bounds()) {
			t.Errorf("%T: want bounds %v, got %v", m, Rect(0, 0, 10, 10), m.Bounds())
			continue
		}
		if !cmp(t, m.ColorModel(), Transparent, m.At(6, 3)) {
			t.Errorf("%T: at (6, 3), want a zero color, got %v", m, m.At(6, 3))
			continue
		}
		m.Set(6, 3, Opaque)
		if !cmp(t, m.ColorModel(), Opaque, m.At(6, 3)) {
			t.Errorf("%T: at (6, 3), want a non-zero color, got %v", m, m.At(6, 3))
			continue
		}
		if !m.SubImage(Rect(6, 3, 7, 4)).(image).Opaque() {
			t.Errorf("%T: at (6, 3) was not opaque", m)
			continue
		}
		m = m.SubImage(Rect(3, 2, 9, 8)).(image)
		if !Rect(3, 2, 9, 8).Eq(m.Bounds()) {
			t.Errorf("%T: sub-image want bounds %v, got %v", m, Rect(3, 2, 9, 8), m.Bounds())
			continue
		}
		if !cmp(t, m.ColorModel(), Opaque, m.At(6, 3)) {
			t.Errorf("%T: sub-image at (6, 3), want a non-zero color, got %v", m, m.At(6, 3))
			continue
		}
		if !cmp(t, m.ColorModel(), Transparent, m.At(3, 3)) {
			t.Errorf("%T: sub-image at (3, 3), want a zero color, got %v", m, m.At(3, 3))
			continue
		}
		m.Set(3, 3, Opaque)
		if !cmp(t, m.ColorModel(), Opaque, m.At(3, 3)) {
			t.Errorf("%T: sub-image at (3, 3), want a non-zero color, got %v", m, m.At(3, 3))
			continue
		}
		// Test that taking an empty sub-image starting at a corner does not panic.
		m.SubImage(Rect(0, 0, 0, 0))
		m.SubImage(Rect(10, 0, 10, 0))
		m.SubImage(Rect(0, 10, 0, 10))
		m.SubImage(Rect(10, 10, 10, 10))
	}
}

func Test16BitsPerColorChannel(t *testing.T) {
	testColorModel := []color.Model{
		color.RGBA64Model,
		color.NRGBA64Model,
		color.Alpha16Model,
		color.Gray16Model,
	}
	for _, cm := range testColorModel {
		c := cm.Convert(color.RGBA64{0x1234, 0x1234, 0x1234, 0x1234}) // Premultiplied alpha.
		r, _, _, _ := c.RGBA()
		if r != 0x1234 {
			t.Errorf("%T: want red value 0x%04x got 0x%04x", c, 0x1234, r)
			continue
		}
	}
	testImage := []image{
		NewRGBA64(Rect(0, 0, 10, 10)),
		NewNRGBA64(Rect(0, 0, 10, 10)),
		NewAlpha16(Rect(0, 0, 10, 10)),
		NewGray16(Rect(0, 0, 10, 10)),
	}
	for _, m := range testImage {
		m.Set(1, 2, color.NRGBA64{0xffff, 0xffff, 0xffff, 0x1357}) // Non-premultiplied alpha.
		r, _, _, _ := m.At(1, 2).RGBA()
		if r != 0x1357 {
			t.Errorf("%T: want red value 0x%04x got 0x%04x", m, 0x1357, r)
			continue
		}
	}
}
