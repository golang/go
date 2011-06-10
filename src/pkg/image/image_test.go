// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package image

import (
	"testing"
)

func cmp(t *testing.T, cm ColorModel, c0, c1 Color) bool {
	r0, g0, b0, a0 := cm.Convert(c0).RGBA()
	r1, g1, b1, a1 := cm.Convert(c1).RGBA()
	return r0 == r1 && g0 == g1 && b0 == b1 && a0 == a1
}

func TestImage(t *testing.T) {
	type buffered interface {
		Image
		Set(int, int, Color)
		SubImage(Rectangle) Image
	}
	testImage := []Image{
		NewRGBA(10, 10),
		NewRGBA64(10, 10),
		NewNRGBA(10, 10),
		NewNRGBA64(10, 10),
		NewAlpha(10, 10),
		NewAlpha16(10, 10),
		NewGray(10, 10),
		NewGray16(10, 10),
		NewPaletted(10, 10, PalettedColorModel{
			Transparent,
			Opaque,
		}),
	}
	for _, m := range testImage {
		b := m.(buffered)
		if !Rect(0, 0, 10, 10).Eq(b.Bounds()) {
			t.Errorf("%T: want bounds %v, got %v", b, Rect(0, 0, 10, 10), b.Bounds())
			continue
		}
		if !cmp(t, b.ColorModel(), Transparent, b.At(6, 3)) {
			t.Errorf("%T: at (6, 3), want a zero color, got %v", b, b.At(6, 3))
			continue
		}
		b.Set(6, 3, Opaque)
		if !cmp(t, b.ColorModel(), Opaque, b.At(6, 3)) {
			t.Errorf("%T: at (6, 3), want a non-zero color, got %v", b, b.At(6, 3))
			continue
		}
		b = b.SubImage(Rect(3, 2, 9, 8)).(buffered)
		if !Rect(3, 2, 9, 8).Eq(b.Bounds()) {
			t.Errorf("%T: sub-image want bounds %v, got %v", b, Rect(3, 2, 9, 8), b.Bounds())
			continue
		}
		if !cmp(t, b.ColorModel(), Opaque, b.At(6, 3)) {
			t.Errorf("%T: sub-image at (6, 3), want a non-zero color, got %v", b, b.At(6, 3))
			continue
		}
		if !cmp(t, b.ColorModel(), Transparent, b.At(3, 3)) {
			t.Errorf("%T: sub-image at (3, 3), want a zero color, got %v", b, b.At(3, 3))
			continue
		}
		b.Set(3, 3, Opaque)
		if !cmp(t, b.ColorModel(), Opaque, b.At(3, 3)) {
			t.Errorf("%T: sub-image at (3, 3), want a non-zero color, got %v", b, b.At(3, 3))
			continue
		}
	}
}
