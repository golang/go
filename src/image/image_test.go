// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package image

import (
	"image/color"
	"image/color/palette"
	"testing"
)

type image interface {
	Image
	Opaque() bool
	Set(int, int, color.Color)
	SubImage(Rectangle) Image
}

func cmp(cm color.Model, c0, c1 color.Color) bool {
	r0, g0, b0, a0 := cm.Convert(c0).RGBA()
	r1, g1, b1, a1 := cm.Convert(c1).RGBA()
	return r0 == r1 && g0 == g1 && b0 == b1 && a0 == a1
}

var testImages = []struct {
	name  string
	image func() image
}{
	{"rgba", func() image { return NewRGBA(Rect(0, 0, 10, 10)) }},
	{"rgba64", func() image { return NewRGBA64(Rect(0, 0, 10, 10)) }},
	{"nrgba", func() image { return NewNRGBA(Rect(0, 0, 10, 10)) }},
	{"nrgba64", func() image { return NewNRGBA64(Rect(0, 0, 10, 10)) }},
	{"alpha", func() image { return NewAlpha(Rect(0, 0, 10, 10)) }},
	{"alpha16", func() image { return NewAlpha16(Rect(0, 0, 10, 10)) }},
	{"gray", func() image { return NewGray(Rect(0, 0, 10, 10)) }},
	{"gray16", func() image { return NewGray16(Rect(0, 0, 10, 10)) }},
	{"paletted", func() image {
		return NewPaletted(Rect(0, 0, 10, 10), color.Palette{
			Transparent,
			Opaque,
		})
	}},
}

func TestImage(t *testing.T) {
	for _, tc := range testImages {
		m := tc.image()
		if !Rect(0, 0, 10, 10).Eq(m.Bounds()) {
			t.Errorf("%T: want bounds %v, got %v", m, Rect(0, 0, 10, 10), m.Bounds())
			continue
		}
		if !cmp(m.ColorModel(), Transparent, m.At(6, 3)) {
			t.Errorf("%T: at (6, 3), want a zero color, got %v", m, m.At(6, 3))
			continue
		}
		m.Set(6, 3, Opaque)
		if !cmp(m.ColorModel(), Opaque, m.At(6, 3)) {
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
		if !cmp(m.ColorModel(), Opaque, m.At(6, 3)) {
			t.Errorf("%T: sub-image at (6, 3), want a non-zero color, got %v", m, m.At(6, 3))
			continue
		}
		if !cmp(m.ColorModel(), Transparent, m.At(3, 3)) {
			t.Errorf("%T: sub-image at (3, 3), want a zero color, got %v", m, m.At(3, 3))
			continue
		}
		m.Set(3, 3, Opaque)
		if !cmp(m.ColorModel(), Opaque, m.At(3, 3)) {
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

func TestNewXxxBadRectangle(t *testing.T) {
	// call calls f(r) and reports whether it ran without panicking.
	call := func(f func(Rectangle), r Rectangle) (ok bool) {
		defer func() {
			if recover() != nil {
				ok = false
			}
		}()
		f(r)
		return true
	}

	testCases := []struct {
		name string
		f    func(Rectangle)
	}{
		{"RGBA", func(r Rectangle) { NewRGBA(r) }},
		{"RGBA64", func(r Rectangle) { NewRGBA64(r) }},
		{"NRGBA", func(r Rectangle) { NewNRGBA(r) }},
		{"NRGBA64", func(r Rectangle) { NewNRGBA64(r) }},
		{"Alpha", func(r Rectangle) { NewAlpha(r) }},
		{"Alpha16", func(r Rectangle) { NewAlpha16(r) }},
		{"Gray", func(r Rectangle) { NewGray(r) }},
		{"Gray16", func(r Rectangle) { NewGray16(r) }},
		{"CMYK", func(r Rectangle) { NewCMYK(r) }},
		{"Paletted", func(r Rectangle) { NewPaletted(r, color.Palette{color.Black, color.White}) }},
		{"YCbCr", func(r Rectangle) { NewYCbCr(r, YCbCrSubsampleRatio422) }},
		{"NYCbCrA", func(r Rectangle) { NewNYCbCrA(r, YCbCrSubsampleRatio444) }},
	}

	for _, tc := range testCases {
		// Calling NewXxx(r) should fail (panic, since NewXxx doesn't return an
		// error) unless r's width and height are both non-negative.
		for _, negDx := range []bool{false, true} {
			for _, negDy := range []bool{false, true} {
				r := Rectangle{
					Min: Point{15, 28},
					Max: Point{16, 29},
				}
				if negDx {
					r.Max.X = 14
				}
				if negDy {
					r.Max.Y = 27
				}

				got := call(tc.f, r)
				want := !negDx && !negDy
				if got != want {
					t.Errorf("New%s: negDx=%t, negDy=%t: got %t, want %t",
						tc.name, negDx, negDy, got, want)
				}
			}
		}

		// Passing a Rectangle whose width and height is MaxInt should also fail
		// (panic), due to overflow.
		{
			zeroAsUint := uint(0)
			maxUint := zeroAsUint - 1
			maxInt := int(maxUint / 2)
			got := call(tc.f, Rectangle{
				Min: Point{0, 0},
				Max: Point{maxInt, maxInt},
			})
			if got {
				t.Errorf("New%s: overflow: got ok, want !ok", tc.name)
			}
		}
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

func TestRGBA64Image(t *testing.T) {
	// memset sets every element of s to v.
	memset := func(s []byte, v byte) {
		for i := range s {
			s[i] = v
		}
	}

	r := Rect(0, 0, 3, 2)
	testCases := []Image{
		NewAlpha(r),
		NewAlpha16(r),
		NewCMYK(r),
		NewGray(r),
		NewGray16(r),
		NewNRGBA(r),
		NewNRGBA64(r),
		NewNYCbCrA(r, YCbCrSubsampleRatio444),
		NewPaletted(r, palette.Plan9),
		NewRGBA(r),
		NewRGBA64(r),
		NewUniform(color.RGBA64{}),
		NewYCbCr(r, YCbCrSubsampleRatio444),
		r,
	}
	for _, tc := range testCases {
		switch tc := tc.(type) {
		// Most of the concrete image types in the testCases implement the
		// draw.RGBA64Image interface: they have a SetRGBA64 method. We use an
		// interface literal here, instead of importing "image/draw", to avoid
		// an import cycle.
		//
		// The YCbCr and NYCbCrA types are special-cased. Chroma subsampling
		// means that setting one pixel can modify neighboring pixels. They
		// don't have Set or SetRGBA64 methods because that side effect could
		// be surprising. Here, we just memset the channel buffers instead.
		//
		// The Uniform and Rectangle types are also special-cased, as they
		// don't have a Set or SetRGBA64 method.
		case interface {
			SetRGBA64(x, y int, c color.RGBA64)
		}:
			tc.SetRGBA64(1, 1, color.RGBA64{0x7FFF, 0x3FFF, 0x0000, 0x7FFF})

		case *NYCbCrA:
			memset(tc.YCbCr.Y, 0x77)
			memset(tc.YCbCr.Cb, 0x88)
			memset(tc.YCbCr.Cr, 0x99)
			memset(tc.A, 0xAA)

		case *Uniform:
			tc.C = color.RGBA64{0x7FFF, 0x3FFF, 0x0000, 0x7FFF}

		case *YCbCr:
			memset(tc.Y, 0x77)
			memset(tc.Cb, 0x88)
			memset(tc.Cr, 0x99)

		case Rectangle:
			// No-op. Rectangle pixels' colors are immutable. They're always
			// color.Opaque.

		default:
			t.Errorf("could not initialize pixels for %T", tc)
			continue
		}

		// Check that RGBA64At(x, y) is equivalent to At(x, y).RGBA().
		rgba64Image, ok := tc.(RGBA64Image)
		if !ok {
			t.Errorf("%T is not an RGBA64Image", tc)
			continue
		}
		got := rgba64Image.RGBA64At(1, 1)
		wantR, wantG, wantB, wantA := tc.At(1, 1).RGBA()
		if (uint32(got.R) != wantR) || (uint32(got.G) != wantG) ||
			(uint32(got.B) != wantB) || (uint32(got.A) != wantA) {
			t.Errorf("%T:\ngot  (0x%04X, 0x%04X, 0x%04X, 0x%04X)\n"+
				"want (0x%04X, 0x%04X, 0x%04X, 0x%04X)", tc,
				got.R, got.G, got.B, got.A,
				wantR, wantG, wantB, wantA)
			continue
		}
	}
}

func BenchmarkAt(b *testing.B) {
	for _, tc := range testImages {
		b.Run(tc.name, func { b ->
			m := tc.image()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m.At(4, 5)
			}
		})
	}
}

func BenchmarkSet(b *testing.B) {
	c := color.Gray{0xff}
	for _, tc := range testImages {
		b.Run(tc.name, func { b ->
			m := tc.image()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m.Set(4, 5, c)
			}
		})
	}
}

func BenchmarkRGBAAt(b *testing.B) {
	m := NewRGBA(Rect(0, 0, 10, 10))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.RGBAAt(4, 5)
	}
}

func BenchmarkRGBASetRGBA(b *testing.B) {
	m := NewRGBA(Rect(0, 0, 10, 10))
	c := color.RGBA{0xff, 0xff, 0xff, 0x13}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.SetRGBA(4, 5, c)
	}
}

func BenchmarkRGBA64At(b *testing.B) {
	m := NewRGBA64(Rect(0, 0, 10, 10))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.RGBA64At(4, 5)
	}
}

func BenchmarkRGBA64SetRGBA64(b *testing.B) {
	m := NewRGBA64(Rect(0, 0, 10, 10))
	c := color.RGBA64{0xffff, 0xffff, 0xffff, 0x1357}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.SetRGBA64(4, 5, c)
	}
}

func BenchmarkNRGBAAt(b *testing.B) {
	m := NewNRGBA(Rect(0, 0, 10, 10))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.NRGBAAt(4, 5)
	}
}

func BenchmarkNRGBASetNRGBA(b *testing.B) {
	m := NewNRGBA(Rect(0, 0, 10, 10))
	c := color.NRGBA{0xff, 0xff, 0xff, 0x13}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.SetNRGBA(4, 5, c)
	}
}

func BenchmarkNRGBA64At(b *testing.B) {
	m := NewNRGBA64(Rect(0, 0, 10, 10))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.NRGBA64At(4, 5)
	}
}

func BenchmarkNRGBA64SetNRGBA64(b *testing.B) {
	m := NewNRGBA64(Rect(0, 0, 10, 10))
	c := color.NRGBA64{0xffff, 0xffff, 0xffff, 0x1357}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.SetNRGBA64(4, 5, c)
	}
}

func BenchmarkAlphaAt(b *testing.B) {
	m := NewAlpha(Rect(0, 0, 10, 10))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.AlphaAt(4, 5)
	}
}

func BenchmarkAlphaSetAlpha(b *testing.B) {
	m := NewAlpha(Rect(0, 0, 10, 10))
	c := color.Alpha{0x13}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.SetAlpha(4, 5, c)
	}
}

func BenchmarkAlpha16At(b *testing.B) {
	m := NewAlpha16(Rect(0, 0, 10, 10))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.Alpha16At(4, 5)
	}
}

func BenchmarkAlphaSetAlpha16(b *testing.B) {
	m := NewAlpha16(Rect(0, 0, 10, 10))
	c := color.Alpha16{0x13}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.SetAlpha16(4, 5, c)
	}
}

func BenchmarkGrayAt(b *testing.B) {
	m := NewGray(Rect(0, 0, 10, 10))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.GrayAt(4, 5)
	}
}

func BenchmarkGraySetGray(b *testing.B) {
	m := NewGray(Rect(0, 0, 10, 10))
	c := color.Gray{0x13}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.SetGray(4, 5, c)
	}
}

func BenchmarkGray16At(b *testing.B) {
	m := NewGray16(Rect(0, 0, 10, 10))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.Gray16At(4, 5)
	}
}

func BenchmarkGraySetGray16(b *testing.B) {
	m := NewGray16(Rect(0, 0, 10, 10))
	c := color.Gray16{0x13}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.SetGray16(4, 5, c)
	}
}
