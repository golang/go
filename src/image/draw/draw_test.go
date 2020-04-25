// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package draw

import (
	"image"
	"image/color"
	"image/png"
	"os"
	"testing"
	"testing/quick"
)

func eq(c0, c1 color.Color) bool {
	r0, g0, b0, a0 := c0.RGBA()
	r1, g1, b1, a1 := c1.RGBA()
	return r0 == r1 && g0 == g1 && b0 == b1 && a0 == a1
}

func fillBlue(alpha int) image.Image {
	return image.NewUniform(color.RGBA{0, 0, uint8(alpha), uint8(alpha)})
}

func fillAlpha(alpha int) image.Image {
	return image.NewUniform(color.Alpha{uint8(alpha)})
}

func vgradGreen(alpha int) image.Image {
	m := image.NewRGBA(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			m.Set(x, y, color.RGBA{0, uint8(y * alpha / 15), 0, uint8(alpha)})
		}
	}
	return m
}

func vgradAlpha(alpha int) image.Image {
	m := image.NewAlpha(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			m.Set(x, y, color.Alpha{uint8(y * alpha / 15)})
		}
	}
	return m
}

func vgradGreenNRGBA(alpha int) image.Image {
	m := image.NewNRGBA(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			m.Set(x, y, color.RGBA{0, uint8(y * 0x11), 0, uint8(alpha)})
		}
	}
	return m
}

func vgradCr() image.Image {
	m := &image.YCbCr{
		Y:              make([]byte, 16*16),
		Cb:             make([]byte, 16*16),
		Cr:             make([]byte, 16*16),
		YStride:        16,
		CStride:        16,
		SubsampleRatio: image.YCbCrSubsampleRatio444,
		Rect:           image.Rect(0, 0, 16, 16),
	}
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			m.Cr[y*m.CStride+x] = uint8(y * 0x11)
		}
	}
	return m
}

func vgradGray() image.Image {
	m := image.NewGray(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			m.Set(x, y, color.Gray{uint8(y * 0x11)})
		}
	}
	return m
}

func vgradMagenta() image.Image {
	m := image.NewCMYK(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			m.Set(x, y, color.CMYK{0, uint8(y * 0x11), 0, 0x3f})
		}
	}
	return m
}

func hgradRed(alpha int) Image {
	m := image.NewRGBA(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			m.Set(x, y, color.RGBA{uint8(x * alpha / 15), 0, 0, uint8(alpha)})
		}
	}
	return m
}

func gradYellow(alpha int) Image {
	m := image.NewRGBA(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			m.Set(x, y, color.RGBA{uint8(x * alpha / 15), uint8(y * alpha / 15), 0, uint8(alpha)})
		}
	}
	return m
}

type drawTest struct {
	desc     string
	src      image.Image
	mask     image.Image
	op       Op
	expected color.Color
}

var drawTests = []drawTest{
	// Uniform mask (0% opaque).
	{"nop", vgradGreen(255), fillAlpha(0), Over, color.RGBA{136, 0, 0, 255}},
	{"clear", vgradGreen(255), fillAlpha(0), Src, color.RGBA{0, 0, 0, 0}},
	// Uniform mask (100%, 75%, nil) and uniform source.
	// At (x, y) == (8, 8):
	// The destination pixel is {136, 0, 0, 255}.
	// The source pixel is {0, 0, 90, 90}.
	{"fill", fillBlue(90), fillAlpha(255), Over, color.RGBA{88, 0, 90, 255}},
	{"fillSrc", fillBlue(90), fillAlpha(255), Src, color.RGBA{0, 0, 90, 90}},
	{"fillAlpha", fillBlue(90), fillAlpha(192), Over, color.RGBA{100, 0, 68, 255}},
	{"fillAlphaSrc", fillBlue(90), fillAlpha(192), Src, color.RGBA{0, 0, 68, 68}},
	{"fillNil", fillBlue(90), nil, Over, color.RGBA{88, 0, 90, 255}},
	{"fillNilSrc", fillBlue(90), nil, Src, color.RGBA{0, 0, 90, 90}},
	// Uniform mask (100%, 75%, nil) and variable source.
	// At (x, y) == (8, 8):
	// The destination pixel is {136, 0, 0, 255}.
	// The source pixel is {0, 48, 0, 90}.
	{"copy", vgradGreen(90), fillAlpha(255), Over, color.RGBA{88, 48, 0, 255}},
	{"copySrc", vgradGreen(90), fillAlpha(255), Src, color.RGBA{0, 48, 0, 90}},
	{"copyAlpha", vgradGreen(90), fillAlpha(192), Over, color.RGBA{100, 36, 0, 255}},
	{"copyAlphaSrc", vgradGreen(90), fillAlpha(192), Src, color.RGBA{0, 36, 0, 68}},
	{"copyNil", vgradGreen(90), nil, Over, color.RGBA{88, 48, 0, 255}},
	{"copyNilSrc", vgradGreen(90), nil, Src, color.RGBA{0, 48, 0, 90}},
	// Uniform mask (100%, 75%, nil) and variable NRGBA source.
	// At (x, y) == (8, 8):
	// The destination pixel is {136, 0, 0, 255}.
	// The source pixel is {0, 136, 0, 90} in NRGBA-space, which is {0, 48, 0, 90} in RGBA-space.
	// The result pixel is different than in the "copy*" test cases because of rounding errors.
	{"nrgba", vgradGreenNRGBA(90), fillAlpha(255), Over, color.RGBA{88, 46, 0, 255}},
	{"nrgbaSrc", vgradGreenNRGBA(90), fillAlpha(255), Src, color.RGBA{0, 46, 0, 90}},
	{"nrgbaAlpha", vgradGreenNRGBA(90), fillAlpha(192), Over, color.RGBA{100, 34, 0, 255}},
	{"nrgbaAlphaSrc", vgradGreenNRGBA(90), fillAlpha(192), Src, color.RGBA{0, 34, 0, 68}},
	{"nrgbaNil", vgradGreenNRGBA(90), nil, Over, color.RGBA{88, 46, 0, 255}},
	{"nrgbaNilSrc", vgradGreenNRGBA(90), nil, Src, color.RGBA{0, 46, 0, 90}},
	// Uniform mask (100%, 75%, nil) and variable YCbCr source.
	// At (x, y) == (8, 8):
	// The destination pixel is {136, 0, 0, 255}.
	// The source pixel is {0, 0, 136} in YCbCr-space, which is {11, 38, 0, 255} in RGB-space.
	{"ycbcr", vgradCr(), fillAlpha(255), Over, color.RGBA{11, 38, 0, 255}},
	{"ycbcrSrc", vgradCr(), fillAlpha(255), Src, color.RGBA{11, 38, 0, 255}},
	{"ycbcrAlpha", vgradCr(), fillAlpha(192), Over, color.RGBA{42, 28, 0, 255}},
	{"ycbcrAlphaSrc", vgradCr(), fillAlpha(192), Src, color.RGBA{8, 28, 0, 192}},
	{"ycbcrNil", vgradCr(), nil, Over, color.RGBA{11, 38, 0, 255}},
	{"ycbcrNilSrc", vgradCr(), nil, Src, color.RGBA{11, 38, 0, 255}},
	// Uniform mask (100%, 75%, nil) and variable Gray source.
	// At (x, y) == (8, 8):
	// The destination pixel is {136, 0, 0, 255}.
	// The source pixel is {136} in Gray-space, which is {136, 136, 136, 255} in RGBA-space.
	{"gray", vgradGray(), fillAlpha(255), Over, color.RGBA{136, 136, 136, 255}},
	{"graySrc", vgradGray(), fillAlpha(255), Src, color.RGBA{136, 136, 136, 255}},
	{"grayAlpha", vgradGray(), fillAlpha(192), Over, color.RGBA{136, 102, 102, 255}},
	{"grayAlphaSrc", vgradGray(), fillAlpha(192), Src, color.RGBA{102, 102, 102, 192}},
	{"grayNil", vgradGray(), nil, Over, color.RGBA{136, 136, 136, 255}},
	{"grayNilSrc", vgradGray(), nil, Src, color.RGBA{136, 136, 136, 255}},
	// Uniform mask (100%, 75%, nil) and variable CMYK source.
	// At (x, y) == (8, 8):
	// The destination pixel is {136, 0, 0, 255}.
	// The source pixel is {0, 136, 0, 63} in CMYK-space, which is {192, 89, 192} in RGB-space.
	{"cmyk", vgradMagenta(), fillAlpha(255), Over, color.RGBA{192, 89, 192, 255}},
	{"cmykSrc", vgradMagenta(), fillAlpha(255), Src, color.RGBA{192, 89, 192, 255}},
	{"cmykAlpha", vgradMagenta(), fillAlpha(192), Over, color.RGBA{178, 67, 145, 255}},
	{"cmykAlphaSrc", vgradMagenta(), fillAlpha(192), Src, color.RGBA{145, 67, 145, 192}},
	{"cmykNil", vgradMagenta(), nil, Over, color.RGBA{192, 89, 192, 255}},
	{"cmykNilSrc", vgradMagenta(), nil, Src, color.RGBA{192, 89, 192, 255}},
	// Variable mask and variable source.
	// At (x, y) == (8, 8):
	// The destination pixel is {136, 0, 0, 255}.
	// The source pixel is {0, 0, 255, 255}.
	// The mask pixel's alpha is 102, or 40%.
	{"generic", fillBlue(255), vgradAlpha(192), Over, color.RGBA{81, 0, 102, 255}},
	{"genericSrc", fillBlue(255), vgradAlpha(192), Src, color.RGBA{0, 0, 102, 102}},
}

func makeGolden(dst image.Image, r image.Rectangle, src image.Image, sp image.Point, mask image.Image, mp image.Point, op Op) image.Image {
	// Since golden is a newly allocated image, we don't have to check if the
	// input source and mask images and the output golden image overlap.
	b := dst.Bounds()
	sb := src.Bounds()
	mb := image.Rect(-1e9, -1e9, 1e9, 1e9)
	if mask != nil {
		mb = mask.Bounds()
	}
	golden := image.NewRGBA(image.Rect(0, 0, b.Max.X, b.Max.Y))
	for y := r.Min.Y; y < r.Max.Y; y++ {
		sy := y + sp.Y - r.Min.Y
		my := y + mp.Y - r.Min.Y
		for x := r.Min.X; x < r.Max.X; x++ {
			if !(image.Pt(x, y).In(b)) {
				continue
			}
			sx := x + sp.X - r.Min.X
			if !(image.Pt(sx, sy).In(sb)) {
				continue
			}
			mx := x + mp.X - r.Min.X
			if !(image.Pt(mx, my).In(mb)) {
				continue
			}

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
			golden.Set(x, y, color.RGBA64{
				uint16((dr*a + sr*ma) / M),
				uint16((dg*a + sg*ma) / M),
				uint16((db*a + sb*ma) / M),
				uint16((da*a + sa*ma) / M),
			})
		}
	}
	return golden.SubImage(b)
}

func TestDraw(t *testing.T) {
	rr := []image.Rectangle{
		image.Rect(0, 0, 0, 0),
		image.Rect(0, 0, 16, 16),
		image.Rect(3, 5, 12, 10),
		image.Rect(0, 0, 9, 9),
		image.Rect(8, 8, 16, 16),
		image.Rect(8, 0, 9, 16),
		image.Rect(0, 8, 16, 9),
		image.Rect(8, 8, 9, 9),
		image.Rect(8, 8, 8, 8),
	}
	for _, r := range rr {
	loop:
		for _, test := range drawTests {
			dst := hgradRed(255).(*image.RGBA).SubImage(r).(Image)
			// Draw the (src, mask, op) onto a copy of dst using a slow but obviously correct implementation.
			golden := makeGolden(dst, image.Rect(0, 0, 16, 16), test.src, image.ZP, test.mask, image.ZP, test.op)
			b := dst.Bounds()
			if !b.Eq(golden.Bounds()) {
				t.Errorf("draw %v %s: bounds %v versus %v", r, test.desc, dst.Bounds(), golden.Bounds())
				continue
			}
			// Draw the same combination onto the actual dst using the optimized DrawMask implementation.
			DrawMask(dst, image.Rect(0, 0, 16, 16), test.src, image.ZP, test.mask, image.ZP, test.op)
			if image.Pt(8, 8).In(r) {
				// Check that the resultant pixel at (8, 8) matches what we expect
				// (the expected value can be verified by hand).
				if !eq(dst.At(8, 8), test.expected) {
					t.Errorf("draw %v %s: at (8, 8) %v versus %v", r, test.desc, dst.At(8, 8), test.expected)
					continue
				}
			}
			// Check that the resultant dst image matches the golden output.
			for y := b.Min.Y; y < b.Max.Y; y++ {
				for x := b.Min.X; x < b.Max.X; x++ {
					if !eq(dst.At(x, y), golden.At(x, y)) {
						t.Errorf("draw %v %s: at (%d, %d), %v versus golden %v", r, test.desc, x, y, dst.At(x, y), golden.At(x, y))
						continue loop
					}
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
				dst := m.SubImage(image.Rect(5, 5, 10, 10)).(*image.RGBA)
				src := m.SubImage(image.Rect(5+xoff, 5+yoff, 10+xoff, 10+yoff)).(*image.RGBA)
				b := dst.Bounds()
				// Draw the (src, mask, op) onto a copy of dst using a slow but obviously correct implementation.
				golden := makeGolden(dst, b, src, src.Bounds().Min, nil, image.ZP, op)
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

// TestNonZeroSrcPt checks drawing with a non-zero src point parameter.
func TestNonZeroSrcPt(t *testing.T) {
	a := image.NewRGBA(image.Rect(0, 0, 1, 1))
	b := image.NewRGBA(image.Rect(0, 0, 2, 2))
	b.Set(0, 0, color.RGBA{0, 0, 0, 5})
	b.Set(1, 0, color.RGBA{0, 0, 5, 5})
	b.Set(0, 1, color.RGBA{0, 5, 0, 5})
	b.Set(1, 1, color.RGBA{5, 0, 0, 5})
	Draw(a, image.Rect(0, 0, 1, 1), b, image.Pt(1, 1), Over)
	if !eq(color.RGBA{5, 0, 0, 5}, a.At(0, 0)) {
		t.Errorf("non-zero src pt: want %v got %v", color.RGBA{5, 0, 0, 5}, a.At(0, 0))
	}
}

func TestFill(t *testing.T) {
	rr := []image.Rectangle{
		image.Rect(0, 0, 0, 0),
		image.Rect(0, 0, 40, 30),
		image.Rect(10, 0, 40, 30),
		image.Rect(0, 20, 40, 30),
		image.Rect(10, 20, 40, 30),
		image.Rect(10, 20, 15, 25),
		image.Rect(10, 0, 35, 30),
		image.Rect(0, 15, 40, 16),
		image.Rect(24, 24, 25, 25),
		image.Rect(23, 23, 26, 26),
		image.Rect(22, 22, 27, 27),
		image.Rect(21, 21, 28, 28),
		image.Rect(20, 20, 29, 29),
	}
	for _, r := range rr {
		m := image.NewRGBA(image.Rect(0, 0, 40, 30)).SubImage(r).(*image.RGBA)
		b := m.Bounds()
		c := color.RGBA{11, 0, 0, 255}
		src := &image.Uniform{C: c}
		check := func(desc string) {
			for y := b.Min.Y; y < b.Max.Y; y++ {
				for x := b.Min.X; x < b.Max.X; x++ {
					if !eq(c, m.At(x, y)) {
						t.Errorf("%s fill: at (%d, %d), sub-image bounds=%v: want %v got %v", desc, x, y, r, c, m.At(x, y))
						return
					}
				}
			}
		}
		// Draw 1 pixel at a time.
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				DrawMask(m, image.Rect(x, y, x+1, y+1), src, image.ZP, nil, image.ZP, Src)
			}
		}
		check("pixel")
		// Draw 1 row at a time.
		c = color.RGBA{0, 22, 0, 255}
		src = &image.Uniform{C: c}
		for y := b.Min.Y; y < b.Max.Y; y++ {
			DrawMask(m, image.Rect(b.Min.X, y, b.Max.X, y+1), src, image.ZP, nil, image.ZP, Src)
		}
		check("row")
		// Draw 1 column at a time.
		c = color.RGBA{0, 0, 33, 255}
		src = &image.Uniform{C: c}
		for x := b.Min.X; x < b.Max.X; x++ {
			DrawMask(m, image.Rect(x, b.Min.Y, x+1, b.Max.Y), src, image.ZP, nil, image.ZP, Src)
		}
		check("column")
		// Draw the whole image at once.
		c = color.RGBA{44, 55, 66, 77}
		src = &image.Uniform{C: c}
		DrawMask(m, b, src, image.ZP, nil, image.ZP, Src)
		check("whole")
	}
}

// TestFloydSteinbergCheckerboard tests that the result of Floyd-Steinberg
// error diffusion of a uniform 50% gray source image with a black-and-white
// palette is a checkerboard pattern.
func TestFloydSteinbergCheckerboard(t *testing.T) {
	b := image.Rect(0, 0, 640, 480)
	// We can't represent 50% exactly, but 0x7fff / 0xffff is close enough.
	src := &image.Uniform{color.Gray16{0x7fff}}
	dst := image.NewPaletted(b, color.Palette{color.Black, color.White})
	FloydSteinberg.Draw(dst, b, src, image.Point{})
	nErr := 0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			got := dst.Pix[dst.PixOffset(x, y)]
			want := uint8(x+y) % 2
			if got != want {
				t.Errorf("at (%d, %d): got %d, want %d", x, y, got, want)
				if nErr++; nErr == 10 {
					t.Fatal("there may be more errors")
				}
			}
		}
	}
}

// embeddedPaletted is an Image that behaves like an *image.Paletted but whose
// type is not *image.Paletted.
type embeddedPaletted struct {
	*image.Paletted
}

// TestPaletted tests that the drawPaletted function behaves the same
// regardless of whether dst is an *image.Paletted.
func TestPaletted(t *testing.T) {
	f, err := os.Open("../testdata/video-001.png")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer f.Close()
	video001, err := png.Decode(f)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	b := video001.Bounds()

	cgaPalette := color.Palette{
		color.RGBA{0x00, 0x00, 0x00, 0xff},
		color.RGBA{0x55, 0xff, 0xff, 0xff},
		color.RGBA{0xff, 0x55, 0xff, 0xff},
		color.RGBA{0xff, 0xff, 0xff, 0xff},
	}
	drawers := map[string]Drawer{
		"src":             Src,
		"floyd-steinberg": FloydSteinberg,
	}
	sources := map[string]image.Image{
		"uniform":  &image.Uniform{color.RGBA{0xff, 0x7f, 0xff, 0xff}},
		"video001": video001,
	}

	for dName, d := range drawers {
	loop:
		for sName, src := range sources {
			dst0 := image.NewPaletted(b, cgaPalette)
			dst1 := image.NewPaletted(b, cgaPalette)
			d.Draw(dst0, b, src, image.Point{})
			d.Draw(embeddedPaletted{dst1}, b, src, image.Point{})
			for y := b.Min.Y; y < b.Max.Y; y++ {
				for x := b.Min.X; x < b.Max.X; x++ {
					if !eq(dst0.At(x, y), dst1.At(x, y)) {
						t.Errorf("%s / %s: at (%d, %d), %v versus %v",
							dName, sName, x, y, dst0.At(x, y), dst1.At(x, y))
						continue loop
					}
				}
			}
		}
	}
}

func TestSqDiff(t *testing.T) {
	// This test is similar to the one from the image/color package, but
	// sqDiff in this package accepts int32 instead of uint32, so test it
	// for appropriate input.

	// canonical sqDiff implementation
	orig := func(x, y int32) uint32 {
		var d uint32
		if x > y {
			d = uint32(x - y)
		} else {
			d = uint32(y - x)
		}
		return (d * d) >> 2
	}
	testCases := []int32{
		0,
		1,
		2,
		0x0fffd,
		0x0fffe,
		0x0ffff,
		0x10000,
		0x10001,
		0x10002,
		0x7ffffffd,
		0x7ffffffe,
		0x7fffffff,
		-0x7ffffffd,
		-0x7ffffffe,
		-0x80000000,
	}
	for _, x := range testCases {
		for _, y := range testCases {
			if got, want := sqDiff(x, y), orig(x, y); got != want {
				t.Fatalf("sqDiff(%#x, %#x): got %d, want %d", x, y, got, want)
			}
		}
	}
	if err := quick.CheckEqual(orig, sqDiff, &quick.Config{MaxCountScale: 10}); err != nil {
		t.Fatal(err)
	}
}
