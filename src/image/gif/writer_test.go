// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gif

import (
	"bytes"
	"image"
	"image/color"
	"image/color/palette"
	_ "image/png"
	"io/ioutil"
	"math/rand"
	"os"
	"reflect"
	"testing"
)

func readImg(filename string) (image.Image, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	m, _, err := image.Decode(f)
	return m, err
}

func readGIF(filename string) (*GIF, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return DecodeAll(f)
}

func delta(u0, u1 uint32) int64 {
	d := int64(u0) - int64(u1)
	if d < 0 {
		return -d
	}
	return d
}

// averageDelta returns the average delta in RGB space. The two images must
// have the same bounds.
func averageDelta(m0, m1 image.Image) int64 {
	b := m0.Bounds()
	var sum, n int64
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c0 := m0.At(x, y)
			c1 := m1.At(x, y)
			r0, g0, b0, _ := c0.RGBA()
			r1, g1, b1, _ := c1.RGBA()
			sum += delta(r0, r1)
			sum += delta(g0, g1)
			sum += delta(b0, b1)
			n += 3
		}
	}
	return sum / n
}

var testCase = []struct {
	filename  string
	tolerance int64
}{
	{"../testdata/video-001.png", 1 << 12},
	{"../testdata/video-001.gif", 0},
	{"../testdata/video-001.interlaced.gif", 0},
}

func TestWriter(t *testing.T) {
	for _, tc := range testCase {
		m0, err := readImg(tc.filename)
		if err != nil {
			t.Error(tc.filename, err)
			continue
		}
		var buf bytes.Buffer
		err = Encode(&buf, m0, nil)
		if err != nil {
			t.Error(tc.filename, err)
			continue
		}
		m1, err := Decode(&buf)
		if err != nil {
			t.Error(tc.filename, err)
			continue
		}
		if m0.Bounds() != m1.Bounds() {
			t.Errorf("%s, bounds differ: %v and %v", tc.filename, m0.Bounds(), m1.Bounds())
			continue
		}
		// Compare the average delta to the tolerance level.
		avgDelta := averageDelta(m0, m1)
		if avgDelta > tc.tolerance {
			t.Errorf("%s: average delta is too high. expected: %d, got %d", tc.filename, tc.tolerance, avgDelta)
			continue
		}
	}
}

func TestSubImage(t *testing.T) {
	m0, err := readImg("../testdata/video-001.gif")
	if err != nil {
		t.Fatalf("readImg: %v", err)
	}
	m0 = m0.(*image.Paletted).SubImage(image.Rect(0, 0, 50, 30))
	var buf bytes.Buffer
	err = Encode(&buf, m0, nil)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	m1, err := Decode(&buf)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if m0.Bounds() != m1.Bounds() {
		t.Fatalf("bounds differ: %v and %v", m0.Bounds(), m1.Bounds())
	}
	if averageDelta(m0, m1) != 0 {
		t.Fatalf("images differ")
	}
}

// palettesEqual reports whether two color.Palette values are equal, ignoring
// any trailing opaque-black palette entries.
func palettesEqual(p, q color.Palette) bool {
	n := len(p)
	if n > len(q) {
		n = len(q)
	}
	for i := 0; i < n; i++ {
		if p[i] != q[i] {
			return false
		}
	}
	for i := n; i < len(p); i++ {
		r, g, b, a := p[i].RGBA()
		if r != 0 || g != 0 || b != 0 || a != 0xffff {
			return false
		}
	}
	for i := n; i < len(q); i++ {
		r, g, b, a := q[i].RGBA()
		if r != 0 || g != 0 || b != 0 || a != 0xffff {
			return false
		}
	}
	return true
}

var frames = []string{
	"../testdata/video-001.gif",
	"../testdata/video-005.gray.gif",
}

func testEncodeAll(t *testing.T, go1Dot5Fields bool, useGlobalColorModel bool) {
	const width, height = 150, 103

	g0 := &GIF{
		Image:     make([]*image.Paletted, len(frames)),
		Delay:     make([]int, len(frames)),
		LoopCount: 5,
	}
	for i, f := range frames {
		g, err := readGIF(f)
		if err != nil {
			t.Fatal(f, err)
		}
		m := g.Image[0]
		if m.Bounds().Dx() != width || m.Bounds().Dy() != height {
			t.Fatalf("frame %d had unexpected bounds: got %v, want width/height = %d/%d",
				i, m.Bounds(), width, height)
		}
		g0.Image[i] = m
	}
	// The GIF.Disposal, GIF.Config and GIF.BackgroundIndex fields were added
	// in Go 1.5. Valid Go 1.4 or earlier code should still produce valid GIFs.
	//
	// On the following line, color.Model is an interface type, and
	// color.Palette is a concrete (slice) type.
	globalColorModel, backgroundIndex := color.Model(color.Palette(nil)), uint8(0)
	if useGlobalColorModel {
		globalColorModel, backgroundIndex = color.Palette(palette.WebSafe), uint8(1)
	}
	if go1Dot5Fields {
		g0.Disposal = make([]byte, len(g0.Image))
		for i := range g0.Disposal {
			g0.Disposal[i] = DisposalNone
		}
		g0.Config = image.Config{
			ColorModel: globalColorModel,
			Width:      width,
			Height:     height,
		}
		g0.BackgroundIndex = backgroundIndex
	}

	var buf bytes.Buffer
	if err := EncodeAll(&buf, g0); err != nil {
		t.Fatal("EncodeAll:", err)
	}
	encoded := buf.Bytes()
	config, err := DecodeConfig(bytes.NewReader(encoded))
	if err != nil {
		t.Fatal("DecodeConfig:", err)
	}
	g1, err := DecodeAll(bytes.NewReader(encoded))
	if err != nil {
		t.Fatal("DecodeAll:", err)
	}

	if !reflect.DeepEqual(config, g1.Config) {
		t.Errorf("DecodeConfig inconsistent with DecodeAll")
	}
	if !palettesEqual(g1.Config.ColorModel.(color.Palette), globalColorModel.(color.Palette)) {
		t.Errorf("unexpected global color model")
	}
	if w, h := g1.Config.Width, g1.Config.Height; w != width || h != height {
		t.Errorf("got config width * height = %d * %d, want %d * %d", w, h, width, height)
	}

	if g0.LoopCount != g1.LoopCount {
		t.Errorf("loop counts differ: %d and %d", g0.LoopCount, g1.LoopCount)
	}
	if backgroundIndex != g1.BackgroundIndex {
		t.Errorf("background indexes differ: %d and %d", backgroundIndex, g1.BackgroundIndex)
	}
	if len(g0.Image) != len(g1.Image) {
		t.Fatalf("image lengths differ: %d and %d", len(g0.Image), len(g1.Image))
	}
	if len(g1.Image) != len(g1.Delay) {
		t.Fatalf("image and delay lengths differ: %d and %d", len(g1.Image), len(g1.Delay))
	}
	if len(g1.Image) != len(g1.Disposal) {
		t.Fatalf("image and disposal lengths differ: %d and %d", len(g1.Image), len(g1.Disposal))
	}

	for i := range g0.Image {
		m0, m1 := g0.Image[i], g1.Image[i]
		if m0.Bounds() != m1.Bounds() {
			t.Errorf("frame %d: bounds differ: %v and %v", i, m0.Bounds(), m1.Bounds())
		}
		d0, d1 := g0.Delay[i], g1.Delay[i]
		if d0 != d1 {
			t.Errorf("frame %d: delay values differ: %d and %d", i, d0, d1)
		}
		p0, p1 := uint8(0), g1.Disposal[i]
		if go1Dot5Fields {
			p0 = DisposalNone
		}
		if p0 != p1 {
			t.Errorf("frame %d: disposal values differ: %d and %d", i, p0, p1)
		}
	}
}

func TestEncodeAllGo1Dot4(t *testing.T)                 { testEncodeAll(t, false, false) }
func TestEncodeAllGo1Dot5(t *testing.T)                 { testEncodeAll(t, true, false) }
func TestEncodeAllGo1Dot5GlobalColorModel(t *testing.T) { testEncodeAll(t, true, true) }

func TestEncodeMismatchDelay(t *testing.T) {
	images := make([]*image.Paletted, 2)
	for i := range images {
		images[i] = image.NewPaletted(image.Rect(0, 0, 5, 5), palette.Plan9)
	}

	g0 := &GIF{
		Image: images,
		Delay: make([]int, 1),
	}
	if err := EncodeAll(ioutil.Discard, g0); err == nil {
		t.Error("expected error from mismatched delay and image slice lengths")
	}

	g1 := &GIF{
		Image:    images,
		Delay:    make([]int, len(images)),
		Disposal: make([]byte, 1),
	}
	for i := range g1.Disposal {
		g1.Disposal[i] = DisposalNone
	}
	if err := EncodeAll(ioutil.Discard, g1); err == nil {
		t.Error("expected error from mismatched disposal and image slice lengths")
	}
}

func TestEncodeZeroGIF(t *testing.T) {
	if err := EncodeAll(ioutil.Discard, &GIF{}); err == nil {
		t.Error("expected error from providing empty gif")
	}
}

func TestEncodeAllFramesOutOfBounds(t *testing.T) {
	images := []*image.Paletted{
		image.NewPaletted(image.Rect(0, 0, 5, 5), palette.Plan9),
		image.NewPaletted(image.Rect(2, 2, 8, 8), palette.Plan9),
		image.NewPaletted(image.Rect(3, 3, 4, 4), palette.Plan9),
	}
	for _, upperBound := range []int{6, 10} {
		g := &GIF{
			Image:    images,
			Delay:    make([]int, len(images)),
			Disposal: make([]byte, len(images)),
			Config: image.Config{
				Width:  upperBound,
				Height: upperBound,
			},
		}
		err := EncodeAll(ioutil.Discard, g)
		if upperBound >= 8 {
			if err != nil {
				t.Errorf("upperBound=%d: %v", upperBound, err)
			}
		} else {
			if err == nil {
				t.Errorf("upperBound=%d: got nil error, want non-nil", upperBound)
			}
		}
	}
}

func TestEncodeNonZeroMinPoint(t *testing.T) {
	points := []image.Point{
		{-8, -9},
		{-4, -4},
		{-3, +3},
		{+0, +0},
		{+2, +2},
	}
	for _, p := range points {
		src := image.NewPaletted(image.Rectangle{Min: p, Max: p.Add(image.Point{6, 6})}, palette.Plan9)
		var buf bytes.Buffer
		if err := Encode(&buf, src, nil); err != nil {
			t.Errorf("p=%v: Encode: %v", p, err)
			continue
		}
		m, err := Decode(&buf)
		if err != nil {
			t.Errorf("p=%v: Decode: %v", p, err)
			continue
		}
		if got, want := m.Bounds(), image.Rect(0, 0, 6, 6); got != want {
			t.Errorf("p=%v: got %v, want %v", p, got, want)
		}
	}
}

func TestEncodeImplicitConfigSize(t *testing.T) {
	// For backwards compatibility for Go 1.4 and earlier code, the Config
	// field is optional, and if zero, the width and height is implied by the
	// first (and in this case only) frame's width and height.
	//
	// A Config only specifies a width and height (two integers) while an
	// image.Image's Bounds method returns an image.Rectangle (four integers).
	// For a gif.GIF, the overall bounds' top-left point is always implicitly
	// (0, 0), and any frame whose bounds have a negative X or Y will be
	// outside those overall bounds, so encoding should fail.
	for _, lowerBound := range []int{-1, 0, 1} {
		images := []*image.Paletted{
			image.NewPaletted(image.Rect(lowerBound, lowerBound, 4, 4), palette.Plan9),
		}
		g := &GIF{
			Image: images,
			Delay: make([]int, len(images)),
		}
		err := EncodeAll(ioutil.Discard, g)
		if lowerBound >= 0 {
			if err != nil {
				t.Errorf("lowerBound=%d: %v", lowerBound, err)
			}
		} else {
			if err == nil {
				t.Errorf("lowerBound=%d: got nil error, want non-nil", lowerBound)
			}
		}
	}
}

func TestEncodePalettes(t *testing.T) {
	const w, h = 5, 5
	pals := []color.Palette{{
		color.RGBA{0x00, 0x00, 0x00, 0xff},
		color.RGBA{0x01, 0x00, 0x00, 0xff},
		color.RGBA{0x02, 0x00, 0x00, 0xff},
	}, {
		color.RGBA{0x00, 0x00, 0x00, 0xff},
		color.RGBA{0x00, 0x01, 0x00, 0xff},
	}, {
		color.RGBA{0x00, 0x00, 0x03, 0xff},
		color.RGBA{0x00, 0x00, 0x02, 0xff},
		color.RGBA{0x00, 0x00, 0x01, 0xff},
		color.RGBA{0x00, 0x00, 0x00, 0xff},
	}, {
		color.RGBA{0x10, 0x07, 0xf0, 0xff},
		color.RGBA{0x20, 0x07, 0xf0, 0xff},
		color.RGBA{0x30, 0x07, 0xf0, 0xff},
		color.RGBA{0x40, 0x07, 0xf0, 0xff},
		color.RGBA{0x50, 0x07, 0xf0, 0xff},
	}}
	g0 := &GIF{
		Image: []*image.Paletted{
			image.NewPaletted(image.Rect(0, 0, w, h), pals[0]),
			image.NewPaletted(image.Rect(0, 0, w, h), pals[1]),
			image.NewPaletted(image.Rect(0, 0, w, h), pals[2]),
			image.NewPaletted(image.Rect(0, 0, w, h), pals[3]),
		},
		Delay:    make([]int, len(pals)),
		Disposal: make([]byte, len(pals)),
		Config: image.Config{
			ColorModel: pals[2],
			Width:      w,
			Height:     h,
		},
	}

	var buf bytes.Buffer
	if err := EncodeAll(&buf, g0); err != nil {
		t.Fatalf("EncodeAll: %v", err)
	}
	g1, err := DecodeAll(&buf)
	if err != nil {
		t.Fatalf("DecodeAll: %v", err)
	}
	if len(g0.Image) != len(g1.Image) {
		t.Fatalf("image lengths differ: %d and %d", len(g0.Image), len(g1.Image))
	}
	for i, m := range g1.Image {
		if got, want := m.Palette, pals[i]; !palettesEqual(got, want) {
			t.Errorf("frame %d:\ngot  %v\nwant %v", i, got, want)
		}
	}
}

func TestEncodeBadPalettes(t *testing.T) {
	const w, h = 5, 5
	for _, n := range []int{256, 257} {
		for _, nilColors := range []bool{false, true} {
			pal := make(color.Palette, n)
			if !nilColors {
				for i := range pal {
					pal[i] = color.Black
				}
			}

			err := EncodeAll(ioutil.Discard, &GIF{
				Image: []*image.Paletted{
					image.NewPaletted(image.Rect(0, 0, w, h), pal),
				},
				Delay:    make([]int, 1),
				Disposal: make([]byte, 1),
				Config: image.Config{
					ColorModel: pal,
					Width:      w,
					Height:     h,
				},
			})

			got := err != nil
			want := n > 256 || nilColors
			if got != want {
				t.Errorf("n=%d, nilColors=%t: err != nil: got %t, want %t", n, nilColors, got, want)
			}
		}
	}
}

func TestEncodeCroppedSubImages(t *testing.T) {
	// This test means to ensure that Encode honors the Bounds and Strides of
	// images correctly when encoding.
	whole := image.NewPaletted(image.Rect(0, 0, 100, 100), palette.Plan9)
	subImages := []image.Rectangle{
		image.Rect(0, 0, 50, 50),
		image.Rect(50, 0, 100, 50),
		image.Rect(0, 50, 50, 50),
		image.Rect(50, 50, 100, 100),
		image.Rect(25, 25, 75, 75),
		image.Rect(0, 0, 100, 50),
		image.Rect(0, 50, 100, 100),
		image.Rect(0, 0, 50, 100),
		image.Rect(50, 0, 100, 100),
	}
	for _, sr := range subImages {
		si := whole.SubImage(sr)
		buf := bytes.NewBuffer(nil)
		if err := Encode(buf, si, nil); err != nil {
			t.Errorf("Encode: sr=%v: %v", sr, err)
			continue
		}
		if _, err := Decode(buf); err != nil {
			t.Errorf("Decode: sr=%v: %v", sr, err)
		}
	}
}

func BenchmarkEncode(b *testing.B) {
	b.StopTimer()

	bo := image.Rect(0, 0, 640, 480)
	rnd := rand.New(rand.NewSource(123))

	// Restrict to a 256-color paletted image to avoid quantization path.
	palette := make(color.Palette, 256)
	for i := range palette {
		palette[i] = color.RGBA{
			uint8(rnd.Intn(256)),
			uint8(rnd.Intn(256)),
			uint8(rnd.Intn(256)),
			255,
		}
	}
	img := image.NewPaletted(image.Rect(0, 0, 640, 480), palette)
	for y := bo.Min.Y; y < bo.Max.Y; y++ {
		for x := bo.Min.X; x < bo.Max.X; x++ {
			img.Set(x, y, palette[rnd.Intn(256)])
		}
	}

	b.SetBytes(640 * 480 * 4)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Encode(ioutil.Discard, img, nil)
	}
}

func BenchmarkQuantizedEncode(b *testing.B) {
	b.StopTimer()
	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	bo := img.Bounds()
	rnd := rand.New(rand.NewSource(123))
	for y := bo.Min.Y; y < bo.Max.Y; y++ {
		for x := bo.Min.X; x < bo.Max.X; x++ {
			img.SetRGBA(x, y, color.RGBA{
				uint8(rnd.Intn(256)),
				uint8(rnd.Intn(256)),
				uint8(rnd.Intn(256)),
				255,
			})
		}
	}
	b.SetBytes(640 * 480 * 4)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Encode(ioutil.Discard, img, nil)
	}
}
