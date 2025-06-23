// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jpeg

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"math/rand"
	"os"
	"strings"
	"testing"
)

// zigzag maps from the natural ordering to the zig-zag ordering. For example,
// zigzag[0*8 + 3] is the zig-zag sequence number of the element in the fourth
// column and first row.
var zigzag = [blockSize]int{
	0, 1, 5, 6, 14, 15, 27, 28,
	2, 4, 7, 13, 16, 26, 29, 42,
	3, 8, 12, 17, 25, 30, 41, 43,
	9, 11, 18, 24, 31, 40, 44, 53,
	10, 19, 23, 32, 39, 45, 52, 54,
	20, 22, 33, 38, 46, 51, 55, 60,
	21, 34, 37, 47, 50, 56, 59, 61,
	35, 36, 48, 49, 57, 58, 62, 63,
}

func TestZigUnzig(t *testing.T) {
	for i := 0; i < blockSize; i++ {
		if unzig[zigzag[i]] != i {
			t.Errorf("unzig[zigzag[%d]] == %d", i, unzig[zigzag[i]])
		}
		if zigzag[unzig[i]] != i {
			t.Errorf("zigzag[unzig[%d]] == %d", i, zigzag[unzig[i]])
		}
	}
}

// unscaledQuantInNaturalOrder are the unscaled quantization tables in
// natural (not zig-zag) order, as specified in section K.1.
var unscaledQuantInNaturalOrder = [nQuantIndex][blockSize]byte{
	// Luminance.
	{
		16, 11, 10, 16, 24, 40, 51, 61,
		12, 12, 14, 19, 26, 58, 60, 55,
		14, 13, 16, 24, 40, 57, 69, 56,
		14, 17, 22, 29, 51, 87, 80, 62,
		18, 22, 37, 56, 68, 109, 103, 77,
		24, 35, 55, 64, 81, 104, 113, 92,
		49, 64, 78, 87, 103, 121, 120, 101,
		72, 92, 95, 98, 112, 100, 103, 99,
	},
	// Chrominance.
	{
		17, 18, 24, 47, 99, 99, 99, 99,
		18, 21, 26, 66, 99, 99, 99, 99,
		24, 26, 56, 99, 99, 99, 99, 99,
		47, 66, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
	},
}

func TestUnscaledQuant(t *testing.T) {
	bad := false
	for i := quantIndex(0); i < nQuantIndex; i++ {
		for zig := 0; zig < blockSize; zig++ {
			got := unscaledQuant[i][zig]
			want := unscaledQuantInNaturalOrder[i][unzig[zig]]
			if got != want {
				t.Errorf("i=%d, zig=%d: got %d, want %d", i, zig, got, want)
				bad = true
			}
		}
	}
	if bad {
		names := [nQuantIndex]string{"Luminance", "Chrominance"}
		buf := &strings.Builder{}
		for i, name := range names {
			fmt.Fprintf(buf, "// %s.\n{\n", name)
			for zig := 0; zig < blockSize; zig++ {
				fmt.Fprintf(buf, "%d, ", unscaledQuantInNaturalOrder[i][unzig[zig]])
				if zig%8 == 7 {
					buf.WriteString("\n")
				}
			}
			buf.WriteString("},\n")
		}
		t.Logf("expected unscaledQuant values:\n%s", buf.String())
	}
}

var testCase = []struct {
	filename  string
	quality   int
	tolerance int64
}{
	{"../testdata/video-001.png", 1, 24 << 8},
	{"../testdata/video-001.png", 20, 12 << 8},
	{"../testdata/video-001.png", 60, 8 << 8},
	{"../testdata/video-001.png", 80, 6 << 8},
	{"../testdata/video-001.png", 90, 4 << 8},
	{"../testdata/video-001.png", 100, 2 << 8},
}

func delta(u0, u1 uint32) int64 {
	d := int64(u0) - int64(u1)
	if d < 0 {
		return -d
	}
	return d
}

func readPng(filename string) (image.Image, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return png.Decode(f)
}

func TestWriter(t *testing.T) {
	for _, tc := range testCase {
		// Read the image.
		m0, err := readPng(tc.filename)
		if err != nil {
			t.Error(tc.filename, err)
			continue
		}
		// Encode that image as JPEG.
		var buf bytes.Buffer
		err = Encode(&buf, m0, &Options{Quality: tc.quality})
		if err != nil {
			t.Error(tc.filename, err)
			continue
		}
		// Decode that JPEG.
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
		if averageDelta(m0, m1) > tc.tolerance {
			t.Errorf("%s, quality=%d: average delta is too high", tc.filename, tc.quality)
			continue
		}
	}
}

// TestWriteGrayscale tests that a grayscale images survives a round-trip
// through encode/decode cycle.
func TestWriteGrayscale(t *testing.T) {
	m0 := image.NewGray(image.Rect(0, 0, 32, 32))
	for i := range m0.Pix {
		m0.Pix[i] = uint8(i)
	}
	var buf bytes.Buffer
	if err := Encode(&buf, m0, nil); err != nil {
		t.Fatal(err)
	}
	m1, err := Decode(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if m0.Bounds() != m1.Bounds() {
		t.Fatalf("bounds differ: %v and %v", m0.Bounds(), m1.Bounds())
	}
	if _, ok := m1.(*image.Gray); !ok {
		t.Errorf("got %T, want *image.Gray", m1)
	}
	// Compare the average delta to the tolerance level.
	want := int64(2 << 8)
	if got := averageDelta(m0, m1); got > want {
		t.Errorf("average delta too high; got %d, want <= %d", got, want)
	}
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

func TestEncodeYCbCr(t *testing.T) {
	bo := image.Rect(0, 0, 640, 480)
	imgRGBA := image.NewRGBA(bo)
	// Must use 444 subsampling to avoid lossy RGBA to YCbCr conversion.
	imgYCbCr := image.NewYCbCr(bo, image.YCbCrSubsampleRatio444)
	rnd := rand.New(rand.NewSource(123))
	// Create identical rgba and ycbcr images.
	for y := bo.Min.Y; y < bo.Max.Y; y++ {
		for x := bo.Min.X; x < bo.Max.X; x++ {
			col := color.RGBA{
				uint8(rnd.Intn(256)),
				uint8(rnd.Intn(256)),
				uint8(rnd.Intn(256)),
				255,
			}
			imgRGBA.SetRGBA(x, y, col)
			yo := imgYCbCr.YOffset(x, y)
			co := imgYCbCr.COffset(x, y)
			cy, ccr, ccb := color.RGBToYCbCr(col.R, col.G, col.B)
			imgYCbCr.Y[yo] = cy
			imgYCbCr.Cb[co] = ccr
			imgYCbCr.Cr[co] = ccb
		}
	}

	// Now check that both images are identical after an encode.
	var bufRGBA, bufYCbCr bytes.Buffer
	Encode(&bufRGBA, imgRGBA, nil)
	Encode(&bufYCbCr, imgYCbCr, nil)
	if !bytes.Equal(bufRGBA.Bytes(), bufYCbCr.Bytes()) {
		t.Errorf("RGBA and YCbCr encoded bytes differ")
	}
}

func BenchmarkEncodeRGBA(b *testing.B) {
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
	b.ReportAllocs()
	b.ResetTimer()
	options := &Options{Quality: 90}
	for i := 0; i < b.N; i++ {
		Encode(io.Discard, img, options)
	}
}

func BenchmarkEncodeYCbCr(b *testing.B) {
	img := image.NewYCbCr(image.Rect(0, 0, 640, 480), image.YCbCrSubsampleRatio420)
	bo := img.Bounds()
	rnd := rand.New(rand.NewSource(123))
	for y := bo.Min.Y; y < bo.Max.Y; y++ {
		for x := bo.Min.X; x < bo.Max.X; x++ {
			cy := img.YOffset(x, y)
			ci := img.COffset(x, y)
			img.Y[cy] = uint8(rnd.Intn(256))
			img.Cb[ci] = uint8(rnd.Intn(256))
			img.Cr[ci] = uint8(rnd.Intn(256))
		}
	}
	b.SetBytes(640 * 480 * 3)
	b.ReportAllocs()
	b.ResetTimer()
	options := &Options{Quality: 90}
	for i := 0; i < b.N; i++ {
		Encode(io.Discard, img, options)
	}
}
