// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jpeg

import (
	"bytes"
	"fmt"
	"image"
	"io/ioutil"
	"os"
	"testing"
)

// TestDecodeProgressive tests that decoding the baseline and progressive
// versions of the same image result in exactly the same pixel data, in YCbCr
// space for color images, and Y space for grayscale images.
func TestDecodeProgressive(t *testing.T) {
	testCases := []string{
		"../testdata/video-001",
		"../testdata/video-001.q50.420",
		"../testdata/video-001.q50.422",
		"../testdata/video-001.q50.440",
		"../testdata/video-001.q50.444",
		"../testdata/video-005.gray.q50",
		"../testdata/video-005.gray.q50.2x2",
	}
	for _, tc := range testCases {
		m0, err := decodeFile(tc + ".jpeg")
		if err != nil {
			t.Errorf("%s: %v", tc+".jpeg", err)
			continue
		}
		m1, err := decodeFile(tc + ".progressive.jpeg")
		if err != nil {
			t.Errorf("%s: %v", tc+".progressive.jpeg", err)
			continue
		}
		if m0.Bounds() != m1.Bounds() {
			t.Errorf("%s: bounds differ: %v and %v", tc, m0.Bounds(), m1.Bounds())
			continue
		}
		switch m0 := m0.(type) {
		case *image.YCbCr:
			m1 := m1.(*image.YCbCr)
			if err := check(m0.Bounds(), m0.Y, m1.Y, m0.YStride, m1.YStride); err != nil {
				t.Errorf("%s (Y): %v", tc, err)
				continue
			}
			if err := check(m0.Bounds(), m0.Cb, m1.Cb, m0.CStride, m1.CStride); err != nil {
				t.Errorf("%s (Cb): %v", tc, err)
				continue
			}
			if err := check(m0.Bounds(), m0.Cr, m1.Cr, m0.CStride, m1.CStride); err != nil {
				t.Errorf("%s (Cr): %v", tc, err)
				continue
			}
		case *image.Gray:
			m1 := m1.(*image.Gray)
			if err := check(m0.Bounds(), m0.Pix, m1.Pix, m0.Stride, m1.Stride); err != nil {
				t.Errorf("%s: %v", tc, err)
				continue
			}
		default:
			t.Errorf("%s: unexpected image type %T", tc, m0)
			continue
		}
	}
}

func decodeFile(filename string) (image.Image, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Decode(f)

}

// check checks that the two pix data are equal, within the given bounds.
func check(bounds image.Rectangle, pix0, pix1 []byte, stride0, stride1 int) error {
	if len(pix0) != len(pix1) {
		return fmt.Errorf("len(pix) %d and %d differ", len(pix0), len(pix1))
	}
	if stride0 != stride1 {
		return fmt.Errorf("strides %d and %d differ", stride0, stride1)
	}
	if stride0%8 != 0 {
		return fmt.Errorf("stride %d is not a multiple of 8", stride0)
	}
	// Compare the two pix data, one 8x8 block at a time.
	for y := 0; y < len(pix0)/stride0; y += 8 {
		for x := 0; x < stride0; x += 8 {
			if x >= bounds.Max.X || y >= bounds.Max.Y {
				// We don't care if the two pix data differ if the 8x8 block is
				// entirely outside of the image's bounds. For example, this can
				// occur with a 4:2:0 chroma subsampling and a 1x1 image. Baseline
				// decoding works on the one 16x16 MCU as a whole; progressive
				// decoding's first pass works on that 16x16 MCU as a whole but
				// refinement passes only process one 8x8 block within the MCU.
				continue
			}

			for j := 0; j < 8; j++ {
				for i := 0; i < 8; i++ {
					index := (y+j)*stride0 + (x + i)
					if pix0[index] != pix1[index] {
						return fmt.Errorf("blocks at (%d, %d) differ:\n%sand\n%s", x, y,
							pixString(pix0, stride0, x, y),
							pixString(pix1, stride1, x, y),
						)
					}
				}
			}
		}
	}
	return nil
}

func pixString(pix []byte, stride, x, y int) string {
	s := bytes.NewBuffer(nil)
	for j := 0; j < 8; j++ {
		fmt.Fprintf(s, "\t")
		for i := 0; i < 8; i++ {
			fmt.Fprintf(s, "%02x ", pix[(y+j)*stride+(x+i)])
		}
		fmt.Fprintf(s, "\n")
	}
	return s.String()
}

func benchmarkDecode(b *testing.B, filename string) {
	b.StopTimer()
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		b.Fatal(err)
	}
	cfg, err := DecodeConfig(bytes.NewReader(data))
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(cfg.Width * cfg.Height * 4))
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Decode(bytes.NewReader(data))
	}
}

func BenchmarkDecodeBaseline(b *testing.B) {
	benchmarkDecode(b, "../testdata/video-001.jpeg")
}

func BenchmarkDecodeProgressive(b *testing.B) {
	benchmarkDecode(b, "../testdata/video-001.progressive.jpeg")
}
