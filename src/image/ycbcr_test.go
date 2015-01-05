// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package image

import (
	"image/color"
	"testing"
)

func TestYCbCr(t *testing.T) {
	rects := []Rectangle{
		Rect(0, 0, 16, 16),
		Rect(1, 0, 16, 16),
		Rect(0, 1, 16, 16),
		Rect(1, 1, 16, 16),
		Rect(1, 1, 15, 16),
		Rect(1, 1, 16, 15),
		Rect(1, 1, 15, 15),
		Rect(2, 3, 14, 15),
		Rect(7, 0, 7, 16),
		Rect(0, 8, 16, 8),
		Rect(0, 0, 10, 11),
		Rect(5, 6, 16, 16),
		Rect(7, 7, 8, 8),
		Rect(7, 8, 8, 9),
		Rect(8, 7, 9, 8),
		Rect(8, 8, 9, 9),
		Rect(7, 7, 17, 17),
		Rect(8, 8, 17, 17),
		Rect(9, 9, 17, 17),
		Rect(10, 10, 17, 17),
	}
	subsampleRatios := []YCbCrSubsampleRatio{
		YCbCrSubsampleRatio444,
		YCbCrSubsampleRatio422,
		YCbCrSubsampleRatio420,
		YCbCrSubsampleRatio440,
	}
	deltas := []Point{
		Pt(0, 0),
		Pt(1000, 1001),
		Pt(5001, -400),
		Pt(-701, -801),
	}
	for _, r := range rects {
		for _, subsampleRatio := range subsampleRatios {
			for _, delta := range deltas {
				testYCbCr(t, r, subsampleRatio, delta)
			}
		}
		if testing.Short() {
			break
		}
	}
}

func testYCbCr(t *testing.T, r Rectangle, subsampleRatio YCbCrSubsampleRatio, delta Point) {
	// Create a YCbCr image m, whose bounds are r translated by (delta.X, delta.Y).
	r1 := r.Add(delta)
	m := NewYCbCr(r1, subsampleRatio)

	// Test that the image buffer is reasonably small even if (delta.X, delta.Y) is far from the origin.
	if len(m.Y) > 100*100 {
		t.Errorf("r=%v, subsampleRatio=%v, delta=%v: image buffer is too large",
			r, subsampleRatio, delta)
		return
	}

	// Initialize m's pixels. For 422 and 420 subsampling, some of the Cb and Cr elements
	// will be set multiple times. That's OK. We just want to avoid a uniform image.
	for y := r1.Min.Y; y < r1.Max.Y; y++ {
		for x := r1.Min.X; x < r1.Max.X; x++ {
			yi := m.YOffset(x, y)
			ci := m.COffset(x, y)
			m.Y[yi] = uint8(16*y + x)
			m.Cb[ci] = uint8(y + 16*x)
			m.Cr[ci] = uint8(y + 16*x)
		}
	}

	// Make various sub-images of m.
	for y0 := delta.Y + 3; y0 < delta.Y+7; y0++ {
		for y1 := delta.Y + 8; y1 < delta.Y+13; y1++ {
			for x0 := delta.X + 3; x0 < delta.X+7; x0++ {
				for x1 := delta.X + 8; x1 < delta.X+13; x1++ {
					subRect := Rect(x0, y0, x1, y1)
					sub := m.SubImage(subRect).(*YCbCr)

					// For each point in the sub-image's bounds, check that m.At(x, y) equals sub.At(x, y).
					for y := sub.Rect.Min.Y; y < sub.Rect.Max.Y; y++ {
						for x := sub.Rect.Min.X; x < sub.Rect.Max.X; x++ {
							color0 := m.At(x, y).(color.YCbCr)
							color1 := sub.At(x, y).(color.YCbCr)
							if color0 != color1 {
								t.Errorf("r=%v, subsampleRatio=%v, delta=%v, x=%d, y=%d, color0=%v, color1=%v",
									r, subsampleRatio, delta, x, y, color0, color1)
								return
							}
						}
					}
				}
			}
		}
	}
}

func TestYCbCrSlicesDontOverlap(t *testing.T) {
	m := NewYCbCr(Rect(0, 0, 8, 8), YCbCrSubsampleRatio420)
	names := []string{"Y", "Cb", "Cr"}
	slices := [][]byte{
		m.Y[:cap(m.Y)],
		m.Cb[:cap(m.Cb)],
		m.Cr[:cap(m.Cr)],
	}
	for i, slice := range slices {
		want := uint8(10 + i)
		for j := range slice {
			slice[j] = want
		}
	}
	for i, slice := range slices {
		want := uint8(10 + i)
		for j, got := range slice {
			if got != want {
				t.Fatalf("m.%s[%d]: got %d, want %d", names[i], j, got, want)
			}
		}
	}
}
