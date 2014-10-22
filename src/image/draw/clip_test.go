// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package draw

import (
	"image"
	"testing"
)

type clipTest struct {
	desc          string
	r, dr, sr, mr image.Rectangle
	sp, mp        image.Point
	nilMask       bool
	r0            image.Rectangle
	sp0, mp0      image.Point
}

var clipTests = []clipTest{
	// The following tests all have a nil mask.
	{
		"basic",
		image.Rect(0, 0, 100, 100),
		image.Rect(0, 0, 100, 100),
		image.Rect(0, 0, 100, 100),
		image.ZR,
		image.ZP,
		image.ZP,
		true,
		image.Rect(0, 0, 100, 100),
		image.ZP,
		image.ZP,
	},
	{
		"clip dr",
		image.Rect(0, 0, 100, 100),
		image.Rect(40, 40, 60, 60),
		image.Rect(0, 0, 100, 100),
		image.ZR,
		image.ZP,
		image.ZP,
		true,
		image.Rect(40, 40, 60, 60),
		image.Pt(40, 40),
		image.ZP,
	},
	{
		"clip sr",
		image.Rect(0, 0, 100, 100),
		image.Rect(0, 0, 100, 100),
		image.Rect(20, 20, 80, 80),
		image.ZR,
		image.ZP,
		image.ZP,
		true,
		image.Rect(20, 20, 80, 80),
		image.Pt(20, 20),
		image.ZP,
	},
	{
		"clip dr and sr",
		image.Rect(0, 0, 100, 100),
		image.Rect(0, 0, 50, 100),
		image.Rect(20, 20, 80, 80),
		image.ZR,
		image.ZP,
		image.ZP,
		true,
		image.Rect(20, 20, 50, 80),
		image.Pt(20, 20),
		image.ZP,
	},
	{
		"clip dr and sr, sp outside sr (top-left)",
		image.Rect(0, 0, 100, 100),
		image.Rect(0, 0, 50, 100),
		image.Rect(20, 20, 80, 80),
		image.ZR,
		image.Pt(15, 8),
		image.ZP,
		true,
		image.Rect(5, 12, 50, 72),
		image.Pt(20, 20),
		image.ZP,
	},
	{
		"clip dr and sr, sp outside sr (middle-left)",
		image.Rect(0, 0, 100, 100),
		image.Rect(0, 0, 50, 100),
		image.Rect(20, 20, 80, 80),
		image.ZR,
		image.Pt(15, 66),
		image.ZP,
		true,
		image.Rect(5, 0, 50, 14),
		image.Pt(20, 66),
		image.ZP,
	},
	{
		"clip dr and sr, sp outside sr (bottom-left)",
		image.Rect(0, 0, 100, 100),
		image.Rect(0, 0, 50, 100),
		image.Rect(20, 20, 80, 80),
		image.ZR,
		image.Pt(15, 91),
		image.ZP,
		true,
		image.ZR,
		image.Pt(15, 91),
		image.ZP,
	},
	{
		"clip dr and sr, sp inside sr",
		image.Rect(0, 0, 100, 100),
		image.Rect(0, 0, 50, 100),
		image.Rect(20, 20, 80, 80),
		image.ZR,
		image.Pt(44, 33),
		image.ZP,
		true,
		image.Rect(0, 0, 36, 47),
		image.Pt(44, 33),
		image.ZP,
	},

	// The following tests all have a non-nil mask.
	{
		"basic mask",
		image.Rect(0, 0, 80, 80),
		image.Rect(20, 0, 100, 80),
		image.Rect(0, 0, 50, 49),
		image.Rect(0, 0, 46, 47),
		image.ZP,
		image.ZP,
		false,
		image.Rect(20, 0, 46, 47),
		image.Pt(20, 0),
		image.Pt(20, 0),
	},
	// TODO(nigeltao): write more tests.
}

func TestClip(t *testing.T) {
	dst0 := image.NewRGBA(image.Rect(0, 0, 100, 100))
	src0 := image.NewRGBA(image.Rect(0, 0, 100, 100))
	mask0 := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for _, c := range clipTests {
		dst := dst0.SubImage(c.dr).(*image.RGBA)
		src := src0.SubImage(c.sr).(*image.RGBA)
		var mask image.Image
		if !c.nilMask {
			mask = mask0.SubImage(c.mr)
		}
		r, sp, mp := c.r, c.sp, c.mp
		clip(dst, &r, src, &sp, mask, &mp)

		// Check that the actual results equal the expected results.
		if !c.r0.Eq(r) {
			t.Errorf("%s: clip rectangle want %v got %v", c.desc, c.r0, r)
			continue
		}
		if !c.sp0.Eq(sp) {
			t.Errorf("%s: sp want %v got %v", c.desc, c.sp0, sp)
			continue
		}
		if !c.nilMask {
			if !c.mp0.Eq(mp) {
				t.Errorf("%s: mp want %v got %v", c.desc, c.mp0, mp)
				continue
			}
		}

		// Check that the clipped rectangle is contained by the dst / src / mask
		// rectangles, in their respective co-ordinate spaces.
		if !r.In(c.dr) {
			t.Errorf("%s: c.dr %v does not contain r %v", c.desc, c.dr, r)
		}
		// sr is r translated into src's co-ordinate space.
		sr := r.Add(c.sp.Sub(c.dr.Min))
		if !sr.In(c.sr) {
			t.Errorf("%s: c.sr %v does not contain sr %v", c.desc, c.sr, sr)
		}
		if !c.nilMask {
			// mr is r translated into mask's co-ordinate space.
			mr := r.Add(c.mp.Sub(c.dr.Min))
			if !mr.In(c.mr) {
				t.Errorf("%s: c.mr %v does not contain mr %v", c.desc, c.mr, mr)
			}
		}
	}
}
