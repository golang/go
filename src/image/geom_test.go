// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package image

import (
	"fmt"
	"testing"
)

func TestRectangle(t *testing.T) {
	// in checks that every point in f is in g.
	in := func(f, g Rectangle) error {
		if !f.In(g) {
			return fmt.Errorf("f=%s, f.In(%s): got false, want true", f, g)
		}
		for y := f.Min.Y; y < f.Max.Y; y++ {
			for x := f.Min.X; x < f.Max.X; x++ {
				p := Point{x, y}
				if !p.In(g) {
					return fmt.Errorf("p=%s, p.In(%s): got false, want true", p, g)
				}
			}
		}
		return nil
	}

	rects := []Rectangle{
		Rect(0, 0, 10, 10),
		Rect(10, 0, 20, 10),
		Rect(1, 2, 3, 4),
		Rect(4, 6, 10, 10),
		Rect(2, 3, 12, 5),
		Rect(-1, -2, 0, 0),
		Rect(-1, -2, 4, 6),
		Rect(-10, -20, 30, 40),
		Rect(8, 8, 8, 8),
		Rect(88, 88, 88, 88),
		Rect(6, 5, 4, 3),
	}

	// r.Eq(s) should be equivalent to every point in r being in s, and every
	// point in s being in r.
	for _, r := range rects {
		for _, s := range rects {
			got := r.Eq(s)
			want := in(r, s) == nil && in(s, r) == nil
			if got != want {
				t.Errorf("Eq: r=%s, s=%s: got %t, want %t", r, s, got, want)
			}
		}
	}

	// The intersection should be the largest rectangle a such that every point
	// in a is both in r and in s.
	for _, r := range rects {
		for _, s := range rects {
			a := r.Intersect(s)
			if err := in(a, r); err != nil {
				t.Errorf("Intersect: r=%s, s=%s, a=%s, a not in r: %v", r, s, a, err)
			}
			if err := in(a, s); err != nil {
				t.Errorf("Intersect: r=%s, s=%s, a=%s, a not in s: %v", r, s, a, err)
			}
			if isZero, overlaps := a == (Rectangle{}), r.Overlaps(s); isZero == overlaps {
				t.Errorf("Intersect: r=%s, s=%s, a=%s: isZero=%t same as overlaps=%t",
					r, s, a, isZero, overlaps)
			}
			largerThanA := [4]Rectangle{a, a, a, a}
			largerThanA[0].Min.X--
			largerThanA[1].Min.Y--
			largerThanA[2].Max.X++
			largerThanA[3].Max.Y++
			for i, b := range largerThanA {
				if b.Empty() {
					// b isn't actually larger than a.
					continue
				}
				if in(b, r) == nil && in(b, s) == nil {
					t.Errorf("Intersect: r=%s, s=%s, a=%s, b=%s, i=%d: intersection could be larger",
						r, s, a, b, i)
				}
			}
		}
	}

	// The union should be the smallest rectangle a such that every point in r
	// is in a and every point in s is in a.
	for _, r := range rects {
		for _, s := range rects {
			a := r.Union(s)
			if err := in(r, a); err != nil {
				t.Errorf("Union: r=%s, s=%s, a=%s, r not in a: %v", r, s, a, err)
			}
			if err := in(s, a); err != nil {
				t.Errorf("Union: r=%s, s=%s, a=%s, s not in a: %v", r, s, a, err)
			}
			if a.Empty() {
				// You can't get any smaller than a.
				continue
			}
			smallerThanA := [4]Rectangle{a, a, a, a}
			smallerThanA[0].Min.X++
			smallerThanA[1].Min.Y++
			smallerThanA[2].Max.X--
			smallerThanA[3].Max.Y--
			for i, b := range smallerThanA {
				if in(r, b) == nil && in(s, b) == nil {
					t.Errorf("Union: r=%s, s=%s, a=%s, b=%s, i=%d: union could be smaller",
						r, s, a, b, i)
				}
			}
		}
	}
}
