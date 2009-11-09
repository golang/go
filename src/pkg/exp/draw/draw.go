// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package draw provides basic graphics and drawing primitives,
// in the style of the Plan 9 graphics library
// (see http://plan9.bell-labs.com/magic/man2html/2/draw)
// and the X Render extension.
package draw

// BUG(rsc): This is a toy library and not ready for production use.

import "image"

// A draw.Image is an image.Image with a Set method to change a single pixel.
type Image interface {
	image.Image;
	Set(x, y int, c image.Color);
}

// Draw aligns r.Min in dst with pt in src and mask
// and then replaces the rectangle r in dst with the
// result of the Porter-Duff compositing operation
// ``(src in mask) over dst.''  If mask is nil, the operation
// simplifies to ``src over dst.''
// The implementation is simple and slow.
func Draw(dst Image, r Rectangle, src, mask image.Image, pt Point) {
	// Plenty of room for optimizations here.

	dx, dy := src.Width(), src.Height();
	if mask != nil {
		if dx > mask.Width() {
			dx = mask.Width()
		}
		if dy > mask.Width() {
			dy = mask.Width()
		}
	}
	dx -= pt.X;
	dy -= pt.Y;
	if r.Dx() > dx {
		r.Max.X = r.Min.X + dx
	}
	if r.Dy() > dy {
		r.Max.Y = r.Min.Y + dy
	}

	x0, x1, dx := r.Min.X, r.Max.X, 1;
	y0, y1, dy := r.Min.Y, r.Max.Y, 1;
	if image.Image(dst) == src && r.Overlaps(r.Add(pt.Sub(r.Min))) {
		// Rectangles overlap: process backward?
		if pt.Y < r.Min.Y || pt.Y == r.Min.Y && pt.X < r.Min.X {
			x0, x1, dx = x1-1, x0-1, -1;
			y0, y1, dy = y1-1, y0-1, -1;
		}
	}

	var out *image.RGBA64Color;
	for y := y0; y != y1; y += dy {
		for x := x0; x != x1; x += dx {
			sx := pt.X + x - r.Min.X;
			sy := pt.Y + y - r.Min.Y;
			if mask == nil {
				dst.Set(x, y, src.At(sx, sy));
				continue;
			}
			_, _, _, ma := mask.At(sx, sy).RGBA();
			switch ma {
			case 0:
				continue
			case 0xFFFFFFFF:
				dst.Set(x, y, src.At(sx, sy))
			default:
				dr, dg, db, da := dst.At(x, y).RGBA();
				dr >>= 16;
				dg >>= 16;
				db >>= 16;
				da >>= 16;
				sr, sg, sb, sa := src.At(sx, sy).RGBA();
				sr >>= 16;
				sg >>= 16;
				sb >>= 16;
				sa >>= 16;
				ma >>= 16;
				const M = 1<<16 - 1;
				a := sa*ma/M;
				dr = (dr*(M-a) + sr*ma)/M;
				dg = (dg*(M-a) + sg*ma)/M;
				db = (db*(M-a) + sb*ma)/M;
				da = (da*(M-a) + sa*ma)/M;
				if out == nil {
					out = new(image.RGBA64Color)
				}
				out.R = uint16(dr);
				out.G = uint16(dg);
				out.B = uint16(db);
				out.A = uint16(da);
				dst.Set(x, y, out);
			}
		}
	}
}

// Border aligns r.Min in dst with sp in src and then replaces pixels
// in a w-pixel border around r in dst with the result of the Porter-Duff compositing
// operation ``src over dst.''  If w is positive, the border extends w pixels inside r.
// If w is negative, the border extends w pixels outside r.
func Border(dst Image, r Rectangle, w int, src image.Image, sp Point) {
	i := w;
	if i > 0 {
		// inside r
		Draw(dst, Rect(r.Min.X, r.Min.Y, r.Max.X, r.Min.Y + i), src, nil, sp);					// top
		Draw(dst, Rect(r.Min.X, r.Min.Y + i, r.Min.X + i, r.Max.Y - i), src, nil, sp.Add(Pt(0, i)));		// left
		Draw(dst, Rect(r.Max.X - i, r.Min.Y + i, r.Max.X, r.Max.Y - i), src, nil, sp.Add(Pt(r.Dx() - i, i)));	// right
		Draw(dst, Rect(r.Min.X, r.Max.Y - i, r.Max.X, r.Max.Y), src, nil, sp.Add(Pt(0, r.Dy() - i)));		// bottom
		return;
	}

	// outside r;
	i = -i;
	Draw(dst, Rect(r.Min.X - i, r.Min.Y - i, r.Max.X + i, r.Min.Y), src, nil, sp.Add(Pt(-i, -i)));	// top
	Draw(dst, Rect(r.Min.X - i, r.Min.Y, r.Min.X, r.Max.Y), src, nil, sp.Add(Pt(-i, 0)));		// left
	Draw(dst, Rect(r.Max.X, r.Min.Y, r.Max.X + i, r.Max.Y), src, nil, sp.Add(Pt(r.Dx(), 0)));	// right
	Draw(dst, Rect(r.Min.X - i, r.Max.Y, r.Max.X + i, r.Max.Y + i), src, nil, sp.Add(Pt(-i, 0)));	// bottom
}
