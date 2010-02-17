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

// A Porter-Duff compositing operator.
type Op int

const (
	// Over specifies ``(src in mask) over dst''.
	Over Op = iota
	// Src specifies ``src in mask''.
	Src
)

var zeroColor image.Color = image.AlphaColor{0}

// A draw.Image is an image.Image with a Set method to change a single pixel.
type Image interface {
	image.Image
	Set(x, y int, c image.Color)
}

// Draw calls DrawMask with a nil mask and an Over op.
func Draw(dst Image, r Rectangle, src image.Image, sp Point) {
	DrawMask(dst, r, src, sp, nil, ZP, Over)
}

// DrawMask aligns r.Min in dst with sp in src and mp in mask and then replaces the rectangle r
// in dst with the result of a Porter-Duff composition. A nil mask is treated as opaque.
// The implementation is simple and slow.
// TODO(nigeltao): Optimize this.
func DrawMask(dst Image, r Rectangle, src image.Image, sp Point, mask image.Image, mp Point, op Op) {
	dx, dy := src.Width()-sp.X, src.Height()-sp.Y
	if mask != nil {
		if dx > mask.Width()-mp.X {
			dx = mask.Width() - mp.X
		}
		if dy > mask.Height()-mp.Y {
			dy = mask.Height() - mp.Y
		}
	}
	if r.Dx() > dx {
		r.Max.X = r.Min.X + dx
	}
	if r.Dy() > dy {
		r.Max.Y = r.Min.Y + dy
	}

	// TODO(nigeltao): Clip r to dst's bounding box, and handle the case when sp or mp has negative X or Y.
	// TODO(nigeltao): Ensure that r is well formed, i.e. r.Max.X >= r.Min.X and likewise for Y.

	// Fast paths for special cases. If none of them apply, then we fall back to a general but slow implementation.
	if dst0, ok := dst.(*image.RGBA); ok {
		if op == Over {
			// TODO(nigeltao): Implement a fast path for font glyphs (i.e. when mask is an image.Alpha).
		} else {
			if mask == nil {
				if src0, ok := src.(image.ColorImage); ok {
					drawFill(dst0, r, src0)
					return
				}
				if src0, ok := src.(*image.RGBA); ok {
					if dst0 == src0 && r.Overlaps(r.Add(sp.Sub(r.Min))) {
						// TODO(nigeltao): Implement a fast path for the overlapping case.
					} else {
						drawCopy(dst0, r, src0, sp)
						return
					}
				}
			}
		}
	}

	x0, x1, dx := r.Min.X, r.Max.X, 1
	y0, y1, dy := r.Min.Y, r.Max.Y, 1
	if image.Image(dst) == src && r.Overlaps(r.Add(sp.Sub(r.Min))) {
		// Rectangles overlap: process backward?
		if sp.Y < r.Min.Y || sp.Y == r.Min.Y && sp.X < r.Min.X {
			x0, x1, dx = x1-1, x0-1, -1
			y0, y1, dy = y1-1, y0-1, -1
		}
	}

	var out *image.RGBA64Color
	sy := sp.Y + y0 - r.Min.Y
	my := mp.Y + y0 - r.Min.Y
	for y := y0; y != y1; y, sy, my = y+dy, sy+dy, my+dy {
		sx := sp.X + x0 - r.Min.X
		mx := mp.X + x0 - r.Min.X
		for x := x0; x != x1; x, sx, mx = x+dx, sx+dx, mx+dx {
			// A nil mask is equivalent to a fully opaque, infinitely large mask.
			// We work in 16-bit color, so that multiplying two values does not overflow a uint32.
			const M = 1<<16 - 1
			ma := uint32(M)
			if mask != nil {
				_, _, _, ma = mask.At(mx, my).RGBA()
				ma >>= 16
			}
			switch {
			case ma == 0:
				if op == Over {
					// No-op.
				} else {
					dst.Set(x, y, zeroColor)
				}
			case ma == M && op == Src:
				dst.Set(x, y, src.At(sx, sy))
			default:
				sr, sg, sb, sa := src.At(sx, sy).RGBA()
				sr >>= 16
				sg >>= 16
				sb >>= 16
				sa >>= 16
				if out == nil {
					out = new(image.RGBA64Color)
				}
				if op == Over {
					dr, dg, db, da := dst.At(x, y).RGBA()
					dr >>= 16
					dg >>= 16
					db >>= 16
					da >>= 16
					a := M - (sa * ma / M)
					out.R = uint16((dr*a + sr*ma) / M)
					out.G = uint16((dg*a + sg*ma) / M)
					out.B = uint16((db*a + sb*ma) / M)
					out.A = uint16((da*a + sa*ma) / M)
				} else {
					out.R = uint16(sr * ma / M)
					out.G = uint16(sg * ma / M)
					out.B = uint16(sb * ma / M)
					out.A = uint16(sa * ma / M)
				}
				dst.Set(x, y, out)
			}
		}
	}
}

func drawFill(dst *image.RGBA, r Rectangle, src image.ColorImage) {
	if r.Dy() < 1 {
		return
	}
	cr, cg, cb, ca := src.RGBA()
	color := image.RGBAColor{uint8(cr >> 24), uint8(cg >> 24), uint8(cb >> 24), uint8(ca >> 24)}
	// The built-in copy function is faster than a straightforward for loop to fill the destination with
	// the color, but copy requires a slice source. We therefore use a for loop to fill the first row, and
	// then use the first row as the slice source for the remaining rows.
	dx0, dx1 := r.Min.X, r.Max.X
	dy0, dy1 := r.Min.Y, r.Max.Y
	firstRow := dst.Pixel[dy0]
	for x := dx0; x < dx1; x++ {
		firstRow[x] = color
	}
	copySrc := firstRow[dx0:dx1]
	for y := dy0 + 1; y < dy1; y++ {
		copy(dst.Pixel[y][dx0:dx1], copySrc)
	}
}

func drawCopy(dst *image.RGBA, r Rectangle, src *image.RGBA, sp Point) {
	dx0, dx1 := r.Min.X, r.Max.X
	dy0, dy1 := r.Min.Y, r.Max.Y
	sx0, sx1 := sp.X, sp.X+dx1-dx0
	for y, sy := dy0, sp.Y; y < dy1; y, sy = y+1, sy+1 {
		copy(dst.Pixel[y][dx0:dx1], src.Pixel[sy][sx0:sx1])
	}
}

// Border aligns r.Min in dst with sp in src and then replaces pixels
// in a w-pixel border around r in dst with the result of the Porter-Duff compositing
// operation ``src over dst.''  If w is positive, the border extends w pixels inside r.
// If w is negative, the border extends w pixels outside r.
func Border(dst Image, r Rectangle, w int, src image.Image, sp Point) {
	i := w
	if i > 0 {
		// inside r
		Draw(dst, Rect(r.Min.X, r.Min.Y, r.Max.X, r.Min.Y+i), src, sp)                          // top
		Draw(dst, Rect(r.Min.X, r.Min.Y+i, r.Min.X+i, r.Max.Y-i), src, sp.Add(Pt(0, i)))        // left
		Draw(dst, Rect(r.Max.X-i, r.Min.Y+i, r.Max.X, r.Max.Y-i), src, sp.Add(Pt(r.Dx()-i, i))) // right
		Draw(dst, Rect(r.Min.X, r.Max.Y-i, r.Max.X, r.Max.Y), src, sp.Add(Pt(0, r.Dy()-i)))     // bottom
		return
	}

	// outside r;
	i = -i
	Draw(dst, Rect(r.Min.X-i, r.Min.Y-i, r.Max.X+i, r.Min.Y), src, sp.Add(Pt(-i, -i))) // top
	Draw(dst, Rect(r.Min.X-i, r.Min.Y, r.Min.X, r.Max.Y), src, sp.Add(Pt(-i, 0)))      // left
	Draw(dst, Rect(r.Max.X, r.Min.Y, r.Max.X+i, r.Max.Y), src, sp.Add(Pt(r.Dx(), 0)))  // right
	Draw(dst, Rect(r.Min.X-i, r.Max.Y, r.Max.X+i, r.Max.Y+i), src, sp.Add(Pt(-i, 0)))  // bottom
}
