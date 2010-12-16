// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package draw provides basic graphics and drawing primitives,
// in the style of the Plan 9 graphics library
// (see http://plan9.bell-labs.com/magic/man2html/2/draw)
// and the X Render extension.
package draw

import "image"

// m is the maximum color value returned by image.Color.RGBA.
const m = 1<<16 - 1

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
func Draw(dst Image, r image.Rectangle, src image.Image, sp image.Point) {
	DrawMask(dst, r, src, sp, nil, image.ZP, Over)
}

// DrawMask aligns r.Min in dst with sp in src and mp in mask and then replaces the rectangle r
// in dst with the result of a Porter-Duff composition. A nil mask is treated as opaque.
func DrawMask(dst Image, r image.Rectangle, src image.Image, sp image.Point, mask image.Image, mp image.Point, op Op) {
	sb := src.Bounds()
	dx, dy := sb.Max.X-sp.X, sb.Max.Y-sp.Y
	if mask != nil {
		mb := mask.Bounds()
		if dx > mb.Max.X-mp.X {
			dx = mb.Max.X - mp.X
		}
		if dy > mb.Max.Y-mp.Y {
			dy = mb.Max.Y - mp.Y
		}
	}
	if r.Dx() > dx {
		r.Max.X = r.Min.X + dx
	}
	if r.Dy() > dy {
		r.Max.Y = r.Min.Y + dy
	}
	r = r.Intersect(dst.Bounds())
	if r.Empty() {
		return
	}

	// Fast paths for special cases. If none of them apply, then we fall back to a general but slow implementation.
	if dst0, ok := dst.(*image.RGBA); ok {
		if op == Over {
			if mask == nil {
				if src0, ok := src.(*image.ColorImage); ok {
					drawFillOver(dst0, r, src0)
					return
				}
				if src0, ok := src.(*image.RGBA); ok {
					drawCopyOver(dst0, r, src0, sp)
					return
				}
			} else if mask0, ok := mask.(*image.Alpha); ok {
				if src0, ok := src.(*image.ColorImage); ok {
					drawGlyphOver(dst0, r, src0, mask0, mp)
					return
				}
			}
		} else {
			if mask == nil {
				if src0, ok := src.(*image.ColorImage); ok {
					drawFillSrc(dst0, r, src0)
					return
				}
				if src0, ok := src.(*image.RGBA); ok {
					drawCopySrc(dst0, r, src0, sp)
					return
				}
			}
		}
		drawRGBA(dst0, r, src, sp, mask, mp, op)
		return
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
			ma := uint32(m)
			if mask != nil {
				_, _, _, ma = mask.At(mx, my).RGBA()
			}
			switch {
			case ma == 0:
				if op == Over {
					// No-op.
				} else {
					dst.Set(x, y, zeroColor)
				}
			case ma == m && op == Src:
				dst.Set(x, y, src.At(sx, sy))
			default:
				sr, sg, sb, sa := src.At(sx, sy).RGBA()
				if out == nil {
					out = new(image.RGBA64Color)
				}
				if op == Over {
					dr, dg, db, da := dst.At(x, y).RGBA()
					a := m - (sa * ma / m)
					out.R = uint16((dr*a + sr*ma) / m)
					out.G = uint16((dg*a + sg*ma) / m)
					out.B = uint16((db*a + sb*ma) / m)
					out.A = uint16((da*a + sa*ma) / m)
				} else {
					out.R = uint16(sr * ma / m)
					out.G = uint16(sg * ma / m)
					out.B = uint16(sb * ma / m)
					out.A = uint16(sa * ma / m)
				}
				dst.Set(x, y, out)
			}
		}
	}
}

func drawFillOver(dst *image.RGBA, r image.Rectangle, src *image.ColorImage) {
	cr, cg, cb, ca := src.RGBA()
	// The 0x101 is here for the same reason as in drawRGBA.
	a := (m - ca) * 0x101
	x0, x1 := r.Min.X, r.Max.X
	y0, y1 := r.Min.Y, r.Max.Y
	for y := y0; y != y1; y++ {
		dbase := y * dst.Stride
		dpix := dst.Pix[dbase+x0 : dbase+x1]
		for i, rgba := range dpix {
			dr := (uint32(rgba.R)*a)/m + cr
			dg := (uint32(rgba.G)*a)/m + cg
			db := (uint32(rgba.B)*a)/m + cb
			da := (uint32(rgba.A)*a)/m + ca
			dpix[i] = image.RGBAColor{uint8(dr >> 8), uint8(dg >> 8), uint8(db >> 8), uint8(da >> 8)}
		}
	}
}

func drawCopyOver(dst *image.RGBA, r image.Rectangle, src *image.RGBA, sp image.Point) {
	dx0, dx1 := r.Min.X, r.Max.X
	dy0, dy1 := r.Min.Y, r.Max.Y
	nrows := dy1 - dy0
	sx0, sx1 := sp.X, sp.X+dx1-dx0
	d0 := dy0*dst.Stride + dx0
	d1 := dy0*dst.Stride + dx1
	s0 := sp.Y*src.Stride + sx0
	s1 := sp.Y*src.Stride + sx1
	var (
		ddelta, sdelta int
		i0, i1, idelta int
	)
	if r.Min.Y < sp.Y || r.Min.Y == sp.Y && r.Min.X <= sp.X {
		ddelta = dst.Stride
		sdelta = src.Stride
		i0, i1, idelta = 0, d1-d0, +1
	} else {
		// If the source start point is higher than the destination start point, or equal height but to the left,
		// then we compose the rows in right-to-left, bottom-up order instead of left-to-right, top-down.
		d0 += (nrows - 1) * dst.Stride
		d1 += (nrows - 1) * dst.Stride
		s0 += (nrows - 1) * src.Stride
		s1 += (nrows - 1) * src.Stride
		ddelta = -dst.Stride
		sdelta = -src.Stride
		i0, i1, idelta = d1-d0-1, -1, -1
	}
	for ; nrows > 0; nrows-- {
		dpix := dst.Pix[d0:d1]
		spix := src.Pix[s0:s1]
		for i := i0; i != i1; i += idelta {
			// For unknown reasons, even though both dpix[i] and spix[i] are
			// image.RGBAColors, on an x86 CPU it seems fastest to call RGBA
			// for the source but to do it manually for the destination.
			sr, sg, sb, sa := spix[i].RGBA()
			rgba := dpix[i]
			dr := uint32(rgba.R)
			dg := uint32(rgba.G)
			db := uint32(rgba.B)
			da := uint32(rgba.A)
			// The 0x101 is here for the same reason as in drawRGBA.
			a := (m - sa) * 0x101
			dr = (dr*a)/m + sr
			dg = (dg*a)/m + sg
			db = (db*a)/m + sb
			da = (da*a)/m + sa
			dpix[i] = image.RGBAColor{uint8(dr >> 8), uint8(dg >> 8), uint8(db >> 8), uint8(da >> 8)}
		}
		d0 += ddelta
		d1 += ddelta
		s0 += sdelta
		s1 += sdelta
	}
}

func drawGlyphOver(dst *image.RGBA, r image.Rectangle, src *image.ColorImage, mask *image.Alpha, mp image.Point) {
	x0, x1 := r.Min.X, r.Max.X
	y0, y1 := r.Min.Y, r.Max.Y
	cr, cg, cb, ca := src.RGBA()
	for y, my := y0, mp.Y; y != y1; y, my = y+1, my+1 {
		dbase := y * dst.Stride
		dpix := dst.Pix[dbase+x0 : dbase+x1]
		mbase := my * mask.Stride
		mpix := mask.Pix[mbase+mp.X:]
		for i, rgba := range dpix {
			ma := uint32(mpix[i].A)
			if ma == 0 {
				continue
			}
			ma |= ma << 8
			dr := uint32(rgba.R)
			dg := uint32(rgba.G)
			db := uint32(rgba.B)
			da := uint32(rgba.A)
			// The 0x101 is here for the same reason as in drawRGBA.
			a := (m - (ca * ma / m)) * 0x101
			dr = (dr*a + cr*ma) / m
			dg = (dg*a + cg*ma) / m
			db = (db*a + cb*ma) / m
			da = (da*a + ca*ma) / m
			dpix[i] = image.RGBAColor{uint8(dr >> 8), uint8(dg >> 8), uint8(db >> 8), uint8(da >> 8)}
		}
	}
}

func drawFillSrc(dst *image.RGBA, r image.Rectangle, src *image.ColorImage) {
	if r.Dy() < 1 {
		return
	}
	cr, cg, cb, ca := src.RGBA()
	color := image.RGBAColor{uint8(cr >> 8), uint8(cg >> 8), uint8(cb >> 8), uint8(ca >> 8)}
	// The built-in copy function is faster than a straightforward for loop to fill the destination with
	// the color, but copy requires a slice source. We therefore use a for loop to fill the first row, and
	// then use the first row as the slice source for the remaining rows.
	dx0, dx1 := r.Min.X, r.Max.X
	dy0, dy1 := r.Min.Y, r.Max.Y
	dbase := dy0 * dst.Stride
	i0, i1 := dbase+dx0, dbase+dx1
	firstRow := dst.Pix[i0:i1]
	for i := range firstRow {
		firstRow[i] = color
	}
	for y := dy0 + 1; y < dy1; y++ {
		i0 += dst.Stride
		i1 += dst.Stride
		copy(dst.Pix[i0:i1], firstRow)
	}
}

func drawCopySrc(dst *image.RGBA, r image.Rectangle, src *image.RGBA, sp image.Point) {
	dx0, dx1 := r.Min.X, r.Max.X
	dy0, dy1 := r.Min.Y, r.Max.Y
	nrows := dy1 - dy0
	sx0, sx1 := sp.X, sp.X+dx1-dx0
	d0 := dy0*dst.Stride + dx0
	d1 := dy0*dst.Stride + dx1
	s0 := sp.Y*src.Stride + sx0
	s1 := sp.Y*src.Stride + sx1
	var ddelta, sdelta int
	if r.Min.Y <= sp.Y {
		ddelta = dst.Stride
		sdelta = src.Stride
	} else {
		// If the source start point is higher than the destination start point, then we compose the rows
		// in bottom-up order instead of top-down. Unlike the drawCopyOver function, we don't have to
		// check the x co-ordinates because the built-in copy function can handle overlapping slices.
		d0 += (nrows - 1) * dst.Stride
		d1 += (nrows - 1) * dst.Stride
		s0 += (nrows - 1) * src.Stride
		s1 += (nrows - 1) * src.Stride
		ddelta = -dst.Stride
		sdelta = -src.Stride
	}
	for ; nrows > 0; nrows-- {
		copy(dst.Pix[d0:d1], src.Pix[s0:s1])
		d0 += ddelta
		d1 += ddelta
		s0 += sdelta
		s1 += sdelta
	}
}

func drawRGBA(dst *image.RGBA, r image.Rectangle, src image.Image, sp image.Point, mask image.Image, mp image.Point, op Op) {
	x0, x1, dx := r.Min.X, r.Max.X, 1
	y0, y1, dy := r.Min.Y, r.Max.Y, 1
	if image.Image(dst) == src && r.Overlaps(r.Add(sp.Sub(r.Min))) {
		if sp.Y < r.Min.Y || sp.Y == r.Min.Y && sp.X < r.Min.X {
			x0, x1, dx = x1-1, x0-1, -1
			y0, y1, dy = y1-1, y0-1, -1
		}
	}

	sy := sp.Y + y0 - r.Min.Y
	my := mp.Y + y0 - r.Min.Y
	for y := y0; y != y1; y, sy, my = y+dy, sy+dy, my+dy {
		sx := sp.X + x0 - r.Min.X
		mx := mp.X + x0 - r.Min.X
		dpix := dst.Pix[y*dst.Stride : (y+1)*dst.Stride]
		for x := x0; x != x1; x, sx, mx = x+dx, sx+dx, mx+dx {
			ma := uint32(m)
			if mask != nil {
				_, _, _, ma = mask.At(mx, my).RGBA()
			}
			sr, sg, sb, sa := src.At(sx, sy).RGBA()
			var dr, dg, db, da uint32
			if op == Over {
				rgba := dpix[x]
				dr = uint32(rgba.R)
				dg = uint32(rgba.G)
				db = uint32(rgba.B)
				da = uint32(rgba.A)
				// dr, dg, db and da are all 8-bit color at the moment, ranging in [0,255].
				// We work in 16-bit color, and so would normally do:
				// dr |= dr << 8
				// and similarly for dg, db and da, but instead we multiply a
				// (which is a 16-bit color, ranging in [0,65535]) by 0x101.
				// This yields the same result, but is fewer arithmetic operations.
				a := (m - (sa * ma / m)) * 0x101
				dr = (dr*a + sr*ma) / m
				dg = (dg*a + sg*ma) / m
				db = (db*a + sb*ma) / m
				da = (da*a + sa*ma) / m
			} else {
				dr = sr * ma / m
				dg = sg * ma / m
				db = sb * ma / m
				da = sa * ma / m
			}
			dpix[x] = image.RGBAColor{uint8(dr >> 8), uint8(dg >> 8), uint8(db >> 8), uint8(da >> 8)}
		}
	}
}
