// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package draw provides image composition functions.
//
// See "The Go image/draw package" for an introduction to this package:
// http://golang.org/doc/articles/image_draw.html
package draw

import (
	"image"
	"image/color"
)

// m is the maximum color value returned by image.Color.RGBA.
const m = 1<<16 - 1

// Op is a Porter-Duff compositing operator.
type Op int

const (
	// Over specifies ``(src in mask) over dst''.
	Over Op = iota
	// Src specifies ``src in mask''.
	Src
)

// A draw.Image is an image.Image with a Set method to change a single pixel.
type Image interface {
	image.Image
	Set(x, y int, c color.Color)
}

// Draw calls DrawMask with a nil mask.
func Draw(dst Image, r image.Rectangle, src image.Image, sp image.Point, op Op) {
	DrawMask(dst, r, src, sp, nil, image.ZP, op)
}

// clip clips r against each image's bounds (after translating into the
// destination image's co-ordinate space) and shifts the points sp and mp by
// the same amount as the change in r.Min.
func clip(dst Image, r *image.Rectangle, src image.Image, sp *image.Point, mask image.Image, mp *image.Point) {
	orig := r.Min
	*r = r.Intersect(dst.Bounds())
	*r = r.Intersect(src.Bounds().Add(orig.Sub(*sp)))
	if mask != nil {
		*r = r.Intersect(mask.Bounds().Add(orig.Sub(*mp)))
	}
	dx := r.Min.X - orig.X
	dy := r.Min.Y - orig.Y
	if dx == 0 && dy == 0 {
		return
	}
	(*sp).X += dx
	(*sp).Y += dy
	(*mp).X += dx
	(*mp).Y += dy
}

// DrawMask aligns r.Min in dst with sp in src and mp in mask and then replaces the rectangle r
// in dst with the result of a Porter-Duff composition. A nil mask is treated as opaque.
func DrawMask(dst Image, r image.Rectangle, src image.Image, sp image.Point, mask image.Image, mp image.Point, op Op) {
	clip(dst, &r, src, &sp, mask, &mp)
	if r.Empty() {
		return
	}

	// Fast paths for special cases. If none of them apply, then we fall back to a general but slow implementation.
	if dst0, ok := dst.(*image.RGBA); ok {
		if op == Over {
			if mask == nil {
				switch src0 := src.(type) {
				case *image.Uniform:
					drawFillOver(dst0, r, src0)
					return
				case *image.RGBA:
					drawCopyOver(dst0, r, src0, sp)
					return
				case *image.NRGBA:
					drawNRGBAOver(dst0, r, src0, sp)
					return
				case *image.YCbCr:
					drawYCbCr(dst0, r, src0, sp)
					return
				}
			} else if mask0, ok := mask.(*image.Alpha); ok {
				switch src0 := src.(type) {
				case *image.Uniform:
					drawGlyphOver(dst0, r, src0, mask0, mp)
					return
				}
			}
		} else {
			if mask == nil {
				switch src0 := src.(type) {
				case *image.Uniform:
					drawFillSrc(dst0, r, src0)
					return
				case *image.RGBA:
					drawCopySrc(dst0, r, src0, sp)
					return
				case *image.NRGBA:
					drawNRGBASrc(dst0, r, src0, sp)
					return
				case *image.YCbCr:
					drawYCbCr(dst0, r, src0, sp)
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

	var out *color.RGBA64
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
					dst.Set(x, y, color.Transparent)
				}
			case ma == m && op == Src:
				dst.Set(x, y, src.At(sx, sy))
			default:
				sr, sg, sb, sa := src.At(sx, sy).RGBA()
				if out == nil {
					out = new(color.RGBA64)
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

func drawFillOver(dst *image.RGBA, r image.Rectangle, src *image.Uniform) {
	sr, sg, sb, sa := src.RGBA()
	// The 0x101 is here for the same reason as in drawRGBA.
	a := (m - sa) * 0x101
	i0 := dst.PixOffset(r.Min.X, r.Min.Y)
	i1 := i0 + r.Dx()*4
	for y := r.Min.Y; y != r.Max.Y; y++ {
		for i := i0; i < i1; i += 4 {
			dr := uint32(dst.Pix[i+0])
			dg := uint32(dst.Pix[i+1])
			db := uint32(dst.Pix[i+2])
			da := uint32(dst.Pix[i+3])

			dst.Pix[i+0] = uint8((dr*a/m + sr) >> 8)
			dst.Pix[i+1] = uint8((dg*a/m + sg) >> 8)
			dst.Pix[i+2] = uint8((db*a/m + sb) >> 8)
			dst.Pix[i+3] = uint8((da*a/m + sa) >> 8)
		}
		i0 += dst.Stride
		i1 += dst.Stride
	}
}

func drawFillSrc(dst *image.RGBA, r image.Rectangle, src *image.Uniform) {
	sr, sg, sb, sa := src.RGBA()
	// The built-in copy function is faster than a straightforward for loop to fill the destination with
	// the color, but copy requires a slice source. We therefore use a for loop to fill the first row, and
	// then use the first row as the slice source for the remaining rows.
	i0 := dst.PixOffset(r.Min.X, r.Min.Y)
	i1 := i0 + r.Dx()*4
	for i := i0; i < i1; i += 4 {
		dst.Pix[i+0] = uint8(sr >> 8)
		dst.Pix[i+1] = uint8(sg >> 8)
		dst.Pix[i+2] = uint8(sb >> 8)
		dst.Pix[i+3] = uint8(sa >> 8)
	}
	firstRow := dst.Pix[i0:i1]
	for y := r.Min.Y + 1; y < r.Max.Y; y++ {
		i0 += dst.Stride
		i1 += dst.Stride
		copy(dst.Pix[i0:i1], firstRow)
	}
}

func drawCopyOver(dst *image.RGBA, r image.Rectangle, src *image.RGBA, sp image.Point) {
	dx, dy := r.Dx(), r.Dy()
	d0 := dst.PixOffset(r.Min.X, r.Min.Y)
	s0 := src.PixOffset(sp.X, sp.Y)
	var (
		ddelta, sdelta int
		i0, i1, idelta int
	)
	if r.Min.Y < sp.Y || r.Min.Y == sp.Y && r.Min.X <= sp.X {
		ddelta = dst.Stride
		sdelta = src.Stride
		i0, i1, idelta = 0, dx*4, +4
	} else {
		// If the source start point is higher than the destination start point, or equal height but to the left,
		// then we compose the rows in right-to-left, bottom-up order instead of left-to-right, top-down.
		d0 += (dy - 1) * dst.Stride
		s0 += (dy - 1) * src.Stride
		ddelta = -dst.Stride
		sdelta = -src.Stride
		i0, i1, idelta = (dx-1)*4, -4, -4
	}
	for ; dy > 0; dy-- {
		dpix := dst.Pix[d0:]
		spix := src.Pix[s0:]
		for i := i0; i != i1; i += idelta {
			sr := uint32(spix[i+0]) * 0x101
			sg := uint32(spix[i+1]) * 0x101
			sb := uint32(spix[i+2]) * 0x101
			sa := uint32(spix[i+3]) * 0x101

			dr := uint32(dpix[i+0])
			dg := uint32(dpix[i+1])
			db := uint32(dpix[i+2])
			da := uint32(dpix[i+3])

			// The 0x101 is here for the same reason as in drawRGBA.
			a := (m - sa) * 0x101

			dpix[i+0] = uint8((dr*a/m + sr) >> 8)
			dpix[i+1] = uint8((dg*a/m + sg) >> 8)
			dpix[i+2] = uint8((db*a/m + sb) >> 8)
			dpix[i+3] = uint8((da*a/m + sa) >> 8)
		}
		d0 += ddelta
		s0 += sdelta
	}
}

func drawCopySrc(dst *image.RGBA, r image.Rectangle, src *image.RGBA, sp image.Point) {
	n, dy := 4*r.Dx(), r.Dy()
	d0 := dst.PixOffset(r.Min.X, r.Min.Y)
	s0 := src.PixOffset(sp.X, sp.Y)
	var ddelta, sdelta int
	if r.Min.Y <= sp.Y {
		ddelta = dst.Stride
		sdelta = src.Stride
	} else {
		// If the source start point is higher than the destination start point, then we compose the rows
		// in bottom-up order instead of top-down. Unlike the drawCopyOver function, we don't have to
		// check the x co-ordinates because the built-in copy function can handle overlapping slices.
		d0 += (dy - 1) * dst.Stride
		s0 += (dy - 1) * src.Stride
		ddelta = -dst.Stride
		sdelta = -src.Stride
	}
	for ; dy > 0; dy-- {
		copy(dst.Pix[d0:d0+n], src.Pix[s0:s0+n])
		d0 += ddelta
		s0 += sdelta
	}
}

func drawNRGBAOver(dst *image.RGBA, r image.Rectangle, src *image.NRGBA, sp image.Point) {
	i0 := (r.Min.X - dst.Rect.Min.X) * 4
	i1 := (r.Max.X - dst.Rect.Min.X) * 4
	si0 := (sp.X - src.Rect.Min.X) * 4
	yMax := r.Max.Y - dst.Rect.Min.Y

	y := r.Min.Y - dst.Rect.Min.Y
	sy := sp.Y - src.Rect.Min.Y
	for ; y != yMax; y, sy = y+1, sy+1 {
		dpix := dst.Pix[y*dst.Stride:]
		spix := src.Pix[sy*src.Stride:]

		for i, si := i0, si0; i < i1; i, si = i+4, si+4 {
			// Convert from non-premultiplied color to pre-multiplied color.
			sa := uint32(spix[si+3]) * 0x101
			sr := uint32(spix[si+0]) * sa / 0xff
			sg := uint32(spix[si+1]) * sa / 0xff
			sb := uint32(spix[si+2]) * sa / 0xff

			dr := uint32(dpix[i+0])
			dg := uint32(dpix[i+1])
			db := uint32(dpix[i+2])
			da := uint32(dpix[i+3])

			// The 0x101 is here for the same reason as in drawRGBA.
			a := (m - sa) * 0x101

			dpix[i+0] = uint8((dr*a/m + sr) >> 8)
			dpix[i+1] = uint8((dg*a/m + sg) >> 8)
			dpix[i+2] = uint8((db*a/m + sb) >> 8)
			dpix[i+3] = uint8((da*a/m + sa) >> 8)
		}
	}
}

func drawNRGBASrc(dst *image.RGBA, r image.Rectangle, src *image.NRGBA, sp image.Point) {
	i0 := (r.Min.X - dst.Rect.Min.X) * 4
	i1 := (r.Max.X - dst.Rect.Min.X) * 4
	si0 := (sp.X - src.Rect.Min.X) * 4
	yMax := r.Max.Y - dst.Rect.Min.Y

	y := r.Min.Y - dst.Rect.Min.Y
	sy := sp.Y - src.Rect.Min.Y
	for ; y != yMax; y, sy = y+1, sy+1 {
		dpix := dst.Pix[y*dst.Stride:]
		spix := src.Pix[sy*src.Stride:]

		for i, si := i0, si0; i < i1; i, si = i+4, si+4 {
			// Convert from non-premultiplied color to pre-multiplied color.
			sa := uint32(spix[si+3]) * 0x101
			sr := uint32(spix[si+0]) * sa / 0xff
			sg := uint32(spix[si+1]) * sa / 0xff
			sb := uint32(spix[si+2]) * sa / 0xff

			dpix[i+0] = uint8(sr >> 8)
			dpix[i+1] = uint8(sg >> 8)
			dpix[i+2] = uint8(sb >> 8)
			dpix[i+3] = uint8(sa >> 8)
		}
	}
}

func drawYCbCr(dst *image.RGBA, r image.Rectangle, src *image.YCbCr, sp image.Point) {
	// An image.YCbCr is always fully opaque, and so if the mask is implicitly nil
	// (i.e. fully opaque) then the op is effectively always Src.
	x0 := (r.Min.X - dst.Rect.Min.X) * 4
	x1 := (r.Max.X - dst.Rect.Min.X) * 4
	y0 := r.Min.Y - dst.Rect.Min.Y
	y1 := r.Max.Y - dst.Rect.Min.Y
	switch src.SubsampleRatio {
	case image.YCbCrSubsampleRatio422:
		for y, sy := y0, sp.Y; y != y1; y, sy = y+1, sy+1 {
			dpix := dst.Pix[y*dst.Stride:]
			yi := (sy-src.Rect.Min.Y)*src.YStride + (sp.X - src.Rect.Min.X)
			ciBase := (sy-src.Rect.Min.Y)*src.CStride - src.Rect.Min.X/2
			for x, sx := x0, sp.X; x != x1; x, sx, yi = x+4, sx+1, yi+1 {
				ci := ciBase + sx/2
				rr, gg, bb := color.YCbCrToRGB(src.Y[yi], src.Cb[ci], src.Cr[ci])
				dpix[x+0] = rr
				dpix[x+1] = gg
				dpix[x+2] = bb
				dpix[x+3] = 255
			}
		}
	case image.YCbCrSubsampleRatio420:
		for y, sy := y0, sp.Y; y != y1; y, sy = y+1, sy+1 {
			dpix := dst.Pix[y*dst.Stride:]
			yi := (sy-src.Rect.Min.Y)*src.YStride + (sp.X - src.Rect.Min.X)
			ciBase := (sy/2-src.Rect.Min.Y/2)*src.CStride - src.Rect.Min.X/2
			for x, sx := x0, sp.X; x != x1; x, sx, yi = x+4, sx+1, yi+1 {
				ci := ciBase + sx/2
				rr, gg, bb := color.YCbCrToRGB(src.Y[yi], src.Cb[ci], src.Cr[ci])
				dpix[x+0] = rr
				dpix[x+1] = gg
				dpix[x+2] = bb
				dpix[x+3] = 255
			}
		}
	default:
		// Default to 4:4:4 subsampling.
		for y, sy := y0, sp.Y; y != y1; y, sy = y+1, sy+1 {
			dpix := dst.Pix[y*dst.Stride:]
			yi := (sy-src.Rect.Min.Y)*src.YStride + (sp.X - src.Rect.Min.X)
			ci := (sy-src.Rect.Min.Y)*src.CStride + (sp.X - src.Rect.Min.X)
			for x := x0; x != x1; x, yi, ci = x+4, yi+1, ci+1 {
				rr, gg, bb := color.YCbCrToRGB(src.Y[yi], src.Cb[ci], src.Cr[ci])
				dpix[x+0] = rr
				dpix[x+1] = gg
				dpix[x+2] = bb
				dpix[x+3] = 255
			}
		}
	}
}

func drawGlyphOver(dst *image.RGBA, r image.Rectangle, src *image.Uniform, mask *image.Alpha, mp image.Point) {
	i0 := dst.PixOffset(r.Min.X, r.Min.Y)
	i1 := i0 + r.Dx()*4
	mi0 := mask.PixOffset(mp.X, mp.Y)
	sr, sg, sb, sa := src.RGBA()
	for y, my := r.Min.Y, mp.Y; y != r.Max.Y; y, my = y+1, my+1 {
		for i, mi := i0, mi0; i < i1; i, mi = i+4, mi+1 {
			ma := uint32(mask.Pix[mi])
			if ma == 0 {
				continue
			}
			ma |= ma << 8

			dr := uint32(dst.Pix[i+0])
			dg := uint32(dst.Pix[i+1])
			db := uint32(dst.Pix[i+2])
			da := uint32(dst.Pix[i+3])

			// The 0x101 is here for the same reason as in drawRGBA.
			a := (m - (sa * ma / m)) * 0x101

			dst.Pix[i+0] = uint8((dr*a + sr*ma) / m >> 8)
			dst.Pix[i+1] = uint8((dg*a + sg*ma) / m >> 8)
			dst.Pix[i+2] = uint8((db*a + sb*ma) / m >> 8)
			dst.Pix[i+3] = uint8((da*a + sa*ma) / m >> 8)
		}
		i0 += dst.Stride
		i1 += dst.Stride
		mi0 += mask.Stride
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
	sx0 := sp.X + x0 - r.Min.X
	mx0 := mp.X + x0 - r.Min.X
	sx1 := sx0 + (x1 - x0)
	i0 := dst.PixOffset(x0, y0)
	di := dx * 4
	for y := y0; y != y1; y, sy, my = y+dy, sy+dy, my+dy {
		for i, sx, mx := i0, sx0, mx0; sx != sx1; i, sx, mx = i+di, sx+dx, mx+dx {
			ma := uint32(m)
			if mask != nil {
				_, _, _, ma = mask.At(mx, my).RGBA()
			}
			sr, sg, sb, sa := src.At(sx, sy).RGBA()
			if op == Over {
				dr := uint32(dst.Pix[i+0])
				dg := uint32(dst.Pix[i+1])
				db := uint32(dst.Pix[i+2])
				da := uint32(dst.Pix[i+3])

				// dr, dg, db and da are all 8-bit color at the moment, ranging in [0,255].
				// We work in 16-bit color, and so would normally do:
				// dr |= dr << 8
				// and similarly for dg, db and da, but instead we multiply a
				// (which is a 16-bit color, ranging in [0,65535]) by 0x101.
				// This yields the same result, but is fewer arithmetic operations.
				a := (m - (sa * ma / m)) * 0x101

				dst.Pix[i+0] = uint8((dr*a + sr*ma) / m >> 8)
				dst.Pix[i+1] = uint8((dg*a + sg*ma) / m >> 8)
				dst.Pix[i+2] = uint8((db*a + sb*ma) / m >> 8)
				dst.Pix[i+3] = uint8((da*a + sa*ma) / m >> 8)

			} else {
				dst.Pix[i+0] = uint8(sr * ma / m >> 8)
				dst.Pix[i+1] = uint8(sg * ma / m >> 8)
				dst.Pix[i+2] = uint8(sb * ma / m >> 8)
				dst.Pix[i+3] = uint8(sa * ma / m >> 8)
			}
		}
		i0 += dy * dst.Stride
	}
}
