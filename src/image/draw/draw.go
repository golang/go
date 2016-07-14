// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package draw provides image composition functions.
//
// See "The Go image/draw package" for an introduction to this package:
// https://golang.org/doc/articles/image_draw.html
package draw

import (
	"image"
	"image/color"
	"image/internal/imageutil"
)

// m is the maximum color value returned by image.Color.RGBA.
const m = 1<<16 - 1

// Image is an image.Image with a Set method to change a single pixel.
type Image interface {
	image.Image
	Set(x, y int, c color.Color)
}

// Quantizer produces a palette for an image.
type Quantizer interface {
	// Quantize appends up to cap(p) - len(p) colors to p and returns the
	// updated palette suitable for converting m to a paletted image.
	Quantize(p color.Palette, m image.Image) color.Palette
}

// Op is a Porter-Duff compositing operator.
type Op int

const (
	// Over specifies ``(src in mask) over dst''.
	Over Op = iota
	// Src specifies ``src in mask''.
	Src
)

// Draw implements the Drawer interface by calling the Draw function with this
// Op.
func (op Op) Draw(dst Image, r image.Rectangle, src image.Image, sp image.Point) {
	DrawMask(dst, r, src, sp, nil, image.Point{}, op)
}

// Drawer contains the Draw method.
type Drawer interface {
	// Draw aligns r.Min in dst with sp in src and then replaces the
	// rectangle r in dst with the result of drawing src on dst.
	Draw(dst Image, r image.Rectangle, src image.Image, sp image.Point)
}

// FloydSteinberg is a Drawer that is the Src Op with Floyd-Steinberg error
// diffusion.
var FloydSteinberg Drawer = floydSteinberg{}

type floydSteinberg struct{}

func (floydSteinberg) Draw(dst Image, r image.Rectangle, src image.Image, sp image.Point) {
	clip(dst, &r, src, &sp, nil, nil)
	if r.Empty() {
		return
	}
	drawPaletted(dst, r, src, sp, true)
}

// clip clips r against each image's bounds (after translating into the
// destination image's coordinate space) and shifts the points sp and mp by
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
	sp.X += dx
	sp.Y += dy
	if mp != nil {
		mp.X += dx
		mp.Y += dy
	}
}

func processBackward(dst Image, r image.Rectangle, src image.Image, sp image.Point) bool {
	return image.Image(dst) == src &&
		r.Overlaps(r.Add(sp.Sub(r.Min))) &&
		(sp.Y < r.Min.Y || (sp.Y == r.Min.Y && sp.X < r.Min.X))
}

// Draw calls DrawMask with a nil mask.
func Draw(dst Image, r image.Rectangle, src image.Image, sp image.Point, op Op) {
	DrawMask(dst, r, src, sp, nil, image.Point{}, op)
}

// DrawMask aligns r.Min in dst with sp in src and mp in mask and then replaces the rectangle r
// in dst with the result of a Porter-Duff composition. A nil mask is treated as opaque.
func DrawMask(dst Image, r image.Rectangle, src image.Image, sp image.Point, mask image.Image, mp image.Point, op Op) {
	clip(dst, &r, src, &sp, mask, &mp)
	if r.Empty() {
		return
	}

	// Fast paths for special cases. If none of them apply, then we fall back to a general but slow implementation.
	switch dst0 := dst.(type) {
	case *image.RGBA:
		if op == Over {
			if mask == nil {
				switch src0 := src.(type) {
				case *image.Uniform:
					sr, sg, sb, sa := src0.RGBA()
					if sa == 0xffff {
						drawFillSrc(dst0, r, sr, sg, sb, sa)
					} else {
						drawFillOver(dst0, r, sr, sg, sb, sa)
					}
					return
				case *image.RGBA:
					drawCopyOver(dst0, r, src0, sp)
					return
				case *image.NRGBA:
					drawNRGBAOver(dst0, r, src0, sp)
					return
				case *image.YCbCr:
					// An image.YCbCr is always fully opaque, and so if the
					// mask is nil (i.e. fully opaque) then the op is
					// effectively always Src. Similarly for image.Gray and
					// image.CMYK.
					if imageutil.DrawYCbCr(dst0, r, src0, sp) {
						return
					}
				case *image.Gray:
					drawGray(dst0, r, src0, sp)
					return
				case *image.CMYK:
					drawCMYK(dst0, r, src0, sp)
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
					sr, sg, sb, sa := src0.RGBA()
					drawFillSrc(dst0, r, sr, sg, sb, sa)
					return
				case *image.RGBA:
					drawCopySrc(dst0, r, src0, sp)
					return
				case *image.NRGBA:
					drawNRGBASrc(dst0, r, src0, sp)
					return
				case *image.YCbCr:
					if imageutil.DrawYCbCr(dst0, r, src0, sp) {
						return
					}
				case *image.Gray:
					drawGray(dst0, r, src0, sp)
					return
				case *image.CMYK:
					drawCMYK(dst0, r, src0, sp)
					return
				}
			}
		}
		drawRGBA(dst0, r, src, sp, mask, mp, op)
		return
	case *image.Paletted:
		if op == Src && mask == nil && !processBackward(dst, r, src, sp) {
			drawPaletted(dst0, r, src, sp, false)
			return
		}
	}

	x0, x1, dx := r.Min.X, r.Max.X, 1
	y0, y1, dy := r.Min.Y, r.Max.Y, 1
	if processBackward(dst, r, src, sp) {
		x0, x1, dx = x1-1, x0-1, -1
		y0, y1, dy = y1-1, y0-1, -1
	}

	var out color.RGBA64
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
				// The third argument is &out instead of out (and out is
				// declared outside of the inner loop) to avoid the implicit
				// conversion to color.Color here allocating memory in the
				// inner loop if sizeof(color.RGBA64) > sizeof(uintptr).
				dst.Set(x, y, &out)
			}
		}
	}
}

func drawFillOver(dst *image.RGBA, r image.Rectangle, sr, sg, sb, sa uint32) {
	// The 0x101 is here for the same reason as in drawRGBA.
	a := (m - sa) * 0x101
	i0 := dst.PixOffset(r.Min.X, r.Min.Y)
	i1 := i0 + r.Dx()*4
	for y := r.Min.Y; y != r.Max.Y; y++ {
		for i := i0; i < i1; i += 4 {
			dr := &dst.Pix[i+0]
			dg := &dst.Pix[i+1]
			db := &dst.Pix[i+2]
			da := &dst.Pix[i+3]

			*dr = uint8((uint32(*dr)*a/m + sr) >> 8)
			*dg = uint8((uint32(*dg)*a/m + sg) >> 8)
			*db = uint8((uint32(*db)*a/m + sb) >> 8)
			*da = uint8((uint32(*da)*a/m + sa) >> 8)
		}
		i0 += dst.Stride
		i1 += dst.Stride
	}
}

func drawFillSrc(dst *image.RGBA, r image.Rectangle, sr, sg, sb, sa uint32) {
	sr8 := uint8(sr >> 8)
	sg8 := uint8(sg >> 8)
	sb8 := uint8(sb >> 8)
	sa8 := uint8(sa >> 8)
	// The built-in copy function is faster than a straightforward for loop to fill the destination with
	// the color, but copy requires a slice source. We therefore use a for loop to fill the first row, and
	// then use the first row as the slice source for the remaining rows.
	i0 := dst.PixOffset(r.Min.X, r.Min.Y)
	i1 := i0 + r.Dx()*4
	for i := i0; i < i1; i += 4 {
		dst.Pix[i+0] = sr8
		dst.Pix[i+1] = sg8
		dst.Pix[i+2] = sb8
		dst.Pix[i+3] = sa8
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

			dr := &dpix[i+0]
			dg := &dpix[i+1]
			db := &dpix[i+2]
			da := &dpix[i+3]

			// The 0x101 is here for the same reason as in drawRGBA.
			a := (m - sa) * 0x101

			*dr = uint8((uint32(*dr)*a/m + sr) >> 8)
			*dg = uint8((uint32(*dg)*a/m + sg) >> 8)
			*db = uint8((uint32(*db)*a/m + sb) >> 8)
			*da = uint8((uint32(*da)*a/m + sa) >> 8)
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
		// If the source start point is higher than the destination start
		// point, then we compose the rows in bottom-up order instead of
		// top-down. Unlike the drawCopyOver function, we don't have to check
		// the x coordinates because the built-in copy function can handle
		// overlapping slices.
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

func drawGray(dst *image.RGBA, r image.Rectangle, src *image.Gray, sp image.Point) {
	i0 := (r.Min.X - dst.Rect.Min.X) * 4
	i1 := (r.Max.X - dst.Rect.Min.X) * 4
	si0 := (sp.X - src.Rect.Min.X) * 1
	yMax := r.Max.Y - dst.Rect.Min.Y

	y := r.Min.Y - dst.Rect.Min.Y
	sy := sp.Y - src.Rect.Min.Y
	for ; y != yMax; y, sy = y+1, sy+1 {
		dpix := dst.Pix[y*dst.Stride:]
		spix := src.Pix[sy*src.Stride:]

		for i, si := i0, si0; i < i1; i, si = i+4, si+1 {
			p := spix[si]
			dpix[i+0] = p
			dpix[i+1] = p
			dpix[i+2] = p
			dpix[i+3] = 255
		}
	}
}

func drawCMYK(dst *image.RGBA, r image.Rectangle, src *image.CMYK, sp image.Point) {
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
			dpix[i+0], dpix[i+1], dpix[i+2] =
				color.CMYKToRGB(spix[si+0], spix[si+1], spix[si+2], spix[si+3])
			dpix[i+3] = 255
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

			dr := &dst.Pix[i+0]
			dg := &dst.Pix[i+1]
			db := &dst.Pix[i+2]
			da := &dst.Pix[i+3]

			// The 0x101 is here for the same reason as in drawRGBA.
			a := (m - (sa * ma / m)) * 0x101

			*dr = uint8((uint32(*dr)*a + sr*ma) / m >> 8)
			*dg = uint8((uint32(*dg)*a + sg*ma) / m >> 8)
			*db = uint8((uint32(*db)*a + sb*ma) / m >> 8)
			*da = uint8((uint32(*da)*a + sa*ma) / m >> 8)
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

// clamp clamps i to the interval [0, 0xffff].
func clamp(i int32) int32 {
	if i < 0 {
		return 0
	}
	if i > 0xffff {
		return 0xffff
	}
	return i
}

// sqDiff returns the squared-difference of x and y, shifted by 2 so that
// adding four of those won't overflow a uint32.
//
// x and y are both assumed to be in the range [0, 0xffff].
func sqDiff(x, y int32) uint32 {
	var d uint32
	if x > y {
		d = uint32(x - y)
	} else {
		d = uint32(y - x)
	}
	return (d * d) >> 2
}

func drawPaletted(dst Image, r image.Rectangle, src image.Image, sp image.Point, floydSteinberg bool) {
	// TODO(nigeltao): handle the case where the dst and src overlap.
	// Does it even make sense to try and do Floyd-Steinberg whilst
	// walking the image backward (right-to-left bottom-to-top)?

	// If dst is an *image.Paletted, we have a fast path for dst.Set and
	// dst.At. The dst.Set equivalent is a batch version of the algorithm
	// used by color.Palette's Index method in image/color/color.go, plus
	// optional Floyd-Steinberg error diffusion.
	palette, pix, stride := [][4]int32(nil), []byte(nil), 0
	if p, ok := dst.(*image.Paletted); ok {
		palette = make([][4]int32, len(p.Palette))
		for i, col := range p.Palette {
			r, g, b, a := col.RGBA()
			palette[i][0] = int32(r)
			palette[i][1] = int32(g)
			palette[i][2] = int32(b)
			palette[i][3] = int32(a)
		}
		pix, stride = p.Pix[p.PixOffset(r.Min.X, r.Min.Y):], p.Stride
	}

	// quantErrorCurr and quantErrorNext are the Floyd-Steinberg quantization
	// errors that have been propagated to the pixels in the current and next
	// rows. The +2 simplifies calculation near the edges.
	var quantErrorCurr, quantErrorNext [][4]int32
	if floydSteinberg {
		quantErrorCurr = make([][4]int32, r.Dx()+2)
		quantErrorNext = make([][4]int32, r.Dx()+2)
	}

	// Loop over each source pixel.
	out := color.RGBA64{A: 0xffff}
	for y := 0; y != r.Dy(); y++ {
		for x := 0; x != r.Dx(); x++ {
			// er, eg and eb are the pixel's R,G,B values plus the
			// optional Floyd-Steinberg error.
			sr, sg, sb, sa := src.At(sp.X+x, sp.Y+y).RGBA()
			er, eg, eb, ea := int32(sr), int32(sg), int32(sb), int32(sa)
			if floydSteinberg {
				er = clamp(er + quantErrorCurr[x+1][0]/16)
				eg = clamp(eg + quantErrorCurr[x+1][1]/16)
				eb = clamp(eb + quantErrorCurr[x+1][2]/16)
				ea = clamp(ea + quantErrorCurr[x+1][3]/16)
			}

			if palette != nil {
				// Find the closest palette color in Euclidean R,G,B,A space:
				// the one that minimizes sum-squared-difference.
				// TODO(nigeltao): consider smarter algorithms.
				bestIndex, bestSum := 0, uint32(1<<32-1)
				for index, p := range palette {
					sum := sqDiff(er, p[0]) + sqDiff(eg, p[1]) + sqDiff(eb, p[2]) + sqDiff(ea, p[3])
					if sum < bestSum {
						bestIndex, bestSum = index, sum
						if sum == 0 {
							break
						}
					}
				}
				pix[y*stride+x] = byte(bestIndex)

				if !floydSteinberg {
					continue
				}
				er -= palette[bestIndex][0]
				eg -= palette[bestIndex][1]
				eb -= palette[bestIndex][2]
				ea -= palette[bestIndex][3]

			} else {
				out.R = uint16(er)
				out.G = uint16(eg)
				out.B = uint16(eb)
				out.A = uint16(ea)
				// The third argument is &out instead of out (and out is
				// declared outside of the inner loop) to avoid the implicit
				// conversion to color.Color here allocating memory in the
				// inner loop if sizeof(color.RGBA64) > sizeof(uintptr).
				dst.Set(r.Min.X+x, r.Min.Y+y, &out)

				if !floydSteinberg {
					continue
				}
				sr, sg, sb, sa = dst.At(r.Min.X+x, r.Min.Y+y).RGBA()
				er -= int32(sr)
				eg -= int32(sg)
				eb -= int32(sb)
				ea -= int32(sa)
			}

			// Propagate the Floyd-Steinberg quantization error.
			quantErrorNext[x+0][0] += er * 3
			quantErrorNext[x+0][1] += eg * 3
			quantErrorNext[x+0][2] += eb * 3
			quantErrorNext[x+0][3] += ea * 3
			quantErrorNext[x+1][0] += er * 5
			quantErrorNext[x+1][1] += eg * 5
			quantErrorNext[x+1][2] += eb * 5
			quantErrorNext[x+1][3] += ea * 5
			quantErrorNext[x+2][0] += er * 1
			quantErrorNext[x+2][1] += eg * 1
			quantErrorNext[x+2][2] += eb * 1
			quantErrorNext[x+2][3] += ea * 1
			quantErrorCurr[x+2][0] += er * 7
			quantErrorCurr[x+2][1] += eg * 7
			quantErrorCurr[x+2][2] += eb * 7
			quantErrorCurr[x+2][3] += ea * 7
		}

		// Recycle the quantization error buffers.
		if floydSteinberg {
			quantErrorCurr, quantErrorNext = quantErrorNext, quantErrorCurr
			for i := range quantErrorNext {
				quantErrorNext[i] = [4]int32{}
			}
		}
	}
}
