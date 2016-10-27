// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package png implements a PNG image decoder and encoder.
//
// The PNG specification is at http://www.w3.org/TR/PNG/.
package png

import (
	"compress/zlib"
	"encoding/binary"
	"fmt"
	"hash"
	"hash/crc32"
	"image"
	"image/color"
	"io"
)

// Color type, as per the PNG spec.
const (
	ctGrayscale      = 0
	ctTrueColor      = 2
	ctPaletted       = 3
	ctGrayscaleAlpha = 4
	ctTrueColorAlpha = 6
)

// A cb is a combination of color type and bit depth.
const (
	cbInvalid = iota
	cbG1
	cbG2
	cbG4
	cbG8
	cbGA8
	cbTC8
	cbP1
	cbP2
	cbP4
	cbP8
	cbTCA8
	cbG16
	cbGA16
	cbTC16
	cbTCA16
)

func cbPaletted(cb int) bool {
	return cbP1 <= cb && cb <= cbP8
}

// Filter type, as per the PNG spec.
const (
	ftNone    = 0
	ftSub     = 1
	ftUp      = 2
	ftAverage = 3
	ftPaeth   = 4
	nFilter   = 5
)

// Interlace type.
const (
	itNone  = 0
	itAdam7 = 1
)

// interlaceScan defines the placement and size of a pass for Adam7 interlacing.
type interlaceScan struct {
	xFactor, yFactor, xOffset, yOffset int
}

// interlacing defines Adam7 interlacing, with 7 passes of reduced images.
// See http://www.w3.org/TR/PNG/#8Interlace
var interlacing = []interlaceScan{
	{8, 8, 0, 0},
	{8, 8, 4, 0},
	{4, 8, 0, 4},
	{4, 4, 2, 0},
	{2, 4, 0, 2},
	{2, 2, 1, 0},
	{1, 2, 0, 1},
}

// Decoding stage.
// The PNG specification says that the IHDR, PLTE (if present), tRNS (if
// present), IDAT and IEND chunks must appear in that order. There may be
// multiple IDAT chunks, and IDAT chunks must be sequential (i.e. they may not
// have any other chunks between them).
// http://www.w3.org/TR/PNG/#5ChunkOrdering
const (
	dsStart = iota
	dsSeenIHDR
	dsSeenPLTE
	dsSeentRNS
	dsSeenIDAT
	dsSeenIEND
)

const pngHeader = "\x89PNG\r\n\x1a\n"

type decoder struct {
	r             io.Reader
	img           image.Image
	crc           hash.Hash32
	width, height int
	depth         int
	palette       color.Palette
	cb            int
	stage         int
	idatLength    uint32
	tmp           [3 * 256]byte
	interlace     int

	// useTransparent and transparent are used for grayscale and truecolor
	// transparency, as opposed to palette transparency.
	useTransparent bool
	transparent    [6]byte
}

// A FormatError reports that the input is not a valid PNG.
type FormatError string

func (e FormatError) Error() string { return "png: invalid format: " + string(e) }

var chunkOrderError = FormatError("chunk out of order")

// An UnsupportedError reports that the input uses a valid but unimplemented PNG feature.
type UnsupportedError string

func (e UnsupportedError) Error() string { return "png: unsupported feature: " + string(e) }

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (d *decoder) parseIHDR(length uint32) error {
	if length != 13 {
		return FormatError("bad IHDR length")
	}
	if _, err := io.ReadFull(d.r, d.tmp[:13]); err != nil {
		return err
	}
	d.crc.Write(d.tmp[:13])
	if d.tmp[10] != 0 {
		return UnsupportedError("compression method")
	}
	if d.tmp[11] != 0 {
		return UnsupportedError("filter method")
	}
	if d.tmp[12] != itNone && d.tmp[12] != itAdam7 {
		return FormatError("invalid interlace method")
	}
	d.interlace = int(d.tmp[12])
	w := int32(binary.BigEndian.Uint32(d.tmp[0:4]))
	h := int32(binary.BigEndian.Uint32(d.tmp[4:8]))
	if w <= 0 || h <= 0 {
		return FormatError("non-positive dimension")
	}
	nPixels := int64(w) * int64(h)
	if nPixels != int64(int(nPixels)) {
		return UnsupportedError("dimension overflow")
	}
	d.cb = cbInvalid
	d.depth = int(d.tmp[8])
	switch d.depth {
	case 1:
		switch d.tmp[9] {
		case ctGrayscale:
			d.cb = cbG1
		case ctPaletted:
			d.cb = cbP1
		}
	case 2:
		switch d.tmp[9] {
		case ctGrayscale:
			d.cb = cbG2
		case ctPaletted:
			d.cb = cbP2
		}
	case 4:
		switch d.tmp[9] {
		case ctGrayscale:
			d.cb = cbG4
		case ctPaletted:
			d.cb = cbP4
		}
	case 8:
		switch d.tmp[9] {
		case ctGrayscale:
			d.cb = cbG8
		case ctTrueColor:
			d.cb = cbTC8
		case ctPaletted:
			d.cb = cbP8
		case ctGrayscaleAlpha:
			d.cb = cbGA8
		case ctTrueColorAlpha:
			d.cb = cbTCA8
		}
	case 16:
		switch d.tmp[9] {
		case ctGrayscale:
			d.cb = cbG16
		case ctTrueColor:
			d.cb = cbTC16
		case ctGrayscaleAlpha:
			d.cb = cbGA16
		case ctTrueColorAlpha:
			d.cb = cbTCA16
		}
	}
	if d.cb == cbInvalid {
		return UnsupportedError(fmt.Sprintf("bit depth %d, color type %d", d.tmp[8], d.tmp[9]))
	}
	d.width, d.height = int(w), int(h)
	return d.verifyChecksum()
}

func (d *decoder) parsePLTE(length uint32) error {
	np := int(length / 3) // The number of palette entries.
	if length%3 != 0 || np <= 0 || np > 256 || np > 1<<uint(d.depth) {
		return FormatError("bad PLTE length")
	}
	n, err := io.ReadFull(d.r, d.tmp[:3*np])
	if err != nil {
		return err
	}
	d.crc.Write(d.tmp[:n])
	switch d.cb {
	case cbP1, cbP2, cbP4, cbP8:
		d.palette = make(color.Palette, 256)
		for i := 0; i < np; i++ {
			d.palette[i] = color.RGBA{d.tmp[3*i+0], d.tmp[3*i+1], d.tmp[3*i+2], 0xff}
		}
		for i := np; i < 256; i++ {
			// Initialize the rest of the palette to opaque black. The spec (section
			// 11.2.3) says that "any out-of-range pixel value found in the image data
			// is an error", but some real-world PNG files have out-of-range pixel
			// values. We fall back to opaque black, the same as libpng 1.5.13;
			// ImageMagick 6.5.7 returns an error.
			d.palette[i] = color.RGBA{0x00, 0x00, 0x00, 0xff}
		}
		d.palette = d.palette[:np]
	case cbTC8, cbTCA8, cbTC16, cbTCA16:
		// As per the PNG spec, a PLTE chunk is optional (and for practical purposes,
		// ignorable) for the ctTrueColor and ctTrueColorAlpha color types (section 4.1.2).
	default:
		return FormatError("PLTE, color type mismatch")
	}
	return d.verifyChecksum()
}

func (d *decoder) parsetRNS(length uint32) error {
	switch d.cb {
	case cbG1, cbG2, cbG4, cbG8, cbG16:
		if length != 2 {
			return FormatError("bad tRNS length")
		}
		n, err := io.ReadFull(d.r, d.tmp[:length])
		if err != nil {
			return err
		}
		d.crc.Write(d.tmp[:n])

		copy(d.transparent[:], d.tmp[:length])
		switch d.cb {
		case cbG1:
			d.transparent[1] *= 0xff
		case cbG2:
			d.transparent[1] *= 0x55
		case cbG4:
			d.transparent[1] *= 0x11
		}
		d.useTransparent = true

	case cbTC8, cbTC16:
		if length != 6 {
			return FormatError("bad tRNS length")
		}
		n, err := io.ReadFull(d.r, d.tmp[:length])
		if err != nil {
			return err
		}
		d.crc.Write(d.tmp[:n])

		copy(d.transparent[:], d.tmp[:length])
		d.useTransparent = true

	case cbP1, cbP2, cbP4, cbP8:
		if length > 256 {
			return FormatError("bad tRNS length")
		}
		n, err := io.ReadFull(d.r, d.tmp[:length])
		if err != nil {
			return err
		}
		d.crc.Write(d.tmp[:n])

		if len(d.palette) < n {
			d.palette = d.palette[:n]
		}
		for i := 0; i < n; i++ {
			rgba := d.palette[i].(color.RGBA)
			d.palette[i] = color.NRGBA{rgba.R, rgba.G, rgba.B, d.tmp[i]}
		}

	default:
		return FormatError("tRNS, color type mismatch")
	}
	return d.verifyChecksum()
}

// Read presents one or more IDAT chunks as one continuous stream (minus the
// intermediate chunk headers and footers). If the PNG data looked like:
//   ... len0 IDAT xxx crc0 len1 IDAT yy crc1 len2 IEND crc2
// then this reader presents xxxyy. For well-formed PNG data, the decoder state
// immediately before the first Read call is that d.r is positioned between the
// first IDAT and xxx, and the decoder state immediately after the last Read
// call is that d.r is positioned between yy and crc1.
func (d *decoder) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	for d.idatLength == 0 {
		// We have exhausted an IDAT chunk. Verify the checksum of that chunk.
		if err := d.verifyChecksum(); err != nil {
			return 0, err
		}
		// Read the length and chunk type of the next chunk, and check that
		// it is an IDAT chunk.
		if _, err := io.ReadFull(d.r, d.tmp[:8]); err != nil {
			return 0, err
		}
		d.idatLength = binary.BigEndian.Uint32(d.tmp[:4])
		if string(d.tmp[4:8]) != "IDAT" {
			return 0, FormatError("not enough pixel data")
		}
		d.crc.Reset()
		d.crc.Write(d.tmp[4:8])
	}
	if int(d.idatLength) < 0 {
		return 0, UnsupportedError("IDAT chunk length overflow")
	}
	n, err := d.r.Read(p[:min(len(p), int(d.idatLength))])
	d.crc.Write(p[:n])
	d.idatLength -= uint32(n)
	return n, err
}

// decode decodes the IDAT data into an image.
func (d *decoder) decode() (image.Image, error) {
	r, err := zlib.NewReader(d)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	var img image.Image
	if d.interlace == itNone {
		img, err = d.readImagePass(r, 0, false)
		if err != nil {
			return nil, err
		}
	} else if d.interlace == itAdam7 {
		// Allocate a blank image of the full size.
		img, err = d.readImagePass(nil, 0, true)
		if err != nil {
			return nil, err
		}
		for pass := 0; pass < 7; pass++ {
			imagePass, err := d.readImagePass(r, pass, false)
			if err != nil {
				return nil, err
			}
			if imagePass != nil {
				d.mergePassInto(img, imagePass, pass)
			}
		}
	}

	// Check for EOF, to verify the zlib checksum.
	n := 0
	for i := 0; n == 0 && err == nil; i++ {
		if i == 100 {
			return nil, io.ErrNoProgress
		}
		n, err = r.Read(d.tmp[:1])
	}
	if err != nil && err != io.EOF {
		return nil, FormatError(err.Error())
	}
	if n != 0 || d.idatLength != 0 {
		return nil, FormatError("too much pixel data")
	}

	return img, nil
}

// readImagePass reads a single image pass, sized according to the pass number.
func (d *decoder) readImagePass(r io.Reader, pass int, allocateOnly bool) (image.Image, error) {
	bitsPerPixel := 0
	pixOffset := 0
	var (
		gray     *image.Gray
		rgba     *image.RGBA
		paletted *image.Paletted
		nrgba    *image.NRGBA
		gray16   *image.Gray16
		rgba64   *image.RGBA64
		nrgba64  *image.NRGBA64
		img      image.Image
	)
	width, height := d.width, d.height
	if d.interlace == itAdam7 && !allocateOnly {
		p := interlacing[pass]
		// Add the multiplication factor and subtract one, effectively rounding up.
		width = (width - p.xOffset + p.xFactor - 1) / p.xFactor
		height = (height - p.yOffset + p.yFactor - 1) / p.yFactor
		// A PNG image can't have zero width or height, but for an interlaced
		// image, an individual pass might have zero width or height. If so, we
		// shouldn't even read a per-row filter type byte, so return early.
		if width == 0 || height == 0 {
			return nil, nil
		}
	}
	switch d.cb {
	case cbG1, cbG2, cbG4, cbG8:
		bitsPerPixel = d.depth
		if d.useTransparent {
			nrgba = image.NewNRGBA(image.Rect(0, 0, width, height))
			img = nrgba
		} else {
			gray = image.NewGray(image.Rect(0, 0, width, height))
			img = gray
		}
	case cbGA8:
		bitsPerPixel = 16
		nrgba = image.NewNRGBA(image.Rect(0, 0, width, height))
		img = nrgba
	case cbTC8:
		bitsPerPixel = 24
		if d.useTransparent {
			nrgba = image.NewNRGBA(image.Rect(0, 0, width, height))
			img = nrgba
		} else {
			rgba = image.NewRGBA(image.Rect(0, 0, width, height))
			img = rgba
		}
	case cbP1, cbP2, cbP4, cbP8:
		bitsPerPixel = d.depth
		paletted = image.NewPaletted(image.Rect(0, 0, width, height), d.palette)
		img = paletted
	case cbTCA8:
		bitsPerPixel = 32
		nrgba = image.NewNRGBA(image.Rect(0, 0, width, height))
		img = nrgba
	case cbG16:
		bitsPerPixel = 16
		if d.useTransparent {
			nrgba64 = image.NewNRGBA64(image.Rect(0, 0, width, height))
			img = nrgba64
		} else {
			gray16 = image.NewGray16(image.Rect(0, 0, width, height))
			img = gray16
		}
	case cbGA16:
		bitsPerPixel = 32
		nrgba64 = image.NewNRGBA64(image.Rect(0, 0, width, height))
		img = nrgba64
	case cbTC16:
		bitsPerPixel = 48
		if d.useTransparent {
			nrgba64 = image.NewNRGBA64(image.Rect(0, 0, width, height))
			img = nrgba64
		} else {
			rgba64 = image.NewRGBA64(image.Rect(0, 0, width, height))
			img = rgba64
		}
	case cbTCA16:
		bitsPerPixel = 64
		nrgba64 = image.NewNRGBA64(image.Rect(0, 0, width, height))
		img = nrgba64
	}
	if allocateOnly {
		return img, nil
	}
	bytesPerPixel := (bitsPerPixel + 7) / 8

	// The +1 is for the per-row filter type, which is at cr[0].
	rowSize := 1 + (bitsPerPixel*width+7)/8
	// cr and pr are the bytes for the current and previous row.
	cr := make([]uint8, rowSize)
	pr := make([]uint8, rowSize)

	for y := 0; y < height; y++ {
		// Read the decompressed bytes.
		_, err := io.ReadFull(r, cr)
		if err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				return nil, FormatError("not enough pixel data")
			}
			return nil, err
		}

		// Apply the filter.
		cdat := cr[1:]
		pdat := pr[1:]
		switch cr[0] {
		case ftNone:
			// No-op.
		case ftSub:
			for i := bytesPerPixel; i < len(cdat); i++ {
				cdat[i] += cdat[i-bytesPerPixel]
			}
		case ftUp:
			for i, p := range pdat {
				cdat[i] += p
			}
		case ftAverage:
			// The first column has no column to the left of it, so it is a
			// special case. We know that the first column exists because we
			// check above that width != 0, and so len(cdat) != 0.
			for i := 0; i < bytesPerPixel; i++ {
				cdat[i] += pdat[i] / 2
			}
			for i := bytesPerPixel; i < len(cdat); i++ {
				cdat[i] += uint8((int(cdat[i-bytesPerPixel]) + int(pdat[i])) / 2)
			}
		case ftPaeth:
			filterPaeth(cdat, pdat, bytesPerPixel)
		default:
			return nil, FormatError("bad filter type")
		}

		// Convert from bytes to colors.
		switch d.cb {
		case cbG1:
			if d.useTransparent {
				ty := d.transparent[1]
				for x := 0; x < width; x += 8 {
					b := cdat[x/8]
					for x2 := 0; x2 < 8 && x+x2 < width; x2++ {
						ycol := (b >> 7) * 0xff
						acol := uint8(0xff)
						if ycol == ty {
							acol = 0x00
						}
						nrgba.SetNRGBA(x+x2, y, color.NRGBA{ycol, ycol, ycol, acol})
						b <<= 1
					}
				}
			} else {
				for x := 0; x < width; x += 8 {
					b := cdat[x/8]
					for x2 := 0; x2 < 8 && x+x2 < width; x2++ {
						gray.SetGray(x+x2, y, color.Gray{(b >> 7) * 0xff})
						b <<= 1
					}
				}
			}
		case cbG2:
			if d.useTransparent {
				ty := d.transparent[1]
				for x := 0; x < width; x += 4 {
					b := cdat[x/4]
					for x2 := 0; x2 < 4 && x+x2 < width; x2++ {
						ycol := (b >> 6) * 0x55
						acol := uint8(0xff)
						if ycol == ty {
							acol = 0x00
						}
						nrgba.SetNRGBA(x+x2, y, color.NRGBA{ycol, ycol, ycol, acol})
						b <<= 2
					}
				}
			} else {
				for x := 0; x < width; x += 4 {
					b := cdat[x/4]
					for x2 := 0; x2 < 4 && x+x2 < width; x2++ {
						gray.SetGray(x+x2, y, color.Gray{(b >> 6) * 0x55})
						b <<= 2
					}
				}
			}
		case cbG4:
			if d.useTransparent {
				ty := d.transparent[1]
				for x := 0; x < width; x += 2 {
					b := cdat[x/2]
					for x2 := 0; x2 < 2 && x+x2 < width; x2++ {
						ycol := (b >> 4) * 0x11
						acol := uint8(0xff)
						if ycol == ty {
							acol = 0x00
						}
						nrgba.SetNRGBA(x+x2, y, color.NRGBA{ycol, ycol, ycol, acol})
						b <<= 4
					}
				}
			} else {
				for x := 0; x < width; x += 2 {
					b := cdat[x/2]
					for x2 := 0; x2 < 2 && x+x2 < width; x2++ {
						gray.SetGray(x+x2, y, color.Gray{(b >> 4) * 0x11})
						b <<= 4
					}
				}
			}
		case cbG8:
			copy(gray.Pix[pixOffset:], cdat)
			pixOffset += gray.Stride
		case cbGA8:
			for x := 0; x < width; x++ {
				ycol := cdat[2*x+0]
				nrgba.SetNRGBA(x, y, color.NRGBA{ycol, ycol, ycol, cdat[2*x+1]})
			}
		case cbTC8:
			if d.useTransparent {
				pix, i, j := nrgba.Pix, pixOffset, 0
				tr, tg, tb := d.transparent[1], d.transparent[3], d.transparent[5]
				for x := 0; x < width; x++ {
					r := cdat[j+0]
					g := cdat[j+1]
					b := cdat[j+2]
					a := uint8(0xff)
					if r == tr && g == tg && b == tb {
						a = 0x00
					}
					pix[i+0] = r
					pix[i+1] = g
					pix[i+2] = b
					pix[i+3] = a
					i += 4
					j += 3
				}
				pixOffset += nrgba.Stride
			} else {
				pix, i, j := rgba.Pix, pixOffset, 0
				for x := 0; x < width; x++ {
					pix[i+0] = cdat[j+0]
					pix[i+1] = cdat[j+1]
					pix[i+2] = cdat[j+2]
					pix[i+3] = 0xff
					i += 4
					j += 3
				}
				pixOffset += rgba.Stride
			}
		case cbP1:
			for x := 0; x < width; x += 8 {
				b := cdat[x/8]
				for x2 := 0; x2 < 8 && x+x2 < width; x2++ {
					idx := b >> 7
					if len(paletted.Palette) <= int(idx) {
						paletted.Palette = paletted.Palette[:int(idx)+1]
					}
					paletted.SetColorIndex(x+x2, y, idx)
					b <<= 1
				}
			}
		case cbP2:
			for x := 0; x < width; x += 4 {
				b := cdat[x/4]
				for x2 := 0; x2 < 4 && x+x2 < width; x2++ {
					idx := b >> 6
					if len(paletted.Palette) <= int(idx) {
						paletted.Palette = paletted.Palette[:int(idx)+1]
					}
					paletted.SetColorIndex(x+x2, y, idx)
					b <<= 2
				}
			}
		case cbP4:
			for x := 0; x < width; x += 2 {
				b := cdat[x/2]
				for x2 := 0; x2 < 2 && x+x2 < width; x2++ {
					idx := b >> 4
					if len(paletted.Palette) <= int(idx) {
						paletted.Palette = paletted.Palette[:int(idx)+1]
					}
					paletted.SetColorIndex(x+x2, y, idx)
					b <<= 4
				}
			}
		case cbP8:
			if len(paletted.Palette) != 255 {
				for x := 0; x < width; x++ {
					if len(paletted.Palette) <= int(cdat[x]) {
						paletted.Palette = paletted.Palette[:int(cdat[x])+1]
					}
				}
			}
			copy(paletted.Pix[pixOffset:], cdat)
			pixOffset += paletted.Stride
		case cbTCA8:
			copy(nrgba.Pix[pixOffset:], cdat)
			pixOffset += nrgba.Stride
		case cbG16:
			if d.useTransparent {
				ty := uint16(d.transparent[0])<<8 | uint16(d.transparent[1])
				for x := 0; x < width; x++ {
					ycol := uint16(cdat[2*x+0])<<8 | uint16(cdat[2*x+1])
					acol := uint16(0xffff)
					if ycol == ty {
						acol = 0x0000
					}
					nrgba64.SetNRGBA64(x, y, color.NRGBA64{ycol, ycol, ycol, acol})
				}
			} else {
				for x := 0; x < width; x++ {
					ycol := uint16(cdat[2*x+0])<<8 | uint16(cdat[2*x+1])
					gray16.SetGray16(x, y, color.Gray16{ycol})
				}
			}
		case cbGA16:
			for x := 0; x < width; x++ {
				ycol := uint16(cdat[4*x+0])<<8 | uint16(cdat[4*x+1])
				acol := uint16(cdat[4*x+2])<<8 | uint16(cdat[4*x+3])
				nrgba64.SetNRGBA64(x, y, color.NRGBA64{ycol, ycol, ycol, acol})
			}
		case cbTC16:
			if d.useTransparent {
				tr := uint16(d.transparent[0])<<8 | uint16(d.transparent[1])
				tg := uint16(d.transparent[2])<<8 | uint16(d.transparent[3])
				tb := uint16(d.transparent[4])<<8 | uint16(d.transparent[5])
				for x := 0; x < width; x++ {
					rcol := uint16(cdat[6*x+0])<<8 | uint16(cdat[6*x+1])
					gcol := uint16(cdat[6*x+2])<<8 | uint16(cdat[6*x+3])
					bcol := uint16(cdat[6*x+4])<<8 | uint16(cdat[6*x+5])
					acol := uint16(0xffff)
					if rcol == tr && gcol == tg && bcol == tb {
						acol = 0x0000
					}
					nrgba64.SetNRGBA64(x, y, color.NRGBA64{rcol, gcol, bcol, acol})
				}
			} else {
				for x := 0; x < width; x++ {
					rcol := uint16(cdat[6*x+0])<<8 | uint16(cdat[6*x+1])
					gcol := uint16(cdat[6*x+2])<<8 | uint16(cdat[6*x+3])
					bcol := uint16(cdat[6*x+4])<<8 | uint16(cdat[6*x+5])
					rgba64.SetRGBA64(x, y, color.RGBA64{rcol, gcol, bcol, 0xffff})
				}
			}
		case cbTCA16:
			for x := 0; x < width; x++ {
				rcol := uint16(cdat[8*x+0])<<8 | uint16(cdat[8*x+1])
				gcol := uint16(cdat[8*x+2])<<8 | uint16(cdat[8*x+3])
				bcol := uint16(cdat[8*x+4])<<8 | uint16(cdat[8*x+5])
				acol := uint16(cdat[8*x+6])<<8 | uint16(cdat[8*x+7])
				nrgba64.SetNRGBA64(x, y, color.NRGBA64{rcol, gcol, bcol, acol})
			}
		}

		// The current row for y is the previous row for y+1.
		pr, cr = cr, pr
	}

	return img, nil
}

// mergePassInto merges a single pass into a full sized image.
func (d *decoder) mergePassInto(dst image.Image, src image.Image, pass int) {
	p := interlacing[pass]
	var (
		srcPix        []uint8
		dstPix        []uint8
		stride        int
		rect          image.Rectangle
		bytesPerPixel int
	)
	switch target := dst.(type) {
	case *image.Alpha:
		srcPix = src.(*image.Alpha).Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 1
	case *image.Alpha16:
		srcPix = src.(*image.Alpha16).Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 2
	case *image.Gray:
		srcPix = src.(*image.Gray).Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 1
	case *image.Gray16:
		srcPix = src.(*image.Gray16).Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 2
	case *image.NRGBA:
		srcPix = src.(*image.NRGBA).Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 4
	case *image.NRGBA64:
		srcPix = src.(*image.NRGBA64).Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 8
	case *image.Paletted:
		srcPix = src.(*image.Paletted).Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 1
	case *image.RGBA:
		srcPix = src.(*image.RGBA).Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 4
	case *image.RGBA64:
		srcPix = src.(*image.RGBA64).Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 8
	}
	s, bounds := 0, src.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		dBase := (y*p.yFactor+p.yOffset-rect.Min.Y)*stride + (p.xOffset-rect.Min.X)*bytesPerPixel
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			d := dBase + x*p.xFactor*bytesPerPixel
			copy(dstPix[d:], srcPix[s:s+bytesPerPixel])
			s += bytesPerPixel
		}
	}
}

func (d *decoder) parseIDAT(length uint32) (err error) {
	d.idatLength = length
	d.img, err = d.decode()
	if err != nil {
		return err
	}
	return d.verifyChecksum()
}

func (d *decoder) parseIEND(length uint32) error {
	if length != 0 {
		return FormatError("bad IEND length")
	}
	return d.verifyChecksum()
}

func (d *decoder) parseChunk() error {
	// Read the length and chunk type.
	n, err := io.ReadFull(d.r, d.tmp[:8])
	if err != nil {
		return err
	}
	length := binary.BigEndian.Uint32(d.tmp[:4])
	d.crc.Reset()
	d.crc.Write(d.tmp[4:8])

	// Read the chunk data.
	switch string(d.tmp[4:8]) {
	case "IHDR":
		if d.stage != dsStart {
			return chunkOrderError
		}
		d.stage = dsSeenIHDR
		return d.parseIHDR(length)
	case "PLTE":
		if d.stage != dsSeenIHDR {
			return chunkOrderError
		}
		d.stage = dsSeenPLTE
		return d.parsePLTE(length)
	case "tRNS":
		if cbPaletted(d.cb) {
			if d.stage != dsSeenPLTE {
				return chunkOrderError
			}
		} else if d.stage != dsSeenIHDR {
			return chunkOrderError
		}
		d.stage = dsSeentRNS
		return d.parsetRNS(length)
	case "IDAT":
		if d.stage < dsSeenIHDR || d.stage > dsSeenIDAT || (d.stage == dsSeenIHDR && cbPaletted(d.cb)) {
			return chunkOrderError
		} else if d.stage == dsSeenIDAT {
			// Ignore trailing zero-length or garbage IDAT chunks.
			//
			// This does not affect valid PNG images that contain multiple IDAT
			// chunks, since the first call to parseIDAT below will consume all
			// consecutive IDAT chunks required for decoding the image.
			break
		}
		d.stage = dsSeenIDAT
		return d.parseIDAT(length)
	case "IEND":
		if d.stage != dsSeenIDAT {
			return chunkOrderError
		}
		d.stage = dsSeenIEND
		return d.parseIEND(length)
	}
	if length > 0x7fffffff {
		return FormatError(fmt.Sprintf("Bad chunk length: %d", length))
	}
	// Ignore this chunk (of a known length).
	var ignored [4096]byte
	for length > 0 {
		n, err = io.ReadFull(d.r, ignored[:min(len(ignored), int(length))])
		if err != nil {
			return err
		}
		d.crc.Write(ignored[:n])
		length -= uint32(n)
	}
	return d.verifyChecksum()
}

func (d *decoder) verifyChecksum() error {
	if _, err := io.ReadFull(d.r, d.tmp[:4]); err != nil {
		return err
	}
	if binary.BigEndian.Uint32(d.tmp[:4]) != d.crc.Sum32() {
		return FormatError("invalid checksum")
	}
	return nil
}

func (d *decoder) checkHeader() error {
	_, err := io.ReadFull(d.r, d.tmp[:len(pngHeader)])
	if err != nil {
		return err
	}
	if string(d.tmp[:len(pngHeader)]) != pngHeader {
		return FormatError("not a PNG file")
	}
	return nil
}

// Decode reads a PNG image from r and returns it as an image.Image.
// The type of Image returned depends on the PNG contents.
func Decode(r io.Reader) (image.Image, error) {
	d := &decoder{
		r:   r,
		crc: crc32.NewIEEE(),
	}
	if err := d.checkHeader(); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return nil, err
	}
	for d.stage != dsSeenIEND {
		if err := d.parseChunk(); err != nil {
			if err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return nil, err
		}
	}
	return d.img, nil
}

// DecodeConfig returns the color model and dimensions of a PNG image without
// decoding the entire image.
func DecodeConfig(r io.Reader) (image.Config, error) {
	d := &decoder{
		r:   r,
		crc: crc32.NewIEEE(),
	}
	if err := d.checkHeader(); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return image.Config{}, err
	}
	for {
		if err := d.parseChunk(); err != nil {
			if err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return image.Config{}, err
		}
		paletted := cbPaletted(d.cb)
		if d.stage == dsSeenIHDR && !paletted {
			break
		}
		if d.stage == dsSeenPLTE && paletted {
			break
		}
	}
	var cm color.Model
	switch d.cb {
	case cbG1, cbG2, cbG4, cbG8:
		cm = color.GrayModel
	case cbGA8:
		cm = color.NRGBAModel
	case cbTC8:
		cm = color.RGBAModel
	case cbP1, cbP2, cbP4, cbP8:
		cm = d.palette
	case cbTCA8:
		cm = color.NRGBAModel
	case cbG16:
		cm = color.Gray16Model
	case cbGA16:
		cm = color.NRGBA64Model
	case cbTC16:
		cm = color.RGBA64Model
	case cbTCA16:
		cm = color.NRGBA64Model
	}
	return image.Config{
		ColorModel: cm,
		Width:      d.width,
		Height:     d.height,
	}, nil
}

func init() {
	image.RegisterFormat("png", pngHeader, Decode, DecodeConfig)
}
