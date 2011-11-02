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

// Filter type, as per the PNG spec.
const (
	ftNone    = 0
	ftSub     = 1
	ftUp      = 2
	ftAverage = 3
	ftPaeth   = 4
	nFilter   = 5
)

// Decoding stage.
// The PNG specification says that the IHDR, PLTE (if present), IDAT and IEND
// chunks must appear in that order. There may be multiple IDAT chunks, and
// IDAT chunks must be sequential (i.e. they may not have any other chunks
// between them).
// http://www.w3.org/TR/PNG/#5ChunkOrdering
const (
	dsStart = iota
	dsSeenIHDR
	dsSeenPLTE
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
}

// A FormatError reports that the input is not a valid PNG.
type FormatError string

func (e FormatError) Error() string { return "png: invalid format: " + string(e) }

var chunkOrderError = FormatError("chunk out of order")

// An UnsupportedError reports that the input uses a valid but unimplemented PNG feature.
type UnsupportedError string

func (e UnsupportedError) Error() string { return "png: unsupported feature: " + string(e) }

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

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
	if d.tmp[10] != 0 || d.tmp[11] != 0 || d.tmp[12] != 0 {
		return UnsupportedError("compression, filter or interlace method")
	}
	w := int32(binary.BigEndian.Uint32(d.tmp[0:4]))
	h := int32(binary.BigEndian.Uint32(d.tmp[4:8]))
	if w < 0 || h < 0 {
		return FormatError("negative dimension")
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
		d.palette = color.Palette(make([]color.Color, np))
		for i := 0; i < np; i++ {
			d.palette[i] = color.RGBA{d.tmp[3*i+0], d.tmp[3*i+1], d.tmp[3*i+2], 0xff}
		}
	case cbTC8, cbTCA8, cbTC16, cbTCA16:
		// As per the PNG spec, a PLTE chunk is optional (and for practical purposes,
		// ignorable) for the ctTrueColor and ctTrueColorAlpha color types (section 4.1.2).
	default:
		return FormatError("PLTE, color type mismatch")
	}
	return d.verifyChecksum()
}

func (d *decoder) parsetRNS(length uint32) error {
	if length > 256 {
		return FormatError("bad tRNS length")
	}
	n, err := io.ReadFull(d.r, d.tmp[:length])
	if err != nil {
		return err
	}
	d.crc.Write(d.tmp[:n])
	switch d.cb {
	case cbG8, cbG16:
		return UnsupportedError("grayscale transparency")
	case cbTC8, cbTC16:
		return UnsupportedError("truecolor transparency")
	case cbP1, cbP2, cbP4, cbP8:
		if n > len(d.palette) {
			return FormatError("bad tRNS length")
		}
		for i := 0; i < n; i++ {
			rgba := d.palette[i].(color.RGBA)
			d.palette[i] = color.RGBA{rgba.R, rgba.G, rgba.B, d.tmp[i]}
		}
	case cbGA8, cbGA16, cbTCA8, cbTCA16:
		return FormatError("tRNS, color type mismatch")
	}
	return d.verifyChecksum()
}

// The Paeth filter function, as per the PNG specification.
func paeth(a, b, c uint8) uint8 {
	p := int(a) + int(b) - int(c)
	pa := abs(p - int(a))
	pb := abs(p - int(b))
	pc := abs(p - int(c))
	if pa <= pb && pa <= pc {
		return a
	} else if pb <= pc {
		return b
	}
	return c
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
	bitsPerPixel := 0
	maxPalette := uint8(0)
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
	switch d.cb {
	case cbG1, cbG2, cbG4, cbG8:
		bitsPerPixel = d.depth
		gray = image.NewGray(image.Rect(0, 0, d.width, d.height))
		img = gray
	case cbGA8:
		bitsPerPixel = 16
		nrgba = image.NewNRGBA(image.Rect(0, 0, d.width, d.height))
		img = nrgba
	case cbTC8:
		bitsPerPixel = 24
		rgba = image.NewRGBA(image.Rect(0, 0, d.width, d.height))
		img = rgba
	case cbP1, cbP2, cbP4, cbP8:
		bitsPerPixel = d.depth
		paletted = image.NewPaletted(image.Rect(0, 0, d.width, d.height), d.palette)
		img = paletted
		maxPalette = uint8(len(d.palette) - 1)
	case cbTCA8:
		bitsPerPixel = 32
		nrgba = image.NewNRGBA(image.Rect(0, 0, d.width, d.height))
		img = nrgba
	case cbG16:
		bitsPerPixel = 16
		gray16 = image.NewGray16(image.Rect(0, 0, d.width, d.height))
		img = gray16
	case cbGA16:
		bitsPerPixel = 32
		nrgba64 = image.NewNRGBA64(image.Rect(0, 0, d.width, d.height))
		img = nrgba64
	case cbTC16:
		bitsPerPixel = 48
		rgba64 = image.NewRGBA64(image.Rect(0, 0, d.width, d.height))
		img = rgba64
	case cbTCA16:
		bitsPerPixel = 64
		nrgba64 = image.NewNRGBA64(image.Rect(0, 0, d.width, d.height))
		img = nrgba64
	}
	bytesPerPixel := (bitsPerPixel + 7) / 8

	// cr and pr are the bytes for the current and previous row.
	// The +1 is for the per-row filter type, which is at cr[0].
	cr := make([]uint8, 1+(bitsPerPixel*d.width+7)/8)
	pr := make([]uint8, 1+(bitsPerPixel*d.width+7)/8)

	for y := 0; y < d.height; y++ {
		// Read the decompressed bytes.
		_, err := io.ReadFull(r, cr)
		if err != nil {
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
			for i := 0; i < len(cdat); i++ {
				cdat[i] += pdat[i]
			}
		case ftAverage:
			for i := 0; i < bytesPerPixel; i++ {
				cdat[i] += pdat[i] / 2
			}
			for i := bytesPerPixel; i < len(cdat); i++ {
				cdat[i] += uint8((int(cdat[i-bytesPerPixel]) + int(pdat[i])) / 2)
			}
		case ftPaeth:
			for i := 0; i < bytesPerPixel; i++ {
				cdat[i] += paeth(0, pdat[i], 0)
			}
			for i := bytesPerPixel; i < len(cdat); i++ {
				cdat[i] += paeth(cdat[i-bytesPerPixel], pdat[i], pdat[i-bytesPerPixel])
			}
		default:
			return nil, FormatError("bad filter type")
		}

		// Convert from bytes to colors.
		switch d.cb {
		case cbG1:
			for x := 0; x < d.width; x += 8 {
				b := cdat[x/8]
				for x2 := 0; x2 < 8 && x+x2 < d.width; x2++ {
					gray.SetGray(x+x2, y, color.Gray{(b >> 7) * 0xff})
					b <<= 1
				}
			}
		case cbG2:
			for x := 0; x < d.width; x += 4 {
				b := cdat[x/4]
				for x2 := 0; x2 < 4 && x+x2 < d.width; x2++ {
					gray.SetGray(x+x2, y, color.Gray{(b >> 6) * 0x55})
					b <<= 2
				}
			}
		case cbG4:
			for x := 0; x < d.width; x += 2 {
				b := cdat[x/2]
				for x2 := 0; x2 < 2 && x+x2 < d.width; x2++ {
					gray.SetGray(x+x2, y, color.Gray{(b >> 4) * 0x11})
					b <<= 4
				}
			}
		case cbG8:
			for x := 0; x < d.width; x++ {
				gray.SetGray(x, y, color.Gray{cdat[x]})
			}
		case cbGA8:
			for x := 0; x < d.width; x++ {
				ycol := cdat[2*x+0]
				nrgba.SetNRGBA(x, y, color.NRGBA{ycol, ycol, ycol, cdat[2*x+1]})
			}
		case cbTC8:
			for x := 0; x < d.width; x++ {
				rgba.SetRGBA(x, y, color.RGBA{cdat[3*x+0], cdat[3*x+1], cdat[3*x+2], 0xff})
			}
		case cbP1:
			for x := 0; x < d.width; x += 8 {
				b := cdat[x/8]
				for x2 := 0; x2 < 8 && x+x2 < d.width; x2++ {
					idx := b >> 7
					if idx > maxPalette {
						return nil, FormatError("palette index out of range")
					}
					paletted.SetColorIndex(x+x2, y, idx)
					b <<= 1
				}
			}
		case cbP2:
			for x := 0; x < d.width; x += 4 {
				b := cdat[x/4]
				for x2 := 0; x2 < 4 && x+x2 < d.width; x2++ {
					idx := b >> 6
					if idx > maxPalette {
						return nil, FormatError("palette index out of range")
					}
					paletted.SetColorIndex(x+x2, y, idx)
					b <<= 2
				}
			}
		case cbP4:
			for x := 0; x < d.width; x += 2 {
				b := cdat[x/2]
				for x2 := 0; x2 < 2 && x+x2 < d.width; x2++ {
					idx := b >> 4
					if idx > maxPalette {
						return nil, FormatError("palette index out of range")
					}
					paletted.SetColorIndex(x+x2, y, idx)
					b <<= 4
				}
			}
		case cbP8:
			for x := 0; x < d.width; x++ {
				if cdat[x] > maxPalette {
					return nil, FormatError("palette index out of range")
				}
				paletted.SetColorIndex(x, y, cdat[x])
			}
		case cbTCA8:
			for x := 0; x < d.width; x++ {
				nrgba.SetNRGBA(x, y, color.NRGBA{cdat[4*x+0], cdat[4*x+1], cdat[4*x+2], cdat[4*x+3]})
			}
		case cbG16:
			for x := 0; x < d.width; x++ {
				ycol := uint16(cdat[2*x+0])<<8 | uint16(cdat[2*x+1])
				gray16.SetGray16(x, y, color.Gray16{ycol})
			}
		case cbGA16:
			for x := 0; x < d.width; x++ {
				ycol := uint16(cdat[4*x+0])<<8 | uint16(cdat[4*x+1])
				acol := uint16(cdat[4*x+2])<<8 | uint16(cdat[4*x+3])
				nrgba64.SetNRGBA64(x, y, color.NRGBA64{ycol, ycol, ycol, acol})
			}
		case cbTC16:
			for x := 0; x < d.width; x++ {
				rcol := uint16(cdat[6*x+0])<<8 | uint16(cdat[6*x+1])
				gcol := uint16(cdat[6*x+2])<<8 | uint16(cdat[6*x+3])
				bcol := uint16(cdat[6*x+4])<<8 | uint16(cdat[6*x+5])
				rgba64.SetRGBA64(x, y, color.RGBA64{rcol, gcol, bcol, 0xffff})
			}
		case cbTCA16:
			for x := 0; x < d.width; x++ {
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

	// Check for EOF, to verify the zlib checksum.
	n, err := r.Read(pr[:1])
	if err != io.EOF {
		return nil, FormatError(err.Error())
	}
	if n != 0 || d.idatLength != 0 {
		return nil, FormatError("too much pixel data")
	}

	return img, nil
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
		if d.stage != dsSeenPLTE {
			return chunkOrderError
		}
		return d.parsetRNS(length)
	case "IDAT":
		if d.stage < dsSeenIHDR || d.stage > dsSeenIDAT || (d.cb == cbP8 && d.stage == dsSeenIHDR) {
			return chunkOrderError
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
		if d.stage == dsSeenIHDR && d.cb != cbP8 {
			break
		}
		if d.stage == dsSeenPLTE && d.cb == cbP8 {
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
	return image.Config{cm, d.width, d.height}, nil
}

func init() {
	image.RegisterFormat("png", pngHeader, Decode, DecodeConfig)
}
