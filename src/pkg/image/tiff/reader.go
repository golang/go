// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tiff implements a TIFF image decoder.
//
// The TIFF specification is at http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf
package tiff

import (
	"compress/lzw"
	"compress/zlib"
	"encoding/binary"
	"image"
	"image/color"
	"io"
	"io/ioutil"
	"os"
)

// A FormatError reports that the input is not a valid TIFF image.
type FormatError string

func (e FormatError) String() string {
	return "tiff: invalid format: " + string(e)
}

// An UnsupportedError reports that the input uses a valid but
// unimplemented feature.
type UnsupportedError string

func (e UnsupportedError) String() string {
	return "tiff: unsupported feature: " + string(e)
}

// An InternalError reports that an internal error was encountered.
type InternalError string

func (e InternalError) String() string {
	return "tiff: internal error: " + string(e)
}

type decoder struct {
	r         io.ReaderAt
	byteOrder binary.ByteOrder
	config    image.Config
	mode      imageMode
	features  map[int][]uint
	palette   []color.Color

	buf   []byte
	off   int    // Current offset in buf.
	v     uint32 // Buffer value for reading with arbitrary bit depths.
	nbits uint   // Remaining number of bits in v.
}

// firstVal returns the first uint of the features entry with the given tag,
// or 0 if the tag does not exist.
func (d *decoder) firstVal(tag int) uint {
	f := d.features[tag]
	if len(f) == 0 {
		return 0
	}
	return f[0]
}

// ifdUint decodes the IFD entry in p, which must be of the Byte, Short
// or Long type, and returns the decoded uint values.
func (d *decoder) ifdUint(p []byte) (u []uint, err os.Error) {
	var raw []byte
	datatype := d.byteOrder.Uint16(p[2:4])
	count := d.byteOrder.Uint32(p[4:8])
	if datalen := lengths[datatype] * count; datalen > 4 {
		// The IFD contains a pointer to the real value.
		raw = make([]byte, datalen)
		_, err = d.r.ReadAt(raw, int64(d.byteOrder.Uint32(p[8:12])))
	} else {
		raw = p[8 : 8+datalen]
	}
	if err != nil {
		return nil, err
	}

	u = make([]uint, count)
	switch datatype {
	case dtByte:
		for i := uint32(0); i < count; i++ {
			u[i] = uint(raw[i])
		}
	case dtShort:
		for i := uint32(0); i < count; i++ {
			u[i] = uint(d.byteOrder.Uint16(raw[2*i : 2*(i+1)]))
		}
	case dtLong:
		for i := uint32(0); i < count; i++ {
			u[i] = uint(d.byteOrder.Uint32(raw[4*i : 4*(i+1)]))
		}
	default:
		return nil, UnsupportedError("data type")
	}
	return u, nil
}

// parseIFD decides whether the the IFD entry in p is "interesting" and
// stows away the data in the decoder.
func (d *decoder) parseIFD(p []byte) os.Error {
	tag := d.byteOrder.Uint16(p[0:2])
	switch tag {
	case tBitsPerSample,
		tExtraSamples,
		tPhotometricInterpretation,
		tCompression,
		tPredictor,
		tStripOffsets,
		tStripByteCounts,
		tRowsPerStrip,
		tImageLength,
		tImageWidth:
		val, err := d.ifdUint(p)
		if err != nil {
			return err
		}
		d.features[int(tag)] = val
	case tColorMap:
		val, err := d.ifdUint(p)
		if err != nil {
			return err
		}
		numcolors := len(val) / 3
		if len(val)%3 != 0 || numcolors <= 0 || numcolors > 256 {
			return FormatError("bad ColorMap length")
		}
		d.palette = make([]color.Color, numcolors)
		for i := 0; i < numcolors; i++ {
			d.palette[i] = color.RGBA64{
				uint16(val[i]),
				uint16(val[i+numcolors]),
				uint16(val[i+2*numcolors]),
				0xffff,
			}
		}
	case tSampleFormat:
		// Page 27 of the spec: If the SampleFormat is present and
		// the value is not 1 [= unsigned integer data], a Baseline
		// TIFF reader that cannot handle the SampleFormat value
		// must terminate the import process gracefully.
		val, err := d.ifdUint(p)
		if err != nil {
			return err
		}
		for _, v := range val {
			if v != 1 {
				return UnsupportedError("sample format")
			}
		}
	}
	return nil
}

// readBits reads n bits from the internal buffer starting at the current offset.
func (d *decoder) readBits(n uint) uint32 {
	for d.nbits < n {
		d.v <<= 8
		d.v |= uint32(d.buf[d.off])
		d.off++
		d.nbits += 8
	}
	d.nbits -= n
	rv := d.v >> d.nbits
	d.v &^= rv << d.nbits
	return rv
}

// flushBits discards the unread bits in the buffer used by readBits.
// It is used at the end of a line.
func (d *decoder) flushBits() {
	d.v = 0
	d.nbits = 0
}

// decode decodes the raw data of an image.
// It reads from d.buf and writes the strip with ymin <= y < ymax into dst.
func (d *decoder) decode(dst image.Image, ymin, ymax int) os.Error {
	d.off = 0

	// Apply horizontal predictor if necessary.
	// In this case, p contains the color difference to the preceding pixel.
	// See page 64-65 of the spec.
	if d.firstVal(tPredictor) == prHorizontal && d.firstVal(tBitsPerSample) == 8 {
		var off int
		spp := len(d.features[tBitsPerSample]) // samples per pixel
		for y := ymin; y < ymax; y++ {
			off += spp
			for x := 0; x < (dst.Bounds().Dx()-1)*spp; x++ {
				d.buf[off] += d.buf[off-spp]
				off++
			}
		}
	}

	switch d.mode {
	case mGray, mGrayInvert:
		img := dst.(*image.Gray)
		bpp := d.firstVal(tBitsPerSample)
		max := uint32((1 << bpp) - 1)
		for y := ymin; y < ymax; y++ {
			for x := img.Rect.Min.X; x < img.Rect.Max.X; x++ {
				v := uint8(d.readBits(bpp) * 0xff / max)
				if d.mode == mGrayInvert {
					v = 0xff - v
				}
				img.SetGray(x, y, color.Gray{v})
			}
			d.flushBits()
		}
	case mPaletted:
		img := dst.(*image.Paletted)
		bpp := d.firstVal(tBitsPerSample)
		for y := ymin; y < ymax; y++ {
			for x := img.Rect.Min.X; x < img.Rect.Max.X; x++ {
				img.SetColorIndex(x, y, uint8(d.readBits(bpp)))
			}
			d.flushBits()
		}
	case mRGB:
		img := dst.(*image.RGBA)
		min := (ymin-img.Rect.Min.Y)*img.Stride - img.Rect.Min.X*4
		max := (ymax-img.Rect.Min.Y)*img.Stride - img.Rect.Min.X*4
		var off int
		for i := min; i < max; i += 4 {
			img.Pix[i+0] = d.buf[off+0]
			img.Pix[i+1] = d.buf[off+1]
			img.Pix[i+2] = d.buf[off+2]
			img.Pix[i+3] = 0xff
			off += 3
		}
	case mNRGBA:
		img := dst.(*image.NRGBA)
		min := (ymin-img.Rect.Min.Y)*img.Stride - img.Rect.Min.X*4
		max := (ymax-img.Rect.Min.Y)*img.Stride - img.Rect.Min.X*4
		if len(d.buf) != max-min {
			return FormatError("short data strip")
		}
		copy(img.Pix[min:max], d.buf)
	case mRGBA:
		img := dst.(*image.RGBA)
		min := (ymin-img.Rect.Min.Y)*img.Stride - img.Rect.Min.X*4
		max := (ymax-img.Rect.Min.Y)*img.Stride - img.Rect.Min.X*4
		if len(d.buf) != max-min {
			return FormatError("short data strip")
		}
		copy(img.Pix[min:max], d.buf)
	}

	return nil
}

func newDecoder(r io.Reader) (*decoder, os.Error) {
	d := &decoder{
		r:        newReaderAt(r),
		features: make(map[int][]uint),
	}

	p := make([]byte, 8)
	if _, err := d.r.ReadAt(p, 0); err != nil {
		return nil, err
	}
	switch string(p[0:4]) {
	case leHeader:
		d.byteOrder = binary.LittleEndian
	case beHeader:
		d.byteOrder = binary.BigEndian
	default:
		return nil, FormatError("malformed header")
	}

	ifdOffset := int64(d.byteOrder.Uint32(p[4:8]))

	// The first two bytes contain the number of entries (12 bytes each).
	if _, err := d.r.ReadAt(p[0:2], ifdOffset); err != nil {
		return nil, err
	}
	numItems := int(d.byteOrder.Uint16(p[0:2]))

	// All IFD entries are read in one chunk.
	p = make([]byte, ifdLen*numItems)
	if _, err := d.r.ReadAt(p, ifdOffset+2); err != nil {
		return nil, err
	}

	for i := 0; i < len(p); i += ifdLen {
		if err := d.parseIFD(p[i : i+ifdLen]); err != nil {
			return nil, err
		}
	}

	d.config.Width = int(d.firstVal(tImageWidth))
	d.config.Height = int(d.firstVal(tImageLength))

	if _, ok := d.features[tBitsPerSample]; !ok {
		return nil, FormatError("BitsPerSample tag missing")
	}

	// Determine the image mode.
	switch d.firstVal(tPhotometricInterpretation) {
	case pRGB:
		for _, b := range d.features[tBitsPerSample] {
			if b != 8 {
				return nil, UnsupportedError("non-8-bit RGB image")
			}
		}
		d.config.ColorModel = color.RGBAModel
		// RGB images normally have 3 samples per pixel.
		// If there are more, ExtraSamples (p. 31-32 of the spec)
		// gives their meaning (usually an alpha channel).
		//
		// This implementation does not support extra samples
		// of an unspecified type.
		switch len(d.features[tBitsPerSample]) {
		case 3:
			d.mode = mRGB
		case 4:
			switch d.firstVal(tExtraSamples) {
			case 1:
				d.mode = mRGBA
			case 2:
				d.mode = mNRGBA
				d.config.ColorModel = color.NRGBAModel
			default:
				return nil, FormatError("wrong number of samples for RGB")
			}
		default:
			return nil, FormatError("wrong number of samples for RGB")
		}
	case pPaletted:
		d.mode = mPaletted
		d.config.ColorModel = color.Palette(d.palette)
	case pWhiteIsZero:
		d.mode = mGrayInvert
		d.config.ColorModel = color.GrayModel
	case pBlackIsZero:
		d.mode = mGray
		d.config.ColorModel = color.GrayModel
	default:
		return nil, UnsupportedError("color model")
	}

	return d, nil
}

// DecodeConfig returns the color model and dimensions of a TIFF image without
// decoding the entire image.
func DecodeConfig(r io.Reader) (image.Config, os.Error) {
	d, err := newDecoder(r)
	if err != nil {
		return image.Config{}, err
	}
	return d.config, nil
}

// Decode reads a TIFF image from r and returns it as an image.Image.
// The type of Image returned depends on the contents of the TIFF.
func Decode(r io.Reader) (img image.Image, err os.Error) {
	d, err := newDecoder(r)
	if err != nil {
		return
	}

	// Check if we have the right number of strips, offsets and counts.
	rps := int(d.firstVal(tRowsPerStrip))
	if rps == 0 {
		// Assume only one strip.
		rps = d.config.Height
	}
	numStrips := (d.config.Height + rps - 1) / rps
	if rps == 0 || len(d.features[tStripOffsets]) < numStrips || len(d.features[tStripByteCounts]) < numStrips {
		return nil, FormatError("inconsistent header")
	}

	switch d.mode {
	case mGray, mGrayInvert:
		img = image.NewGray(image.Rect(0, 0, d.config.Width, d.config.Height))
	case mPaletted:
		img = image.NewPaletted(image.Rect(0, 0, d.config.Width, d.config.Height), d.palette)
	case mNRGBA:
		img = image.NewNRGBA(image.Rect(0, 0, d.config.Width, d.config.Height))
	case mRGB, mRGBA:
		img = image.NewRGBA(image.Rect(0, 0, d.config.Width, d.config.Height))
	}

	for i := 0; i < numStrips; i++ {
		ymin := i * rps
		// The last strip may be shorter.
		if i == numStrips-1 && d.config.Height%rps != 0 {
			rps = d.config.Height % rps
		}
		offset := int64(d.features[tStripOffsets][i])
		n := int64(d.features[tStripByteCounts][i])
		switch d.firstVal(tCompression) {
		case cNone:
			// TODO(bsiegert): Avoid copy if r is a tiff.buffer.
			d.buf = make([]byte, n)
			_, err = d.r.ReadAt(d.buf, offset)
		case cLZW:
			r := lzw.NewReader(io.NewSectionReader(d.r, offset, n), lzw.MSB, 8)
			d.buf, err = ioutil.ReadAll(r)
			r.Close()
		case cDeflate, cDeflateOld:
			r, err := zlib.NewReader(io.NewSectionReader(d.r, offset, n))
			if err != nil {
				return nil, err
			}
			d.buf, err = ioutil.ReadAll(r)
			r.Close()
		case cPackBits:
			d.buf, err = unpackBits(io.NewSectionReader(d.r, offset, n))
		default:
			err = UnsupportedError("compression")
		}
		if err != nil {
			return
		}
		err = d.decode(img, ymin, ymin+rps)
	}
	return
}

func init() {
	image.RegisterFormat("tiff", leHeader, Decode, DecodeConfig)
	image.RegisterFormat("tiff", beHeader, Decode, DecodeConfig)
}
