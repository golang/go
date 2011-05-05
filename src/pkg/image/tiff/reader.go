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
	palette   []image.Color
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
		d.palette = make([]image.Color, numcolors)
		for i := 0; i < numcolors; i++ {
			d.palette[i] = image.RGBA64Color{
				uint16(val[i]),
				uint16(val[i+numcolors]),
				uint16(val[i+2*numcolors]),
				0xffff,
			}
		}
	}
	return nil
}

// decode decodes the raw data of an image with 8 bits in each sample.
// It reads from p and writes the strip with ymin <= y < ymax into dst.
func (d *decoder) decode(dst image.Image, p []byte, ymin, ymax int) os.Error {
	spp := len(d.features[tBitsPerSample]) // samples per pixel
	off := 0
	width := dst.Bounds().Dx()

	if len(p) < spp*(ymax-ymin)*width {
		return FormatError("short data strip")
	}

	// Apply horizontal predictor if necessary.
	// In this case, p contains the color difference to the preceding pixel.
	// See page 64-65 of the spec.
	if d.firstVal(tPredictor) == prHorizontal {
		for y := ymin; y < ymax; y++ {
			off += spp
			for x := 0; x < (width-1)*spp; x++ {
				p[off] += p[off-spp]
				off++
			}
		}
		off = 0
	}

	switch d.mode {
	case mGray:
		img := dst.(*image.Gray)
		for y := ymin; y < ymax; y++ {
			for x := img.Rect.Min.X; x < img.Rect.Max.X; x++ {
				img.Set(x, y, image.GrayColor{p[off]})
				off += spp
			}
		}
	case mGrayInvert:
		img := dst.(*image.Gray)
		for y := ymin; y < ymax; y++ {
			for x := img.Rect.Min.X; x < img.Rect.Max.X; x++ {
				img.Set(x, y, image.GrayColor{0xff - p[off]})
				off += spp
			}
		}
	case mPaletted:
		img := dst.(*image.Paletted)
		for y := ymin; y < ymax; y++ {
			for x := img.Rect.Min.X; x < img.Rect.Max.X; x++ {
				img.SetColorIndex(x, y, p[off])
				off += spp
			}
		}
	case mRGB:
		img := dst.(*image.RGBA)
		for y := ymin; y < ymax; y++ {
			for x := img.Rect.Min.X; x < img.Rect.Max.X; x++ {
				img.Set(x, y, image.RGBAColor{p[off], p[off+1], p[off+2], 0xff})
				off += spp
			}
		}
	case mNRGBA:
		img := dst.(*image.NRGBA)
		for y := ymin; y < ymax; y++ {
			for x := img.Rect.Min.X; x < img.Rect.Max.X; x++ {
				img.Set(x, y, image.NRGBAColor{p[off], p[off+1], p[off+2], p[off+3]})
				off += spp
			}
		}
	case mRGBA:
		img := dst.(*image.RGBA)
		for y := ymin; y < ymax; y++ {
			for x := img.Rect.Min.X; x < img.Rect.Max.X; x++ {
				img.Set(x, y, image.RGBAColor{p[off], p[off+1], p[off+2], p[off+3]})
				off += spp
			}
		}
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

	// Determine the image mode.
	switch d.firstVal(tPhotometricInterpretation) {
	case pRGB:
		d.config.ColorModel = image.RGBAColorModel
		// RGB images normally have 3 samples per pixel.
		// If there are more, ExtraSamples (p. 31-32 of the spec)
		// gives their meaning (usually an alpha channel).
		switch len(d.features[tBitsPerSample]) {
		case 3:
			d.mode = mRGB
		case 4:
			switch d.firstVal(tExtraSamples) {
			case 1:
				d.mode = mRGBA
			case 2:
				d.mode = mNRGBA
				d.config.ColorModel = image.NRGBAColorModel
			default:
				// The extra sample is discarded.
				d.mode = mRGB
			}
		default:
			return nil, FormatError("wrong number of samples for RGB")
		}
	case pPaletted:
		d.mode = mPaletted
		d.config.ColorModel = image.PalettedColorModel(d.palette)
	case pWhiteIsZero:
		d.mode = mGrayInvert
		d.config.ColorModel = image.GrayColorModel
	case pBlackIsZero:
		d.mode = mGray
		d.config.ColorModel = image.GrayColorModel
	default:
		return nil, UnsupportedError("color model")
	}

	if _, ok := d.features[tBitsPerSample]; !ok {
		return nil, FormatError("BitsPerSample tag missing")
	}
	for _, b := range d.features[tBitsPerSample] {
		if b != 8 {
			return nil, UnsupportedError("not an 8-bit image")
		}
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
	numStrips := (d.config.Height + rps - 1) / rps
	if rps == 0 || len(d.features[tStripOffsets]) < numStrips || len(d.features[tStripByteCounts]) < numStrips {
		return nil, FormatError("inconsistent header")
	}

	switch d.mode {
	case mGray, mGrayInvert:
		img = image.NewGray(d.config.Width, d.config.Height)
	case mPaletted:
		img = image.NewPaletted(d.config.Width, d.config.Height, d.palette)
	case mNRGBA:
		img = image.NewNRGBA(d.config.Width, d.config.Height)
	case mRGB, mRGBA:
		img = image.NewRGBA(d.config.Width, d.config.Height)
	}

	var p []byte
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
			p = make([]byte, 0, n)
			_, err = d.r.ReadAt(p, offset)
		case cLZW:
			r := lzw.NewReader(io.NewSectionReader(d.r, offset, n), lzw.MSB, 8)
			p, err = ioutil.ReadAll(r)
			r.Close()
		case cDeflate, cDeflateOld:
			r, err := zlib.NewReader(io.NewSectionReader(d.r, offset, n))
			if err != nil {
				return nil, err
			}
			p, err = ioutil.ReadAll(r)
			r.Close()
		default:
			err = UnsupportedError("compression")
		}
		if err != nil {
			return
		}
		err = d.decode(img, p, ymin, ymin+rps)
	}
	return
}

func init() {
	image.RegisterFormat("tiff", leHeader, Decode, DecodeConfig)
	image.RegisterFormat("tiff", beHeader, Decode, DecodeConfig)
}
