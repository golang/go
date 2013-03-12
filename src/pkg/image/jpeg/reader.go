// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package jpeg implements a JPEG image decoder and encoder.
//
// JPEG is defined in ITU-T T.81: http://www.w3.org/Graphics/JPEG/itu-t81.pdf.
package jpeg

import (
	"bufio"
	"image"
	"image/color"
	"io"
)

// TODO(nigeltao): fix up the doc comment style so that sentences start with
// the name of the type or function that they annotate.

// A FormatError reports that the input is not a valid JPEG.
type FormatError string

func (e FormatError) Error() string { return "invalid JPEG format: " + string(e) }

// An UnsupportedError reports that the input uses a valid but unimplemented JPEG feature.
type UnsupportedError string

func (e UnsupportedError) Error() string { return "unsupported JPEG feature: " + string(e) }

// Component specification, specified in section B.2.2.
type component struct {
	h  int   // Horizontal sampling factor.
	v  int   // Vertical sampling factor.
	c  uint8 // Component identifier.
	tq uint8 // Quantization table destination selector.
}

const (
	dcTable = 0
	acTable = 1
	maxTc   = 1
	maxTh   = 3
	maxTq   = 3

	// A grayscale JPEG image has only a Y component.
	nGrayComponent = 1
	// A color JPEG image has Y, Cb and Cr components.
	nColorComponent = 3

	// We only support 4:4:4, 4:4:0, 4:2:2 and 4:2:0 downsampling, and therefore the
	// number of luma samples per chroma sample is at most 2 in the horizontal
	// and 2 in the vertical direction.
	maxH = 2
	maxV = 2
)

const (
	soiMarker   = 0xd8 // Start Of Image.
	eoiMarker   = 0xd9 // End Of Image.
	sof0Marker  = 0xc0 // Start Of Frame (Baseline).
	sof2Marker  = 0xc2 // Start Of Frame (Progressive).
	dhtMarker   = 0xc4 // Define Huffman Table.
	dqtMarker   = 0xdb // Define Quantization Table.
	sosMarker   = 0xda // Start Of Scan.
	driMarker   = 0xdd // Define Restart Interval.
	rst0Marker  = 0xd0 // ReSTart (0).
	rst7Marker  = 0xd7 // ReSTart (7).
	app0Marker  = 0xe0 // APPlication specific (0).
	app15Marker = 0xef // APPlication specific (15).
	comMarker   = 0xfe // COMment.
)

// unzig maps from the zig-zag ordering to the natural ordering. For example,
// unzig[3] is the column and row of the fourth element in zig-zag order. The
// value is 16, which means first column (16%8 == 0) and third row (16/8 == 2).
var unzig = [blockSize]int{
	0, 1, 8, 16, 9, 2, 3, 10,
	17, 24, 32, 25, 18, 11, 4, 5,
	12, 19, 26, 33, 40, 48, 41, 34,
	27, 20, 13, 6, 7, 14, 21, 28,
	35, 42, 49, 56, 57, 50, 43, 36,
	29, 22, 15, 23, 30, 37, 44, 51,
	58, 59, 52, 45, 38, 31, 39, 46,
	53, 60, 61, 54, 47, 55, 62, 63,
}

// If the passed in io.Reader does not also have ReadByte, then Decode will introduce its own buffering.
type Reader interface {
	io.Reader
	ReadByte() (c byte, err error)
}

type decoder struct {
	r             Reader
	b             bits
	width, height int
	img1          *image.Gray
	img3          *image.YCbCr
	ri            int // Restart Interval.
	nComp         int
	progressive   bool
	eobRun        uint16 // End-of-Band run, specified in section G.1.2.2.
	comp          [nColorComponent]component
	progCoeffs    [nColorComponent][]block // Saved state between progressive-mode scans.
	huff          [maxTc + 1][maxTh + 1]huffman
	quant         [maxTq + 1]block // Quantization tables, in zig-zag order.
	tmp           [1024]byte
}

// Reads and ignores the next n bytes.
func (d *decoder) ignore(n int) error {
	for n > 0 {
		m := len(d.tmp)
		if m > n {
			m = n
		}
		_, err := io.ReadFull(d.r, d.tmp[0:m])
		if err != nil {
			return err
		}
		n -= m
	}
	return nil
}

// Specified in section B.2.2.
func (d *decoder) processSOF(n int) error {
	switch n {
	case 6 + 3*nGrayComponent:
		d.nComp = nGrayComponent
	case 6 + 3*nColorComponent:
		d.nComp = nColorComponent
	default:
		return UnsupportedError("SOF has wrong length")
	}
	_, err := io.ReadFull(d.r, d.tmp[:n])
	if err != nil {
		return err
	}
	// We only support 8-bit precision.
	if d.tmp[0] != 8 {
		return UnsupportedError("precision")
	}
	d.height = int(d.tmp[1])<<8 + int(d.tmp[2])
	d.width = int(d.tmp[3])<<8 + int(d.tmp[4])
	if int(d.tmp[5]) != d.nComp {
		return UnsupportedError("SOF has wrong number of image components")
	}
	for i := 0; i < d.nComp; i++ {
		d.comp[i].c = d.tmp[6+3*i]
		d.comp[i].tq = d.tmp[8+3*i]
		if d.nComp == nGrayComponent {
			// If a JPEG image has only one component, section A.2 says "this data
			// is non-interleaved by definition" and section A.2.2 says "[in this
			// case...] the order of data units within a scan shall be left-to-right
			// and top-to-bottom... regardless of the values of H_1 and V_1". Section
			// 4.8.2 also says "[for non-interleaved data], the MCU is defined to be
			// one data unit". Similarly, section A.1.1 explains that it is the ratio
			// of H_i to max_j(H_j) that matters, and similarly for V. For grayscale
			// images, H_1 is the maximum H_j for all components j, so that ratio is
			// always 1. The component's (h, v) is effectively always (1, 1): even if
			// the nominal (h, v) is (2, 1), a 20x5 image is encoded in three 8x8
			// MCUs, not two 16x8 MCUs.
			d.comp[i].h = 1
			d.comp[i].v = 1
			continue
		}
		hv := d.tmp[7+3*i]
		d.comp[i].h = int(hv >> 4)
		d.comp[i].v = int(hv & 0x0f)
		// For color images, we only support 4:4:4, 4:4:0, 4:2:2 or 4:2:0 chroma
		// downsampling ratios. This implies that the (h, v) values for the Y
		// component are either (1, 1), (1, 2), (2, 1) or (2, 2), and the (h, v)
		// values for the Cr and Cb components must be (1, 1).
		if i == 0 {
			if hv != 0x11 && hv != 0x21 && hv != 0x22 && hv != 0x12 {
				return UnsupportedError("luma downsample ratio")
			}
		} else if hv != 0x11 {
			return UnsupportedError("chroma downsample ratio")
		}
	}
	return nil
}

// Specified in section B.2.4.1.
func (d *decoder) processDQT(n int) error {
	const qtLength = 1 + blockSize
	for ; n >= qtLength; n -= qtLength {
		_, err := io.ReadFull(d.r, d.tmp[0:qtLength])
		if err != nil {
			return err
		}
		pq := d.tmp[0] >> 4
		if pq != 0 {
			return UnsupportedError("bad Pq value")
		}
		tq := d.tmp[0] & 0x0f
		if tq > maxTq {
			return FormatError("bad Tq value")
		}
		for i := range d.quant[tq] {
			d.quant[tq][i] = int32(d.tmp[i+1])
		}
	}
	if n != 0 {
		return FormatError("DQT has wrong length")
	}
	return nil
}

// Specified in section B.2.4.4.
func (d *decoder) processDRI(n int) error {
	if n != 2 {
		return FormatError("DRI has wrong length")
	}
	_, err := io.ReadFull(d.r, d.tmp[0:2])
	if err != nil {
		return err
	}
	d.ri = int(d.tmp[0])<<8 + int(d.tmp[1])
	return nil
}

// decode reads a JPEG image from r and returns it as an image.Image.
func (d *decoder) decode(r io.Reader, configOnly bool) (image.Image, error) {
	if rr, ok := r.(Reader); ok {
		d.r = rr
	} else {
		d.r = bufio.NewReader(r)
	}

	// Check for the Start Of Image marker.
	_, err := io.ReadFull(d.r, d.tmp[0:2])
	if err != nil {
		return nil, err
	}
	if d.tmp[0] != 0xff || d.tmp[1] != soiMarker {
		return nil, FormatError("missing SOI marker")
	}

	// Process the remaining segments until the End Of Image marker.
	for {
		_, err := io.ReadFull(d.r, d.tmp[0:2])
		if err != nil {
			return nil, err
		}
		for d.tmp[0] != 0xff {
			// Strictly speaking, this is a format error. However, libjpeg is
			// liberal in what it accepts. As of version 9, next_marker in
			// jdmarker.c treats this as a warning (JWRN_EXTRANEOUS_DATA) and
			// continues to decode the stream. Even before next_marker sees
			// extraneous data, jpeg_fill_bit_buffer in jdhuff.c reads as many
			// bytes as it can, possibly past the end of a scan's data. It
			// effectively puts back any markers that it overscanned (e.g. an
			// "\xff\xd9" EOI marker), but it does not put back non-marker data,
			// and thus it can silently ignore a small number of extraneous
			// non-marker bytes before next_marker has a chance to see them (and
			// print a warning).
			//
			// We are therefore also liberal in what we accept. Extraneous data
			// is silently ignored.
			//
			// This is similar to, but not exactly the same as, the restart
			// mechanism within a scan (the RST[0-7] markers).
			//
			// Note that extraneous 0xff bytes in e.g. SOS data are escaped as
			// "\xff\x00", and so are detected a little further down below.
			d.tmp[0] = d.tmp[1]
			d.tmp[1], err = d.r.ReadByte()
			if err != nil {
				return nil, err
			}
		}
		marker := d.tmp[1]
		if marker == 0 {
			// Treat "\xff\x00" as extraneous data.
			continue
		}
		for marker == 0xff {
			// Section B.1.1.2 says, "Any marker may optionally be preceded by any
			// number of fill bytes, which are bytes assigned code X'FF'".
			marker, err = d.r.ReadByte()
			if err != nil {
				return nil, err
			}
		}
		if marker == eoiMarker { // End Of Image.
			break
		}
		if rst0Marker <= marker && marker <= rst7Marker {
			// Figures B.2 and B.16 of the specification suggest that restart markers should
			// only occur between Entropy Coded Segments and not after the final ECS.
			// However, some encoders may generate incorrect JPEGs with a final restart
			// marker. That restart marker will be seen here instead of inside the processSOS
			// method, and is ignored as a harmless error. Restart markers have no extra data,
			// so we check for this before we read the 16-bit length of the segment.
			continue
		}

		// Read the 16-bit length of the segment. The value includes the 2 bytes for the
		// length itself, so we subtract 2 to get the number of remaining bytes.
		_, err = io.ReadFull(d.r, d.tmp[0:2])
		if err != nil {
			return nil, err
		}
		n := int(d.tmp[0])<<8 + int(d.tmp[1]) - 2
		if n < 0 {
			return nil, FormatError("short segment length")
		}

		switch {
		case marker == sof0Marker || marker == sof2Marker: // Start Of Frame.
			d.progressive = marker == sof2Marker
			err = d.processSOF(n)
			if configOnly {
				return nil, err
			}
		case marker == dhtMarker: // Define Huffman Table.
			err = d.processDHT(n)
		case marker == dqtMarker: // Define Quantization Table.
			err = d.processDQT(n)
		case marker == sosMarker: // Start Of Scan.
			err = d.processSOS(n)
		case marker == driMarker: // Define Restart Interval.
			err = d.processDRI(n)
		case app0Marker <= marker && marker <= app15Marker || marker == comMarker: // APPlication specific, or COMment.
			err = d.ignore(n)
		default:
			err = UnsupportedError("unknown marker")
		}
		if err != nil {
			return nil, err
		}
	}
	if d.img1 != nil {
		return d.img1, nil
	}
	if d.img3 != nil {
		return d.img3, nil
	}
	return nil, FormatError("missing SOS marker")
}

// Decode reads a JPEG image from r and returns it as an image.Image.
func Decode(r io.Reader) (image.Image, error) {
	var d decoder
	return d.decode(r, false)
}

// DecodeConfig returns the color model and dimensions of a JPEG image without
// decoding the entire image.
func DecodeConfig(r io.Reader) (image.Config, error) {
	var d decoder
	if _, err := d.decode(r, true); err != nil {
		return image.Config{}, err
	}
	switch d.nComp {
	case nGrayComponent:
		return image.Config{
			ColorModel: color.GrayModel,
			Width:      d.width,
			Height:     d.height,
		}, nil
	case nColorComponent:
		return image.Config{
			ColorModel: color.YCbCrModel,
			Width:      d.width,
			Height:     d.height,
		}, nil
	}
	return image.Config{}, FormatError("missing SOF marker")
}

func init() {
	image.RegisterFormat("jpeg", "\xff\xd8", Decode, DecodeConfig)
}
