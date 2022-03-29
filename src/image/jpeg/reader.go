// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package jpeg implements a JPEG image decoder and encoder.
//
// JPEG is defined in ITU-T T.81: https://www.w3.org/Graphics/JPEG/itu-t81.pdf.
package jpeg

import (
	"image"
	"image/color"
	"image/internal/imageutil"
	"io"
)

// A FormatError reports that the input is not a valid JPEG.
type FormatError string

func (e FormatError) Error() string { return "invalid JPEG format: " + string(e) }

// An UnsupportedError reports that the input uses a valid but unimplemented JPEG feature.
type UnsupportedError string

func (e UnsupportedError) Error() string { return "unsupported JPEG feature: " + string(e) }

var errUnsupportedSubsamplingRatio = UnsupportedError("luma/chroma subsampling ratio")

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

	maxComponents = 4
)

const (
	sof0Marker = 0xc0 // Start Of Frame (Baseline Sequential).
	sof1Marker = 0xc1 // Start Of Frame (Extended Sequential).
	sof2Marker = 0xc2 // Start Of Frame (Progressive).
	dhtMarker  = 0xc4 // Define Huffman Table.
	rst0Marker = 0xd0 // ReSTart (0).
	rst7Marker = 0xd7 // ReSTart (7).
	soiMarker  = 0xd8 // Start Of Image.
	eoiMarker  = 0xd9 // End Of Image.
	sosMarker  = 0xda // Start Of Scan.
	dqtMarker  = 0xdb // Define Quantization Table.
	driMarker  = 0xdd // Define Restart Interval.
	comMarker  = 0xfe // COMment.
	// "APPlication specific" markers aren't part of the JPEG spec per se,
	// but in practice, their use is described at
	// https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/JPEG.html
	app0Marker  = 0xe0
	app14Marker = 0xee
	app15Marker = 0xef
)

// See https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/JPEG.html#Adobe
const (
	adobeTransformUnknown = 0
	adobeTransformYCbCr   = 1
	adobeTransformYCbCrK  = 2
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

// Deprecated: Reader is not used by the image/jpeg package and should
// not be used by others. It is kept for compatibility.
type Reader interface {
	io.ByteReader
	io.Reader
}

// bits holds the unprocessed bits that have been taken from the byte-stream.
// The n least significant bits of a form the unread bits, to be read in MSB to
// LSB order.
type bits struct {
	a uint32 // accumulator.
	m uint32 // mask. m==1<<(n-1) when n>0, with m==0 when n==0.
	n int32  // the number of unread bits in a.
}

type decoder struct {
	r    io.Reader
	bits bits
	// bytes is a byte buffer, similar to a bufio.Reader, except that it
	// has to be able to unread more than 1 byte, due to byte stuffing.
	// Byte stuffing is specified in section F.1.2.3.
	bytes struct {
		// buf[i:j] are the buffered bytes read from the underlying
		// io.Reader that haven't yet been passed further on.
		buf  [4096]byte
		i, j int
		// nUnreadable is the number of bytes to back up i after
		// overshooting. It can be 0, 1 or 2.
		nUnreadable int
	}
	width, height int

	img1        *image.Gray
	img3        *image.YCbCr
	blackPix    []byte
	blackStride int

	ri    int // Restart Interval.
	nComp int

	// As per section 4.5, there are four modes of operation (selected by the
	// SOF? markers): sequential DCT, progressive DCT, lossless and
	// hierarchical, although this implementation does not support the latter
	// two non-DCT modes. Sequential DCT is further split into baseline and
	// extended, as per section 4.11.
	baseline    bool
	progressive bool

	jfif                bool
	adobeTransformValid bool
	adobeTransform      uint8
	eobRun              uint16 // End-of-Band run, specified in section G.1.2.2.

	comp       [maxComponents]component
	progCoeffs [maxComponents][]block // Saved state between progressive-mode scans.
	huff       [maxTc + 1][maxTh + 1]huffman
	quant      [maxTq + 1]block // Quantization tables, in zig-zag order.
	tmp        [2 * blockSize]byte
}

// fill fills up the d.bytes.buf buffer from the underlying io.Reader. It
// should only be called when there are no unread bytes in d.bytes.
func (d *decoder) fill() error {
	if d.bytes.i != d.bytes.j {
		panic("jpeg: fill called when unread bytes exist")
	}
	// Move the last 2 bytes to the start of the buffer, in case we need
	// to call unreadByteStuffedByte.
	if d.bytes.j > 2 {
		d.bytes.buf[0] = d.bytes.buf[d.bytes.j-2]
		d.bytes.buf[1] = d.bytes.buf[d.bytes.j-1]
		d.bytes.i, d.bytes.j = 2, 2
	}
	// Fill in the rest of the buffer.
	n, err := d.r.Read(d.bytes.buf[d.bytes.j:])
	d.bytes.j += n
	if n > 0 {
		err = nil
	}
	return err
}

// unreadByteStuffedByte undoes the most recent readByteStuffedByte call,
// giving a byte of data back from d.bits to d.bytes. The Huffman look-up table
// requires at least 8 bits for look-up, which means that Huffman decoding can
// sometimes overshoot and read one or two too many bytes. Two-byte overshoot
// can happen when expecting to read a 0xff 0x00 byte-stuffed byte.
func (d *decoder) unreadByteStuffedByte() {
	d.bytes.i -= d.bytes.nUnreadable
	d.bytes.nUnreadable = 0
	if d.bits.n >= 8 {
		d.bits.a >>= 8
		d.bits.n -= 8
		d.bits.m >>= 8
	}
}

// readByte returns the next byte, whether buffered or not buffered. It does
// not care about byte stuffing.
func (d *decoder) readByte() (x byte, err error) {
	for d.bytes.i == d.bytes.j {
		if err = d.fill(); err != nil {
			return 0, err
		}
	}
	x = d.bytes.buf[d.bytes.i]
	d.bytes.i++
	d.bytes.nUnreadable = 0
	return x, nil
}

// errMissingFF00 means that readByteStuffedByte encountered an 0xff byte (a
// marker byte) that wasn't the expected byte-stuffed sequence 0xff, 0x00.
var errMissingFF00 = FormatError("missing 0xff00 sequence")

// readByteStuffedByte is like readByte but is for byte-stuffed Huffman data.
func (d *decoder) readByteStuffedByte() (x byte, err error) {
	// Take the fast path if d.bytes.buf contains at least two bytes.
	if d.bytes.i+2 <= d.bytes.j {
		x = d.bytes.buf[d.bytes.i]
		d.bytes.i++
		d.bytes.nUnreadable = 1
		if x != 0xff {
			return x, err
		}
		if d.bytes.buf[d.bytes.i] != 0x00 {
			return 0, errMissingFF00
		}
		d.bytes.i++
		d.bytes.nUnreadable = 2
		return 0xff, nil
	}

	d.bytes.nUnreadable = 0

	x, err = d.readByte()
	if err != nil {
		return 0, err
	}
	d.bytes.nUnreadable = 1
	if x != 0xff {
		return x, nil
	}

	x, err = d.readByte()
	if err != nil {
		return 0, err
	}
	d.bytes.nUnreadable = 2
	if x != 0x00 {
		return 0, errMissingFF00
	}
	return 0xff, nil
}

// readFull reads exactly len(p) bytes into p. It does not care about byte
// stuffing.
func (d *decoder) readFull(p []byte) error {
	// Unread the overshot bytes, if any.
	if d.bytes.nUnreadable != 0 {
		if d.bits.n >= 8 {
			d.unreadByteStuffedByte()
		}
		d.bytes.nUnreadable = 0
	}

	for {
		n := copy(p, d.bytes.buf[d.bytes.i:d.bytes.j])
		p = p[n:]
		d.bytes.i += n
		if len(p) == 0 {
			break
		}
		if err := d.fill(); err != nil {
			if err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return err
		}
	}
	return nil
}

// ignore ignores the next n bytes.
func (d *decoder) ignore(n int) error {
	// Unread the overshot bytes, if any.
	if d.bytes.nUnreadable != 0 {
		if d.bits.n >= 8 {
			d.unreadByteStuffedByte()
		}
		d.bytes.nUnreadable = 0
	}

	for {
		m := d.bytes.j - d.bytes.i
		if m > n {
			m = n
		}
		d.bytes.i += m
		n -= m
		if n == 0 {
			break
		}
		if err := d.fill(); err != nil {
			if err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return err
		}
	}
	return nil
}

// Specified in section B.2.2.
func (d *decoder) processSOF(n int) error {
	if d.nComp != 0 {
		return FormatError("multiple SOF markers")
	}
	switch n {
	case 6 + 3*1: // Grayscale image.
		d.nComp = 1
	case 6 + 3*3: // YCbCr or RGB image.
		d.nComp = 3
	case 6 + 3*4: // YCbCrK or CMYK image.
		d.nComp = 4
	default:
		return UnsupportedError("number of components")
	}
	if err := d.readFull(d.tmp[:n]); err != nil {
		return err
	}
	// We only support 8-bit precision.
	if d.tmp[0] != 8 {
		return UnsupportedError("precision")
	}
	d.height = int(d.tmp[1])<<8 + int(d.tmp[2])
	d.width = int(d.tmp[3])<<8 + int(d.tmp[4])
	if int(d.tmp[5]) != d.nComp {
		return FormatError("SOF has wrong length")
	}

	for i := 0; i < d.nComp; i++ {
		d.comp[i].c = d.tmp[6+3*i]
		// Section B.2.2 states that "the value of C_i shall be different from
		// the values of C_1 through C_(i-1)".
		for j := 0; j < i; j++ {
			if d.comp[i].c == d.comp[j].c {
				return FormatError("repeated component identifier")
			}
		}

		d.comp[i].tq = d.tmp[8+3*i]
		if d.comp[i].tq > maxTq {
			return FormatError("bad Tq value")
		}

		hv := d.tmp[7+3*i]
		h, v := int(hv>>4), int(hv&0x0f)
		if h < 1 || 4 < h || v < 1 || 4 < v {
			return FormatError("luma/chroma subsampling ratio")
		}
		if h == 3 || v == 3 {
			return errUnsupportedSubsamplingRatio
		}
		switch d.nComp {
		case 1:
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
			h, v = 1, 1

		case 3:
			// For YCbCr images, we only support 4:4:4, 4:4:0, 4:2:2, 4:2:0,
			// 4:1:1 or 4:1:0 chroma subsampling ratios. This implies that the
			// (h, v) values for the Y component are either (1, 1), (1, 2),
			// (2, 1), (2, 2), (4, 1) or (4, 2), and the Y component's values
			// must be a multiple of the Cb and Cr component's values. We also
			// assume that the two chroma components have the same subsampling
			// ratio.
			switch i {
			case 0: // Y.
				// We have already verified, above, that h and v are both
				// either 1, 2 or 4, so invalid (h, v) combinations are those
				// with v == 4.
				if v == 4 {
					return errUnsupportedSubsamplingRatio
				}
			case 1: // Cb.
				if d.comp[0].h%h != 0 || d.comp[0].v%v != 0 {
					return errUnsupportedSubsamplingRatio
				}
			case 2: // Cr.
				if d.comp[1].h != h || d.comp[1].v != v {
					return errUnsupportedSubsamplingRatio
				}
			}

		case 4:
			// For 4-component images (either CMYK or YCbCrK), we only support two
			// hv vectors: [0x11 0x11 0x11 0x11] and [0x22 0x11 0x11 0x22].
			// Theoretically, 4-component JPEG images could mix and match hv values
			// but in practice, those two combinations are the only ones in use,
			// and it simplifies the applyBlack code below if we can assume that:
			//	- for CMYK, the C and K channels have full samples, and if the M
			//	  and Y channels subsample, they subsample both horizontally and
			//	  vertically.
			//	- for YCbCrK, the Y and K channels have full samples.
			switch i {
			case 0:
				if hv != 0x11 && hv != 0x22 {
					return errUnsupportedSubsamplingRatio
				}
			case 1, 2:
				if hv != 0x11 {
					return errUnsupportedSubsamplingRatio
				}
			case 3:
				if d.comp[0].h != h || d.comp[0].v != v {
					return errUnsupportedSubsamplingRatio
				}
			}
		}

		d.comp[i].h = h
		d.comp[i].v = v
	}
	return nil
}

// Specified in section B.2.4.1.
func (d *decoder) processDQT(n int) error {
loop:
	for n > 0 {
		n--
		x, err := d.readByte()
		if err != nil {
			return err
		}
		tq := x & 0x0f
		if tq > maxTq {
			return FormatError("bad Tq value")
		}
		switch x >> 4 {
		default:
			return FormatError("bad Pq value")
		case 0:
			if n < blockSize {
				break loop
			}
			n -= blockSize
			if err := d.readFull(d.tmp[:blockSize]); err != nil {
				return err
			}
			for i := range d.quant[tq] {
				d.quant[tq][i] = int32(d.tmp[i])
			}
		case 1:
			if n < 2*blockSize {
				break loop
			}
			n -= 2 * blockSize
			if err := d.readFull(d.tmp[:2*blockSize]); err != nil {
				return err
			}
			for i := range d.quant[tq] {
				d.quant[tq][i] = int32(d.tmp[2*i])<<8 | int32(d.tmp[2*i+1])
			}
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
	if err := d.readFull(d.tmp[:2]); err != nil {
		return err
	}
	d.ri = int(d.tmp[0])<<8 + int(d.tmp[1])
	return nil
}

func (d *decoder) processApp0Marker(n int) error {
	if n < 5 {
		return d.ignore(n)
	}
	if err := d.readFull(d.tmp[:5]); err != nil {
		return err
	}
	n -= 5

	d.jfif = d.tmp[0] == 'J' && d.tmp[1] == 'F' && d.tmp[2] == 'I' && d.tmp[3] == 'F' && d.tmp[4] == '\x00'

	if n > 0 {
		return d.ignore(n)
	}
	return nil
}

func (d *decoder) processApp14Marker(n int) error {
	if n < 12 {
		return d.ignore(n)
	}
	if err := d.readFull(d.tmp[:12]); err != nil {
		return err
	}
	n -= 12

	if d.tmp[0] == 'A' && d.tmp[1] == 'd' && d.tmp[2] == 'o' && d.tmp[3] == 'b' && d.tmp[4] == 'e' {
		d.adobeTransformValid = true
		d.adobeTransform = d.tmp[11]
	}

	if n > 0 {
		return d.ignore(n)
	}
	return nil
}

// decode reads a JPEG image from r and returns it as an image.Image.
func (d *decoder) decode(r io.Reader, configOnly bool) (image.Image, error) {
	d.r = r

	// Check for the Start Of Image marker.
	if err := d.readFull(d.tmp[:2]); err != nil {
		return nil, err
	}
	if d.tmp[0] != 0xff || d.tmp[1] != soiMarker {
		return nil, FormatError("missing SOI marker")
	}

	// Process the remaining segments until the End Of Image marker.
	for {
		err := d.readFull(d.tmp[:2])
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
			d.tmp[1], err = d.readByte()
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
			marker, err = d.readByte()
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
		if err = d.readFull(d.tmp[:2]); err != nil {
			return nil, err
		}
		n := int(d.tmp[0])<<8 + int(d.tmp[1]) - 2
		if n < 0 {
			return nil, FormatError("short segment length")
		}

		switch marker {
		case sof0Marker, sof1Marker, sof2Marker:
			d.baseline = marker == sof0Marker
			d.progressive = marker == sof2Marker
			err = d.processSOF(n)
			if configOnly && d.jfif {
				return nil, err
			}
		case dhtMarker:
			if configOnly {
				err = d.ignore(n)
			} else {
				err = d.processDHT(n)
			}
		case dqtMarker:
			if configOnly {
				err = d.ignore(n)
			} else {
				err = d.processDQT(n)
			}
		case sosMarker:
			if configOnly {
				return nil, nil
			}
			err = d.processSOS(n)
		case driMarker:
			if configOnly {
				err = d.ignore(n)
			} else {
				err = d.processDRI(n)
			}
		case app0Marker:
			err = d.processApp0Marker(n)
		case app14Marker:
			err = d.processApp14Marker(n)
		default:
			if app0Marker <= marker && marker <= app15Marker || marker == comMarker {
				err = d.ignore(n)
			} else if marker < 0xc0 { // See Table B.1 "Marker code assignments".
				err = FormatError("unknown marker")
			} else {
				err = UnsupportedError("unknown marker")
			}
		}
		if err != nil {
			return nil, err
		}
	}

	if d.progressive {
		if err := d.reconstructProgressiveImage(); err != nil {
			return nil, err
		}
	}
	if d.img1 != nil {
		return d.img1, nil
	}
	if d.img3 != nil {
		if d.blackPix != nil {
			return d.applyBlack()
		} else if d.isRGB() {
			return d.convertToRGB()
		}
		return d.img3, nil
	}
	return nil, FormatError("missing SOS marker")
}

// applyBlack combines d.img3 and d.blackPix into a CMYK image. The formula
// used depends on whether the JPEG image is stored as CMYK or YCbCrK,
// indicated by the APP14 (Adobe) metadata.
//
// Adobe CMYK JPEG images are inverted, where 255 means no ink instead of full
// ink, so we apply "v = 255 - v" at various points. Note that a double
// inversion is a no-op, so inversions might be implicit in the code below.
func (d *decoder) applyBlack() (image.Image, error) {
	if !d.adobeTransformValid {
		return nil, UnsupportedError("unknown color model: 4-component JPEG doesn't have Adobe APP14 metadata")
	}

	// If the 4-component JPEG image isn't explicitly marked as "Unknown (RGB
	// or CMYK)" as per
	// https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/JPEG.html#Adobe
	// we assume that it is YCbCrK. This matches libjpeg's jdapimin.c.
	if d.adobeTransform != adobeTransformUnknown {
		// Convert the YCbCr part of the YCbCrK to RGB, invert the RGB to get
		// CMY, and patch in the original K. The RGB to CMY inversion cancels
		// out the 'Adobe inversion' described in the applyBlack doc comment
		// above, so in practice, only the fourth channel (black) is inverted.
		bounds := d.img3.Bounds()
		img := image.NewRGBA(bounds)
		imageutil.DrawYCbCr(img, bounds, d.img3, bounds.Min)
		for iBase, y := 0, bounds.Min.Y; y < bounds.Max.Y; iBase, y = iBase+img.Stride, y+1 {
			for i, x := iBase+3, bounds.Min.X; x < bounds.Max.X; i, x = i+4, x+1 {
				img.Pix[i] = 255 - d.blackPix[(y-bounds.Min.Y)*d.blackStride+(x-bounds.Min.X)]
			}
		}
		return &image.CMYK{
			Pix:    img.Pix,
			Stride: img.Stride,
			Rect:   img.Rect,
		}, nil
	}

	// The first three channels (cyan, magenta, yellow) of the CMYK
	// were decoded into d.img3, but each channel was decoded into a separate
	// []byte slice, and some channels may be subsampled. We interleave the
	// separate channels into an image.CMYK's single []byte slice containing 4
	// contiguous bytes per pixel.
	bounds := d.img3.Bounds()
	img := image.NewCMYK(bounds)

	translations := [4]struct {
		src    []byte
		stride int
	}{
		{d.img3.Y, d.img3.YStride},
		{d.img3.Cb, d.img3.CStride},
		{d.img3.Cr, d.img3.CStride},
		{d.blackPix, d.blackStride},
	}
	for t, translation := range translations {
		subsample := d.comp[t].h != d.comp[0].h || d.comp[t].v != d.comp[0].v
		for iBase, y := 0, bounds.Min.Y; y < bounds.Max.Y; iBase, y = iBase+img.Stride, y+1 {
			sy := y - bounds.Min.Y
			if subsample {
				sy /= 2
			}
			for i, x := iBase+t, bounds.Min.X; x < bounds.Max.X; i, x = i+4, x+1 {
				sx := x - bounds.Min.X
				if subsample {
					sx /= 2
				}
				img.Pix[i] = 255 - translation.src[sy*translation.stride+sx]
			}
		}
	}
	return img, nil
}

func (d *decoder) isRGB() bool {
	if d.jfif {
		return false
	}
	if d.adobeTransformValid && d.adobeTransform == adobeTransformUnknown {
		// https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/JPEG.html#Adobe
		// says that 0 means Unknown (and in practice RGB) and 1 means YCbCr.
		return true
	}
	return d.comp[0].c == 'R' && d.comp[1].c == 'G' && d.comp[2].c == 'B'
}

func (d *decoder) convertToRGB() (image.Image, error) {
	cScale := d.comp[0].h / d.comp[1].h
	bounds := d.img3.Bounds()
	img := image.NewRGBA(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		po := img.PixOffset(bounds.Min.X, y)
		yo := d.img3.YOffset(bounds.Min.X, y)
		co := d.img3.COffset(bounds.Min.X, y)
		for i, iMax := 0, bounds.Max.X-bounds.Min.X; i < iMax; i++ {
			img.Pix[po+4*i+0] = d.img3.Y[yo+i]
			img.Pix[po+4*i+1] = d.img3.Cb[co+i/cScale]
			img.Pix[po+4*i+2] = d.img3.Cr[co+i/cScale]
			img.Pix[po+4*i+3] = 255
		}
	}
	return img, nil
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
	case 1:
		return image.Config{
			ColorModel: color.GrayModel,
			Width:      d.width,
			Height:     d.height,
		}, nil
	case 3:
		cm := color.YCbCrModel
		if d.isRGB() {
			cm = color.RGBAModel
		}
		return image.Config{
			ColorModel: cm,
			Width:      d.width,
			Height:     d.height,
		}, nil
	case 4:
		return image.Config{
			ColorModel: color.CMYKModel,
			Width:      d.width,
			Height:     d.height,
		}, nil
	}
	return image.Config{}, FormatError("missing SOF marker")
}

func init() {
	image.RegisterFormat("jpeg", "\xff\xd8", Decode, DecodeConfig)
}
