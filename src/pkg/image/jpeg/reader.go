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

type block [blockSize]int

const (
	blockSize = 64 // A DCT block is 8x8.

	dcTable = 0
	acTable = 1
	maxTc   = 1
	maxTh   = 3
	maxTq   = 3

	// A grayscale JPEG image has only a Y component.
	nGrayComponent = 1
	// A color JPEG image has Y, Cb and Cr components.
	nColorComponent = 3

	// We only support 4:4:4, 4:2:2 and 4:2:0 downsampling, and therefore the
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

// Maps from the zig-zag ordering to the natural ordering.
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
	width, height int
	img1          *image.Gray
	img3          *image.YCbCr
	ri            int // Restart Interval.
	nComp         int
	comp          [nColorComponent]component
	huff          [maxTc + 1][maxTh + 1]huffman
	quant         [maxTq + 1]block
	b             bits
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
		hv := d.tmp[7+3*i]
		d.comp[i].h = int(hv >> 4)
		d.comp[i].v = int(hv & 0x0f)
		d.comp[i].c = d.tmp[6+3*i]
		d.comp[i].tq = d.tmp[8+3*i]
		if d.nComp == nGrayComponent {
			continue
		}
		// For color images, we only support 4:4:4, 4:2:2 or 4:2:0 chroma
		// downsampling ratios. This implies that the (h, v) values for the Y
		// component are either (1, 1), (2, 1) or (2, 2), and the (h, v)
		// values for the Cr and Cb components must be (1, 1).
		if i == 0 {
			if hv != 0x11 && hv != 0x21 && hv != 0x22 {
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
			d.quant[tq][i] = int(d.tmp[i+1])
		}
	}
	if n != 0 {
		return FormatError("DQT has wrong length")
	}
	return nil
}

// makeImg allocates and initializes the destination image.
func (d *decoder) makeImg(h0, v0, mxx, myy int) {
	if d.nComp == nGrayComponent {
		m := image.NewGray(image.Rect(0, 0, 8*mxx, 8*myy))
		d.img1 = m.SubImage(image.Rect(0, 0, d.width, d.height)).(*image.Gray)
		return
	}
	var subsampleRatio image.YCbCrSubsampleRatio
	switch h0 * v0 {
	case 1:
		subsampleRatio = image.YCbCrSubsampleRatio444
	case 2:
		subsampleRatio = image.YCbCrSubsampleRatio422
	case 4:
		subsampleRatio = image.YCbCrSubsampleRatio420
	default:
		panic("unreachable")
	}
	m := image.NewYCbCr(image.Rect(0, 0, 8*h0*mxx, 8*v0*myy), subsampleRatio)
	d.img3 = m.SubImage(image.Rect(0, 0, d.width, d.height)).(*image.YCbCr)
}

// Specified in section B.2.3.
func (d *decoder) processSOS(n int) error {
	if d.nComp == 0 {
		return FormatError("missing SOF marker")
	}
	if n != 4+2*d.nComp {
		return UnsupportedError("SOS has wrong length")
	}
	_, err := io.ReadFull(d.r, d.tmp[0:4+2*d.nComp])
	if err != nil {
		return err
	}
	if int(d.tmp[0]) != d.nComp {
		return UnsupportedError("SOS has wrong number of image components")
	}
	var scan [nColorComponent]struct {
		td uint8 // DC table selector.
		ta uint8 // AC table selector.
	}
	for i := 0; i < d.nComp; i++ {
		cs := d.tmp[1+2*i] // Component selector.
		if cs != d.comp[i].c {
			return UnsupportedError("scan components out of order")
		}
		scan[i].td = d.tmp[2+2*i] >> 4
		scan[i].ta = d.tmp[2+2*i] & 0x0f
	}
	// mxx and myy are the number of MCUs (Minimum Coded Units) in the image.
	h0, v0 := d.comp[0].h, d.comp[0].v // The h and v values from the Y components.
	mxx := (d.width + 8*h0 - 1) / (8 * h0)
	myy := (d.height + 8*v0 - 1) / (8 * v0)
	if d.img1 == nil && d.img3 == nil {
		d.makeImg(h0, v0, mxx, myy)
	}

	mcu, expectedRST := 0, uint8(rst0Marker)
	var (
		b  block
		dc [nColorComponent]int
	)
	for my := 0; my < myy; my++ {
		for mx := 0; mx < mxx; mx++ {
			for i := 0; i < d.nComp; i++ {
				qt := &d.quant[d.comp[i].tq]
				for j := 0; j < d.comp[i].h*d.comp[i].v; j++ {
					// TODO(nigeltao): make this a "var b block" once the compiler's escape
					// analysis is good enough to allocate it on the stack, not the heap.
					b = block{}

					// Decode the DC coefficient, as specified in section F.2.2.1.
					value, err := d.decodeHuffman(&d.huff[dcTable][scan[i].td])
					if err != nil {
						return err
					}
					if value > 16 {
						return UnsupportedError("excessive DC component")
					}
					dcDelta, err := d.receiveExtend(value)
					if err != nil {
						return err
					}
					dc[i] += dcDelta
					b[0] = dc[i] * qt[0]

					// Decode the AC coefficients, as specified in section F.2.2.2.
					for k := 1; k < blockSize; k++ {
						value, err := d.decodeHuffman(&d.huff[acTable][scan[i].ta])
						if err != nil {
							return err
						}
						val0 := value >> 4
						val1 := value & 0x0f
						if val1 != 0 {
							k += int(val0)
							if k > blockSize {
								return FormatError("bad DCT index")
							}
							ac, err := d.receiveExtend(val1)
							if err != nil {
								return err
							}
							b[unzig[k]] = ac * qt[k]
						} else {
							if val0 != 0x0f {
								break
							}
							k += 0x0f
						}
					}

					// Perform the inverse DCT and store the MCU component to the image.
					if d.nComp == nGrayComponent {
						idct(d.img1.Pix[8*(my*d.img1.Stride+mx):], d.img1.Stride, &b)
					} else {
						switch i {
						case 0:
							mx0 := h0*mx + (j % 2)
							my0 := v0*my + (j / 2)
							idct(d.img3.Y[8*(my0*d.img3.YStride+mx0):], d.img3.YStride, &b)
						case 1:
							idct(d.img3.Cb[8*(my*d.img3.CStride+mx):], d.img3.CStride, &b)
						case 2:
							idct(d.img3.Cr[8*(my*d.img3.CStride+mx):], d.img3.CStride, &b)
						}
					}
				} // for j
			} // for i
			mcu++
			if d.ri > 0 && mcu%d.ri == 0 && mcu < mxx*myy {
				// A more sophisticated decoder could use RST[0-7] markers to resynchronize from corrupt input,
				// but this one assumes well-formed input, and hence the restart marker follows immediately.
				_, err := io.ReadFull(d.r, d.tmp[0:2])
				if err != nil {
					return err
				}
				if d.tmp[0] != 0xff || d.tmp[1] != expectedRST {
					return FormatError("bad RST marker")
				}
				expectedRST++
				if expectedRST == rst7Marker+1 {
					expectedRST = rst0Marker
				}
				// Reset the Huffman decoder.
				d.b = bits{}
				// Reset the DC components, as per section F.2.1.3.1.
				dc = [nColorComponent]int{}
			}
		} // for mx
	} // for my

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
		if d.tmp[0] != 0xff {
			return nil, FormatError("missing 0xff marker start")
		}
		marker := d.tmp[1]
		if marker == eoiMarker { // End Of Image.
			break
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
		case marker == sof0Marker: // Start Of Frame (Baseline).
			err = d.processSOF(n)
			if configOnly {
				return nil, err
			}
		case marker == sof2Marker: // Start Of Frame (Progressive).
			err = UnsupportedError("progressive mode")
		case marker == dhtMarker: // Define Huffman Table.
			err = d.processDHT(n)
		case marker == dqtMarker: // Define Quantization Table.
			err = d.processDQT(n)
		case marker == sosMarker: // Start Of Scan.
			err = d.processSOS(n)
		case marker == driMarker: // Define Restart Interval.
			err = d.processDRI(n)
		case marker >= app0Marker && marker <= app15Marker || marker == comMarker: // APPlication specific, or COMment.
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
		return image.Config{color.GrayModel, d.width, d.height}, nil
	case nColorComponent:
		return image.Config{color.YCbCrModel, d.width, d.height}, nil
	}
	return image.Config{}, FormatError("missing SOF marker")
}

func init() {
	image.RegisterFormat("jpeg", "\xff\xd8", Decode, DecodeConfig)
}
