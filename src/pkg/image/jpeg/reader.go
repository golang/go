// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The jpeg package implements a decoder for JPEG images, as defined in ITU-T T.81.
package jpeg

// See http://www.w3.org/Graphics/JPEG/itu-t81.pdf

import (
	"bufio"
	"image"
	"io"
	"os"
)

// A FormatError reports that the input is not a valid JPEG.
type FormatError string

func (e FormatError) String() string { return "invalid JPEG format: " + string(e) }

// An UnsupportedError reports that the input uses a valid but unimplemented JPEG feature.
type UnsupportedError string

func (e UnsupportedError) String() string { return "unsupported JPEG feature: " + string(e) }

// Component specification, specified in section B.2.2.
type component struct {
	c  uint8 // Component identifier.
	h  uint8 // Horizontal sampling factor.
	v  uint8 // Vertical sampling factor.
	tq uint8 // Quantization table destination selector.
}

const (
	blockSize = 64 // A DCT block is 8x8.

	dcTableClass = 0
	acTableClass = 1
	maxTc        = 1
	maxTh        = 3
	maxTq        = 3

	// We only support 4:4:4, 4:2:2 and 4:2:0 downsampling, and assume that the components are Y, Cb, Cr.
	nComponent = 3
	maxH       = 2
	maxV       = 2
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
	ReadByte() (c byte, err os.Error)
}

type decoder struct {
	r             Reader
	width, height int
	image         *image.RGBA
	ri            int // Restart Interval.
	comps         [nComponent]component
	huff          [maxTc + 1][maxTh + 1]huffman
	quant         [maxTq + 1][blockSize]int
	b             bits
	blocks        [nComponent][maxH * maxV][blockSize]int
	tmp           [1024]byte
}

// Reads and ignores the next n bytes.
func (d *decoder) ignore(n int) os.Error {
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
func (d *decoder) processSOF(n int) os.Error {
	if n != 6+3*nComponent {
		return UnsupportedError("SOF has wrong length")
	}
	_, err := io.ReadFull(d.r, d.tmp[0:6+3*nComponent])
	if err != nil {
		return err
	}
	// We only support 8-bit precision.
	if d.tmp[0] != 8 {
		return UnsupportedError("precision")
	}
	d.height = int(d.tmp[1])<<8 + int(d.tmp[2])
	d.width = int(d.tmp[3])<<8 + int(d.tmp[4])
	if d.tmp[5] != nComponent {
		return UnsupportedError("SOF has wrong number of image components")
	}
	for i := 0; i < nComponent; i++ {
		hv := d.tmp[7+3*i]
		d.comps[i].c = d.tmp[6+3*i]
		d.comps[i].h = hv >> 4
		d.comps[i].v = hv & 0x0f
		d.comps[i].tq = d.tmp[8+3*i]
		// We only support YCbCr images, and 4:4:4, 4:2:2 or 4:2:0 chroma downsampling ratios. This implies that
		// the (h, v) values for the Y component are either (1, 1), (2, 1) or (2, 2), and the
		// (h, v) values for the Cr and Cb components must be (1, 1).
		if i == 0 {
			if hv != 0x11 && hv != 0x21 && hv != 0x22 {
				return UnsupportedError("luma downsample ratio")
			}
		} else {
			if hv != 0x11 {
				return UnsupportedError("chroma downsample ratio")
			}
		}
	}
	return nil
}

// Specified in section B.2.4.1.
func (d *decoder) processDQT(n int) os.Error {
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

// Set the Pixel (px, py)'s RGB value, based on its YCbCr value.
func (d *decoder) calcPixel(px, py, lumaBlock, lumaIndex, chromaIndex int) {
	y, cb, cr := d.blocks[0][lumaBlock][lumaIndex], d.blocks[1][0][chromaIndex], d.blocks[2][0][chromaIndex]
	// The JFIF specification (http://www.w3.org/Graphics/JPEG/jfif3.pdf, page 3) gives the formula
	// for translating YCbCr to RGB as:
	//   R = Y + 1.402 (Cr-128)
	//   G = Y - 0.34414 (Cb-128) - 0.71414 (Cr-128)
	//   B = Y + 1.772 (Cb-128)
	yPlusHalf := 100000*y + 50000
	cb -= 128
	cr -= 128
	r := (yPlusHalf + 140200*cr) / 100000
	g := (yPlusHalf - 34414*cb - 71414*cr) / 100000
	b := (yPlusHalf + 177200*cb) / 100000
	if r < 0 {
		r = 0
	} else if r > 255 {
		r = 255
	}
	if g < 0 {
		g = 0
	} else if g > 255 {
		g = 255
	}
	if b < 0 {
		b = 0
	} else if b > 255 {
		b = 255
	}
	d.image.Pix[py*d.image.Stride+px] = image.RGBAColor{uint8(r), uint8(g), uint8(b), 0xff}
}

// Convert the MCU from YCbCr to RGB.
func (d *decoder) convertMCU(mx, my, h0, v0 int) {
	lumaBlock := 0
	for v := 0; v < v0; v++ {
		for h := 0; h < h0; h++ {
			chromaBase := 8*4*v + 4*h
			py := 8 * (v0*my + v)
			for y := 0; y < 8 && py < d.height; y++ {
				px := 8 * (h0*mx + h)
				lumaIndex := 8 * y
				chromaIndex := chromaBase + 8*(y/v0)
				for x := 0; x < 8 && px < d.width; x++ {
					d.calcPixel(px, py, lumaBlock, lumaIndex, chromaIndex)
					if h0 == 1 {
						chromaIndex += 1
					} else {
						chromaIndex += x % 2
					}
					lumaIndex++
					px++
				}
				py++
			}
			lumaBlock++
		}
	}
}

// Specified in section B.2.3.
func (d *decoder) processSOS(n int) os.Error {
	if d.image == nil {
		d.image = image.NewRGBA(d.width, d.height)
	}
	if n != 4+2*nComponent {
		return UnsupportedError("SOS has wrong length")
	}
	_, err := io.ReadFull(d.r, d.tmp[0:4+2*nComponent])
	if err != nil {
		return err
	}
	if d.tmp[0] != nComponent {
		return UnsupportedError("SOS has wrong number of image components")
	}
	var scanComps [nComponent]struct {
		td uint8 // DC table selector.
		ta uint8 // AC table selector.
	}
	h0, v0 := int(d.comps[0].h), int(d.comps[0].v) // The h and v values from the Y components.
	for i := 0; i < nComponent; i++ {
		cs := d.tmp[1+2*i] // Component selector.
		if cs != d.comps[i].c {
			return UnsupportedError("scan components out of order")
		}
		scanComps[i].td = d.tmp[2+2*i] >> 4
		scanComps[i].ta = d.tmp[2+2*i] & 0x0f
	}
	// mxx and myy are the number of MCUs (Minimum Coded Units) in the image.
	mxx := (d.width + 8*int(h0) - 1) / (8 * int(h0))
	myy := (d.height + 8*int(v0) - 1) / (8 * int(v0))

	mcu, expectedRST := 0, uint8(rst0Marker)
	var allZeroes [blockSize]int
	var dc [nComponent]int
	for my := 0; my < myy; my++ {
		for mx := 0; mx < mxx; mx++ {
			for i := 0; i < nComponent; i++ {
				qt := &d.quant[d.comps[i].tq]
				for j := 0; j < int(d.comps[i].h*d.comps[i].v); j++ {
					d.blocks[i][j] = allZeroes

					// Decode the DC coefficient, as specified in section F.2.2.1.
					value, err := d.decodeHuffman(&d.huff[dcTableClass][scanComps[i].td])
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
					d.blocks[i][j][0] = dc[i] * qt[0]

					// Decode the AC coefficients, as specified in section F.2.2.2.
					for k := 1; k < blockSize; k++ {
						value, err := d.decodeHuffman(&d.huff[acTableClass][scanComps[i].ta])
						if err != nil {
							return err
						}
						v0 := value >> 4
						v1 := value & 0x0f
						if v1 != 0 {
							k += int(v0)
							if k > blockSize {
								return FormatError("bad DCT index")
							}
							ac, err := d.receiveExtend(v1)
							if err != nil {
								return err
							}
							d.blocks[i][j][unzig[k]] = ac * qt[k]
						} else {
							if v0 != 0x0f {
								break
							}
							k += 0x0f
						}
					}

					idct(&d.blocks[i][j])
				} // for j
			} // for i
			d.convertMCU(mx, my, int(d.comps[0].h), int(d.comps[0].v))
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
				for i := 0; i < nComponent; i++ {
					dc[i] = 0
				}
			}
		} // for mx
	} // for my

	return nil
}

// Specified in section B.2.4.4.
func (d *decoder) processDRI(n int) os.Error {
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
func (d *decoder) decode(r io.Reader, configOnly bool) (image.Image, os.Error) {
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
	return d.image, nil
}

// Decode reads a JPEG image from r and returns it as an image.Image.
func Decode(r io.Reader) (image.Image, os.Error) {
	var d decoder
	return d.decode(r, false)
}

// DecodeConfig returns the color model and dimensions of a JPEG image without
// decoding the entire image.
func DecodeConfig(r io.Reader) (image.Config, os.Error) {
	var d decoder
	if _, err := d.decode(r, true); err != nil {
		return image.Config{}, err
	}
	return image.Config{image.RGBAColorModel, d.width, d.height}, nil
}

func init() {
	image.RegisterFormat("jpeg", "\xff\xd8", Decode, DecodeConfig)
}
