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
		hv := d.tmp[7+3*i]
		d.comp[i].h = int(hv >> 4)
		d.comp[i].v = int(hv & 0x0f)
		d.comp[i].c = d.tmp[6+3*i]
		d.comp[i].tq = d.tmp[8+3*i]
		if d.nComp == nGrayComponent {
			continue
		}
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
	switch {
	case h0 == 1 && v0 == 1:
		subsampleRatio = image.YCbCrSubsampleRatio444
	case h0 == 1 && v0 == 2:
		subsampleRatio = image.YCbCrSubsampleRatio440
	case h0 == 2 && v0 == 1:
		subsampleRatio = image.YCbCrSubsampleRatio422
	case h0 == 2 && v0 == 2:
		subsampleRatio = image.YCbCrSubsampleRatio420
	default:
		panic("unreachable")
	}
	m := image.NewYCbCr(image.Rect(0, 0, 8*h0*mxx, 8*v0*myy), subsampleRatio)
	d.img3 = m.SubImage(image.Rect(0, 0, d.width, d.height)).(*image.YCbCr)
}

// TODO(nigeltao): move processSOS to scan.go.

// Specified in section B.2.3.
func (d *decoder) processSOS(n int) error {
	if d.nComp == 0 {
		return FormatError("missing SOF marker")
	}
	if n < 6 || 4+2*d.nComp < n || n%2 != 0 {
		return FormatError("SOS has wrong length")
	}
	_, err := io.ReadFull(d.r, d.tmp[:n])
	if err != nil {
		return err
	}
	nComp := int(d.tmp[0])
	if n != 4+2*nComp {
		return FormatError("SOS length inconsistent with number of components")
	}
	var scan [nColorComponent]struct {
		compIndex uint8
		td        uint8 // DC table selector.
		ta        uint8 // AC table selector.
	}
	for i := 0; i < nComp; i++ {
		cs := d.tmp[1+2*i] // Component selector.
		compIndex := -1
		for j, comp := range d.comp {
			if cs == comp.c {
				compIndex = j
			}
		}
		if compIndex < 0 {
			return FormatError("unknown component selector")
		}
		scan[i].compIndex = uint8(compIndex)
		scan[i].td = d.tmp[2+2*i] >> 4
		scan[i].ta = d.tmp[2+2*i] & 0x0f
	}

	// zigStart and zigEnd are the spectral selection bounds.
	// ah and al are the successive approximation high and low values.
	// The spec calls these values Ss, Se, Ah and Al.
	//
	// For progressive JPEGs, these are the two more-or-less independent
	// aspects of progression. Spectral selection progression is when not
	// all of a block's 64 DCT coefficients are transmitted in one pass.
	// For example, three passes could transmit coefficient 0 (the DC
	// component), coefficients 1-5, and coefficients 6-63, in zig-zag
	// order. Successive approximation is when not all of the bits of a
	// band of coefficients are transmitted in one pass. For example,
	// three passes could transmit the 6 most significant bits, followed
	// by the second-least significant bit, followed by the least
	// significant bit.
	//
	// For baseline JPEGs, these parameters are hard-coded to 0/63/0/0.
	zigStart, zigEnd, ah, al := 0, blockSize-1, uint(0), uint(0)
	if d.progressive {
		zigStart = int(d.tmp[1+2*nComp])
		zigEnd = int(d.tmp[2+2*nComp])
		ah = uint(d.tmp[3+2*nComp] >> 4)
		al = uint(d.tmp[3+2*nComp] & 0x0f)
		if (zigStart == 0 && zigEnd != 0) || zigStart > zigEnd || blockSize <= zigEnd {
			return FormatError("bad spectral selection bounds")
		}
		if zigStart != 0 && nComp != 1 {
			return FormatError("progressive AC coefficients for more than one component")
		}
		if ah != 0 && ah != al+1 {
			return FormatError("bad successive approximation values")
		}
	}

	// mxx and myy are the number of MCUs (Minimum Coded Units) in the image.
	h0, v0 := d.comp[0].h, d.comp[0].v // The h and v values from the Y components.
	mxx := (d.width + 8*h0 - 1) / (8 * h0)
	myy := (d.height + 8*v0 - 1) / (8 * v0)
	if d.img1 == nil && d.img3 == nil {
		d.makeImg(h0, v0, mxx, myy)
		if d.progressive {
			for i := 0; i < nComp; i++ {
				compIndex := scan[i].compIndex
				d.progCoeffs[compIndex] = make([]block, mxx*myy*d.comp[compIndex].h*d.comp[compIndex].v)
			}
		}
	}

	d.b = bits{}
	mcu, expectedRST := 0, uint8(rst0Marker)
	var (
		// b is the decoded coefficients, in natural (not zig-zag) order.
		b  block
		dc [nColorComponent]int
		// mx0 and my0 are the location of the current (in terms of 8x8 blocks).
		// For example, with 4:2:0 chroma subsampling, the block whose top left
		// pixel co-ordinates are (16, 8) is the third block in the first row:
		// mx0 is 2 and my0 is 0, even though the pixel is in the second MCU.
		// TODO(nigeltao): rename mx0 and my0 to bx and by?
		mx0, my0   int
		blockCount int
	)
	for my := 0; my < myy; my++ {
		for mx := 0; mx < mxx; mx++ {
			for i := 0; i < nComp; i++ {
				compIndex := scan[i].compIndex
				qt := &d.quant[d.comp[compIndex].tq]
				for j := 0; j < d.comp[compIndex].h*d.comp[compIndex].v; j++ {
					// The blocks are traversed one MCU at a time. For 4:2:0 chroma
					// subsampling, there are four Y 8x8 blocks in every 16x16 MCU.
					// For a baseline 32x16 pixel image, the Y blocks visiting order is:
					//	0 1 4 5
					//	2 3 6 7
					//
					// For progressive images, the DC data blocks (zigStart == 0) are traversed
					// as above, but AC data blocks are traversed left to right, top to bottom:
					//	0 1 2 3
					//	4 5 6 7
					//
					// To further complicate matters, there is no AC data for any blocks that
					// are inside the image at the MCU level but outside the image at the pixel
					// level. For example, a 24x16 pixel 4:2:0 progressive image consists of
					// two 16x16 MCUs. The earlier scans will process 8 Y blocks:
					//	0 1 4 5
					//	2 3 6 7
					// The later scans will process only 6 Y blocks:
					//	0 1 2
					//	3 4 5
					if zigStart == 0 {
						mx0, my0 = d.comp[compIndex].h*mx, d.comp[compIndex].v*my
						if h0 == 1 {
							my0 += j
						} else {
							mx0 += j % 2
							my0 += j / 2
						}
					} else {
						q := mxx * d.comp[compIndex].h
						mx0 = blockCount % q
						my0 = blockCount / q
						blockCount++
						if mx0*8 >= d.width || my0*8 >= d.height {
							continue
						}
					}

					// Load the previous partially decoded coefficients, if applicable.
					if d.progressive {
						b = d.progCoeffs[compIndex][my0*mxx*d.comp[compIndex].h+mx0]
					} else {
						b = block{}
					}

					if ah != 0 {
						if err := d.refine(&b, &d.huff[acTable][scan[i].ta], zigStart, zigEnd, 1<<al); err != nil {
							return err
						}
					} else {
						zig := zigStart
						if zig == 0 {
							zig++
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
							dc[compIndex] += dcDelta
							b[0] = dc[compIndex] << al
						}

						if zig <= zigEnd && d.eobRun > 0 {
							d.eobRun--
						} else {
							// Decode the AC coefficients, as specified in section F.2.2.2.
							for ; zig <= zigEnd; zig++ {
								value, err := d.decodeHuffman(&d.huff[acTable][scan[i].ta])
								if err != nil {
									return err
								}
								val0 := value >> 4
								val1 := value & 0x0f
								if val1 != 0 {
									zig += int(val0)
									if zig > zigEnd {
										break
									}
									ac, err := d.receiveExtend(val1)
									if err != nil {
										return err
									}
									b[unzig[zig]] = ac << al
								} else {
									if val0 != 0x0f {
										d.eobRun = uint16(1 << val0)
										if val0 != 0 {
											bits, err := d.decodeBits(int(val0))
											if err != nil {
												return err
											}
											d.eobRun |= uint16(bits)
										}
										d.eobRun--
										break
									}
									zig += 0x0f
								}
							}
						}
					}

					if d.progressive {
						if zigEnd != blockSize-1 || al != 0 {
							// We haven't completely decoded this 8x8 block. Save the coefficients.
							d.progCoeffs[compIndex][my0*mxx*d.comp[compIndex].h+mx0] = b
							// At this point, we could execute the rest of the loop body to dequantize and
							// perform the inverse DCT, to save early stages of a progressive image to the
							// *image.YCbCr buffers (the whole point of progressive encoding), but in Go,
							// the jpeg.Decode function does not return until the entire image is decoded,
							// so we "continue" here to avoid wasted computation.
							continue
						}
					}

					// Dequantize, perform the inverse DCT and store the block to the image.
					for zig := 0; zig < blockSize; zig++ {
						b[unzig[zig]] *= qt[zig]
					}
					idct(&b)
					dst, stride := []byte(nil), 0
					if d.nComp == nGrayComponent {
						dst, stride = d.img1.Pix[8*(my0*d.img1.Stride+mx0):], d.img1.Stride
					} else {
						switch compIndex {
						case 0:
							dst, stride = d.img3.Y[8*(my0*d.img3.YStride+mx0):], d.img3.YStride
						case 1:
							dst, stride = d.img3.Cb[8*(my0*d.img3.CStride+mx0):], d.img3.CStride
						case 2:
							dst, stride = d.img3.Cr[8*(my0*d.img3.CStride+mx0):], d.img3.CStride
						default:
							return UnsupportedError("too many components")
						}
					}
					// Level shift by +128, clip to [0, 255], and write to dst.
					for y := 0; y < 8; y++ {
						y8 := y * 8
						yStride := y * stride
						for x := 0; x < 8; x++ {
							c := b[y8+x]
							if c < -128 {
								c = 0
							} else if c > 127 {
								c = 255
							} else {
								c += 128
							}
							dst[yStride+x] = uint8(c)
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
				// Reset the progressive decoder state, as per section G.1.2.2.
				d.eobRun = 0
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
