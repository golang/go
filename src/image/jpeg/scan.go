// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jpeg

import (
	"image"
)

// makeImg allocates and initializes the destination image.
func (d *decoder) makeImg(mxx, myy int) {
	if d.nComp == 1 {
		m := image.NewGray(image.Rect(0, 0, 8*mxx, 8*myy))
		d.img1 = m.SubImage(image.Rect(0, 0, d.width, d.height)).(*image.Gray)
		return
	}

	h0 := d.comp[0].h
	v0 := d.comp[0].v
	hRatio := h0 / d.comp[1].h
	vRatio := v0 / d.comp[1].v
	var subsampleRatio image.YCbCrSubsampleRatio
	switch hRatio<<4 | vRatio {
	case 0x11:
		subsampleRatio = image.YCbCrSubsampleRatio444
	case 0x12:
		subsampleRatio = image.YCbCrSubsampleRatio440
	case 0x21:
		subsampleRatio = image.YCbCrSubsampleRatio422
	case 0x22:
		subsampleRatio = image.YCbCrSubsampleRatio420
	case 0x41:
		subsampleRatio = image.YCbCrSubsampleRatio411
	case 0x42:
		subsampleRatio = image.YCbCrSubsampleRatio410
	default:
		panic("unreachable")
	}
	m := image.NewYCbCr(image.Rect(0, 0, 8*h0*mxx, 8*v0*myy), subsampleRatio)
	d.img3 = m.SubImage(image.Rect(0, 0, d.width, d.height)).(*image.YCbCr)

	if d.nComp == 4 {
		h3, v3 := d.comp[3].h, d.comp[3].v
		d.blackPix = make([]byte, 8*h3*mxx*8*v3*myy)
		d.blackStride = 8 * h3 * mxx
	}
}

// Specified in section B.2.3.
func (d *decoder) processSOS(n int) error {
	if d.nComp == 0 {
		return FormatError("missing SOF marker")
	}
	if n < 6 || 4+2*d.nComp < n || n%2 != 0 {
		return FormatError("SOS has wrong length")
	}
	if err := d.readFull(d.tmp[:n]); err != nil {
		return err
	}
	nComp := int(d.tmp[0])
	if n != 4+2*nComp {
		return FormatError("SOS length inconsistent with number of components")
	}
	var scan [maxComponents]struct {
		compIndex uint8
		td        uint8 // DC table selector.
		ta        uint8 // AC table selector.
	}
	totalHV := 0
	for i := 0; i < nComp; i++ {
		cs := d.tmp[1+2*i] // Component selector.
		compIndex := -1
		for j, comp := range d.comp[:d.nComp] {
			if cs == comp.c {
				compIndex = j
			}
		}
		if compIndex < 0 {
			return FormatError("unknown component selector")
		}
		scan[i].compIndex = uint8(compIndex)
		// Section B.2.3 states that "the value of Cs_j shall be different from
		// the values of Cs_1 through Cs_(j-1)". Since we have previously
		// verified that a frame's component identifiers (C_i values in section
		// B.2.2) are unique, it suffices to check that the implicit indexes
		// into d.comp are unique.
		for j := 0; j < i; j++ {
			if scan[i].compIndex == scan[j].compIndex {
				return FormatError("repeated component selector")
			}
		}
		totalHV += d.comp[compIndex].h * d.comp[compIndex].v

		// The baseline t <= 1 restriction is specified in table B.3.
		scan[i].td = d.tmp[2+2*i] >> 4
		if t := scan[i].td; t > maxTh || (d.baseline && t > 1) {
			return FormatError("bad Td value")
		}
		scan[i].ta = d.tmp[2+2*i] & 0x0f
		if t := scan[i].ta; t > maxTh || (d.baseline && t > 1) {
			return FormatError("bad Ta value")
		}
	}
	// Section B.2.3 states that if there is more than one component then the
	// total H*V values in a scan must be <= 10.
	if d.nComp > 1 && totalHV > 10 {
		return FormatError("total sampling factors too large")
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
	// For sequential JPEGs, these parameters are hard-coded to 0/63/0/0, as
	// per table B.3.
	zigStart, zigEnd, ah, al := int32(0), int32(blockSize-1), uint32(0), uint32(0)
	if d.progressive {
		zigStart = int32(d.tmp[1+2*nComp])
		zigEnd = int32(d.tmp[2+2*nComp])
		ah = uint32(d.tmp[3+2*nComp] >> 4)
		al = uint32(d.tmp[3+2*nComp] & 0x0f)
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
		d.makeImg(mxx, myy)
	}
	if d.progressive {
		for i := 0; i < nComp; i++ {
			compIndex := scan[i].compIndex
			if d.progCoeffs[compIndex] == nil {
				d.progCoeffs[compIndex] = make([]block, mxx*myy*d.comp[compIndex].h*d.comp[compIndex].v)
			}
		}
	}

	d.bits = bits{}
	mcu, expectedRST := 0, uint8(rst0Marker)
	var (
		// b is the decoded coefficients, in natural (not zig-zag) order.
		b  block
		dc [maxComponents]int32
		// bx and by are the location of the current block, in units of 8x8
		// blocks: the third block in the first row has (bx, by) = (2, 0).
		bx, by     int
		blockCount int
	)
	for my := 0; my < myy; my++ {
		for mx := 0; mx < mxx; mx++ {
			for i := 0; i < nComp; i++ {
				compIndex := scan[i].compIndex
				hi := d.comp[compIndex].h
				vi := d.comp[compIndex].v
				for j := 0; j < hi*vi; j++ {
					// The blocks are traversed one MCU at a time. For 4:2:0 chroma
					// subsampling, there are four Y 8x8 blocks in every 16x16 MCU.
					//
					// For a sequential 32x16 pixel image, the Y blocks visiting order is:
					//	0 1 4 5
					//	2 3 6 7
					//
					// For progressive images, the interleaved scans (those with nComp > 1)
					// are traversed as above, but non-interleaved scans are traversed left
					// to right, top to bottom:
					//	0 1 2 3
					//	4 5 6 7
					// Only DC scans (zigStart == 0) can be interleaved. AC scans must have
					// only one component.
					//
					// To further complicate matters, for non-interleaved scans, there is no
					// data for any blocks that are inside the image at the MCU level but
					// outside the image at the pixel level. For example, a 24x16 pixel 4:2:0
					// progressive image consists of two 16x16 MCUs. The interleaved scans
					// will process 8 Y blocks:
					//	0 1 4 5
					//	2 3 6 7
					// The non-interleaved scans will process only 6 Y blocks:
					//	0 1 2
					//	3 4 5
					if nComp != 1 {
						bx = hi*mx + j%hi
						by = vi*my + j/hi
					} else {
						q := mxx * hi
						bx = blockCount % q
						by = blockCount / q
						blockCount++
						if bx*8 >= d.width || by*8 >= d.height {
							continue
						}
					}

					// Load the previous partially decoded coefficients, if applicable.
					if d.progressive {
						b = d.progCoeffs[compIndex][by*mxx*hi+bx]
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
							huff := &d.huff[acTable][scan[i].ta]
							for ; zig <= zigEnd; zig++ {
								value, err := d.decodeHuffman(huff)
								if err != nil {
									return err
								}
								val0 := value >> 4
								val1 := value & 0x0f
								if val1 != 0 {
									zig += int32(val0)
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
											bits, err := d.decodeBits(int32(val0))
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
						// Save the coefficients.
						d.progCoeffs[compIndex][by*mxx*hi+bx] = b
						// At this point, we could call reconstructBlock to dequantize and perform the
						// inverse DCT, to save early stages of a progressive image to the *image.YCbCr
						// buffers (the whole point of progressive encoding), but in Go, the jpeg.Decode
						// function does not return until the entire image is decoded, so we "continue"
						// here to avoid wasted computation. Instead, reconstructBlock is called on each
						// accumulated block by the reconstructProgressiveImage method after all of the
						// SOS markers are processed.
						continue
					}
					if err := d.reconstructBlock(&b, bx, by, int(compIndex)); err != nil {
						return err
					}
				} // for j
			} // for i
			mcu++
			if d.ri > 0 && mcu%d.ri == 0 && mcu < mxx*myy {
				// For well-formed input, the RST[0-7] restart marker follows
				// immediately. For corrupt input, call findRST to try to
				// resynchronize.
				if err := d.readFull(d.tmp[:2]); err != nil {
					return err
				} else if d.tmp[0] != 0xff || d.tmp[1] != expectedRST {
					if err := d.findRST(expectedRST); err != nil {
						return err
					}
				}
				expectedRST++
				if expectedRST == rst7Marker+1 {
					expectedRST = rst0Marker
				}
				// Reset the Huffman decoder.
				d.bits = bits{}
				// Reset the DC components, as per section F.2.1.3.1.
				dc = [maxComponents]int32{}
				// Reset the progressive decoder state, as per section G.1.2.2.
				d.eobRun = 0
			}
		} // for mx
	} // for my

	return nil
}

// refine decodes a successive approximation refinement block, as specified in
// section G.1.2.
func (d *decoder) refine(b *block, h *huffman, zigStart, zigEnd, delta int32) error {
	// Refining a DC component is trivial.
	if zigStart == 0 {
		if zigEnd != 0 {
			panic("unreachable")
		}
		bit, err := d.decodeBit()
		if err != nil {
			return err
		}
		if bit {
			b[0] |= delta
		}
		return nil
	}

	// Refining AC components is more complicated; see sections G.1.2.2 and G.1.2.3.
	zig := zigStart
	if d.eobRun == 0 {
	loop:
		for ; zig <= zigEnd; zig++ {
			z := int32(0)
			value, err := d.decodeHuffman(h)
			if err != nil {
				return err
			}
			val0 := value >> 4
			val1 := value & 0x0f

			switch val1 {
			case 0:
				if val0 != 0x0f {
					d.eobRun = uint16(1 << val0)
					if val0 != 0 {
						bits, err := d.decodeBits(int32(val0))
						if err != nil {
							return err
						}
						d.eobRun |= uint16(bits)
					}
					break loop
				}
			case 1:
				z = delta
				bit, err := d.decodeBit()
				if err != nil {
					return err
				}
				if !bit {
					z = -z
				}
			default:
				return FormatError("unexpected Huffman code")
			}

			zig, err = d.refineNonZeroes(b, zig, zigEnd, int32(val0), delta)
			if err != nil {
				return err
			}
			if zig > zigEnd {
				return FormatError("too many coefficients")
			}
			if z != 0 {
				b[unzig[zig]] = z
			}
		}
	}
	if d.eobRun > 0 {
		d.eobRun--
		if _, err := d.refineNonZeroes(b, zig, zigEnd, -1, delta); err != nil {
			return err
		}
	}
	return nil
}

// refineNonZeroes refines non-zero entries of b in zig-zag order. If nz >= 0,
// the first nz zero entries are skipped over.
func (d *decoder) refineNonZeroes(b *block, zig, zigEnd, nz, delta int32) (int32, error) {
	for ; zig <= zigEnd; zig++ {
		u := unzig[zig]
		if b[u] == 0 {
			if nz == 0 {
				break
			}
			nz--
			continue
		}
		bit, err := d.decodeBit()
		if err != nil {
			return 0, err
		}
		if !bit {
			continue
		}
		if b[u] >= 0 {
			b[u] += delta
		} else {
			b[u] -= delta
		}
	}
	return zig, nil
}

func (d *decoder) reconstructProgressiveImage() error {
	// The h0, mxx, by and bx variables have the same meaning as in the
	// processSOS method.
	h0 := d.comp[0].h
	mxx := (d.width + 8*h0 - 1) / (8 * h0)
	for i := 0; i < d.nComp; i++ {
		if d.progCoeffs[i] == nil {
			continue
		}
		v := 8 * d.comp[0].v / d.comp[i].v
		h := 8 * d.comp[0].h / d.comp[i].h
		stride := mxx * d.comp[i].h
		for by := 0; by*v < d.height; by++ {
			for bx := 0; bx*h < d.width; bx++ {
				if err := d.reconstructBlock(&d.progCoeffs[i][by*stride+bx], bx, by, i); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// reconstructBlock dequantizes, performs the inverse DCT and stores the block
// to the image.
func (d *decoder) reconstructBlock(b *block, bx, by, compIndex int) error {
	qt := &d.quant[d.comp[compIndex].tq]

	// This sequence exactly follows the indexes of the unzig mapping.
	b[0] *= qt[0]
	b[1] *= qt[1]
	b[8] *= qt[2]
	b[16] *= qt[3]
	b[9] *= qt[4]
	b[2] *= qt[5]
	b[3] *= qt[6]
	b[10] *= qt[7]
	b[17] *= qt[8]
	b[24] *= qt[9]
	b[32] *= qt[10]
	b[25] *= qt[11]
	b[18] *= qt[12]
	b[11] *= qt[13]
	b[4] *= qt[14]
	b[5] *= qt[15]
	b[12] *= qt[16]
	b[19] *= qt[17]
	b[26] *= qt[18]
	b[33] *= qt[19]
	b[40] *= qt[20]
	b[48] *= qt[21]
	b[41] *= qt[22]
	b[34] *= qt[23]
	b[27] *= qt[24]
	b[20] *= qt[25]
	b[13] *= qt[26]
	b[6] *= qt[27]
	b[7] *= qt[28]
	b[14] *= qt[29]
	b[21] *= qt[30]
	b[28] *= qt[31]
	b[35] *= qt[32]
	b[42] *= qt[33]
	b[49] *= qt[34]
	b[56] *= qt[35]
	b[57] *= qt[36]
	b[50] *= qt[37]
	b[43] *= qt[38]
	b[36] *= qt[39]
	b[29] *= qt[40]
	b[22] *= qt[41]
	b[15] *= qt[42]
	b[23] *= qt[43]
	b[30] *= qt[44]
	b[37] *= qt[45]
	b[44] *= qt[46]
	b[51] *= qt[47]
	b[58] *= qt[48]
	b[59] *= qt[49]
	b[52] *= qt[50]
	b[45] *= qt[51]
	b[38] *= qt[52]
	b[31] *= qt[53]
	b[39] *= qt[54]
	b[46] *= qt[55]
	b[53] *= qt[56]
	b[60] *= qt[57]
	b[61] *= qt[58]
	b[54] *= qt[59]
	b[47] *= qt[60]
	b[55] *= qt[61]
	b[62] *= qt[62]
	b[63] *= qt[63]

	idct(b)
	dst, stride := []byte(nil), 0
	if d.nComp == 1 {
		dst, stride = d.img1.Pix[8*(by*d.img1.Stride+bx):], d.img1.Stride
	} else {
		switch compIndex {
		case 0:
			dst, stride = d.img3.Y[8*(by*d.img3.YStride+bx):], d.img3.YStride
		case 1:
			dst, stride = d.img3.Cb[8*(by*d.img3.CStride+bx):], d.img3.CStride
		case 2:
			dst, stride = d.img3.Cr[8*(by*d.img3.CStride+bx):], d.img3.CStride
		case 3:
			dst, stride = d.blackPix[8*(by*d.blackStride+bx):], d.blackStride
		default:
			return UnsupportedError("too many components")
		}
	}

	// Level shift by +128, clip to [0, 255], and write to dst.
	writeDst := func(index int) {
		c := (*b)[index] + 128
		if c < 0 {
			c = 0
		} else if c > 255 {
			c = 255
		}
		dst[(index/8)*stride+(index%8)] = uint8(c)
	}
	writeDst(0)
	writeDst(1)
	writeDst(2)
	writeDst(3)
	writeDst(4)
	writeDst(5)
	writeDst(6)
	writeDst(7)
	writeDst(8)
	writeDst(9)
	writeDst(10)
	writeDst(11)
	writeDst(12)
	writeDst(13)
	writeDst(14)
	writeDst(15)
	writeDst(16)
	writeDst(17)
	writeDst(18)
	writeDst(19)
	writeDst(20)
	writeDst(21)
	writeDst(22)
	writeDst(23)
	writeDst(24)
	writeDst(25)
	writeDst(26)
	writeDst(27)
	writeDst(28)
	writeDst(29)
	writeDst(30)
	writeDst(31)
	writeDst(32)
	writeDst(33)
	writeDst(34)
	writeDst(35)
	writeDst(36)
	writeDst(37)
	writeDst(38)
	writeDst(39)
	writeDst(40)
	writeDst(41)
	writeDst(42)
	writeDst(43)
	writeDst(44)
	writeDst(45)
	writeDst(46)
	writeDst(47)
	writeDst(48)
	writeDst(49)
	writeDst(50)
	writeDst(51)
	writeDst(52)
	writeDst(53)
	writeDst(54)
	writeDst(55)
	writeDst(56)
	writeDst(57)
	writeDst(58)
	writeDst(59)
	writeDst(60)
	writeDst(61)
	writeDst(62)
	writeDst(63)

	return nil
}

// findRST advances past the next RST restart marker that matches expectedRST.
// Other than I/O errors, it is also an error if we encounter an {0xFF, M}
// two-byte marker sequence where M is not 0x00, 0xFF or the expectedRST.
//
// This is similar to libjpeg's jdmarker.c's next_marker function.
// https://github.com/libjpeg-turbo/libjpeg-turbo/blob/2dfe6c0fe9e18671105e94f7cbf044d4a1d157e6/jdmarker.c#L892-L935
//
// Precondition: d.tmp[:2] holds the next two bytes of JPEG-encoded input
// (input in the d.readFull sense).
func (d *decoder) findRST(expectedRST uint8) error {
	for {
		// i is the index such that, at the bottom of the loop, we read 2-i
		// bytes into d.tmp[i:2], maintaining the invariant that d.tmp[:2]
		// holds the next two bytes of JPEG-encoded input. It is either 0 or 1,
		// so that each iteration advances by 1 or 2 bytes (or returns).
		i := 0

		if d.tmp[0] == 0xff {
			if d.tmp[1] == expectedRST {
				return nil
			} else if d.tmp[1] == 0xff {
				i = 1
			} else if d.tmp[1] != 0x00 {
				// libjpeg's jdmarker.c's jpeg_resync_to_restart does something
				// fancy here, treating RST markers within two (modulo 8) of
				// expectedRST differently from RST markers that are 'more
				// distant'. Until we see evidence that recovering from such
				// cases is frequent enough to be worth the complexity, we take
				// a simpler approach for now. Any marker that's not 0x00, 0xff
				// or expectedRST is a fatal FormatError.
				return FormatError("bad RST marker")
			}

		} else if d.tmp[1] == 0xff {
			d.tmp[0] = 0xff
			i = 1
		}

		if err := d.readFull(d.tmp[i:2]); err != nil {
			return err
		}
	}
}
