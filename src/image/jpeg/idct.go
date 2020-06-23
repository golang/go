// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jpeg

// This is a Go translation of idct.c from
//
// http://standards.iso.org/ittf/PubliclyAvailableStandards/ISO_IEC_13818-4_2004_Conformance_Testing/Video/verifier/mpeg2decode_960109.tar.gz
//
// which carries the following notice:

/* Copyright (C) 1996, MPEG Software Simulation Group. All Rights Reserved. */

/*
 * Disclaimer of Warranty
 *
 * These software programs are available to the user without any license fee or
 * royalty on an "as is" basis.  The MPEG Software Simulation Group disclaims
 * any and all warranties, whether express, implied, or statuary, including any
 * implied warranties or merchantability or of fitness for a particular
 * purpose.  In no event shall the copyright-holder be liable for any
 * incidental, punitive, or consequential damages of any kind whatsoever
 * arising from the use of these programs.
 *
 * This disclaimer of warranty extends to the user of these programs and user's
 * customers, employees, agents, transferees, successors, and assigns.
 *
 * The MPEG Software Simulation Group does not represent or warrant that the
 * programs furnished hereunder are free of infringement of any third-party
 * patents.
 *
 * Commercial implementations of MPEG-1 and MPEG-2 video, including shareware,
 * are subject to royalty fees to patent holders.  Many of these patents are
 * general enough such that they are unavoidable regardless of implementation
 * design.
 *
 */

const blockSize = 64 // A DCT block is 8x8.

type block [blockSize]int32

const (
	w1 = 2841 // 2048*sqrt(2)*cos(1*pi/16)
	w2 = 2676 // 2048*sqrt(2)*cos(2*pi/16)
	w3 = 2408 // 2048*sqrt(2)*cos(3*pi/16)
	w5 = 1609 // 2048*sqrt(2)*cos(5*pi/16)
	w6 = 1108 // 2048*sqrt(2)*cos(6*pi/16)
	w7 = 565  // 2048*sqrt(2)*cos(7*pi/16)

	w1pw7 = w1 + w7
	w1mw7 = w1 - w7
	w2pw6 = w2 + w6
	w2mw6 = w2 - w6
	w3pw5 = w3 + w5
	w3mw5 = w3 - w5

	r2 = 181 // 256/sqrt(2)
)

// idct performs a 2-D Inverse Discrete Cosine Transformation.
//
// The input coefficients should already have been multiplied by the
// appropriate quantization table. We use fixed-point computation, with the
// number of bits for the fractional component varying over the intermediate
// stages.
//
// For more on the actual algorithm, see Z. Wang, "Fast algorithms for the
// discrete W transform and for the discrete Fourier transform", IEEE Trans. on
// ASSP, Vol. ASSP- 32, pp. 803-816, Aug. 1984.
func idct(src *block) {
	// Horizontal 1-D IDCT.
	for y := 0; y < 8; y++ {
		y8 := y * 8
		s := src[y8 : y8+8 : y8+8] // Small cap improves performance, see https://golang.org/issue/27857
		// If all the AC components are zero, then the IDCT is trivial.
		if s[1] == 0 && s[2] == 0 && s[3] == 0 &&
			s[4] == 0 && s[5] == 0 && s[6] == 0 && s[7] == 0 {
			dc := s[0] << 3
			s[0] = dc
			s[1] = dc
			s[2] = dc
			s[3] = dc
			s[4] = dc
			s[5] = dc
			s[6] = dc
			s[7] = dc
			continue
		}

		// Prescale.
		x0 := (s[0] << 11) + 128
		x1 := s[4] << 11
		x2 := s[6]
		x3 := s[2]
		x4 := s[1]
		x5 := s[7]
		x6 := s[5]
		x7 := s[3]

		// Stage 1.
		x8 := w7 * (x4 + x5)
		x4 = x8 + w1mw7*x4
		x5 = x8 - w1pw7*x5
		x8 = w3 * (x6 + x7)
		x6 = x8 - w3mw5*x6
		x7 = x8 - w3pw5*x7

		// Stage 2.
		x8 = x0 + x1
		x0 -= x1
		x1 = w6 * (x3 + x2)
		x2 = x1 - w2pw6*x2
		x3 = x1 + w2mw6*x3
		x1 = x4 + x6
		x4 -= x6
		x6 = x5 + x7
		x5 -= x7

		// Stage 3.
		x7 = x8 + x3
		x8 -= x3
		x3 = x0 + x2
		x0 -= x2
		x2 = (r2*(x4+x5) + 128) >> 8
		x4 = (r2*(x4-x5) + 128) >> 8

		// Stage 4.
		s[0] = (x7 + x1) >> 8
		s[1] = (x3 + x2) >> 8
		s[2] = (x0 + x4) >> 8
		s[3] = (x8 + x6) >> 8
		s[4] = (x8 - x6) >> 8
		s[5] = (x0 - x4) >> 8
		s[6] = (x3 - x2) >> 8
		s[7] = (x7 - x1) >> 8
	}

	// Vertical 1-D IDCT.
	for x := 0; x < 8; x++ {
		// Similar to the horizontal 1-D IDCT case, if all the AC components are zero, then the IDCT is trivial.
		// However, after performing the horizontal 1-D IDCT, there are typically non-zero AC components, so
		// we do not bother to check for the all-zero case.
		s := src[x : x+57 : x+57] // Small cap improves performance, see https://golang.org/issue/27857

		// Prescale.
		y0 := (s[8*0] << 8) + 8192
		y1 := s[8*4] << 8
		y2 := s[8*6]
		y3 := s[8*2]
		y4 := s[8*1]
		y5 := s[8*7]
		y6 := s[8*5]
		y7 := s[8*3]

		// Stage 1.
		y8 := w7*(y4+y5) + 4
		y4 = (y8 + w1mw7*y4) >> 3
		y5 = (y8 - w1pw7*y5) >> 3
		y8 = w3*(y6+y7) + 4
		y6 = (y8 - w3mw5*y6) >> 3
		y7 = (y8 - w3pw5*y7) >> 3

		// Stage 2.
		y8 = y0 + y1
		y0 -= y1
		y1 = w6*(y3+y2) + 4
		y2 = (y1 - w2pw6*y2) >> 3
		y3 = (y1 + w2mw6*y3) >> 3
		y1 = y4 + y6
		y4 -= y6
		y6 = y5 + y7
		y5 -= y7

		// Stage 3.
		y7 = y8 + y3
		y8 -= y3
		y3 = y0 + y2
		y0 -= y2
		y2 = (r2*(y4+y5) + 128) >> 8
		y4 = (r2*(y4-y5) + 128) >> 8

		// Stage 4.
		s[8*0] = (y7 + y1) >> 14
		s[8*1] = (y3 + y2) >> 14
		s[8*2] = (y0 + y4) >> 14
		s[8*3] = (y8 + y6) >> 14
		s[8*4] = (y8 - y6) >> 14
		s[8*5] = (y0 - y4) >> 14
		s[8*6] = (y3 - y2) >> 14
		s[8*7] = (y7 - y1) >> 14
	}
}
