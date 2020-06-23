// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jpeg

// This file implements a Forward Discrete Cosine Transformation.

/*
It is based on the code in jfdctint.c from the Independent JPEG Group,
found at http://www.ijg.org/files/jpegsrc.v8c.tar.gz.

The "LEGAL ISSUES" section of the README in that archive says:

In plain English:

1. We don't promise that this software works.  (But if you find any bugs,
   please let us know!)
2. You can use this software for whatever you want.  You don't have to pay us.
3. You may not pretend that you wrote this software.  If you use it in a
   program, you must acknowledge somewhere in your documentation that
   you've used the IJG code.

In legalese:

The authors make NO WARRANTY or representation, either express or implied,
with respect to this software, its quality, accuracy, merchantability, or
fitness for a particular purpose.  This software is provided "AS IS", and you,
its user, assume the entire risk as to its quality and accuracy.

This software is copyright (C) 1991-2011, Thomas G. Lane, Guido Vollbeding.
All Rights Reserved except as specified below.

Permission is hereby granted to use, copy, modify, and distribute this
software (or portions thereof) for any purpose, without fee, subject to these
conditions:
(1) If any part of the source code for this software is distributed, then this
README file must be included, with this copyright and no-warranty notice
unaltered; and any additions, deletions, or changes to the original files
must be clearly indicated in accompanying documentation.
(2) If only executable code is distributed, then the accompanying
documentation must state that "this software is based in part on the work of
the Independent JPEG Group".
(3) Permission for use of this software is granted only if the user accepts
full responsibility for any undesirable consequences; the authors accept
NO LIABILITY for damages of any kind.

These conditions apply to any software derived from or based on the IJG code,
not just to the unmodified library.  If you use our work, you ought to
acknowledge us.

Permission is NOT granted for the use of any IJG author's name or company name
in advertising or publicity relating to this software or products derived from
it.  This software may be referred to only as "the Independent JPEG Group's
software".

We specifically permit and encourage the use of this software as the basis of
commercial products, provided that all warranty or liability claims are
assumed by the product vendor.
*/

// Trigonometric constants in 13-bit fixed point format.
const (
	fix_0_298631336 = 2446
	fix_0_390180644 = 3196
	fix_0_541196100 = 4433
	fix_0_765366865 = 6270
	fix_0_899976223 = 7373
	fix_1_175875602 = 9633
	fix_1_501321110 = 12299
	fix_1_847759065 = 15137
	fix_1_961570560 = 16069
	fix_2_053119869 = 16819
	fix_2_562915447 = 20995
	fix_3_072711026 = 25172
)

const (
	constBits     = 13
	pass1Bits     = 2
	centerJSample = 128
)

// fdct performs a forward DCT on an 8x8 block of coefficients, including a
// level shift.
func fdct(b *block) {
	// Pass 1: process rows.
	for y := 0; y < 8; y++ {
		y8 := y * 8
		s := b[y8 : y8+8 : y8+8] // Small cap improves performance, see https://golang.org/issue/27857
		x0 := s[0]
		x1 := s[1]
		x2 := s[2]
		x3 := s[3]
		x4 := s[4]
		x5 := s[5]
		x6 := s[6]
		x7 := s[7]

		tmp0 := x0 + x7
		tmp1 := x1 + x6
		tmp2 := x2 + x5
		tmp3 := x3 + x4

		tmp10 := tmp0 + tmp3
		tmp12 := tmp0 - tmp3
		tmp11 := tmp1 + tmp2
		tmp13 := tmp1 - tmp2

		tmp0 = x0 - x7
		tmp1 = x1 - x6
		tmp2 = x2 - x5
		tmp3 = x3 - x4

		s[0] = (tmp10 + tmp11 - 8*centerJSample) << pass1Bits
		s[4] = (tmp10 - tmp11) << pass1Bits
		z1 := (tmp12 + tmp13) * fix_0_541196100
		z1 += 1 << (constBits - pass1Bits - 1)
		s[2] = (z1 + tmp12*fix_0_765366865) >> (constBits - pass1Bits)
		s[6] = (z1 - tmp13*fix_1_847759065) >> (constBits - pass1Bits)

		tmp10 = tmp0 + tmp3
		tmp11 = tmp1 + tmp2
		tmp12 = tmp0 + tmp2
		tmp13 = tmp1 + tmp3
		z1 = (tmp12 + tmp13) * fix_1_175875602
		z1 += 1 << (constBits - pass1Bits - 1)
		tmp0 *= fix_1_501321110
		tmp1 *= fix_3_072711026
		tmp2 *= fix_2_053119869
		tmp3 *= fix_0_298631336
		tmp10 *= -fix_0_899976223
		tmp11 *= -fix_2_562915447
		tmp12 *= -fix_0_390180644
		tmp13 *= -fix_1_961570560

		tmp12 += z1
		tmp13 += z1
		s[1] = (tmp0 + tmp10 + tmp12) >> (constBits - pass1Bits)
		s[3] = (tmp1 + tmp11 + tmp13) >> (constBits - pass1Bits)
		s[5] = (tmp2 + tmp11 + tmp12) >> (constBits - pass1Bits)
		s[7] = (tmp3 + tmp10 + tmp13) >> (constBits - pass1Bits)
	}
	// Pass 2: process columns.
	// We remove pass1Bits scaling, but leave results scaled up by an overall factor of 8.
	for x := 0; x < 8; x++ {
		tmp0 := b[0*8+x] + b[7*8+x]
		tmp1 := b[1*8+x] + b[6*8+x]
		tmp2 := b[2*8+x] + b[5*8+x]
		tmp3 := b[3*8+x] + b[4*8+x]

		tmp10 := tmp0 + tmp3 + 1<<(pass1Bits-1)
		tmp12 := tmp0 - tmp3
		tmp11 := tmp1 + tmp2
		tmp13 := tmp1 - tmp2

		tmp0 = b[0*8+x] - b[7*8+x]
		tmp1 = b[1*8+x] - b[6*8+x]
		tmp2 = b[2*8+x] - b[5*8+x]
		tmp3 = b[3*8+x] - b[4*8+x]

		b[0*8+x] = (tmp10 + tmp11) >> pass1Bits
		b[4*8+x] = (tmp10 - tmp11) >> pass1Bits

		z1 := (tmp12 + tmp13) * fix_0_541196100
		z1 += 1 << (constBits + pass1Bits - 1)
		b[2*8+x] = (z1 + tmp12*fix_0_765366865) >> (constBits + pass1Bits)
		b[6*8+x] = (z1 - tmp13*fix_1_847759065) >> (constBits + pass1Bits)

		tmp10 = tmp0 + tmp3
		tmp11 = tmp1 + tmp2
		tmp12 = tmp0 + tmp2
		tmp13 = tmp1 + tmp3
		z1 = (tmp12 + tmp13) * fix_1_175875602
		z1 += 1 << (constBits + pass1Bits - 1)
		tmp0 *= fix_1_501321110
		tmp1 *= fix_3_072711026
		tmp2 *= fix_2_053119869
		tmp3 *= fix_0_298631336
		tmp10 *= -fix_0_899976223
		tmp11 *= -fix_2_562915447
		tmp12 *= -fix_0_390180644
		tmp13 *= -fix_1_961570560

		tmp12 += z1
		tmp13 += z1
		b[1*8+x] = (tmp0 + tmp10 + tmp12) >> (constBits + pass1Bits)
		b[3*8+x] = (tmp1 + tmp11 + tmp13) >> (constBits + pass1Bits)
		b[5*8+x] = (tmp2 + tmp11 + tmp12) >> (constBits + pass1Bits)
		b[7*8+x] = (tmp3 + tmp10 + tmp13) >> (constBits + pass1Bits)
	}
}
