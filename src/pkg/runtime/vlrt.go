// Inferno's libkern/vlrt-arm.c
// http://code.google.com/p/inferno-os/source/browse/libkern/vlrt-arm.c
//
//         Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//         Revisions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).  All rights reserved.
//         Portions Copyright 2009 The Go Authors. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// +build arm

package runtime

import "unsafe"

func float64toint64(d float64, y uint64) {
	_d2v(&y, d)
}

func float64touint64(d float64, y uint64) {
	_d2v(&y, d)
}

const (
	sign64 = 1 << (64 - 1)
)

func _d2v(y *uint64, d float64) {
	x := *(*uint64)(unsafe.Pointer(&d))

	xhi := uint32(x>>32)&0xfffff | 0x100000
	xlo := uint32(x)
	sh := 1075 - int32(uint32(x>>52)&0x7ff)

	var ylo, yhi uint32
	if sh >= 0 {
		sh := uint32(sh)
		/* v = (hi||lo) >> sh */
		if sh < 32 {
			if sh == 0 {
				ylo = xlo
				yhi = xhi
			} else {
				ylo = xlo>>sh | xhi<<(32-sh)
				yhi = xhi >> sh
			}
		} else {
			if sh == 32 {
				ylo = xhi
			} else if sh < 64 {
				ylo = xhi >> (sh - 32)
			}
		}
	} else {
		/* v = (hi||lo) << -sh */
		sh := uint32(-sh)
		if sh <= 11 {
			ylo = xlo << sh
			yhi = xhi<<sh | xlo>>(32-sh)
		} else {
			/* overflow */
			yhi = uint32(d) /* causes something awful */
		}
	}
	if x&sign64 != 0 {
		if ylo != 0 {
			ylo = -ylo
			yhi = ^yhi
		} else {
			yhi = -yhi
		}
	}

	*y = uint64(yhi)<<32 | uint64(ylo)
}
