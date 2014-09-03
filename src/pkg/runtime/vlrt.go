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

// +build arm 386

package runtime

import "unsafe"

const (
	sign32 = 1 << (32 - 1)
	sign64 = 1 << (64 - 1)
)

func float64toint64(d float64) (y uint64) {
	_d2v(&y, d)
	return
}

func float64touint64(d float64) (y uint64) {
	_d2v(&y, d)
	return
}

func int64tofloat64(y int64) float64 {
	if y < 0 {
		return -uint64tofloat64(-uint64(y))
	}
	return uint64tofloat64(uint64(y))
}

func uint64tofloat64(y uint64) float64 {
	hi := float64(uint32(y >> 32))
	lo := float64(uint32(y))
	d := hi*(1<<32) + lo
	return d
}

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

func uint64div(n, d uint64) uint64 {
	// Check for 32 bit operands
	if uint32(n>>32) == 0 && uint32(d>>32) == 0 {
		if uint32(d) == 0 {
			panicdivide()
		}
		return uint64(uint32(n) / uint32(d))
	}
	q, _ := dodiv(n, d)
	return q
}

func uint64mod(n, d uint64) uint64 {
	// Check for 32 bit operands
	if uint32(n>>32) == 0 && uint32(d>>32) == 0 {
		if uint32(d) == 0 {
			panicdivide()
		}
		return uint64(uint32(n) % uint32(d))
	}
	_, r := dodiv(n, d)
	return r
}

func int64div(n, d int64) int64 {
	// Check for 32 bit operands
	if int64(int32(n)) == n && int64(int32(d)) == d {
		if int32(n) == -0x80000000 && int32(d) == -1 {
			// special case: 32-bit -0x80000000 / -1 = -0x80000000,
			// but 64-bit -0x80000000 / -1 = 0x80000000.
			return 0x80000000
		}
		if int32(d) == 0 {
			panicdivide()
		}
		return int64(int32(n) / int32(d))
	}

	nneg := n < 0
	dneg := d < 0
	if nneg {
		n = -n
	}
	if dneg {
		d = -d
	}
	uq, _ := dodiv(uint64(n), uint64(d))
	q := int64(uq)
	if nneg != dneg {
		q = -q
	}
	return q
}

func int64mod(n, d int64) int64 {
	// Check for 32 bit operands
	if int64(int32(n)) == n && int64(int32(d)) == d {
		if int32(d) == 0 {
			panicdivide()
		}
		return int64(int32(n) % int32(d))
	}

	nneg := n < 0
	if nneg {
		n = -n
	}
	if d < 0 {
		d = -d
	}
	_, ur := dodiv(uint64(n), uint64(d))
	r := int64(ur)
	if nneg {
		r = -r
	}
	return r
}

//go:noescape
func _mul64by32(lo64 *uint64, a uint64, b uint32) (hi32 uint32)

//go:noescape
func _div64by32(a uint64, b uint32, r *uint32) (q uint32)

func dodiv(n, d uint64) (q, r uint64) {
	if GOARCH == "arm" {
		// arm doesn't have a division instruction, so
		// slowdodiv is the best that we can do.
		// TODO: revisit for arm64.
		return slowdodiv(n, d)
	}

	if d > n {
		return 0, n
	}

	if uint32(d>>32) != 0 {
		t := uint32(n>>32) / uint32(d>>32)
		var lo64 uint64
		hi32 := _mul64by32(&lo64, d, t)
		if hi32 != 0 || lo64 > n {
			return slowdodiv(n, d)
		}
		return uint64(t), n - lo64
	}

	// d is 32 bit
	var qhi uint32
	if uint32(n>>32) >= uint32(d) {
		if uint32(d) == 0 {
			panicdivide()
		}
		qhi = uint32(n>>32) / uint32(d)
		n -= uint64(uint32(d)*qhi) << 32
	} else {
		qhi = 0
	}

	var rlo uint32
	qlo := _div64by32(n, uint32(d), &rlo)
	return uint64(qhi)<<32 + uint64(qlo), uint64(rlo)
}

func slowdodiv(n, d uint64) (q, r uint64) {
	if d == 0 {
		panicdivide()
	}

	// Set up the divisor and find the number of iterations needed.
	capn := n
	if n >= sign64 {
		capn = sign64
	}
	i := 0
	for d < capn {
		d <<= 1
		i++
	}

	for ; i >= 0; i-- {
		q <<= 1
		if n >= d {
			n -= d
			q |= 1
		}
		d >>= 1
	}
	return q, n
}
