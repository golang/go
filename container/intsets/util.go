// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package intsets

// From Hacker's Delight, fig 5.2.
func popcountHD(x uint32) int {
	x -= (x >> 1) & 0x55555555
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
	x = (x + (x >> 4)) & 0x0f0f0f0f
	x = x + (x >> 8)
	x = x + (x >> 16)
	return int(x & 0x0000003f)
}

var a [1 << 8]byte

func init() {
	for i := range a {
		var n byte
		for x := i; x != 0; x >>= 1 {
			if x&1 != 0 {
				n++
			}
		}
		a[i] = n
	}
}

func popcountTable(x word) int {
	return int(a[byte(x>>(0*8))] +
		a[byte(x>>(1*8))] +
		a[byte(x>>(2*8))] +
		a[byte(x>>(3*8))] +
		a[byte(x>>(4*8))] +
		a[byte(x>>(5*8))] +
		a[byte(x>>(6*8))] +
		a[byte(x>>(7*8))])
}

// nlz returns the number of leading zeros of x.
// From Hacker's Delight, fig 5.11.
func nlz(x word) int {
	x |= (x >> 1)
	x |= (x >> 2)
	x |= (x >> 4)
	x |= (x >> 8)
	x |= (x >> 16)
	x |= (x >> 32)
	return popcount(^x)
}

// ntz returns the number of trailing zeros of x.
// From Hacker's Delight, fig 5.13.
func ntz(x word) int {
	if x == 0 {
		return bitsPerWord
	}
	n := 1
	if bitsPerWord == 64 {
		if (x & 0xffffffff) == 0 {
			n = n + 32
			x = x >> 32
		}
	}
	if (x & 0x0000ffff) == 0 {
		n = n + 16
		x = x >> 16
	}
	if (x & 0x000000ff) == 0 {
		n = n + 8
		x = x >> 8
	}
	if (x & 0x0000000f) == 0 {
		n = n + 4
		x = x >> 4
	}
	if (x & 0x00000003) == 0 {
		n = n + 2
		x = x >> 2
	}
	return n - int(x&1)
}
