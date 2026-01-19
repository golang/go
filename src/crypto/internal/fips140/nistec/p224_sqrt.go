// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nistec

import (
	"crypto/internal/fips140/nistec/fiat"
	"sync"
)

var p224GG *[96]fiat.P224Element
var p224GGOnce sync.Once

// p224SqrtCandidate sets r to a square root candidate for x. r and x must not overlap.
func p224SqrtCandidate(r, x *fiat.P224Element) {
	// Since p = 1 mod 4, we can't use the exponentiation by (p + 1) / 4 like
	// for the other primes. Instead, implement a variation of Tonelli–Shanks.
	// The constant-time implementation is adapted from Thomas Pornin's ecGFp5.
	//
	// https://github.com/pornin/ecgfp5/blob/82325b965/rust/src/field.rs#L337-L385

	// p = q*2^n + 1 with q odd -> q = 2^128 - 1 and n = 96
	// g^(2^n) = 1 -> g = 11 ^ q (where 11 is the smallest non-square)
	// GG[j] = g^(2^j) for j = 0 to n-1

	p224GGOnce.Do(func() {
		p224GG = new([96]fiat.P224Element)
		for i := range p224GG {
			if i == 0 {
				p224GG[i].SetBytes([]byte{0x6a, 0x0f, 0xec, 0x67,
					0x85, 0x98, 0xa7, 0x92, 0x0c, 0x55, 0xb2, 0xd4,
					0x0b, 0x2d, 0x6f, 0xfb, 0xbe, 0xa3, 0xd8, 0xce,
					0xf3, 0xfb, 0x36, 0x32, 0xdc, 0x69, 0x1b, 0x74})
			} else {
				p224GG[i].Square(&p224GG[i-1])
			}
		}
	})

	// r <- x^((q+1)/2) = x^(2^127)
	// v <- x^q = x^(2^128-1)

	// Compute x^(2^127-1) first.
	//
	// The sequence of 10 multiplications and 126 squarings is derived from the
	// following addition chain generated with github.com/mmcloughlin/addchain v0.4.0.
	//
	//	_10      = 2*1
	//	_11      = 1 + _10
	//	_110     = 2*_11
	//	_111     = 1 + _110
	//	_111000  = _111 << 3
	//	_111111  = _111 + _111000
	//	_1111110 = 2*_111111
	//	_1111111 = 1 + _1111110
	//	x12      = _1111110 << 5 + _111111
	//	x24      = x12 << 12 + x12
	//	i36      = x24 << 7
	//	x31      = _1111111 + i36
	//	x48      = i36 << 17 + x24
	//	x96      = x48 << 48 + x48
	//	return     x96 << 31 + x31
	//
	var t0 = new(fiat.P224Element)
	var t1 = new(fiat.P224Element)

	r.Square(x)
	r.Mul(x, r)
	r.Square(r)
	r.Mul(x, r)
	t0.Square(r)
	for s := 1; s < 3; s++ {
		t0.Square(t0)
	}
	t0.Mul(r, t0)
	t1.Square(t0)
	r.Mul(x, t1)
	for s := 0; s < 5; s++ {
		t1.Square(t1)
	}
	t0.Mul(t0, t1)
	t1.Square(t0)
	for s := 1; s < 12; s++ {
		t1.Square(t1)
	}
	t0.Mul(t0, t1)
	t1.Square(t0)
	for s := 1; s < 7; s++ {
		t1.Square(t1)
	}
	r.Mul(r, t1)
	for s := 0; s < 17; s++ {
		t1.Square(t1)
	}
	t0.Mul(t0, t1)
	t1.Square(t0)
	for s := 1; s < 48; s++ {
		t1.Square(t1)
	}
	t0.Mul(t0, t1)
	for s := 0; s < 31; s++ {
		t0.Square(t0)
	}
	r.Mul(r, t0)

	// v = x^(2^127-1)^2 * x
	v := new(fiat.P224Element).Square(r)
	v.Mul(v, x)

	// r = x^(2^127-1) * x
	r.Mul(r, x)

	// for i = n-1 down to 1:
	//     w = v^(2^(i-1))
	//     if w == -1 then:
	//         v <- v*GG[n-i]
	//         r <- r*GG[n-i-1]

	var p224MinusOne = new(fiat.P224Element).Sub(
		new(fiat.P224Element), new(fiat.P224Element).One())

	for i := 96 - 1; i >= 1; i-- {
		w := new(fiat.P224Element).Set(v)
		for j := 0; j < i-1; j++ {
			w.Square(w)
		}
		cond := w.Equal(p224MinusOne)
		v.Select(t0.Mul(v, &p224GG[96-i]), v, cond)
		r.Select(t0.Mul(r, &p224GG[96-i-1]), r, cond)
	}
}
