// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm64

package nistec

import "errors"

// Montgomery multiplication modulo org(G). Sets res = in1 * in2 * R⁻¹.
//
//go:noescape
func p256OrdMul(res, in1, in2 *p256OrdElement)

// Montgomery square modulo org(G), repeated n times (n >= 1).
//
//go:noescape
func p256OrdSqr(res, in *p256OrdElement, n int)

func P256OrdInverse(k []byte) ([]byte, error) {
	if len(k) != 32 {
		return nil, errors.New("invalid scalar length")
	}

	x := new(p256OrdElement)
	p256OrdBigToLittle(x, (*[32]byte)(k))
	p256OrdReduce(x)

	// Inversion is implemented as exponentiation by n - 2, per Fermat's little theorem.
	//
	// The sequence of 38 multiplications and 254 squarings is derived from
	// https://briansmith.org/ecc-inversion-addition-chains-01#p256_scalar_inversion
	_1 := new(p256OrdElement)
	_11 := new(p256OrdElement)
	_101 := new(p256OrdElement)
	_111 := new(p256OrdElement)
	_1111 := new(p256OrdElement)
	_10101 := new(p256OrdElement)
	_101111 := new(p256OrdElement)
	t := new(p256OrdElement)

	// This code operates in the Montgomery domain where R = 2²⁵⁶ mod n and n is
	// the order of the scalar field. Elements in the Montgomery domain take the
	// form a×R and p256OrdMul calculates (a × b × R⁻¹) mod n. RR is R in the
	// domain, or R×R mod n, thus p256OrdMul(x, RR) gives x×R, i.e. converts x
	// into the Montgomery domain.
	RR := &p256OrdElement{0x83244c95be79eea2, 0x4699799c49bd6fa6,
		0x2845b2392b6bec59, 0x66e12d94f3d95620}

	p256OrdMul(_1, x, RR)      // _1
	p256OrdSqr(x, _1, 1)       // _10
	p256OrdMul(_11, x, _1)     // _11
	p256OrdMul(_101, x, _11)   // _101
	p256OrdMul(_111, x, _101)  // _111
	p256OrdSqr(x, _101, 1)     // _1010
	p256OrdMul(_1111, _101, x) // _1111

	p256OrdSqr(t, x, 1)          // _10100
	p256OrdMul(_10101, t, _1)    // _10101
	p256OrdSqr(x, _10101, 1)     // _101010
	p256OrdMul(_101111, _101, x) // _101111
	p256OrdMul(x, _10101, x)     // _111111 = x6
	p256OrdSqr(t, x, 2)          // _11111100
	p256OrdMul(t, t, _11)        // _11111111 = x8
	p256OrdSqr(x, t, 8)          // _ff00
	p256OrdMul(x, x, t)          // _ffff = x16
	p256OrdSqr(t, x, 16)         // _ffff0000
	p256OrdMul(t, t, x)          // _ffffffff = x32

	p256OrdSqr(x, t, 64)
	p256OrdMul(x, x, t)
	p256OrdSqr(x, x, 32)
	p256OrdMul(x, x, t)

	sqrs := []int{
		6, 5, 4, 5, 5,
		4, 3, 3, 5, 9,
		6, 2, 5, 6, 5,
		4, 5, 5, 3, 10,
		2, 5, 5, 3, 7, 6}
	muls := []*p256OrdElement{
		_101111, _111, _11, _1111, _10101,
		_101, _101, _101, _111, _101111,
		_1111, _1, _1, _1111, _111,
		_111, _111, _101, _11, _101111,
		_11, _11, _11, _1, _10101, _1111}

	for i, s := range sqrs {
		p256OrdSqr(x, x, s)
		p256OrdMul(x, x, muls[i])
	}

	// Montgomery multiplication by R⁻¹, or 1 outside the domain as R⁻¹×R = 1,
	// converts a Montgomery value out of the domain.
	one := &p256OrdElement{1}
	p256OrdMul(x, x, one)

	var xOut [32]byte
	p256OrdLittleToBig(&xOut, x)
	return xOut[:], nil
}
