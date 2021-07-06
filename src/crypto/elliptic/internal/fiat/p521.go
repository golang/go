// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fiat implements prime order fields using formally verified algorithms
// from the Fiat Cryptography project.
package fiat

import (
	"crypto/subtle"
	"errors"
)

// P521Element is an integer modulo 2^521 - 1.
//
// The zero value is a valid zero element.
type P521Element struct {
	// This element has the following bounds, which are tighter than
	// the output bounds of some operations. Those operations must be
	// followed by a carry.
	//
	// [0x0 ~> 0x400000000000000], [0x0 ~> 0x400000000000000], [0x0 ~> 0x400000000000000],
	// [0x0 ~> 0x400000000000000], [0x0 ~> 0x400000000000000], [0x0 ~> 0x400000000000000],
	// [0x0 ~> 0x400000000000000], [0x0 ~> 0x400000000000000], [0x0 ~> 0x200000000000000]
	x [9]uint64
}

// One sets e = 1, and returns e.
func (e *P521Element) One() *P521Element {
	*e = P521Element{}
	e.x[0] = 1
	return e
}

// Equal returns 1 if e == t, and zero otherwise.
func (e *P521Element) Equal(t *P521Element) int {
	eBytes := e.Bytes()
	tBytes := t.Bytes()
	return subtle.ConstantTimeCompare(eBytes, tBytes)
}

var p521ZeroEncoding = new(P521Element).Bytes()

// IsZero returns 1 if e == 0, and zero otherwise.
func (e *P521Element) IsZero() int {
	eBytes := e.Bytes()
	return subtle.ConstantTimeCompare(eBytes, p521ZeroEncoding)
}

// Set sets e = t, and returns e.
func (e *P521Element) Set(t *P521Element) *P521Element {
	e.x = t.x
	return e
}

// Bytes returns the 66-byte little-endian encoding of e.
func (e *P521Element) Bytes() []byte {
	// This function must be inlined to move the allocation to the parent and
	// save it from escaping to the heap.
	var out [66]byte
	p521ToBytes(&out, &e.x)
	return out[:]
}

// SetBytes sets e = v, where v is a little-endian 66-byte encoding, and returns
// e. If v is not 66 bytes or it encodes a value higher than 2^521 - 1, SetBytes
// returns nil and an error, and e is unchanged.
func (e *P521Element) SetBytes(v []byte) (*P521Element, error) {
	if len(v) != 66 || v[65] > 1 {
		return nil, errors.New("invalid P-521 field encoding")
	}
	var in [66]byte
	copy(in[:], v)
	p521FromBytes(&e.x, &in)
	return e, nil
}

// Add sets e = t1 + t2, and returns e.
func (e *P521Element) Add(t1, t2 *P521Element) *P521Element {
	p521Add(&e.x, &t1.x, &t2.x)
	p521Carry(&e.x, &e.x)
	return e
}

// Sub sets e = t1 - t2, and returns e.
func (e *P521Element) Sub(t1, t2 *P521Element) *P521Element {
	p521Sub(&e.x, &t1.x, &t2.x)
	p521Carry(&e.x, &e.x)
	return e
}

// Mul sets e = t1 * t2, and returns e.
func (e *P521Element) Mul(t1, t2 *P521Element) *P521Element {
	p521CarryMul(&e.x, &t1.x, &t2.x)
	return e
}

// Square sets e = t * t, and returns e.
func (e *P521Element) Square(t *P521Element) *P521Element {
	p521CarrySquare(&e.x, &t.x)
	return e
}

// Select sets e to a if cond == 1, and to b if cond == 0.
func (v *P521Element) Select(a, b *P521Element, cond int) *P521Element {
	p521Selectznz(&v.x, p521Uint1(cond), &b.x, &a.x)
	return v
}

// Invert sets e = 1/t, and returns e.
//
// If t == 0, Invert returns e = 0.
func (e *P521Element) Invert(t *P521Element) *P521Element {
	// Inversion is implemented as exponentiation with exponent p − 2.
	// The sequence of multiplications and squarings was generated with
	// github.com/mmcloughlin/addchain v0.2.0.

	var t1, t2 = new(P521Element), new(P521Element)

	// _10 = 2 * 1
	t1.Square(t)

	// _11 = 1 + _10
	t1.Mul(t, t1)

	// _1100 = _11 << 2
	t2.Square(t1)
	t2.Square(t2)

	// _1111 = _11 + _1100
	t1.Mul(t1, t2)

	// _11110000 = _1111 << 4
	t2.Square(t1)
	for i := 0; i < 3; i++ {
		t2.Square(t2)
	}

	// _11111111 = _1111 + _11110000
	t1.Mul(t1, t2)

	// x16 = _11111111<<8 + _11111111
	t2.Square(t1)
	for i := 0; i < 7; i++ {
		t2.Square(t2)
	}
	t1.Mul(t1, t2)

	// x32 = x16<<16 + x16
	t2.Square(t1)
	for i := 0; i < 15; i++ {
		t2.Square(t2)
	}
	t1.Mul(t1, t2)

	// x64 = x32<<32 + x32
	t2.Square(t1)
	for i := 0; i < 31; i++ {
		t2.Square(t2)
	}
	t1.Mul(t1, t2)

	// x65 = 2*x64 + 1
	t2.Square(t1)
	t2.Mul(t2, t)

	// x129 = x65<<64 + x64
	for i := 0; i < 64; i++ {
		t2.Square(t2)
	}
	t1.Mul(t1, t2)

	// x130 = 2*x129 + 1
	t2.Square(t1)
	t2.Mul(t2, t)

	// x259 = x130<<129 + x129
	for i := 0; i < 129; i++ {
		t2.Square(t2)
	}
	t1.Mul(t1, t2)

	// x260 = 2*x259 + 1
	t2.Square(t1)
	t2.Mul(t2, t)

	// x519 = x260<<259 + x259
	for i := 0; i < 259; i++ {
		t2.Square(t2)
	}
	t1.Mul(t1, t2)

	// return x519<<2 + 1
	t1.Square(t1)
	t1.Square(t1)
	return e.Mul(t1, t)
}
