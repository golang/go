// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package curve25519 provides an implementation of the X25519 function, which
// performs scalar multiplication on the elliptic curve known as Curve25519.
// See RFC 7748.
package curve25519 // import "golang.org/x/crypto/curve25519"

import (
	"crypto/subtle"
	"fmt"

	"golang.org/x/crypto/curve25519/internal/field"
)

// ScalarMult sets dst to the product scalar * point.
//
// Deprecated: when provided a low-order point, ScalarMult will set dst to all
// zeroes, irrespective of the scalar. Instead, use the X25519 function, which
// will return an error.
func ScalarMult(dst, scalar, point *[32]byte) {
	var e [32]byte

	copy(e[:], scalar[:])
	e[0] &= 248
	e[31] &= 127
	e[31] |= 64

	var x1, x2, z2, x3, z3, tmp0, tmp1 field.Element
	x1.SetBytes(point[:])
	x2.One()
	x3.Set(&x1)
	z3.One()

	swap := 0
	for pos := 254; pos >= 0; pos-- {
		b := e[pos/8] >> uint(pos&7)
		b &= 1
		swap ^= int(b)
		x2.Swap(&x3, swap)
		z2.Swap(&z3, swap)
		swap = int(b)

		tmp0.Subtract(&x3, &z3)
		tmp1.Subtract(&x2, &z2)
		x2.Add(&x2, &z2)
		z2.Add(&x3, &z3)
		z3.Multiply(&tmp0, &x2)
		z2.Multiply(&z2, &tmp1)
		tmp0.Square(&tmp1)
		tmp1.Square(&x2)
		x3.Add(&z3, &z2)
		z2.Subtract(&z3, &z2)
		x2.Multiply(&tmp1, &tmp0)
		tmp1.Subtract(&tmp1, &tmp0)
		z2.Square(&z2)

		z3.Mult32(&tmp1, 121666)
		x3.Square(&x3)
		tmp0.Add(&tmp0, &z3)
		z3.Multiply(&x1, &z2)
		z2.Multiply(&tmp1, &tmp0)
	}

	x2.Swap(&x3, swap)
	z2.Swap(&z3, swap)

	z2.Invert(&z2)
	x2.Multiply(&x2, &z2)
	copy(dst[:], x2.Bytes())
}

// ScalarBaseMult sets dst to the product scalar * base where base is the
// standard generator.
//
// It is recommended to use the X25519 function with Basepoint instead, as
// copying into fixed size arrays can lead to unexpected bugs.
func ScalarBaseMult(dst, scalar *[32]byte) {
	ScalarMult(dst, scalar, &basePoint)
}

const (
	// ScalarSize is the size of the scalar input to X25519.
	ScalarSize = 32
	// PointSize is the size of the point input to X25519.
	PointSize = 32
)

// Basepoint is the canonical Curve25519 generator.
var Basepoint []byte

var basePoint = [32]byte{9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

func init() { Basepoint = basePoint[:] }

func checkBasepoint() {
	if subtle.ConstantTimeCompare(Basepoint, []byte{
		0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	}) != 1 {
		panic("curve25519: global Basepoint value was modified")
	}
}

// X25519 returns the result of the scalar multiplication (scalar * point),
// according to RFC 7748, Section 5. scalar, point and the return value are
// slices of 32 bytes.
//
// scalar can be generated at random, for example with crypto/rand. point should
// be either Basepoint or the output of another X25519 call.
//
// If point is Basepoint (but not if it's a different slice with the same
// contents) a precomputed implementation might be used for performance.
func X25519(scalar, point []byte) ([]byte, error) {
	// Outline the body of function, to let the allocation be inlined in the
	// caller, and possibly avoid escaping to the heap.
	var dst [32]byte
	return x25519(&dst, scalar, point)
}

func x25519(dst *[32]byte, scalar, point []byte) ([]byte, error) {
	var in [32]byte
	if l := len(scalar); l != 32 {
		return nil, fmt.Errorf("bad scalar length: %d, expected %d", l, 32)
	}
	if l := len(point); l != 32 {
		return nil, fmt.Errorf("bad point length: %d, expected %d", l, 32)
	}
	copy(in[:], scalar)
	if &point[0] == &Basepoint[0] {
		checkBasepoint()
		ScalarBaseMult(dst, &in)
	} else {
		var base, zero [32]byte
		copy(base[:], point)
		ScalarMult(dst, &in, &base)
		if subtle.ConstantTimeCompare(dst[:], zero[:]) == 1 {
			return nil, fmt.Errorf("bad input point: low order point")
		}
	}
	return dst[:], nil
}
