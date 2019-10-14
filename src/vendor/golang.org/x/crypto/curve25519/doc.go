// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package curve25519 provides an implementation of scalar multiplication on
// the elliptic curve known as curve25519. See https://cr.yp.to/ecdh.html
package curve25519 // import "golang.org/x/crypto/curve25519"

// basePoint is the x coordinate of the generator of the curve.
var basePoint = [32]byte{9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

// ScalarMult sets dst to the product in*base where dst and base are the x
// coordinates of group points and all values are in little-endian form.
func ScalarMult(dst, in, base *[32]byte) {
	scalarMult(dst, in, base)
}

// ScalarBaseMult sets dst to the product in*base where dst and base are the x
// coordinates of group points, base is the standard generator and all values
// are in little-endian form.
func ScalarBaseMult(dst, in *[32]byte) {
	ScalarMult(dst, in, &basePoint)
}
