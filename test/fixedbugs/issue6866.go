// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// WARNING: GENERATED FILE - DO NOT MODIFY MANUALLY!
// (To generate, in go/types directory: go test -run=Hilbert -H=2 -out="h2.src")

// This program tests arbitrary precision constant arithmetic
// by generating the constant elements of a Hilbert matrix H,
// its inverse I, and the product P = H*I. The product should
// be the identity matrix.
package main

func main() {
	if !ok {
		print()
		return
	}
}

// Hilbert matrix, n = 2
const (
	h0_0, h0_1 = 1.0 / (iota + 1), 1.0 / (iota + 2)
	h1_0, h1_1
)

// Inverse Hilbert matrix
const (
	i0_0 = +1 * b2_1 * b2_1 * b0_0 * b0_0
	i0_1 = -2 * b2_0 * b3_1 * b1_0 * b1_0

	i1_0 = -2 * b3_1 * b2_0 * b1_1 * b1_1
	i1_1 = +3 * b3_0 * b3_0 * b2_1 * b2_1
)

// Product matrix
const (
	p0_0 = h0_0*i0_0 + h0_1*i1_0
	p0_1 = h0_0*i0_1 + h0_1*i1_1

	p1_0 = h1_0*i0_0 + h1_1*i1_0
	p1_1 = h1_0*i0_1 + h1_1*i1_1
)

// Verify that product is identity matrix
const ok = p0_0 == 1 && p0_1 == 0 &&
	p1_0 == 0 && p1_1 == 1 &&
	true

func print() {
	println(p0_0, p0_1)
	println(p1_0, p1_1)
}

// Binomials
const (
	b0_0 = f0 / (f0 * f0)

	b1_0 = f1 / (f0 * f1)
	b1_1 = f1 / (f1 * f0)

	b2_0 = f2 / (f0 * f2)
	b2_1 = f2 / (f1 * f1)
	b2_2 = f2 / (f2 * f0)

	b3_0 = f3 / (f0 * f3)
	b3_1 = f3 / (f1 * f2)
	b3_2 = f3 / (f2 * f1)
	b3_3 = f3 / (f3 * f0)
)

// Factorials
const (
	f0 = 1
	f1 = 1
	f2 = f1 * 2
	f3 = f2 * 3
)
