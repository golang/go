// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

/*
	Floating-point arctangent.

	Atan returns the value of the arctangent of its
	argument in the range [-pi/2,pi/2].
	There are no error returns.
	Coefficients are #5077 from Hart & Cheney. (19.56D)
*/

// xatan evaluates a series valid in the
// range [-0.414...,+0.414...]. (tan(pi/8))
func xatan(arg float64) float64 {
	const (
		P4 = .161536412982230228262e2
		P3 = .26842548195503973794141e3
		P2 = .11530293515404850115428136e4
		P1 = .178040631643319697105464587e4
		P0 = .89678597403663861959987488e3
		Q4 = .5895697050844462222791e2
		Q3 = .536265374031215315104235e3
		Q2 = .16667838148816337184521798e4
		Q1 = .207933497444540981287275926e4
		Q0 = .89678597403663861962481162e3
	)
	sq := arg * arg
	value := ((((P4*sq+P3)*sq+P2)*sq+P1)*sq + P0)
	value = value / (((((sq+Q4)*sq+Q3)*sq+Q2)*sq+Q1)*sq + Q0)
	return value * arg
}

// satan reduces its argument (known to be positive)
// to the range [0,0.414...] and calls xatan.
func satan(arg float64) float64 {
	if arg < Sqrt2-1 {
		return xatan(arg)
	}
	if arg > Sqrt2+1 {
		return Pi/2 - xatan(1/arg)
	}
	return Pi/4 + xatan((arg-1)/(arg+1))
}

// Atan returns the arctangent of x.
//
// Special cases are:
//	Atan(±0) = ±0
//	Atan(±Inf) = ±Pi/2
func Atan(x float64) float64 {
	if x == 0 {
		return x
	}
	if x > 0 {
		return satan(x)
	}
	return -satan(-x)
}
