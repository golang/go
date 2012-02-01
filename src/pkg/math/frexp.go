// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Frexp breaks f into a normalized fraction
// and an integral power of two.
// It returns frac and exp satisfying f == frac × 2**exp,
// with the absolute value of frac in the interval [½, 1).
//
// Special cases are:
//	Frexp(±0) = ±0, 0
//	Frexp(±Inf) = ±Inf, 0
//	Frexp(NaN) = NaN, 0
func Frexp(f float64) (frac float64, exp int)

func frexp(f float64) (frac float64, exp int) {
	// special cases
	switch {
	case f == 0:
		return f, 0 // correctly return -0
	case IsInf(f, 0) || IsNaN(f):
		return f, 0
	}
	f, exp = normalize(f)
	x := Float64bits(f)
	exp += int((x>>shift)&mask) - bias + 1
	x &^= mask << shift
	x |= (-1 + bias) << shift
	frac = Float64frombits(x)
	return
}
