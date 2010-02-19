// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Frexp breaks f into a normalized fraction
// and an integral power of two.
// It returns frac and exp satisfying f == frac × 2<sup>exp</sup>,
// with the absolute value of frac in the interval [½, 1).
func Frexp(f float64) (frac float64, exp int) {
	// TODO(rsc): Remove manual inlining of IsNaN, IsInf
	// when compiler does it for us
	// special cases
	switch {
	case f == 0:
		return
	case f < -MaxFloat64 || f > MaxFloat64 || f != f: // IsInf(f, 0) || IsNaN(f):
		frac = f
		return
	}
	x := Float64bits(f)
	exp = int((x>>shift)&mask) - bias
	x &^= mask << shift
	x |= bias << shift
	frac = Float64frombits(x)
	return
}
