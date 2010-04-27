// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Ldexp is the inverse of Frexp.
// It returns frac × 2**exp.
func Ldexp(frac float64, exp int) float64 {
	// TODO(rsc): Remove manual inlining of IsNaN, IsInf
	// when compiler does it for us
	// special cases
	switch {
	case frac == 0:
		return frac // correctly return -0
	case frac != frac: // IsNaN(frac):
		return NaN()
	}
	x := Float64bits(frac)
	exp += int(x>>shift) & mask
	if exp <= 0 {
		return 0 // underflow
	}
	if exp >= mask { // overflow
		if frac < 0 {
			return Inf(-1)
		}
		return Inf(1)
	}
	x &^= mask << shift
	x |= uint64(exp) << shift
	return Float64frombits(x)
}
