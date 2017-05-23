// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

func isposinf(f float64) bool { return f > maxFloat64 }
func isneginf(f float64) bool { return f < -maxFloat64 }
func isnan(f float64) bool    { return f != f }

func nan() float64 {
	var f float64 = 0
	return f / f
}

func posinf() float64 {
	var f float64 = maxFloat64
	return f * f
}

func neginf() float64 {
	var f float64 = maxFloat64
	return -f * f
}

func complex128div(n complex128, d complex128) complex128 {
	// Special cases as in C99.
	ninf := isposinf(real(n)) || isneginf(real(n)) ||
		isposinf(imag(n)) || isneginf(imag(n))
	dinf := isposinf(real(d)) || isneginf(real(d)) ||
		isposinf(imag(d)) || isneginf(imag(d))

	nnan := !ninf && (isnan(real(n)) || isnan(imag(n)))
	dnan := !dinf && (isnan(real(d)) || isnan(imag(d)))

	switch {
	case nnan || dnan:
		return complex(nan(), nan())
	case ninf && !dinf:
		return complex(posinf(), posinf())
	case !ninf && dinf:
		return complex(0, 0)
	case real(d) == 0 && imag(d) == 0:
		if real(n) == 0 && imag(n) == 0 {
			return complex(nan(), nan())
		} else {
			return complex(posinf(), posinf())
		}
	default:
		// Standard complex arithmetic, factored to avoid unnecessary overflow.
		a := real(d)
		if a < 0 {
			a = -a
		}
		b := imag(d)
		if b < 0 {
			b = -b
		}
		if a <= b {
			ratio := real(d) / imag(d)
			denom := real(d)*ratio + imag(d)
			return complex((real(n)*ratio+imag(n))/denom,
				(imag(n)*ratio-real(n))/denom)
		} else {
			ratio := imag(d) / real(d)
			denom := imag(d)*ratio + real(d)
			return complex((imag(n)*ratio+real(n))/denom,
				(imag(n)-real(n)*ratio)/denom)
		}
	}
}
