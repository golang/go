// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

func complex128div(n complex128, d complex128) complex128 {
	// Special cases as in C99.
	ninf := real(n) == posinf || real(n) == neginf ||
		imag(n) == posinf || imag(n) == neginf
	dinf := real(d) == posinf || real(d) == neginf ||
		imag(d) == posinf || imag(d) == neginf

	nnan := !ninf && (real(n) != real(n) || imag(n) != imag(n))
	dnan := !dinf && (real(d) != real(d) || imag(d) != imag(d))

	switch {
	case nnan || dnan:
		return complex(nan, nan)
	case ninf && !dinf:
		return complex(posinf, posinf)
	case !ninf && dinf:
		return complex(0, 0)
	case real(d) == 0 && imag(d) == 0:
		if real(n) == 0 && imag(n) == 0 {
			return complex(nan, nan)
		} else {
			return complex(posinf, posinf)
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
