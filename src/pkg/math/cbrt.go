// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

/*
	The algorithm is based in part on "Optimal Partitioning of
	Newton's Method for Calculating Roots", by Gunter Meinardus
	and G. D. Taylor, Mathematics of Computation © 1980 American
	Mathematical Society.
	(http://www.jstor.org/stable/2006387?seq=9, accessed 11-Feb-2010)
*/

// Cbrt returns the cube root of its argument.
//
// Special cases are:
//	Cbrt(±0) = ±0
//	Cbrt(±Inf) = ±Inf
//	Cbrt(NaN) = NaN
func Cbrt(x float64) float64 {
	const (
		A1 = 1.662848358e-01
		A2 = 1.096040958e+00
		A3 = 4.105032829e-01
		A4 = 5.649335816e-01
		B1 = 2.639607233e-01
		B2 = 8.699282849e-01
		B3 = 1.629083358e-01
		B4 = 2.824667908e-01
		C1 = 4.190115298e-01
		C2 = 6.904625373e-01
		C3 = 6.46502159e-02
		C4 = 1.412333954e-01
	)
	// TODO(rsc): Remove manual inlining of IsNaN, IsInf
	// when compiler does it for us
	// special cases
	switch {
	case x == 0 || x != x || x < -MaxFloat64 || x > MaxFloat64: // x == 0 || IsNaN(x) || IsInf(x, 0):
		return x
	}
	sign := false
	if x < 0 {
		x = -x
		sign = true
	}
	// Reduce argument and estimate cube root
	f, e := Frexp(x) // 0.5 <= f < 1.0
	m := e % 3
	if m > 0 {
		m -= 3
		e -= m // e is multiple of 3
	}
	switch m {
	case 0: // 0.5 <= f < 1.0
		f = A1*f + A2 - A3/(A4+f)
	case -1:
		f *= 0.5 // 0.25 <= f < 0.5
		f = B1*f + B2 - B3/(B4+f)
	default: // m == -2
		f *= 0.25 // 0.125 <= f < 0.25
		f = C1*f + C2 - C3/(C4+f)
	}
	y := Ldexp(f, e/3) // e/3 = exponent of cube root

	// Iterate
	s := y * y * y
	t := s + x
	y *= (t + x) / (s + t)
	// Reiterate
	s = (y*y*y - x) / x
	y -= y * (((14.0/81.0)*s-(2.0/9.0))*s + (1.0 / 3.0)) * s
	if sign {
		y = -y
	}
	return y
}
