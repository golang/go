// Copyright 2009-2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

/*
	Hypot -- sqrt(p*p + q*q), but overflows only if the result does.
	See:
		Cleve Moler and Donald Morrison,
		Replacing Square Roots by Pythagorean Sums
		IBM Journal of Research and Development,
		Vol. 27, Number 6, pp. 577-581, Nov. 1983
*/

// Hypot computes Sqrt(p*p + q*q), taking care to avoid
// unnecessary overflow and underflow.
//
// Special cases are:
//	Hypot(p, q) = +Inf if p or q is infinite
//	Hypot(p, q) = NaN if p or q is NaN
func hypotGo(p, q float64) float64 {
	// TODO(rsc): Remove manual inlining of IsNaN, IsInf
	// when compiler does it for us
	// special cases
	switch {
	case p < -MaxFloat64 || p > MaxFloat64 || q < -MaxFloat64 || q > MaxFloat64: // IsInf(p, 0) || IsInf(q, 0):
		return Inf(1)
	case p != p || q != q: // IsNaN(p) || IsNaN(q):
		return NaN()
	}
	if p < 0 {
		p = -p
	}
	if q < 0 {
		q = -q
	}

	if p < q {
		p, q = q, p
	}

	if p == 0 {
		return 0
	}

	pfac := p
	q = q / p
	r := q
	p = 1
	for {
		r = r * r
		s := r + 4
		if s == 4 {
			return p * pfac
		}
		r = r / s
		p = p + 2*r*p
		q = q * r
		r = q / p
	}
	panic("unreachable")
}
