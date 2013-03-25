// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

/*
	Hypot -- sqrt(p*p + q*q), but overflows only if the result does.
*/

// Hypot returns Sqrt(p*p + q*q), taking care to avoid
// unnecessary overflow and underflow.
//
// Special cases are:
//	Hypot(±Inf, q) = +Inf
//	Hypot(p, ±Inf) = +Inf
//	Hypot(NaN, q) = NaN
//	Hypot(p, NaN) = NaN
func Hypot(p, q float64) float64

func hypot(p, q float64) float64 {
	// special cases
	switch {
	case IsInf(p, 0) || IsInf(q, 0):
		return Inf(1)
	case IsNaN(p) || IsNaN(q):
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
	q = q / p
	return p * Sqrt(1+q*q)
}
