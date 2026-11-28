// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

import "math/bits"

/*
	Hypot -- sqrt(p*p + q*q), but overflows only if the result does.
*/

// If both magnitudes are below 2^1023, their norm cannot overflow.
const hypotOverflowCheck = 0x1p1023

// Hypot returns [Sqrt](p*p + q*q), taking care to avoid
// unnecessary overflow and underflow.
//
// Special cases are:
//
//	Hypot(±Inf, q) = +Inf
//	Hypot(p, ±Inf) = +Inf
//	Hypot(NaN, q) = NaN
//	Hypot(p, NaN) = NaN
func Hypot(p, q float64) float64 {
	if haveArchHypot && Abs(p) < hypotOverflowCheck && Abs(q) < hypotOverflowCheck {
		return archHypot(p, q)
	}
	return hypot(p, q)
}

func hypot(p, q float64) float64 {
	p, q = Abs(p), Abs(q)
	// special cases
	switch {
	case IsInf(p, 1) || IsInf(q, 1):
		return Inf(1)
	case IsNaN(p) || IsNaN(q):
		return NaN()
	}
	if p < q {
		p, q = q, p
	}
	if p == 0 {
		return 0
	}
	ratio := q / p
	r := p * Sqrt(1+ratio*ratio)
	if p < hypotOverflowCheck {
		return r
	}
	// Rounding the intermediate values can move the norm across the
	// overflow midpoint in either direction.
	if hypotOverflows(p, q) {
		return Inf(1)
	}
	return min(r, MaxFloat64)
}

// hypotOverflows reports whether the exact norm rounds to +Inf.
// It assumes p and q are finite and p >= 2^1023 and p >= q >= 0.
func hypotOverflows(p, q float64) bool {
	pb, qb := Float64bits(p), Float64bits(q)
	qShift := 2 * ((pb >> shift) - (qb >> shift))
	if qShift >= 54 {
		// Even at MaxFloat64, the gap to the squared midpoint exceeds
		// q*q when q < 2^997.
		return false
	}

	// Use units of 2^970 for the operands. The overflow midpoint is
	// halfway between MaxFloat64 and 2^1024, or 2^54 - 1 in these units.
	// The doubled significands put p and q in these units before
	// accounting for q's smaller exponent.
	const midpoint = 1<<54 - 1
	mhi, mlo := bits.Mul64(midpoint, midpoint)
	ps := (pb&fracMask | 1<<shift) << 1
	qs := (qb&fracMask | 1<<shift) << 1
	phi, plo := bits.Mul64(ps, ps)
	qhi, qlo := bits.Mul64(qs, qs)

	// The comparison boundary is an integer, so truncating the
	// fractional part of q*q does not change the overflow decision.
	lo, carry := bits.Add64(plo, qlo>>qShift|qhi<<(64-qShift), 0)
	hi := phi + qhi>>qShift + carry
	return hi > mhi || hi == mhi && lo >= mlo
}
