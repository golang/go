// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// The original C code, the long comment, and the constants
// below are from FreeBSD's /usr/src/lib/msun/src/e_acosh.c
// and came with this notice.  The go code is a simplified
// version of the original C.
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
//
//
// __ieee754_acosh(x)
// Method :
//	Based on
//	        acosh(x) = log [ x + sqrt(x*x-1) ]
//	we have
//	        acosh(x) := log(x)+ln2,	if x is large; else
//	        acosh(x) := log(2x-1/(sqrt(x*x-1)+x)) if x>2; else
//	        acosh(x) := log1p(t+sqrt(2.0*t+t*t)); where t=x-1.
//
// Special cases:
//	acosh(x) is NaN with signal if x<1.
//	acosh(NaN) is NaN without signal.
//

// Acosh(x) calculates the inverse hyperbolic cosine of x.
//
// Special cases are:
//	Acosh(x) = NaN if x < 1
//	Acosh(NaN) = NaN
func Acosh(x float64) float64 {
	const (
		Ln2   = 6.93147180559945286227e-01 // 0x3FE62E42FEFA39EF
		Large = 1 << 28                    // 2**28
	)
	// TODO(rsc): Remove manual inlining of IsNaN
	// when compiler does it for us
	// first case is special case
	switch {
	case x < 1 || x != x: // x < 1 || IsNaN(x):
		return NaN()
	case x == 1:
		return 0
	case x >= Large:
		return Log(x) + Ln2 // x > 2**28
	case x > 2:
		return Log(2*x - 1/(x+Sqrt(x*x-1))) // 2**28 > x > 2
	}
	t := x - 1
	return Log1p(t + Sqrt(2*t+t*t)) // 2 >= x > 1
}
