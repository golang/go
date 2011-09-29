// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

/*
	Bessel function of the first and second kinds of order n.
*/

// The original C code and the long comment below are
// from FreeBSD's /usr/src/lib/msun/src/e_jn.c and
// came with this notice.  The go code is a simplified
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
// __ieee754_jn(n, x), __ieee754_yn(n, x)
// floating point Bessel's function of the 1st and 2nd kind
// of order n
//
// Special cases:
//      y0(0)=y1(0)=yn(n,0) = -inf with division by zero signal;
//      y0(-ve)=y1(-ve)=yn(n,-ve) are NaN with invalid signal.
// Note 2. About jn(n,x), yn(n,x)
//      For n=0, j0(x) is called,
//      for n=1, j1(x) is called,
//      for n<x, forward recursion is used starting
//      from values of j0(x) and j1(x).
//      for n>x, a continued fraction approximation to
//      j(n,x)/j(n-1,x) is evaluated and then backward
//      recursion is used starting from a supposed value
//      for j(n,x). The resulting value of j(0,x) is
//      compared with the actual value to correct the
//      supposed value of j(n,x).
//
//      yn(n,x) is similar in all respects, except
//      that forward recursion is used for all
//      values of n>1.

// Jn returns the order-n Bessel function of the first kind.
//
// Special cases are:
//	Jn(n, ±Inf) = 0
//	Jn(n, NaN) = NaN
func Jn(n int, x float64) float64 {
	const (
		TwoM29 = 1.0 / (1 << 29) // 2**-29 0x3e10000000000000
		Two302 = 1 << 302        // 2**302 0x52D0000000000000
	)
	// TODO(rsc): Remove manual inlining of IsNaN, IsInf
	// when compiler does it for us
	// special cases
	switch {
	case x != x: // IsNaN(x)
		return x
	case x < -MaxFloat64 || x > MaxFloat64: // IsInf(x, 0):
		return 0
	}
	// J(-n, x) = (-1)**n * J(n, x), J(n, -x) = (-1)**n * J(n, x)
	// Thus, J(-n, x) = J(n, -x)

	if n == 0 {
		return J0(x)
	}
	if x == 0 {
		return 0
	}
	if n < 0 {
		n, x = -n, -x
	}
	if n == 1 {
		return J1(x)
	}
	sign := false
	if x < 0 {
		x = -x
		if n&1 == 1 {
			sign = true // odd n and negative x
		}
	}
	var b float64
	if float64(n) <= x {
		// Safe to use J(n+1,x)=2n/x *J(n,x)-J(n-1,x)
		if x >= Two302 { // x > 2**302

			// (x >> n**2)
			//          Jn(x) = cos(x-(2n+1)*pi/4)*sqrt(2/x*pi)
			//          Yn(x) = sin(x-(2n+1)*pi/4)*sqrt(2/x*pi)
			//          Let s=sin(x), c=cos(x),
			//              xn=x-(2n+1)*pi/4, sqt2 = sqrt(2),then
			//
			//                 n    sin(xn)*sqt2    cos(xn)*sqt2
			//              ----------------------------------
			//                 0     s-c             c+s
			//                 1    -s-c            -c+s
			//                 2    -s+c            -c-s
			//                 3     s+c             c-s

			var temp float64
			switch n & 3 {
			case 0:
				temp = Cos(x) + Sin(x)
			case 1:
				temp = -Cos(x) + Sin(x)
			case 2:
				temp = -Cos(x) - Sin(x)
			case 3:
				temp = Cos(x) - Sin(x)
			}
			b = (1 / SqrtPi) * temp / Sqrt(x)
		} else {
			b = J1(x)
			for i, a := 1, J0(x); i < n; i++ {
				a, b = b, b*(float64(i+i)/x)-a // avoid underflow
			}
		}
	} else {
		if x < TwoM29 { // x < 2**-29
			// x is tiny, return the first Taylor expansion of J(n,x)
			// J(n,x) = 1/n!*(x/2)**n  - ...

			if n > 33 { // underflow
				b = 0
			} else {
				temp := x * 0.5
				b = temp
				a := 1.0
				for i := 2; i <= n; i++ {
					a *= float64(i) // a = n!
					b *= temp       // b = (x/2)**n
				}
				b /= a
			}
		} else {
			// use backward recurrence
			//                      x      x**2      x**2
			//  J(n,x)/J(n-1,x) =  ----   ------   ------   .....
			//                      2n  - 2(n+1) - 2(n+2)
			//
			//                      1      1        1
			//  (for large x)   =  ----  ------   ------   .....
			//                      2n   2(n+1)   2(n+2)
			//                      -- - ------ - ------ -
			//                       x     x         x
			//
			// Let w = 2n/x and h=2/x, then the above quotient
			// is equal to the continued fraction:
			//                  1
			//      = -----------------------
			//                     1
			//         w - -----------------
			//                        1
			//              w+h - ---------
			//                     w+2h - ...
			//
			// To determine how many terms needed, let
			// Q(0) = w, Q(1) = w(w+h) - 1,
			// Q(k) = (w+k*h)*Q(k-1) - Q(k-2),
			// When Q(k) > 1e4	good for single
			// When Q(k) > 1e9	good for double
			// When Q(k) > 1e17	good for quadruple

			// determine k
			w := float64(n+n) / x
			h := 2 / x
			q0 := w
			z := w + h
			q1 := w*z - 1
			k := 1
			for q1 < 1e9 {
				k += 1
				z += h
				q0, q1 = q1, z*q1-q0
			}
			m := n + n
			t := 0.0
			for i := 2 * (n + k); i >= m; i -= 2 {
				t = 1 / (float64(i)/x - t)
			}
			a := t
			b = 1
			//  estimate log((2/x)**n*n!) = n*log(2/x)+n*ln(n)
			//  Hence, if n*(log(2n/x)) > ...
			//  single 8.8722839355e+01
			//  double 7.09782712893383973096e+02
			//  long double 1.1356523406294143949491931077970765006170e+04
			//  then recurrent value may overflow and the result is
			//  likely underflow to zero

			tmp := float64(n)
			v := 2 / x
			tmp = tmp * Log(Abs(v*tmp))
			if tmp < 7.09782712893383973096e+02 {
				for i := n - 1; i > 0; i-- {
					di := float64(i + i)
					a, b = b, b*di/x-a
					di -= 2
				}
			} else {
				for i := n - 1; i > 0; i-- {
					di := float64(i + i)
					a, b = b, b*di/x-a
					di -= 2
					// scale b to avoid spurious overflow
					if b > 1e100 {
						a /= b
						t /= b
						b = 1
					}
				}
			}
			b = t * J0(x) / b
		}
	}
	if sign {
		return -b
	}
	return b
}

// Yn returns the order-n Bessel function of the second kind.
//
// Special cases are:
//	Yn(n, +Inf) = 0
//	Yn(n > 0, 0) = -Inf
//	Yn(n < 0, 0) = +Inf if n is odd, -Inf if n is even
//	Y1(n, x < 0) = NaN
//	Y1(n, NaN) = NaN
func Yn(n int, x float64) float64 {
	const Two302 = 1 << 302 // 2**302 0x52D0000000000000
	// TODO(rsc): Remove manual inlining of IsNaN, IsInf
	// when compiler does it for us
	// special cases
	switch {
	case x < 0 || x != x: // x < 0 || IsNaN(x):
		return NaN()
	case x > MaxFloat64: // IsInf(x, 1)
		return 0
	}

	if n == 0 {
		return Y0(x)
	}
	if x == 0 {
		if n < 0 && n&1 == 1 {
			return Inf(1)
		}
		return Inf(-1)
	}
	sign := false
	if n < 0 {
		n = -n
		if n&1 == 1 {
			sign = true // sign true if n < 0 && |n| odd
		}
	}
	if n == 1 {
		if sign {
			return -Y1(x)
		}
		return Y1(x)
	}
	var b float64
	if x >= Two302 { // x > 2**302
		// (x >> n**2)
		//	    Jn(x) = cos(x-(2n+1)*pi/4)*sqrt(2/x*pi)
		//	    Yn(x) = sin(x-(2n+1)*pi/4)*sqrt(2/x*pi)
		//	    Let s=sin(x), c=cos(x),
		//		xn=x-(2n+1)*pi/4, sqt2 = sqrt(2),then
		//
		//		   n	sin(xn)*sqt2	cos(xn)*sqt2
		//		----------------------------------
		//		   0	 s-c		 c+s
		//		   1	-s-c 		-c+s
		//		   2	-s+c		-c-s
		//		   3	 s+c		 c-s

		var temp float64
		switch n & 3 {
		case 0:
			temp = Sin(x) - Cos(x)
		case 1:
			temp = -Sin(x) - Cos(x)
		case 2:
			temp = -Sin(x) + Cos(x)
		case 3:
			temp = Sin(x) + Cos(x)
		}
		b = (1 / SqrtPi) * temp / Sqrt(x)
	} else {
		a := Y0(x)
		b = Y1(x)
		// quit if b is -inf
		for i := 1; i < n && b >= -MaxFloat64; i++ { // for i := 1; i < n && !IsInf(b, -1); i++ {
			a, b = b, (float64(i+i)/x)*b-a
		}
	}
	if sign {
		return -b
	}
	return b
}
