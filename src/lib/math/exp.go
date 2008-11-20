// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

import "math"

/*
 *	exp returns the exponential func of its
 *	floating-point argument.
 *
 *	The coefficients are #1069 from Hart and Cheney. (22.35D)
 */

const
(
	p0	= .2080384346694663001443843411e7;
	p1	= .3028697169744036299076048876e5;
	p2	= .6061485330061080841615584556e2;
	q0	= .6002720360238832528230907598e7;
	q1	= .3277251518082914423057964422e6;
	q2	= .1749287689093076403844945335e4;
	log2e	= .14426950408889634073599247e1;
	sqrt2	= .14142135623730950488016887e1;
	maxf	= 10000;
)

export func Exp(arg float64) float64 {
	if arg == 0. {
		return 1;
	}
	if arg < -maxf {
		return 0;
	}
	if arg > maxf {
		return sys.Inf(1)
	}

	x := arg*log2e;
	ent := int(Floor(x));
	fract := (x-float64(ent)) - 0.5;
	xsq := fract*fract;
	temp1 := ((p2*xsq+p1)*xsq+p0)*fract;
	temp2 := ((xsq+q2)*xsq+q1)*xsq + q0;
	return sys.ldexp(sqrt2*(temp2+temp1)/(temp2-temp1), ent);
}
