// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

export	hypot

/*
	hypot -- sqrt(p*p + q*q), but overflows only if the result does.
	See Cleve Moler and Donald Morrison,
	Replacing Square Roots by Pythagorean Sums
	IBM Journal of Research and Development,
	Vol. 27, Number 6, pp. 577-581, Nov. 1983
 */

func
hypot(p, q double) double
{
	var r, s, pfac double;

	if p < 0 {
		p = -p;
	}
	if q < 0 {
		q = -q;
	}

	if p < q {
		r = p;
		p = q;
		q = r;
	}

	if p == 0 {
		return 0;
	}

	pfac = p;
	q = q/p;
	r = q;
	p = 1;
	for ;; {
		r = r*r;
		s = r+4;
		if s == 4 {
			return p*pfac;
		}
		r = r/s;
		p = p + 2*r*p;
		q = q*r;
		r = q/p;
	}
}
