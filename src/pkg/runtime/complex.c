// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

typedef struct Complex128 Complex128;

void
·complex128div(Complex128 n, Complex128 d, Complex128 q)
{
	int32 ninf, dinf, nnan, dnan;
	float64 a, b, ratio, denom;

	// Special cases as in C99.
	ninf = isInf(n.real, 0) || isInf(n.imag, 0);
	dinf = isInf(d.real, 0) || isInf(d.imag, 0);

	nnan = !ninf && (isNaN(n.real) || isNaN(n.imag));
	dnan = !dinf && (isNaN(d.real) || isNaN(d.imag));

	if(nnan || dnan) {
		q.real = NaN();
		q.imag = NaN();
	} else if(ninf && !dinf && !dnan) {
		q.real = Inf(0);
		q.imag = Inf(0);
	} else if(!ninf && !nnan && dinf) {
		q.real = 0;
		q.imag = 0;
	} else if(d.real == 0 && d.imag == 0) {
		if(n.real == 0 && n.imag == 0) {
			q.real = NaN();
			q.imag = NaN();
		} else {
			q.real = Inf(0);
			q.imag = Inf(0);
		}
	} else {
		// Standard complex arithmetic, factored to avoid unnecessary overflow.
		a = d.real;
		if(a < 0)
			a = -a;
		b = d.imag;
		if(b < 0)
			b = -b;
		if(a <= b) {
			ratio = d.real/d.imag;
			denom = d.real*ratio + d.imag;
			q.real = (n.real*ratio + n.imag) / denom;
			q.imag = (n.imag*ratio - n.real) / denom;
		} else {
			ratio = d.imag/d.real;
			denom = d.imag*ratio + d.real;
			q.real = (n.imag*ratio + n.real) / denom;
			q.imag = (n.imag - n.real*ratio) / denom;
		}
	}
	FLUSH(&q);
}
