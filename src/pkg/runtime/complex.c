// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

// complex128div(num, den complex128) (quo complex128)
void
·complex128div(float64 numreal, float64 numimag,
	float64 denreal, float64 denimag,
	float64 quoreal, float64 quoimag)
{
	float64 a, b, ratio, denom;

	a = denreal;
	if(a < 0)
		a = -a;
	b = denimag;
	if(b < 0)
		b = -b;
	if(a <= b) {
		if(b == 0)
			panicstring("complex divide by zero");
		ratio = denreal/denimag;
		denom = denreal*ratio + denimag;
		quoreal = (numreal*ratio + numimag) / denom;
		quoimag = (numimag*ratio - numreal) / denom;
	} else {
		ratio = denimag/denreal;
		denom = denimag*ratio + denreal;
		quoreal = (numimag*ratio + numreal) / denom;
		quoimag = (numimag - numreal*ratio) / denom;
	}
	FLUSH(&quoreal);
	FLUSH(&quoimag);
}
