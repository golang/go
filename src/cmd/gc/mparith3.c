// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go.h"

/// implements float arihmetic

void
mpaddfltflt(Mpflt *a, Mpflt *b)
{
	a->val += b->val;
}

void
mpmulfltflt(Mpflt *a, Mpflt *b)
{
	a->val *= b->val;
}

void
mpdivfltflt(Mpflt *a, Mpflt *b)
{
	a->val /= b->val;
}

double
mpgetflt(Mpflt *a)
{
	return a->val;
}

void
mpmovecflt(Mpflt *a, double c)
{
	a->val = c;
}

void
mpnegflt(Mpflt *a)
{
	a->val = -a->val;
}

int
mptestflt(Mpflt *a)
{
	if(a->val < 0)
		return -1;
	if(a->val > 0)
		return +1;
	return 0;
}
