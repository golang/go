// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include "cc.h"

enum {
	WORDSIZE = sizeof(uint32),
	WORDBITS = 32,
};

uintptr
bvsize(uintptr n)
{
	return ((n + WORDBITS - 1) / WORDBITS) * WORDSIZE;
}

Bvec*
bvalloc(int32 n)
{
	Bvec *bv;
	uintptr nbytes;

	if(n < 0)
		fatal(Z, "bvalloc: initial size is negative\n");
	nbytes = sizeof(Bvec) + bvsize(n);
	bv = malloc(nbytes);
	if(bv == nil)
		fatal(Z, "bvalloc: malloc failed\n");
	memset(bv, 0, nbytes);
	bv->n = n;
	return bv;
}

void
bvset(Bvec *bv, int32 i)
{
	uint32 mask;

	if(i < 0 || i >= bv->n)
		fatal(Z, "bvset: index %d is out of bounds with length %d\n", i, bv->n);
	mask = 1UL << (i % WORDBITS);
	bv->b[i / WORDBITS] |= mask;
}
