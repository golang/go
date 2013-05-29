// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include "go.h"

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
		fatal("bvalloc: initial size is negative\n");
	nbytes = sizeof(Bvec) + bvsize(n);
	bv = malloc(nbytes);
	if(bv == nil)
		fatal("bvalloc: malloc failed\n");
	memset(bv, 0, nbytes);
	bv->n = n;
	return bv;
}

void
bvset(Bvec *bv, int32 i)
{
	uint32 mask;

	if(i < 0 || i >= bv->n)
		fatal("bvset: index %d is out of bounds with length %d\n", i, bv->n);
	mask = 1 << (i % WORDBITS);
	bv->b[i / WORDBITS] |= mask;
}

void
bvres(Bvec *bv, int32 i)
{
	uint32 mask;

	if(i < 0 || i >= bv->n)
		fatal("bvres: index %d is out of bounds with length %d\n", i, bv->n);
	mask = ~(1 << (i % WORDBITS));
	bv->b[i / WORDBITS] &= mask;
}

int
bvget(Bvec *bv, int32 i)
{
	uint32 mask, word;

	if(i < 0 || i >= bv->n)
		fatal("bvget: index %d is out of bounds with length %d\n", i, bv->n);
	mask = 1 << (i % WORDBITS);
	word = bv->b[i / WORDBITS] & mask;
	return word ? 1 : 0;
}

int
bvisempty(Bvec *bv)
{
	int32 i;

	for(i = 0; i < bv->n; i += WORDBITS)
		if(bv->b[i / WORDBITS] != 0)
			return 0;
	return 1;
}

int bvcmp(Bvec *bv1, Bvec *bv2)
{
	int32 i;

	if(bv1->n != bv2->n) {
		fatal("bvcmp: size %d != %d\n", bv1->n, bv2->n);
	}
	for(i = 0; i < bv1->n; i += WORDBITS) {
		if(bv1->b[i / WORDBITS] != bv2->b[i / WORDBITS]) {
			fatal("bvcmp: element %x != %x @ %d\n", bv1->b[i/WORDBITS], bv2->b[i/WORDBITS], i/WORDBITS);
		}
	}
	return 0;
}
