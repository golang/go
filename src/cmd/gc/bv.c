// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include "go.h"

enum {
	WORDSIZE = 4,
	WORDBITS = 32,
	WORDMASK = WORDBITS - 1,
	WORDSHIFT = 5,
};

static uintptr
bvsize(uintptr n)
{
	return ((n + WORDBITS - 1) / WORDBITS) * WORDSIZE;
}

int32
bvbits(Bvec *bv)
{
	return bv->n;
}

int32
bvwords(Bvec *bv)
{
	return (bv->n + WORDBITS - 1) / WORDBITS;
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

/* difference */
void
bvandnot(Bvec *dst, Bvec *src1, Bvec *src2)
{
	int32 i, w;

	if(dst->n != src1->n || dst->n != src2->n)
		fatal("bvand: lengths %d, %d, and %d are not equal", dst->n, src1->n, src2->n);
	for(i = 0, w = 0; i < dst->n; i += WORDBITS, w++)
		dst->b[w] = src1->b[w] & ~src2->b[w];
}

int
bvcmp(Bvec *bv1, Bvec *bv2)
{
	uintptr nbytes;

	if(bv1->n != bv2->n)
		fatal("bvequal: lengths %d and %d are not equal", bv1->n, bv2->n);
	nbytes = bvsize(bv1->n);
	return memcmp(bv1->b, bv2->b, nbytes);
}

void
bvcopy(Bvec *dst, Bvec *src)
{
	memmove(dst->b, src->b, bvsize(dst->n));
}

Bvec*
bvconcat(Bvec *src1, Bvec *src2)
{
	Bvec *dst;
	int32 i;

	dst = bvalloc(src1->n + src2->n);
	for(i = 0; i < src1->n; i++)
		if(bvget(src1, i))
			bvset(dst, i);
	for(i = 0; i < src2->n; i++)
		if(bvget(src2, i))
			bvset(dst, i + src1->n);
	return dst;
}

int
bvget(Bvec *bv, int32 i)
{
	if(i < 0 || i >= bv->n)
		fatal("bvget: index %d is out of bounds with length %d\n", i, bv->n);
	return (bv->b[i>>WORDSHIFT] >> (i&WORDMASK)) & 1;
}

// bvnext returns the smallest index >= i for which bvget(bv, i) == 1.
// If there is no such index, bvnext returns -1.
int
bvnext(Bvec *bv, int32 i)
{
	uint32 w;

	if(i >= bv->n)
		return -1;

	// Jump i ahead to next word with bits.
	if((bv->b[i>>WORDSHIFT]>>(i&WORDMASK)) == 0) {
		i &= ~WORDMASK;
		i += WORDBITS;
		while(i < bv->n && bv->b[i>>WORDSHIFT] == 0)
			i += WORDBITS;
	}
	if(i >= bv->n)
		return -1;

	// Find 1 bit.
	w = bv->b[i>>WORDSHIFT]>>(i&WORDMASK);
	while((w&1) == 0) {
		w>>=1;
		i++;
	}
	return i;
}

int
bvisempty(Bvec *bv)
{
	int32 i;

	for(i = 0; i < bv->n; i += WORDBITS)
		if(bv->b[i>>WORDSHIFT] != 0)
			return 0;
	return 1;
}

void
bvnot(Bvec *bv)
{
	int32 i, w;

	for(i = 0, w = 0; i < bv->n; i += WORDBITS, w++)
		bv->b[w] = ~bv->b[w];
}

/* union */
void
bvor(Bvec *dst, Bvec *src1, Bvec *src2)
{
	int32 i, w;

	if(dst->n != src1->n || dst->n != src2->n)
		fatal("bvor: lengths %d, %d, and %d are not equal", dst->n, src1->n, src2->n);
	for(i = 0, w = 0; i < dst->n; i += WORDBITS, w++)
		dst->b[w] = src1->b[w] | src2->b[w];
}

/* intersection */
void
bvand(Bvec *dst, Bvec *src1, Bvec *src2)
{
	int32 i, w;

	if(dst->n != src1->n || dst->n != src2->n)
		fatal("bvor: lengths %d, %d, and %d are not equal", dst->n, src1->n, src2->n);
	for(i = 0, w = 0; i < dst->n; i += WORDBITS, w++)
		dst->b[w] = src1->b[w] & src2->b[w];
}

void
bvprint(Bvec *bv)
{
	int32 i;

	print("#*");
	for(i = 0; i < bv->n; i++)
		print("%d", bvget(bv, i));
}

void
bvreset(Bvec *bv, int32 i)
{
	uint32 mask;

	if(i < 0 || i >= bv->n)
		fatal("bvreset: index %d is out of bounds with length %d\n", i, bv->n);
	mask = ~(1 << (i % WORDBITS));
	bv->b[i / WORDBITS] &= mask;
}

void
bvresetall(Bvec *bv)
{
	memset(bv->b, 0x00, bvsize(bv->n));
}

void
bvset(Bvec *bv, int32 i)
{
	uint32 mask;

	if(i < 0 || i >= bv->n)
		fatal("bvset: index %d is out of bounds with length %d\n", i, bv->n);
	mask = 1U << (i % WORDBITS);
	bv->b[i / WORDBITS] |= mask;
}
