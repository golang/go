// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "type.h"

/*
 * map and chan helpers for
 * dealing with unknown types
 */
void
runtime·memhash(uintptr *h, uintptr s, void *a)
{
	byte *b;
	uintptr hash;

	b = a;
	if(sizeof(hash) == 4)
		hash = 2860486313U;
	else
		hash = 33054211828000289ULL;
	while(s > 0) {
		if(sizeof(hash) == 4)
			hash = (hash ^ *b) * 3267000013UL;
		else
			hash = (hash ^ *b) * 23344194077549503ULL;
		b++;
		s--;
	}
	*h ^= hash;
}

void
runtime·memequal(bool *eq, uintptr s, void *a, void *b)
{
	byte *ba, *bb, *aend;

	if(a == b) {
		*eq = 1;
		return;
	}
	ba = a;
	bb = b;
	aend = ba+s;
	while(ba != aend) {
		if(*ba != *bb) {
			*eq = 0;
			return;
		}
		ba++;
		bb++;
	}
	*eq = 1;
	return;
}

void
runtime·memprint(uintptr s, void *a)
{
	uint64 v;

	v = 0xbadb00b;
	switch(s) {
	case 1:
		v = *(uint8*)a;
		break;
	case 2:
		v = *(uint16*)a;
		break;
	case 4:
		v = *(uint32*)a;
		break;
	case 8:
		v = *(uint64*)a;
		break;
	}
	runtime·printint(v);
}

void
runtime·memcopy(uintptr s, void *a, void *b)
{
	if(b == nil) {
		runtime·memclr(a, s);
		return;
	}
	runtime·memmove(a, b, s);
}

void
runtime·memequal0(bool *eq, uintptr s, void *a, void *b)
{
	USED(s);
	USED(a);
	USED(b);
	*eq = true;
}

void
runtime·memcopy0(uintptr s, void *a, void *b)
{
	USED(s);
	USED(a);
	USED(b);
}

void
runtime·memequal8(bool *eq, uintptr s, void *a, void *b)
{
	USED(s);
	*eq = *(uint8*)a == *(uint8*)b;
}

void
runtime·memcopy8(uintptr s, void *a, void *b)
{
	USED(s);
	if(b == nil) {
		*(uint8*)a = 0;
		return;
	}
	*(uint8*)a = *(uint8*)b;
}

void
runtime·memequal16(bool *eq, uintptr s, void *a, void *b)
{
	USED(s);
	*eq = *(uint16*)a == *(uint16*)b;
}

void
runtime·memcopy16(uintptr s, void *a, void *b)
{
	USED(s);
	if(b == nil) {
		*(uint16*)a = 0;
		return;
	}
	*(uint16*)a = *(uint16*)b;
}

void
runtime·memequal32(bool *eq, uintptr s, void *a, void *b)
{
	USED(s);
	*eq = *(uint32*)a == *(uint32*)b;
}

void
runtime·memcopy32(uintptr s, void *a, void *b)
{
	USED(s);
	if(b == nil) {
		*(uint32*)a = 0;
		return;
	}
	*(uint32*)a = *(uint32*)b;
}

void
runtime·memequal64(bool *eq, uintptr s, void *a, void *b)
{
	USED(s);
	*eq = *(uint64*)a == *(uint64*)b;
}

void
runtime·memcopy64(uintptr s, void *a, void *b)
{
	USED(s);
	if(b == nil) {
		*(uint64*)a = 0;
		return;
	}
	*(uint64*)a = *(uint64*)b;
}

void
runtime·memequal128(bool *eq, uintptr s, void *a, void *b)
{
	USED(s);
	*eq = ((uint64*)a)[0] == ((uint64*)b)[0] && ((uint64*)a)[1] == ((uint64*)b)[1];
}

void
runtime·memcopy128(uintptr s, void *a, void *b)
{
	USED(s);
	if(b == nil) {
		((uint64*)a)[0] = 0;
		((uint64*)a)[1] = 0;
		return;
	}
	((uint64*)a)[0] = ((uint64*)b)[0];
	((uint64*)a)[1] = ((uint64*)b)[1];
}

void
runtime·f32equal(bool *eq, uintptr s, void *a, void *b)
{
	USED(s);
	*eq = *(float32*)a == *(float32*)b;
}

void
runtime·f64equal(bool *eq, uintptr s, void *a, void *b)
{
	USED(s);
	*eq = *(float64*)a == *(float64*)b;
}

void
runtime·c64equal(bool *eq, uintptr s, void *a, void *b)
{	
	Complex64 *ca, *cb;
	
	USED(s);
	ca = a;
	cb = b;
	*eq = ca->real == cb->real && ca->imag == cb->imag;
}

void
runtime·c128equal(bool *eq, uintptr s, void *a, void *b)
{	
	Complex128 *ca, *cb;
	
	USED(s);
	ca = a;
	cb = b;
	*eq = ca->real == cb->real && ca->imag == cb->imag;
}

// NOTE: Because NaN != NaN, a map can contain any
// number of (mostly useless) entries keyed with NaNs.
// To avoid long hash chains, we assign a random number
// as the hash value for a NaN.

void
runtime·f32hash(uintptr *h, uintptr s, void *a)
{
	uintptr hash;
	float32 f;

	USED(s);
	f = *(float32*)a;
	if(f == 0)
		hash = 0;  // +0, -0
	else if(f != f)
		hash = runtime·fastrand1();  // any kind of NaN
	else
		hash = *(uint32*)a;
	*h ^= (*h ^ hash ^ 2860486313U) * 3267000013U;
}

void
runtime·f64hash(uintptr *h, uintptr s, void *a)
{
	uintptr hash;
	float64 f;
	uint64 u;

	USED(s);
	f = *(float32*)a;
	if(f == 0)
		hash = 0;	// +0, -0
	else if(f != f)
		hash = runtime·fastrand1();  // any kind of NaN
	else {
		u = *(uint64*)a;
		if(sizeof(uintptr) == 4)
			hash = ((uint32)(u>>32) ^ 2860486313) * (uint32)u;
		else
			hash = u;
	}
	if(sizeof(uintptr) == 4)
		*h = (*h ^ hash ^ 2860486313U) * 3267000013U;
	else
		*h = (*h ^ hash ^ 33054211828000289ULL) * 23344194077549503ULL;
}

void
runtime·c64hash(uintptr *h, uintptr s, void *a)
{
	USED(s);
	runtime·f32hash(h, 0, a);
	runtime·f32hash(h, 0, (float32*)a+1);
}

void
runtime·c128hash(uintptr *h, uintptr s, void *a)
{
	USED(s);
	runtime·f64hash(h, 0, a);
	runtime·f64hash(h, 0, (float64*)a+1);
}

void
runtime·slicecopy(uintptr s, void *a, void *b)
{
	USED(s);
	if(b == nil) {
		((Slice*)a)->array = 0;
		((Slice*)a)->len = 0;
		((Slice*)a)->cap = 0;
		return;
	}
	((Slice*)a)->array = ((Slice*)b)->array;
	((Slice*)a)->len = ((Slice*)b)->len;
	((Slice*)a)->cap = ((Slice*)b)->cap;
}

void
runtime·strhash(uintptr *h, uintptr s, void *a)
{
	USED(s);
	runtime·memhash(h, ((String*)a)->len, ((String*)a)->str);
}

void
runtime·strequal(bool *eq, uintptr s, void *a, void *b)
{
	int32 alen;

	USED(s);
	alen = ((String*)a)->len;
	if(alen != ((String*)b)->len) {
		*eq = false;
		return;
	}
	runtime·memequal(eq, alen, ((String*)a)->str, ((String*)b)->str);
}

void
runtime·strprint(uintptr s, void *a)
{
	USED(s);
	runtime·printstring(*(String*)a);
}

void
runtime·strcopy(uintptr s, void *a, void *b)
{
	USED(s);
	if(b == nil) {
		((String*)a)->str = 0;
		((String*)a)->len = 0;
		return;
	}
	((String*)a)->str = ((String*)b)->str;
	((String*)a)->len = ((String*)b)->len;
}

void
runtime·interhash(uintptr *h, uintptr s, void *a)
{
	USED(s);
	*h ^= runtime·ifacehash(*(Iface*)a);
}

void
runtime·interprint(uintptr s, void *a)
{
	USED(s);
	runtime·printiface(*(Iface*)a);
}

void
runtime·interequal(bool *eq, uintptr s, void *a, void *b)
{
	USED(s);
	*eq = runtime·ifaceeq_c(*(Iface*)a, *(Iface*)b);
}

void
runtime·intercopy(uintptr s, void *a, void *b)
{
	USED(s);
	if(b == nil) {
		((Iface*)a)->tab = 0;
		((Iface*)a)->data = 0;
		return;
	}
	((Iface*)a)->tab = ((Iface*)b)->tab;
	((Iface*)a)->data = ((Iface*)b)->data;
}

void
runtime·nilinterhash(uintptr *h, uintptr s, void *a)
{
	USED(s);
	*h ^= runtime·efacehash(*(Eface*)a);
}

void
runtime·nilinterprint(uintptr s, void *a)
{
	USED(s);
	runtime·printeface(*(Eface*)a);
}

void
runtime·nilinterequal(bool *eq, uintptr s, void *a, void *b)
{
	USED(s);
	*eq = runtime·efaceeq_c(*(Eface*)a, *(Eface*)b);
}

void
runtime·nilintercopy(uintptr s, void *a, void *b)
{
	USED(s);
	if(b == nil) {
		((Eface*)a)->type = 0;
		((Eface*)a)->data = 0;
		return;
	}
	((Eface*)a)->type = ((Eface*)b)->type;
	((Eface*)a)->data = ((Eface*)b)->data;
}

void
runtime·nohash(uintptr *h, uintptr s, void *a)
{
	USED(s);
	USED(a);
	USED(h);
	runtime·panicstring("hash of unhashable type");
}

void
runtime·noequal(bool *eq, uintptr s, void *a, void *b)
{
	USED(s);
	USED(a);
	USED(b);
	USED(eq);
	runtime·panicstring("comparing uncomparable types");
}

Alg
runtime·algarray[] =
{
[AMEM]		{ runtime·memhash, runtime·memequal, runtime·memprint, runtime·memcopy },
[ANOEQ]		{ runtime·nohash, runtime·noequal, runtime·memprint, runtime·memcopy },
[ASTRING]	{ runtime·strhash, runtime·strequal, runtime·strprint, runtime·strcopy },
[AINTER]	{ runtime·interhash, runtime·interequal, runtime·interprint, runtime·intercopy },
[ANILINTER]	{ runtime·nilinterhash, runtime·nilinterequal, runtime·nilinterprint, runtime·nilintercopy },
[ASLICE]	{ runtime·nohash, runtime·noequal, runtime·memprint, runtime·slicecopy },
[AFLOAT32]	{ runtime·f32hash, runtime·f32equal, runtime·memprint, runtime·memcopy },
[AFLOAT64]	{ runtime·f64hash, runtime·f64equal, runtime·memprint, runtime·memcopy },
[ACPLX64]	{ runtime·c64hash, runtime·c64equal, runtime·memprint, runtime·memcopy },
[ACPLX128]	{ runtime·c128hash, runtime·c128equal, runtime·memprint, runtime·memcopy },
[AMEM0]		{ runtime·memhash, runtime·memequal0, runtime·memprint, runtime·memcopy0 },
[AMEM8]		{ runtime·memhash, runtime·memequal8, runtime·memprint, runtime·memcopy8 },
[AMEM16]	{ runtime·memhash, runtime·memequal16, runtime·memprint, runtime·memcopy16 },
[AMEM32]	{ runtime·memhash, runtime·memequal32, runtime·memprint, runtime·memcopy32 },
[AMEM64]	{ runtime·memhash, runtime·memequal64, runtime·memprint, runtime·memcopy64 },
[AMEM128]	{ runtime·memhash, runtime·memequal128, runtime·memprint, runtime·memcopy128 },
[ANOEQ0]	{ runtime·nohash, runtime·noequal, runtime·memprint, runtime·memcopy0 },
[ANOEQ8]	{ runtime·nohash, runtime·noequal, runtime·memprint, runtime·memcopy8 },
[ANOEQ16]	{ runtime·nohash, runtime·noequal, runtime·memprint, runtime·memcopy16 },
[ANOEQ32]	{ runtime·nohash, runtime·noequal, runtime·memprint, runtime·memcopy32 },
[ANOEQ64]	{ runtime·nohash, runtime·noequal, runtime·memprint, runtime·memcopy64 },
[ANOEQ128]	{ runtime·nohash, runtime·noequal, runtime·memprint, runtime·memcopy128 },
};

// Runtime helpers.

// func equal(t *Type, x T, y T) (ret bool)
#pragma textflag 7
void
runtime·equal(Type *t, ...)
{
	byte *x, *y;
	bool *ret;
	
	x = (byte*)(&t+1);
	y = x + t->size;
	ret = (bool*)(y + t->size);
	t->alg->equal(ret, t->size, x, y);
}
