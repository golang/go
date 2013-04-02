// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "type.h"

#define M0 (sizeof(uintptr)==4 ? 2860486313UL : 33054211828000289ULL)
#define M1 (sizeof(uintptr)==4 ? 3267000013UL : 23344194077549503ULL)

static bool use_aeshash;

/*
 * map and chan helpers for
 * dealing with unknown types
 */
void
runtime·memhash(uintptr *h, uintptr s, void *a)
{
	byte *b;
	uintptr hash;
	if(use_aeshash) {
		runtime·aeshash(h, s, a);
		return;
	}

	b = a;
	hash = M0 ^ *h;
	while(s > 0) {
		hash = (hash ^ *b) * M1;
		b++;
		s--;
	}
	*h = hash;
}

void
runtime·memequal(bool *eq, uintptr s, void *a, void *b)
{
	if(a == b) {
		*eq = 1;
		return;
	}
	*eq = runtime·memeq(a, b, s);
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
	*h = (*h ^ hash ^ M0) * M1;
}

void
runtime·f64hash(uintptr *h, uintptr s, void *a)
{
	uintptr hash;
	float64 f;
	uint64 u;

	USED(s);
	f = *(float64*)a;
	if(f == 0)
		hash = 0;	// +0, -0
	else if(f != f)
		hash = runtime·fastrand1();  // any kind of NaN
	else {
		u = *(uint64*)a;
		if(sizeof(uintptr) == 4)
			hash = ((uint32)(u>>32) * M1) ^ (uint32)u;
		else
			hash = u;
	}
	*h = (*h ^ hash ^ M0) * M1;
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
	intgo alen;
	byte *s1, *s2;

	USED(s);
	alen = ((String*)a)->len;
	if(alen != ((String*)b)->len) {
		*eq = false;
		return;
	}
	s1 = ((String*)a)->str;
	s2 = ((String*)b)->str;
	if(s1 == s2) {
		*eq = true;
		return;
	}
	*eq = runtime·memeq(s1, s2, alen);
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
	*h = runtime·ifacehash(*(Iface*)a, *h ^ M0) * M1;
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
	*h = runtime·efacehash(*(Eface*)a, *h ^ M0) * M1;
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

// used in asm_{386,amd64}.s
byte runtime·aeskeysched[HashRandomBytes];

void
runtime·hashinit(void)
{
	// Install aes hash algorithm if we have the instructions we need
	if((runtime·cpuid_ecx & (1 << 25)) != 0 &&  // aes (aesenc)
	   (runtime·cpuid_ecx & (1 << 9)) != 0 &&   // sse3 (pshufb)
	   (runtime·cpuid_ecx & (1 << 19)) != 0) {  // sse4.1 (pinsr{d,q})
		byte *rnd;
		int32 n;
		use_aeshash = true;
		runtime·algarray[AMEM].hash = runtime·aeshash;
		runtime·algarray[AMEM8].hash = runtime·aeshash;
		runtime·algarray[AMEM16].hash = runtime·aeshash;
		runtime·algarray[AMEM32].hash = runtime·aeshash32;
		runtime·algarray[AMEM64].hash = runtime·aeshash64;
		runtime·algarray[AMEM128].hash = runtime·aeshash;
		runtime·algarray[ASTRING].hash = runtime·aeshashstr;

		// Initialize with random data so hash collisions will be hard to engineer.
		runtime·get_random_data(&rnd, &n);
		if(n > HashRandomBytes)
			n = HashRandomBytes;
		runtime·memmove(runtime·aeskeysched, rnd, n);
		if(n < HashRandomBytes) {
			// Not very random, but better than nothing.
			int64 t = runtime·nanotime();
			while (n < HashRandomBytes) {
				runtime·aeskeysched[n++] = (int8)(t >> (8 * (n % 8)));
			}
		}
	}
}

// func equal(t *Type, x T, y T) (ret bool)
#pragma textflag 7
void
runtime·equal(Type *t, ...)
{
	byte *x, *y;
	uintptr ret;
	
	x = (byte*)(&t+1);
	y = x + t->size;
	ret = (uintptr)(y + t->size);
	ret = ROUND(ret, Structrnd);
	t->alg->equal((bool*)ret, t->size, x, y);
}
