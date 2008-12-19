// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

int32	panicking	= 0;
int32	maxround	= 8;

int32
gotraceback(void)
{
	byte *p;

	p = getenv("GOTRACEBACK");
	if(p == nil || p[0] == '\0')
		return 1;	// default is on
	return atoi(p);
}

void
sys·panicl(int32 lno)
{
	uint8 *sp;

	prints("\npanic ");
	sys·printpc(&lno);
	prints("\n");
	sp = (uint8*)&lno;
	if(gotraceback()){
		traceback(sys·getcallerpc(&lno), sp, g);
		tracebackothers(g);
	}
	panicking = 1;
	sys·breakpoint();  // so we can grab it in a debugger
	sys·exit(2);
}

void
sys·throwindex(void)
{
	throw("index out of range");
}

void
sys·throwreturn(void)
{
	throw("no return at end of a typed function");
}

void
throw(int8 *s)
{
	prints("throw: ");
	prints(s);
	prints("\n");
	*(int32*)0 = 0;
	sys·exit(1);
}

void
mcpy(byte *t, byte *f, uint32 n)
{
	while(n > 0) {
		*t = *f;
		t++;
		f++;
		n--;
	}
}

void
mmov(byte *t, byte *f, uint32 n)
{
	if(t < f) {
		while(n > 0) {
			*t = *f;
			t++;
			f++;
			n--;
		}
	} else {
		t += n;
		f += n;
		while(n > 0) {
			t--;
			f--;
			*t = *f;
			n--;
		}
	}
}

byte*
mchr(byte *p, byte c, byte *ep)
{
	for(; p < ep; p++)
		if(*p == c)
			return p;
	return nil;
}

uint32
rnd(uint32 n, uint32 m)
{
	uint32 r;

	if(m > maxround)
		m = maxround;
	r = n % m;
	if(r)
		n += m-r;
	return n;
}

static	uint64	uvnan		= 0x7FF0000000000001ULL;
static	uint64	uvinf		= 0x7FF0000000000000ULL;
static	uint64	uvneginf	= 0xFFF0000000000000ULL;

static uint32
float32tobits(float32 f)
{
	// The obvious cast-and-pointer code is technically
	// not valid, and gcc miscompiles it.  Use a union instead.
	union {
		float32 f;
		uint32 i;
	} u;
	u.f = f;
	return u.i;
}

static uint64
float64tobits(float64 f)
{
	// The obvious cast-and-pointer code is technically
	// not valid, and gcc miscompiles it.  Use a union instead.
	union {
		float64 f;
		uint64 i;
	} u;
	u.f = f;
	return u.i;
}

static float64
float64frombits(uint64 i)
{
	// The obvious cast-and-pointer code is technically
	// not valid, and gcc miscompiles it.  Use a union instead.
	union {
		float64 f;
		uint64 i;
	} u;
	u.i = i;
	return u.f;
}

static float32
float32frombits(uint32 i)
{
	// The obvious cast-and-pointer code is technically
	// not valid, and gcc miscompiles it.  Use a union instead.
	union {
		float32 f;
		uint32 i;
	} u;
	u.i = i;
	return u.f;
}

bool
isInf(float64 f, int32 sign)
{
	uint64 x;

	x = float64tobits(f);
	if(sign == 0)
		return x == uvinf || x == uvneginf;
	if(sign > 0)
		return x == uvinf;
	return x == uvneginf;
}

static float64
NaN(void)
{
	return float64frombits(uvnan);
}

bool
isNaN(float64 f)
{
	uint64 x;

	x = float64tobits(f);
	return ((uint32)(x>>52) & 0x7FF) == 0x7FF && !isInf(f, 0);
}

static float64
Inf(int32 sign)
{
	if(sign >= 0)
		return float64frombits(uvinf);
	else
		return float64frombits(uvneginf);
}

enum
{
	MASK	= 0x7ffL,
	SHIFT	= 64-11-1,
	BIAS	= 1022L,
};

static float64
frexp(float64 d, int32 *ep)
{
	uint64 x;

	if(d == 0) {
		*ep = 0;
		return 0;
	}
	x = float64tobits(d);
	*ep = (int32)((x >> SHIFT) & MASK) - BIAS;
	x &= ~((uint64)MASK << SHIFT);
	x |= (uint64)BIAS << SHIFT;
	return float64frombits(x);
}

static float64
ldexp(float64 d, int32 e)
{
	uint64 x;

	if(d == 0)
		return 0;
	x = float64tobits(d);
	e += (int32)(x >> SHIFT) & MASK;
	if(e <= 0)
		return 0;	/* underflow */
	if(e >= MASK){		/* overflow */
		if(d < 0)
			return Inf(-1);
		return Inf(1);
	}
	x &= ~((uint64)MASK << SHIFT);
	x |= (uint64)e << SHIFT;
	return float64frombits(x);
}

static float64
modf(float64 d, float64 *ip)
{
	float64 dd;
	uint64 x;
	int32 e;

	if(d < 1) {
		if(d < 0) {
			d = modf(-d, ip);
			*ip = -*ip;
			return -d;
		}
		*ip = 0;
		return d;
	}

	x = float64tobits(d);
	e = (int32)((x >> SHIFT) & MASK) - BIAS;

	/*
	 * Keep the top 11+e bits; clear the rest.
	 */
	if(e <= 64-11)
		x &= ~(((uint64)1 << (64LL-11LL-e))-1);
	dd = float64frombits(x);
	*ip = dd;
	return d - dd;
}

// func frexp(float64) (float64, int32); // break fp into exp,frac
void
sys·frexp(float64 din, float64 dou, int32 iou)
{
	dou = frexp(din, &iou);
	FLUSH(&dou);
}

//func	ldexp(int32, float64) float64;	// make fp from exp,frac
void
sys·ldexp(float64 din, int32 ein, float64 dou)
{
	dou = ldexp(din, ein);
	FLUSH(&dou);
}

//func	modf(float64) (float64, float64);	// break fp into double+double
void
sys·modf(float64 din, float64 integer, float64 fraction)
{
	fraction = modf(din, &integer);
	FLUSH(&fraction);
}

//func	isinf(float64, int32 sign) bool;  // test for infinity
void
sys·isInf(float64 din, int32 signin, bool out)
{
	out = isInf(din, signin);
	FLUSH(&out);
}

//func	isnan(float64) bool;  // test for NaN
void
sys·isNaN(float64 din, bool out)
{
	out = isNaN(din);
	FLUSH(&out);
}

//func	inf(int32 sign) float64;  // signed infinity
void
sys·Inf(int32 signin, float64 out)
{
	out = Inf(signin);
	FLUSH(&out);
}

//func	nan() float64;  // NaN
void
sys·NaN(float64 out)
{
	out = NaN();
	FLUSH(&out);
}

// func float32bits(float32) uint32; // raw bits of float32
void
sys·float32bits(float32 din, uint32 iou)
{
	iou = float32tobits(din);
	FLUSH(&iou);
}

// func float64bits(float64) uint64; // raw bits of float64
void
sys·float64bits(float64 din, uint64 iou)
{
	iou = float64tobits(din);
	FLUSH(&iou);
}

// func float32frombits(uint32) float32; // raw bits to float32
void
sys·float32frombits(uint32 uin, float32 dou)
{
	dou = float32frombits(uin);
	FLUSH(&dou);
}

// func float64frombits(uint64) float64; // raw bits to float64
void
sys·float64frombits(uint64 uin, float64 dou)
{
	dou = float64frombits(uin);
	FLUSH(&dou);
}

static int32	argc;
static uint8**	argv;
static int32	envc;
static uint8**	envv;

void
args(int32 c, uint8 **v)
{
	argc = c;
	argv = v;
	envv = v + argc + 1;  // skip 0 at end of argv
	for (envc = 0; envv[envc] != 0; envc++)
		;
}

int32
getenvc(void)
{
	return envc;
}

byte*
getenv(int8 *s)
{
	int32 i, j, len;
	byte *v, *bs;

	bs = (byte*)s;
	len = findnull(bs);
	for(i=0; i<envc; i++){
		v = envv[i];
		for(j=0; j<len; j++)
			if(bs[j] != v[j])
				goto nomatch;
		if(v[len] != '=')
			goto nomatch;
		return v+len+1;
	nomatch:;
	}
	return nil;
}

int32
atoi(byte *p)
{
	int32 n;

	n = 0;
	while('0' <= *p && *p <= '9')
		n = n*10 + *p++ - '0';
	return n;
}

//func argc() int32;  // return number of arguments
void
sys·argc(int32 v)
{
	v = argc;
	FLUSH(&v);
}

//func envc() int32;  // return number of environment variables
void
sys·envc(int32 v)
{
	v = envc;
	FLUSH(&v);
}

//func argv(i) string;  // return argument i
void
sys·argv(int32 i, string s)
{
	if(i >= 0 && i < argc)
		s = gostring(argv[i]);
	else
		s = emptystring;
	FLUSH(&s);
}

//func envv(i) string;  // return environment variable i
void
sys·envv(int32 i, string s)
{
	if(i >= 0 && i < envc)
		s = gostring(envv[i]);
	else
		s = emptystring;
	FLUSH(&s);
}

void
check(void)
{
	int8 a;
	uint8 b;
	int16 c;
	uint16 d;
	int32 e;
	uint32 f;
	int64 g;
	uint64 h;
	float32 i;
	float64 j;
	void* k;
	uint16* l;

	if(sizeof(a) != 1) throw("bad a");
	if(sizeof(b) != 1) throw("bad b");
	if(sizeof(c) != 2) throw("bad c");
	if(sizeof(d) != 2) throw("bad d");
	if(sizeof(e) != 4) throw("bad e");
	if(sizeof(f) != 4) throw("bad f");
	if(sizeof(g) != 8) throw("bad g");
	if(sizeof(h) != 8) throw("bad h");
	if(sizeof(i) != 4) throw("bad i");
	if(sizeof(j) != 8) throw("bad j");
	if(sizeof(k) != 8) throw("bad k");
	if(sizeof(l) != 8) throw("bad l");
//	prints(1"check ok\n");

	uint32 z;
	z = 1;
	if(!cas(&z, 1, 2))
		throw("cas1");
	if(z != 2)
		throw("cas2");

	z = 4;
	if(cas(&z, 5, 6))
		throw("cas3");
	if(z != 4)
		throw("cas4");

	initsig();
}

/*
 * map and chan helpers for
 * dealing with unknown types
 */
static uint64
memhash(uint32 s, void *a)
{
	byte *b;
	uint64 hash;

	b = a;
	hash = 33054211828000289ULL;
	while(s > 0) {
		hash = (hash ^ *b) * 23344194077549503ULL;
		b++;
		s--;
	}
	return hash;
}

static uint32
memequal(uint32 s, void *a, void *b)
{
	byte *ba, *bb;
	uint32 i;

	ba = a;
	bb = b;
	for(i=0; i<s; i++)
		if(ba[i] != bb[i])
			return 0;
	return 1;
}

static void
memprint(uint32 s, void *a)
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
	sys·printint(v);
}

static void
memcopy(uint32 s, void *a, void *b)
{
	byte *ba, *bb;
	uint32 i;

	ba = a;
	bb = b;
	if(bb == nil) {
		for(i=0; i<s; i++)
			ba[i] = 0;
		return;
	}
	for(i=0; i<s; i++)
		ba[i] = bb[i];
}

static uint64
strhash(uint32 s, string *a)
{
	USED(s);
	if(*a == nil)
		return memhash(emptystring->len, emptystring->str);
	return memhash((*a)->len, (*a)->str);
}

static uint32
strequal(uint32 s, string *a, string *b)
{
	USED(s);
	return cmpstring(*a, *b) == 0;
}

static void
strprint(uint32 s, string *a)
{
	USED(s);
	sys·printstring(*a);
}

static void
strcopy(uint32 s, string *a, string *b)
{
	USED(s);
	if(b == nil) {
		*a = nil;
		return;
	}
	*a = *b;
}

static uint64
ptrhash(uint32 s, void **a)
{
	return memhash(s, *a);
}

static uint32
ptrequal(uint32 s, void **a, void **b)
{
	USED(s, a, b);
	prints("ptrequal\n");
	return 0;
}

static void
ptrprint(uint32 s, void **a)
{
	USED(s, a);
	prints("ptrprint\n");
}

static void
ptrcopy(uint32 s, void **a, void **b)
{
	USED(s);
	if(b == nil) {
		*a = nil;
		return;
	}
	*a = *b;
}

Alg
algarray[] =
{
[ASIMP]		{ memhash, memequal, memprint, memcopy },
[ASTRING]	{ strhash, strequal, strprint, strcopy },
[APTR]		{ memhash, memequal, memprint, memcopy },	// TODO: ptr routines
[AINTER]	{ memhash, memequal, memprint, memcopy },	// TODO: interface routines
[ASTRUCT]	{ memhash, memequal, memprint, memcopy },	// TODO: what goes here?
[AARRAY]	{ memhash, memequal, memprint, memcopy },	// TODO: what goes here?
};

