// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

int32	debug	= 0;

void
sys·panicl(int32 lno)
{
	uint8 *sp;

	prints("\npanic on line ");
	sys·printint(lno);
	prints(" ");
	sys·printpc(&lno);
	prints("\n");
	sp = (uint8*)&lno;
	traceback(sys·getcallerpc(&lno), sp, g);
	tracebackothers(g);
	sys·breakpoint();
	sys·exit(2);
}

enum
{
	NHUNK		= 20<<20,

	PROT_NONE	= 0x00,
	PROT_READ	= 0x01,
	PROT_WRITE	= 0x02,
	PROT_EXEC	= 0x04,

	MAP_FILE	= 0x0000,
	MAP_SHARED	= 0x0001,
	MAP_PRIVATE	= 0x0002,
	MAP_FIXED	= 0x0010,
	MAP_ANON	= 0x1000,
};

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

uint32
rnd(uint32 n, uint32 m)
{
	uint32 r;

	r = n % m;
	if(r)
		n += m-r;
	return n;
}

// Convenient wrapper around mmap.
static void*
brk(uint32 n)
{
	byte *v;

	v = sys·mmap(nil, n, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, 0, 0);
	m->mem.nmmap += n;
	return v;
}

// Allocate n bytes of memory.  Note that this gets used
// to allocate new stack segments, so at each call to a function
// you have to ask yourself "would it be okay to call mal recursively
// right here?"  The answer is yes unless we're in the middle of
// editing the malloc state in m->mem.
void*
mal(uint32 n)
{
	byte* v;

	// round to keep everything 64-bit aligned
	n = rnd(n, 8);

	// be careful.  calling any function might invoke
	// mal to allocate more stack.
	if(n > NHUNK) {
		v = brk(n);
	} else {
		// allocate a new hunk if this one is too small
		if(n > m->mem.nhunk) {
			// here we're in the middle of editing m->mem
			// (we're about to overwrite m->mem.hunk),
			// so we can't call brk - it might call mal to grow the
			// stack, and the recursive call would allocate a new
			// hunk, and then once brk returned we'd immediately
			// overwrite that hunk with our own.
			// (the net result would be a memory leak, not a crash.)
			// so we have to call sys·mmap directly - it is written
			// in assembly and tagged not to grow the stack.
			m->mem.hunk =
				sys·mmap(nil, NHUNK, PROT_READ|PROT_WRITE,
					MAP_ANON|MAP_PRIVATE, 0, 0);
			m->mem.nhunk = NHUNK;
			m->mem.nmmap += NHUNK;
		}
		v = m->mem.hunk;
		m->mem.hunk += n;
		m->mem.nhunk -= n;
	}
	m->mem.nmal += n;
	return v;
}

void
sys·mal(uint32 n, uint8 *ret)
{
	ret = mal(n);
	FLUSH(&ret);
}

static	Map*	hash[1009];

static Map*
hashmap(Sigi *si, Sigs *ss)
{
	int32 ns, ni;
	uint32 ihash, h;
	byte *sname, *iname;
	Map *m;

	h = ((uint32)(uint64)si + (uint32)(uint64)ss) % nelem(hash);
	for(m=hash[h]; m!=nil; m=m->link) {
		if(m->si == si && m->ss == ss) {
			if(m->bad) {
				throw("bad hashmap");
				m = nil;
			}
			// prints("old hashmap\n");
			return m;
		}
	}

	ni = si[0].offset;	// first word has size
	m = mal(sizeof(*m) + ni*sizeof(m->fun[0]));
	m->si = si;
	m->ss = ss;

	ni = 1;			// skip first word
	ns = 0;

loop1:
	// pick up next name from
	// interface signature
	iname = si[ni].name;
	if(iname == nil) {
		m->link = hash[h];
		hash[h] = m;
		// prints("new hashmap\n");
		return m;
	}
	ihash = si[ni].hash;

loop2:
	// pick up and comapre next name
	// from structure signature
	sname = ss[ns].name;
	if(sname == nil) {
		prints((int8*)iname);
		prints(": ");
		throw("hashmap: failed to find method");
		m->bad = 1;
		m->link = hash[h];
		hash[h] = m;
		return nil;
	}
	if(ihash != ss[ns].hash ||
	   strcmp(sname, iname) != 0) {
		ns++;
		goto loop2;
	}

	m->fun[si[ni].offset] = ss[ns].fun;
	ni++;
	goto loop1;
}

void
sys·ifaces2i(Sigi *si, Sigs *ss, Map *m, void *s)
{

	if(debug) {
		prints("s2i sigi=");
		sys·printpointer(si);
		prints(" sigs=");
		sys·printpointer(ss);
		prints(" s=");
		sys·printpointer(s);
	}

	if(s == nil) {
		throw("ifaces2i: nil pointer");
		m = nil;
		FLUSH(&m);
		return;
	}

	m = hashmap(si, ss);

	if(debug) {
		prints(" returning m=");
		sys·printpointer(m);
		prints(" s=");
		sys·printpointer(s);
		prints("\n");
		dump((byte*)m, 64);
	}

	FLUSH(&m);
}

void
sys·ifacei2i(Sigi *si, Map *m, void *s)
{

	if(debug) {
		prints("i2i sigi=");
		sys·printpointer(si);
		prints(" m=");
		sys·printpointer(m);
		prints(" s=");
		sys·printpointer(s);
	}

	if(m == nil) {
		throw("ifacei2i: nil map");
		s = nil;
		FLUSH(&s);
		return;
	}

	if(m->si == nil) {
		throw("ifacei2i: nil pointer");
		return;
	}

	if(m->si != si) {
		m = hashmap(si, m->ss);
		FLUSH(&m);
	}

	if(debug) {
		prints(" returning m=");
		sys·printpointer(m);
		prints(" s=");
		sys·printpointer(s);
		prints("\n");
		dump((byte*)m, 64);
	}
}

void
sys·ifacei2s(Sigs *ss, Map *m, void *s)
{

	if(debug) {
		prints("i2s m=");
		sys·printpointer(m);
		prints(" s=");
		sys·printpointer(s);
		prints("\n");
	}

	if(m == nil) {
		throw("ifacei2s: nil map");
		s = nil;
		FLUSH(&s);
		return;
	}

	if(m->ss != ss) {
		dump((byte*)m, 64);
		throw("ifacei2s: wrong pointer");
		s = nil;
		FLUSH(&s);
		return;
	}
}

enum
{
	NANEXP		= 2047<<20,
	NANMASK		= 2047<<20,
	NANSIGN		= 1<<31,
};

static	uint64	uvnan		= 0x7FF0000000000001ULL;
static	uint64	uvinf		= 0x7FF0000000000000ULL;
static	uint64	uvneginf	= 0xFFF0000000000000ULL;

static int32
isInf(float64 d, int32 sign)
{
	uint64 x;

	x = *(uint64*)&d;
	if(sign == 0) {
		if(x == uvinf || x == uvneginf)
			return 1;
		return 0;
	}
	if(sign > 0) {
		if(x == uvinf)
			return 1;
		return 0;
	}
	if(x == uvneginf)
		return 1;
	return 0;
}

static float64
NaN(void)
{
	return *(float64*)&uvnan;
}

static int32
isNaN(float64 d)
{
	uint64 x;

	x = *(uint64*)&d;
	return (uint32)(x>>32)==0x7FF00000 && !isInf(d, 0);
}

static float64
Inf(int32 sign)
{
	if(sign >= 0)
		return *(float64*)&uvinf;
	else
		return *(float64*)&uvneginf;
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
	x = *(uint64*)&d;
	*ep = (int32)((x >> SHIFT) & MASK) - BIAS;
	x &= ~((uint64)MASK << SHIFT);
	x |= (uint64)BIAS << SHIFT;
	return *(float64*)&x;
}

static float64
ldexp(float64 d, int32 e)
{
	uint64 x;

	if(d == 0)
		return 0;
	x = *(uint64*)&d;
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
	return *(float64*)&x;
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

	x = *(uint64*)&d;
	e = (int32)((x >> SHIFT) & MASK) - BIAS;

	/*
	 * Keep the top 11+e bits; clear the rest.
	 */
	if(e <= 64-11)
		x &= ~(((uint64)1 << (64LL-11LL-e))-1);
	dd = *(float64*)&x;
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
	len = findnull(s);
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
	uint8* str;
	int32 l;

	if(i < 0 || i >= argc) {
		s = emptystring;
		goto out;
	}

	str = argv[i];
	l = findnull((int8*)str);
	s = mal(sizeof(s->len)+l);
	s->len = l;
	mcpy(s->str, str, l);

out:
	FLUSH(&s);
}

//func envv(i) string;  // return environment variable i
void
sys·envv(int32 i, string s)
{
	uint8* str;
	int32 l;

	if(i < 0 || i >= envc) {
		s = emptystring;
		goto out;
	}

	str = envv[i];
	l = findnull((int8*)str);
	s = mal(sizeof(s->len)+l);
	s->len = l;
	mcpy(s->str, str, l);

out:
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
	USED(s, a);
	prints("memhash\n");
	return 0x12345;
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
stringhash(uint32 s, string *a)
{
	USED(s, a);
	prints("stringhash\n");
	return 0x12345;
}

static uint32
stringequal(uint32 s, string *a, string *b)
{
	USED(s);
	return cmpstring(*a, *b) == 0;
}

static void
stringprint(uint32 s, string *a)
{
	USED(s);
	sys·printstring(*a);
}

static void
stringcopy(uint32 s, string *a, string *b)
{
	USED(s);
	if(b == nil) {
		*a = nil;
		return;
	}
	*a = *b;
}

static uint64
pointerhash(uint32 s, void **a)
{
	USED(s, a);
	prints("pointerhash\n");
	return 0x12345;
}

static uint32
pointerequal(uint32 s, void **a, void **b)
{
	USED(s, a, b);
	prints("pointerequal\n");
	return 0;
}

static void
pointerprint(uint32 s, void **a)
{
	USED(s, a);
	prints("pointerprint\n");
}

static void
pointercopy(uint32 s, void **a, void **b)
{
	USED(s);
	if(b == nil) {
		*a = nil;
		return;
	}
	*a = *b;
}

Alg
algarray[3] =
{
	{	memhash,	memequal,	memprint,	memcopy	},  // 0
	{	stringhash,	stringequal,	stringprint,	stringcopy	},  // 1
//	{	pointerhash,	pointerequal,	pointerprint,	pointercopy	},  // 2
	{	memhash,	memequal,	memprint,	memcopy	},  // 2 - treat pointers as ints
};
