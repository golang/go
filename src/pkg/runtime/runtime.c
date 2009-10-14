// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

int32	panicking	= 0;
int32	maxround	= sizeof(uintptr);
int32	fd		= 1;

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

	fd = 2;
	if(panicking) {
		printf("double panic\n");
		exit(3);
	}
	panicking++;

	printf("\npanic PC=%X\n", (uint64)(uintptr)&lno);
	sp = (uint8*)&lno;
	if(gotraceback()){
		traceback(sys·getcallerpc(&lno), sp, g);
		tracebackothers(g);
	}
	breakpoint();  // so we can grab it in a debugger
	exit(2);
}

void
sys·throwindex(void)
{
	throw("index out of range");
}

void
sys·throwslice(void)
{
	throw("slice out of range");
}

void
sys·throwreturn(void)
{
	throw("no return at end of a typed function");
}

void
sys·throwinit(void)
{
	throw("recursive call during initialization");
}

void
throw(int8 *s)
{
	fd = 2;
	printf("throw: %s\n", s);
	sys·panicl(-1);
	*(int32*)0 = 0;	// not reached
	exit(1);	// even more not reached
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

int32
mcmp(byte *s1, byte *s2, uint32 n)
{
	uint32 i;
	byte c1, c2;

	for(i=0; i<n; i++) {
		c1 = s1[i];
		c2 = s2[i];
		if(c1 < c2)
			return -1;
		if(c1 > c2)
			return +1;
	}
	return 0;
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

static int32	argc;
static uint8**	argv;

Slice os·Args;
Slice os·Envs;

void
args(int32 c, uint8 **v)
{
	argc = c;
	argv = v;
}

void
goargs(void)
{
	String *gargv;
	String *genvv;
	int32 i, envc;

	for(envc=0; argv[argc+1+envc] != 0; envc++)
		;

	gargv = malloc(argc*sizeof gargv[0]);
	genvv = malloc(envc*sizeof genvv[0]);

	for(i=0; i<argc; i++)
		gargv[i] = gostring(argv[i]);
	os·Args.array = (byte*)gargv;
	os·Args.len = argc;
	os·Args.cap = argc;

	for(i=0; i<envc; i++)
		genvv[i] = gostring(argv[argc+1+i]);
	os·Envs.array = (byte*)genvv;
	os·Envs.len = envc;
	os·Envs.cap = envc;
}

byte*
getenv(int8 *s)
{
	int32 i, j, len;
	byte *v, *bs;
	String* envv;
	int32 envc;

	bs = (byte*)s;
	len = findnull(bs);
	envv = (String*)os·Envs.array;
	envc = os·Envs.len;
	for(i=0; i<envc; i++){
		if(envv[i].len <= len)
			continue;
		v = envv[i].str;
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
	if(sizeof(k) != sizeof(uintptr)) throw("bad k");
	if(sizeof(l) != sizeof(uintptr)) throw("bad l");
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
static uintptr
memhash(uint32 s, void *a)
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

static uintptr
strhash(uint32 s, String *a)
{
	USED(s);
	return memhash((*a).len, (*a).str);
}

static uint32
strequal(uint32 s, String *a, String *b)
{
	USED(s);
	return cmpstring(*a, *b) == 0;
}

static void
strprint(uint32 s, String *a)
{
	USED(s);
	sys·printstring(*a);
}

static uintptr
interhash(uint32 s, Iface *a)
{
	USED(s);
	return ifacehash(*a);
}

static void
interprint(uint32 s, Iface *a)
{
	USED(s);
	sys·printiface(*a);
}

static uint32
interequal(uint32 s, Iface *a, Iface *b)
{
	USED(s);
	return ifaceeq(*a, *b);
}

static uintptr
nilinterhash(uint32 s, Eface *a)
{
	USED(s);
	return efacehash(*a);
}

static void
nilinterprint(uint32 s, Eface *a)
{
	USED(s);
	sys·printeface(*a);
}

static uint32
nilinterequal(uint32 s, Eface *a, Eface *b)
{
	USED(s);
	return efaceeq(*a, *b);
}

uintptr
nohash(uint32 s, void *a)
{
	USED(s);
	USED(a);
	throw("hash of unhashable type");
	return 0;
}

uint32
noequal(uint32 s, void *a, void *b)
{
	USED(s);
	USED(a);
	USED(b);
	throw("comparing uncomparable types");
	return 0;
}

static void
noprint(uint32 s, void *a)
{
	USED(s);
	USED(a);
	throw("print of unprintable type");
}

static void
nocopy(uint32 s, void *a, void *b)
{
	USED(s);
	USED(a);
	USED(b);
	throw("copy of uncopyable type");
}

Alg
algarray[] =
{
[AMEM]	{ memhash, memequal, memprint, memcopy },
[ANOEQ]	{ nohash, noequal, memprint, memcopy },
[ASTRING]	{ strhash, strequal, strprint, memcopy },
[AINTER]		{ interhash, interequal, interprint, memcopy },
[ANILINTER]	{ nilinterhash, nilinterequal, nilinterprint, memcopy },
[AFAKE]	{ nohash, noequal, noprint, nocopy },
};

#pragma textflag 7
void
FLUSH(void *v)
{
	USED(v);
}

