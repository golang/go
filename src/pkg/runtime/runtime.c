// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "stack.h"

enum {
	maxround = sizeof(uintptr),
};

uint32	runtime·panicking;

int32
runtime·gotraceback(void)
{
	byte *p;

	p = runtime·getenv("GOTRACEBACK");
	if(p == nil || p[0] == '\0')
		return 1;	// default is on
	return runtime·atoi(p);
}

static Lock paniclk;

void
runtime·startpanic(void)
{
	if(m->dying) {
		runtime·printf("panic during panic\n");
		runtime·exit(3);
	}
	m->dying = 1;
	runtime·xadd(&runtime·panicking, 1);
	runtime·lock(&paniclk);
}

void
runtime·dopanic(int32 unused)
{
	static bool didothers;

	if(g->sig != 0)
		runtime·printf("\n[signal %x code=%p addr=%p pc=%p]\n",
			g->sig, g->sigcode0, g->sigcode1, g->sigpc);

	runtime·printf("\n");
	if(runtime·gotraceback()){
		runtime·traceback(runtime·getcallerpc(&unused), runtime·getcallersp(&unused), 0, g);
		if(!didothers) {
			didothers = true;
			runtime·tracebackothers(g);
		}
	}
	runtime·unlock(&paniclk);
	if(runtime·xadd(&runtime·panicking, -1) != 0) {
		// Some other m is panicking too.
		// Let it print what it needs to print.
		// Wait forever without chewing up cpu.
		// It will exit when it's done.
		static Lock deadlock;
		runtime·lock(&deadlock);
		runtime·lock(&deadlock);
	}

	runtime·exit(2);
}

void
runtime·panicindex(void)
{
	runtime·panicstring("index out of range");
}

void
runtime·panicslice(void)
{
	runtime·panicstring("slice bounds out of range");
}

void
runtime·throwreturn(void)
{
	// can only happen if compiler is broken
	runtime·throw("no return at end of a typed function - compiler is broken");
}

void
runtime·throwinit(void)
{
	// can only happen with linker skew
	runtime·throw("recursive call during initialization - linker skew");
}

void
runtime·throw(int8 *s)
{
	runtime·startpanic();
	runtime·printf("throw: %s\n", s);
	runtime·dopanic(0);
	*(int32*)0 = 0;	// not reached
	runtime·exit(1);	// even more not reached
}

void
runtime·panicstring(int8 *s)
{
	Eface err;
	
	if(m->gcing) {
		runtime·printf("panic: %s\n", s);
		runtime·throw("panic during gc");
	}
	runtime·newErrorString(runtime·gostringnocopy((byte*)s), &err);
	runtime·panic(err);
}

int32
runtime·mcmp(byte *s1, byte *s2, uint32 n)
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


byte*
runtime·mchr(byte *p, byte c, byte *ep)
{
	for(; p < ep; p++)
		if(*p == c)
			return p;
	return nil;
}

uint32
runtime·rnd(uint32 n, uint32 m)
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
runtime·args(int32 c, uint8 **v)
{
	argc = c;
	argv = v;
}

int32 runtime·isplan9;
int32 runtime·iswindows;

void
runtime·goargs(void)
{
	String *s;
	int32 i;
	
	// for windows implementation see "os" package
	if(Windows)
		return;

	s = runtime·malloc(argc*sizeof s[0]);
	for(i=0; i<argc; i++)
		s[i] = runtime·gostringnocopy(argv[i]);
	os·Args.array = (byte*)s;
	os·Args.len = argc;
	os·Args.cap = argc;
}

void
runtime·goenvs_unix(void)
{
	String *s;
	int32 i, n;
	
	for(n=0; argv[argc+1+n] != 0; n++)
		;

	s = runtime·malloc(n*sizeof s[0]);
	for(i=0; i<n; i++)
		s[i] = runtime·gostringnocopy(argv[argc+1+i]);
	os·Envs.array = (byte*)s;
	os·Envs.len = n;
	os·Envs.cap = n;
}

// Atomic add and return new value.
uint32
runtime·xadd(uint32 volatile *val, int32 delta)
{
	uint32 oval, nval;

	for(;;){
		oval = *val;
		nval = oval + delta;
		if(runtime·cas(val, oval, nval))
			return nval;
	}
}

byte*
runtime·getenv(int8 *s)
{
	int32 i, j, len;
	byte *v, *bs;
	String* envv;
	int32 envc;

	bs = (byte*)s;
	len = runtime·findnull(bs);
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

void
runtime·getgoroot(String out)
{
	byte *p;

	p = runtime·getenv("GOROOT");
	out = runtime·gostringnocopy(p);
	FLUSH(&out);
}

int32
runtime·atoi(byte *p)
{
	int32 n;

	n = 0;
	while('0' <= *p && *p <= '9')
		n = n*10 + *p++ - '0';
	return n;
}

void
runtime·check(void)
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
	struct x1 {
		byte x;
	};
	struct y1 {
		struct x1 x1;
		byte y;
	};

	if(sizeof(a) != 1) runtime·throw("bad a");
	if(sizeof(b) != 1) runtime·throw("bad b");
	if(sizeof(c) != 2) runtime·throw("bad c");
	if(sizeof(d) != 2) runtime·throw("bad d");
	if(sizeof(e) != 4) runtime·throw("bad e");
	if(sizeof(f) != 4) runtime·throw("bad f");
	if(sizeof(g) != 8) runtime·throw("bad g");
	if(sizeof(h) != 8) runtime·throw("bad h");
	if(sizeof(i) != 4) runtime·throw("bad i");
	if(sizeof(j) != 8) runtime·throw("bad j");
	if(sizeof(k) != sizeof(uintptr)) runtime·throw("bad k");
	if(sizeof(l) != sizeof(uintptr)) runtime·throw("bad l");
	if(sizeof(struct x1) != 1) runtime·throw("bad sizeof x1");
	if(offsetof(struct y1, y) != 1) runtime·throw("bad offsetof y1.y");
	if(sizeof(struct y1) != 2) runtime·throw("bad sizeof y1");

	uint32 z;
	z = 1;
	if(!runtime·cas(&z, 1, 2))
		runtime·throw("cas1");
	if(z != 2)
		runtime·throw("cas2");

	z = 4;
	if(runtime·cas(&z, 5, 6))
		runtime·throw("cas3");
	if(z != 4)
		runtime·throw("cas4");

	runtime·initsig(0);
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
	byte *ba, *bb, *aend;

	if(a == b)
	  return 1;
	ba = a;
	bb = b;
	aend = ba+s;
	while(ba != aend) {
		if(*ba != *bb)
			return 0;
		ba++;
		bb++;
	}
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
	runtime·printint(v);
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

static uint32
memwordequal(uint32 s, void *a, void *b)
{
	USED(s);
	return *(uintptr*)(a) == *(uintptr*)(b);
}

static void
memwordcopy(uint32 s, void *a, void *b)
{
	USED(s);
	if (b == nil) {
		*(uintptr*)(a) = 0;
		return;
	}
	*(uintptr*)(a) = *(uintptr*)(b);
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
	int32 alen;

	USED(s);
	alen = a->len;
	if(alen != b->len)
		return false;
	return memequal(alen, a->str, b->str);
}

static void
strprint(uint32 s, String *a)
{
	USED(s);
	runtime·printstring(*a);
}

static uintptr
interhash(uint32 s, Iface *a)
{
	USED(s);
	return runtime·ifacehash(*a);
}

static void
interprint(uint32 s, Iface *a)
{
	USED(s);
	runtime·printiface(*a);
}

static uint32
interequal(uint32 s, Iface *a, Iface *b)
{
	USED(s);
	return runtime·ifaceeq_c(*a, *b);
}

static uintptr
nilinterhash(uint32 s, Eface *a)
{
	USED(s);
	return runtime·efacehash(*a);
}

static void
nilinterprint(uint32 s, Eface *a)
{
	USED(s);
	runtime·printeface(*a);
}

static uint32
nilinterequal(uint32 s, Eface *a, Eface *b)
{
	USED(s);
	return runtime·efaceeq_c(*a, *b);
}

uintptr
runtime·nohash(uint32 s, void *a)
{
	USED(s);
	USED(a);
	runtime·panicstring("hash of unhashable type");
	return 0;
}

uint32
runtime·noequal(uint32 s, void *a, void *b)
{
	USED(s);
	USED(a);
	USED(b);
	runtime·panicstring("comparing uncomparable types");
	return 0;
}

Alg
runtime·algarray[] =
{
[AMEM]	{ memhash, memequal, memprint, memcopy },
[ANOEQ]	{ runtime·nohash, runtime·noequal, memprint, memcopy },
[ASTRING]	{ strhash, strequal, strprint, memcopy },
[AINTER]		{ interhash, interequal, interprint, memcopy },
[ANILINTER]	{ nilinterhash, nilinterequal, nilinterprint, memcopy },
[AMEMWORD] { memhash, memwordequal, memprint, memwordcopy },
};

int64
runtime·nanotime(void)
{
	int64 sec;
	int32 usec;

	sec = 0;
	usec = 0;
	runtime·gettime(&sec, &usec);
	return sec*1000000000 + (int64)usec*1000;
}

void
runtime·Caller(int32 skip, uintptr retpc, String retfile, int32 retline, bool retbool)
{
	Func *f;
	uintptr pc;

	if(runtime·callers(1+skip, &retpc, 1) == 0) {
		retfile = runtime·emptystring;
		retline = 0;
		retbool = false;
	} else if((f = runtime·findfunc(retpc)) == nil) {
		retfile = runtime·emptystring;
		retline = 0;
		retbool = true;  // have retpc at least
	} else {
		retfile = f->src;
		pc = retpc;
		if(pc > f->entry)
			pc--;
		retline = runtime·funcline(f, pc);
		retbool = true;
	}
	FLUSH(&retfile);
	FLUSH(&retline);
	FLUSH(&retbool);
}

void
runtime·Callers(int32 skip, Slice pc, int32 retn)
{
	retn = runtime·callers(skip, (uintptr*)pc.array, pc.len);
	FLUSH(&retn);
}

void
runtime·FuncForPC(uintptr pc, void *retf)
{
	retf = runtime·findfunc(pc);
	FLUSH(&retf);
}

uint32
runtime·fastrand1(void)
{
	uint32 x;

	x = m->fastrand;
	x += x;
	if(x & 0x80000000L)
		x ^= 0x88888eefUL;
	m->fastrand = x;
	return x;
}
