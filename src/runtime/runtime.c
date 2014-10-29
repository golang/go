// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "stack.h"
#include "arch_GOARCH.h"
#include "textflag.h"
#include "malloc.h"

// Keep a cached value to make gotraceback fast,
// since we call it on every call to gentraceback.
// The cached value is a uint32 in which the low bit
// is the "crash" setting and the top 31 bits are the
// gotraceback value.
static uint32 traceback_cache = 2<<1;

// The GOTRACEBACK environment variable controls the
// behavior of a Go program that is crashing and exiting.
//	GOTRACEBACK=0   suppress all tracebacks
//	GOTRACEBACK=1   default behavior - show tracebacks but exclude runtime frames
//	GOTRACEBACK=2   show tracebacks including runtime frames
//	GOTRACEBACK=crash   show tracebacks including runtime frames, then crash (core dump etc)
#pragma textflag NOSPLIT
int32
runtime·gotraceback(bool *crash)
{
	if(crash != nil)
		*crash = false;
	if(g->m->traceback != 0)
		return g->m->traceback;
	if(crash != nil)
		*crash = traceback_cache&1;
	return traceback_cache>>1;
}

int32
runtime·mcmp(byte *s1, byte *s2, uintptr n)
{
	uintptr i;
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

static int32	argc;

#pragma dataflag NOPTR /* argv not a heap pointer */
static uint8**	argv;

extern Slice runtime·argslice;
extern Slice runtime·envs;

void (*runtime·sysargs)(int32, uint8**);

void
runtime·args(int32 c, uint8 **v)
{
	argc = c;
	argv = v;
	if(runtime·sysargs != nil)
		runtime·sysargs(c, v);
}

int32 runtime·isplan9;
int32 runtime·issolaris;
int32 runtime·iswindows;

// Information about what cpu features are available.
// Set on startup in asm_{x86/amd64}.s.
uint32 runtime·cpuid_ecx;
uint32 runtime·cpuid_edx;

void
runtime·goargs(void)
{
	String *s;
	int32 i;

	// for windows implementation see "os" package
	if(Windows)
		return;

	runtime·argslice = runtime·makeStringSlice(argc);
	s = (String*)runtime·argslice.array;
	for(i=0; i<argc; i++)
		s[i] = runtime·gostringnocopy(argv[i]);
}

void
runtime·goenvs_unix(void)
{
	String *s;
	int32 i, n;

	for(n=0; argv[argc+1+n] != 0; n++)
		;

	runtime·envs = runtime·makeStringSlice(n);
	s = (String*)runtime·envs.array;
	for(i=0; i<n; i++)
		s[i] = runtime·gostringnocopy(argv[argc+1+i]);
}

#pragma textflag NOSPLIT
Slice
runtime·environ()
{
	return runtime·envs;
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

static void
TestAtomic64(void)
{
	uint64 z64, x64;

	z64 = 42;
	x64 = 0;
	PREFETCH(&z64);
	if(runtime·cas64(&z64, x64, 1))
		runtime·throw("cas64 failed");
	if(x64 != 0)
		runtime·throw("cas64 failed");
	x64 = 42;
	if(!runtime·cas64(&z64, x64, 1))
		runtime·throw("cas64 failed");
	if(x64 != 42 || z64 != 1)
		runtime·throw("cas64 failed");
	if(runtime·atomicload64(&z64) != 1)
		runtime·throw("load64 failed");
	runtime·atomicstore64(&z64, (1ull<<40)+1);
	if(runtime·atomicload64(&z64) != (1ull<<40)+1)
		runtime·throw("store64 failed");
	if(runtime·xadd64(&z64, (1ull<<40)+1) != (2ull<<40)+2)
		runtime·throw("xadd64 failed");
	if(runtime·atomicload64(&z64) != (2ull<<40)+2)
		runtime·throw("xadd64 failed");
	if(runtime·xchg64(&z64, (3ull<<40)+3) != (2ull<<40)+2)
		runtime·throw("xchg64 failed");
	if(runtime·atomicload64(&z64) != (3ull<<40)+3)
		runtime·throw("xchg64 failed");
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
	float32 i, i1;
	float64 j, j1;
	byte *k, *k1;
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

	if(runtime·timediv(12345LL*1000000000+54321, 1000000000, &e) != 12345 || e != 54321)
		runtime·throw("bad timediv");

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

	k = (byte*)0xfedcb123;
	if(sizeof(void*) == 8)
		k = (byte*)((uintptr)k<<10);
	if(runtime·casp((void**)&k, nil, nil))
		runtime·throw("casp1");
	k1 = k+1;
	if(!runtime·casp((void**)&k, k, k1))
		runtime·throw("casp2");
	if(k != k1)
		runtime·throw("casp3");

	*(uint64*)&j = ~0ULL;
	if(j == j)
		runtime·throw("float64nan");
	if(!(j != j))
		runtime·throw("float64nan1");

	*(uint64*)&j1 = ~1ULL;
	if(j == j1)
		runtime·throw("float64nan2");
	if(!(j != j1))
		runtime·throw("float64nan3");

	*(uint32*)&i = ~0UL;
	if(i == i)
		runtime·throw("float32nan");
	if(!(i != i))
		runtime·throw("float32nan1");

	*(uint32*)&i1 = ~1UL;
	if(i == i1)
		runtime·throw("float32nan2");
	if(!(i != i1))
		runtime·throw("float32nan3");

	TestAtomic64();

	if(FixedStack != runtime·round2(FixedStack))
		runtime·throw("FixedStack is not power-of-2");
}

#pragma dataflag NOPTR
DebugVars	runtime·debug;

typedef struct DbgVar DbgVar;
struct DbgVar
{
	int8*	name;
	int32*	value;
};

// Do we report invalid pointers found during stack or heap scans?
int32 runtime·invalidptr = 1;

#pragma dataflag NOPTR /* dbgvar has no heap pointers */
static DbgVar dbgvar[] = {
	{"allocfreetrace", &runtime·debug.allocfreetrace},
	{"invalidptr", &runtime·invalidptr},
	{"efence", &runtime·debug.efence},
	{"gctrace", &runtime·debug.gctrace},
	{"gcdead", &runtime·debug.gcdead},
	{"scheddetail", &runtime·debug.scheddetail},
	{"schedtrace", &runtime·debug.schedtrace},
	{"scavenge", &runtime·debug.scavenge},
};

void
runtime·parsedebugvars(void)
{
	byte *p;
	intgo i, n;

	p = runtime·getenv("GODEBUG");
	if(p != nil){
		for(;;) {
			for(i=0; i<nelem(dbgvar); i++) {
				n = runtime·findnull((byte*)dbgvar[i].name);
				if(runtime·mcmp(p, (byte*)dbgvar[i].name, n) == 0 && p[n] == '=')
					*dbgvar[i].value = runtime·atoi(p+n+1);
			}
			p = runtime·strstr(p, (byte*)",");
			if(p == nil)
				break;
			p++;
		}
	}

	p = runtime·getenv("GOTRACEBACK");
	if(p == nil)
		p = (byte*)"";
	if(p[0] == '\0')
		traceback_cache = 1<<1;
	else if(runtime·strcmp(p, (byte*)"crash") == 0)
		traceback_cache = (2<<1) | 1;
	else
		traceback_cache = runtime·atoi(p)<<1;	
}

// Poor mans 64-bit division.
// This is a very special function, do not use it if you are not sure what you are doing.
// int64 division is lowered into _divv() call on 386, which does not fit into nosplit functions.
// Handles overflow in a time-specific manner.
#pragma textflag NOSPLIT
int32
runtime·timediv(int64 v, int32 div, int32 *rem)
{
	int32 res, bit;

	res = 0;
	for(bit = 30; bit >= 0; bit--) {
		if(v >= ((int64)div<<bit)) {
			v = v - ((int64)div<<bit);
			res += 1<<bit;
		}
	}
	if(v >= (int64)div) {
		if(rem != nil)
			*rem = 0;
		return 0x7fffffff;
	}
	if(rem != nil)
		*rem = v;
	return res;
}

// Helpers for Go. Must be NOSPLIT, must only call NOSPLIT functions, and must not block.

#pragma textflag NOSPLIT
G*
runtime·getg(void)
{
	return g;
}

#pragma textflag NOSPLIT
M*
runtime·acquirem(void)
{
	g->m->locks++;
	return g->m;
}

#pragma textflag NOSPLIT
void
runtime·releasem(M *mp)
{
	mp->locks--;
	if(mp->locks == 0 && g->preempt) {
		// restore the preemption request in case we've cleared it in newstack
		g->stackguard0 = StackPreempt;
	}
}

#pragma textflag NOSPLIT
MCache*
runtime·gomcache(void)
{
	return g->m->mcache;
}

#pragma textflag NOSPLIT
Slice
reflect·typelinks(void)
{
	extern Type *runtime·typelink[], *runtime·etypelink[];
	Slice ret;

	ret.array = (byte*)runtime·typelink;
	ret.len = runtime·etypelink - runtime·typelink;
	ret.cap = ret.len;
	return ret;
}
