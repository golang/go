// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"

enum {
	maxround = sizeof(uintptr),
};

/*
 * We assume that all architectures turn faults and the like
 * into apparent calls to runtime.sigpanic.  If we see a "call"
 * to runtime.sigpanic, we do not back up the PC to find the
 * line number of the CALL instruction, because there is no CALL.
 */
void	runtime·sigpanic(void);

// The GOTRACEBACK environment variable controls the
// behavior of a Go program that is crashing and exiting.
//	GOTRACEBACK=0   suppress all tracebacks
//	GOTRACEBACK=1   default behavior - show tracebacks but exclude runtime frames
//	GOTRACEBACK=2   show tracebacks including runtime frames
//	GOTRACEBACK=crash   show tracebacks including runtime frames, then crash (core dump etc)
int32
runtime·gotraceback(bool *crash)
{
	byte *p;

	if(crash != nil)
		*crash = false;
	p = runtime·getenv("GOTRACEBACK");
	if(p == nil || p[0] == '\0')
		return 1;	// default is on
	if(runtime·strcmp(p, (byte*)"crash") == 0) {
		if(crash != nil)
			*crash = true;
		return 2;	// extra information
	}
	return runtime·atoi(p);
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
static uint8**	argv;

Slice os·Args;
Slice syscall·envs;

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
	syscall·envs.array = (byte*)s;
	syscall·envs.len = n;
	syscall·envs.cap = n;
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

static void
TestAtomic64(void)
{
	uint64 z64, x64;

	z64 = 42;
	x64 = 0;
	PREFETCH(&z64);
	if(runtime·cas64(&z64, &x64, 1))
		runtime·throw("cas64 failed");
	if(x64 != 42)
		runtime·throw("cas64 failed");
	if(!runtime·cas64(&z64, &x64, 1))
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
}

void
runtime·Caller(intgo skip, uintptr retpc, String retfile, intgo retline, bool retbool)
{
	Func *f, *g;
	uintptr pc;
	uintptr rpc[2];

	/*
	 * Ask for two PCs: the one we were asked for
	 * and what it called, so that we can see if it
	 * "called" sigpanic.
	 */
	retpc = 0;
	if(runtime·callers(1+skip-1, rpc, 2) < 2) {
		retfile = runtime·emptystring;
		retline = 0;
		retbool = false;
	} else if((f = runtime·findfunc(rpc[1])) == nil) {
		retfile = runtime·emptystring;
		retline = 0;
		retbool = true;  // have retpc at least
	} else {
		retpc = rpc[1];
		retfile = f->src;
		pc = retpc;
		g = runtime·findfunc(rpc[0]);
		if(pc > f->entry && (g == nil || g->entry != (uintptr)runtime·sigpanic))
			pc--;
		retline = runtime·funcline(f, pc);
		retbool = true;
	}
	FLUSH(&retpc);
	FLUSH(&retfile);
	FLUSH(&retline);
	FLUSH(&retbool);
}

void
runtime·Callers(intgo skip, Slice pc, intgo retn)
{
	// runtime.callers uses pc.array==nil as a signal
	// to print a stack trace.  Pick off 0-length pc here
	// so that we don't let a nil pc slice get to it.
	if(pc.len == 0)
		retn = 0;
	else
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

static Lock ticksLock;
static int64 ticks;

int64
runtime·tickspersecond(void)
{
	int64 res, t0, t1, c0, c1;

	res = (int64)runtime·atomicload64((uint64*)&ticks);
	if(res != 0)
		return ticks;
	runtime·lock(&ticksLock);
	res = ticks;
	if(res == 0) {
		t0 = runtime·nanotime();
		c0 = runtime·cputicks();
		runtime·usleep(100*1000);
		t1 = runtime·nanotime();
		c1 = runtime·cputicks();
		if(t1 == t0)
			t1++;
		res = (c1-c0)*1000*1000*1000/(t1-t0);
		if(res == 0)
			res++;
		runtime·atomicstore64((uint64*)&ticks, res);
	}
	runtime·unlock(&ticksLock);
	return res;
}

void
runtime∕pprof·runtime_cyclesPerSecond(int64 res)
{
	res = runtime·tickspersecond();
	FLUSH(&res);
}
