// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

G	g0;			// idle goroutine
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
	sys·breakpoint();
	sys·exit(2);
}

static	uint8*	hunk;
static	uint32	nhunk;
static	uint64	nmmap;
static	uint64	nmal;
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

static byte*
brk(uint32 n)
{
	byte* v;

	v = sys·mmap(nil, n, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, 0, 0);
	sys·memclr(v, n);
	nmmap += n;
	return v;
}

void*
mal(uint32 n)
{
	byte* v;

	// round to keep everything 64-bit aligned
	n = (n+7) & ~7;
	nmal += n;

	// do we have enough in contiguous hunk
	if(n > nhunk) {

		// if it is big allocate it separately
		if(n > NHUNK)
			return brk(n);

		// allocate a new contiguous hunk
		hunk = brk(NHUNK);
		nhunk = NHUNK;
	}

	// allocate from the contiguous hunk
	v = hunk;
	hunk += n;
	nhunk -= n;
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

	h = ((uint32)si + (uint32)ss) % nelem(hash);
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

static	uint64	uvnan		= 0x7FF0000000000001;
static	uint64	uvinf		= 0x7FF0000000000000;
static	uint64	uvneginf	= 0xFFF0000000000000;

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
	return ((uint32)x>>32)==0x7FF00000 && !isInf(d, 0);
}

static float64
Inf(int32 sign)
{
	if(sign < 0)
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

// func frexp(float64) (float64, int32); // break fp into exp,fract
void
sys·frexp(float64 din, float64 dou, int32 iou)
{
	dou = frexp(din, &iou);
	FLUSH(&dou);
}

//func	ldexp(int32, float64) float64;	// make fp from exp,fract
void
sys·ldexp(float64 din, int32 ein, float64 dou)
{
	dou = ldexp(din, ein);
	FLUSH(&dou);
}

//func	modf(float64) (float64, float64);	// break fp into double+double
float64
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
	initsig();
}

void
sys·goexit(void)
{
//prints("goexit goid=");
//sys·printint(g->goid);
//prints("\n");
	g->status = Gdead;
	sys·gosched();
}

void
sys·newproc(int32 siz, byte* fn, byte* arg0)
{
	byte *stk, *sp;
	G *newg;

//prints("newproc siz=");
//sys·printint(siz);
//prints(" fn=");
//sys·printpointer(fn);

	siz = (siz+7) & ~7;
	if(siz > 1024) {
		prints("sys·newproc: too many args: ");
		sys·printint(siz);
		prints("\n");
		sys·panicl(123);
	}

	newg = mal(sizeof(G));
	stk = mal(4096);
	newg->stackguard = stk+160;

	sp = stk + 4096 - 4*8;
	newg->stackbase = sp;

	sp -= siz;
	mcpy(sp, (byte*)&arg0, siz);

	sp -= 8;
	*(byte**)sp = (byte*)sys·goexit;

	sp -= 8;	// retpc used by gogo
	newg->sched.SP = sp;
	newg->sched.PC = fn;

	goidgen++;
	newg->goid = goidgen;

	newg->status = Grunnable;
	newg->link = allg;
	allg = newg;

//prints(" goid=");
//sys·printint(newg->goid);
//prints("\n");
}

G*
select(void)
{
	G *gp, *bestg;

	bestg = nil;
	for(gp=allg; gp!=nil; gp=gp->link) {
		if(gp->status != Grunnable)
			continue;
		if(bestg == nil || gp->pri < bestg->pri)
			bestg = gp;
	}
	if(bestg != nil)
		bestg->pri++;
	return bestg;
}

void
gom0init(void)
{
	gosave(&m->sched);
	sys·gosched();
}

void
sys·gosched(void)
{
	G* gp;

	if(g != m->g0) {
		if(gosave(&g->sched))
			return;
		g = m->g0;
		gogo(&m->sched);
	}
	gp = select();
	if(gp == nil) {
//		prints("sched: no more work\n");
		sys·exit(0);
	}

	m->curg = gp;
	g = gp;
	gogo(&gp->sched);
}

//
// the calling sequence for a routine that
// needs N bytes stack, A args.
//
//	N1 = (N+160 > 4096)? N+160: 0
//	A1 = A
//
// if N <= 75
//	CMPQ	SP, 0(R15)
//	JHI	4(PC)
//	MOVQ	$(N1<<0) | (A1<<32)), AX
//	MOVQ	AX, 0(R14)
//	CALL	sys·morestack(SB)
//
// if N > 75
//	LEAQ	(-N-75)(SP), AX
//	CMPQ	AX, 0(R15)
//	JHI	4(PC)
//	MOVQ	$(N1<<0) | (A1<<32)), AX
//	MOVQ	AX, 0(R14)
//	CALL	sys·morestack(SB)
//

int32 debug = 0;

void
morestack2(void)
{
	Stktop *top;
	uint32 siz2;
	byte *sp;
if(debug) prints("morestack2\n");

	top = (Stktop*)m->curg->stackbase;

	m->curg->stackbase = top->oldbase;
	m->curg->stackguard = top->oldguard;
	siz2 = (top->magic>>32) & 0xffffLL;

	sp = (byte*)top;
	if(siz2 > 0) {
		siz2 = (siz2+7) & ~7;
		sp -= siz2;
		mcpy(top->oldsp+16, sp, siz2);
	}

	m->morestack.SP = top->oldsp+8;
	m->morestack.PC = (byte*)(*(uint64*)(top->oldsp+8));
if(debug) prints("morestack2 sp=");
if(debug) sys·printpointer(m->morestack.SP);
if(debug) prints(" pc=");
if(debug) sys·printpointer(m->morestack.PC);
if(debug) prints("\n");
	gogo(&m->morestack);
}

void
morestack1(void)
{
	int32 siz1, siz2;
	Stktop *top;
	byte *stk, *sp;
	void (*fn)(void);

	siz1 = m->morearg & 0xffffffffLL;
	siz2 = (m->morearg>>32) & 0xffffLL;

if(debug) prints("morestack1 siz1=");
if(debug) sys·printint(siz1);
if(debug) prints(" siz2=");
if(debug) sys·printint(siz2);
if(debug) prints(" moresp=");
if(debug) sys·printpointer(m->moresp);
if(debug) prints("\n");

	if(siz1 < 4096)
		siz1 = 4096;
	stk = mal(siz1 + 1024);
	stk += 512;

	top = (Stktop*)(stk+siz1-sizeof(*top));

	top->oldbase = m->curg->stackbase;
	top->oldguard = m->curg->stackguard;
	top->oldsp = m->moresp;
	top->magic = m->morearg;

	m->curg->stackbase = (byte*)top;
	m->curg->stackguard = stk + 160;

	sp = (byte*)top;
	
	if(siz2 > 0) {
		siz2 = (siz2+7) & ~7;
		sp -= siz2;
		mcpy(sp, m->moresp+16, siz2);
	}

	g = m->curg;
	fn = (void(*)(void))(*(uint64*)m->moresp);
if(debug) prints("fn=");
if(debug) sys·printpointer(fn);
if(debug) prints("\n");
	setspgoto(sp, fn, morestack2);

	*(int32*)345 = 123;
}

void
sys·morestack(uint64 u)
{
	while(g == m->g0) {
		// very bad news
		*(int32*)123 = 123;
	}

	g = m->g0;
	m->moresp = (byte*)(&u-1);
	setspgoto(m->sched.SP, morestack1, nil);

	*(int32*)234 = 123;
}
