// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

int32	debug	= 0;

void
sys_printbool(bool v)
{
	if(v) {
		sys_write(1, (byte*)"true", 4);
		return;
	}
	sys_write(1, (byte*)"false", 5);
}

void
sys_printfloat(float64 v)
{
	sys_write(1, "printfloat", 10);
}

void
sys_printint(int64 v)
{
	byte buf[100];
	int32 i, s;

	s = 0;
	if(v < 0) {
		v = -v;
		s = 1;
		if(v < 0) {
			sys_write(1, (byte*)"-oo", 3);
			return;
		}
	}

	for(i=nelem(buf)-1; i>0; i--) {
		buf[i] = v%10 + '0';
		if(v < 10)
			break;
		v = v/10;
	}
	if(s) {
		i--;
		buf[i] = '-';
	}
	sys_write(1, buf+i, nelem(buf)-i);
}

void
sys_printpointer(void *p)
{
	uint64 v;
	byte buf[100];
	int32 i;

	v = (int64)p;
	for(i=nelem(buf)-1; i>0; i--) {
		buf[i] = v%16 + '0';
		if(buf[i] > '9')
			buf[i] += 'a'-'0'-10;
		if(v < 16)
			break;
		v = v/16;
	}
	sys_write(1, buf+i, nelem(buf)-i);
}

void
sys_printstring(string v)
{
	sys_write(1, v->str, v->len);
}

int32
strlen(int8 *s)
{
	int32 l;

	for(l=0; s[l]!=0; l++)
		;
	return l;
}

void
prints(int8 *s)
{
	sys_write(1, s, strlen(s));
}

void
sys_printpc(void *p)
{
	prints("PC=0x");
	sys_printpointer(sys_getcallerpc(p));
}

void
sys_panicl(int32 lno)
{
	uint8 *pc;
	uint8 *sp;
	uint8 *retpc;
	int32 spoff;
	static int8 spmark[] = "\xa7\xf1\xd9\x2a\x82\xc8\xd8\xfe";
	int8* spp;
	int32 counter;
	int32 i;
	int8* name;

	prints("\npanic on line ");
	sys_printint(lno);
	prints(" ");
	sys_printpc(&lno);
	prints("\n");
	sp = (uint8*)&lno;
	pc = (uint8*)sys_panicl;
	counter = 0;
	name = "panic";
	while((pc = ((uint8**)sp)[-1]) > (uint8*)0x1000) {
		/* print args for this frame */
		prints("\t");
		prints(name);
		prints("(");
		for(i = 0; i < 3; i++){
			if(i != 0)
				prints(", ");
			sys_printint(((uint32*)sp)[i]);
		}
		prints(", ...)\n");
		prints("\t");
		prints(name);
		prints("(");
		for(i = 0; i < 3; i++){
			if(i != 0)
				prints(", ");
			prints("0x");
			sys_printpointer(((void**)sp)[i]);
		}
		prints(", ...)\n");
		/* print pc for next frame */
		prints("0x");
		sys_printpointer(pc);
		prints(" ");
		/* next word down on stack is PC */
		retpc = pc;
		/* find SP offset by stepping back through instructions to SP offset marker */
		while(pc > (uint8*)0x1000+11) {
			for(spp = spmark; *spp != '\0' && *pc++ == (uint8)*spp++; )
				;
			if(*spp == '\0'){
				spoff = *pc++;
				spoff += *pc++ << 8;
				spoff += *pc++ << 16;
				name = (int8*)pc;
				prints(name);
				prints("+");
				sys_printint(pc-retpc);
				prints("?zi\n");
				sp += spoff + 8;
				break;
			}
		}
		if(counter++ > 100){
			prints("stack trace terminated\n");
			break;
		}
	}
	*(int32*)0 = 0;
}

dump(byte *p, int32 n)
{
	uint32 v;
	int32 i;

	for(i=0; i<n; i++) {
		sys_printpointer((byte*)(p[i]>>4));
		sys_printpointer((byte*)(p[i]&0xf));
		if((i&15) == 15)
			prints("\n");
		else
			prints(" ");
	}
	if(n & 15)
		prints("\n");
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
	sys_exit(1);
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

	v = sys_mmap(nil, NHUNK, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, 0, 0);
	sys_memclr(v, n);
	nmmap += n;
	return v;
}

void*
mal(uint32 n)
{
	byte* v;

	// round to keep everything 64-bit alligned
	while(n & 7)
		n++;

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

uint32
cmpstring(string s1, string s2)
{
	uint32 i, l;
	byte c1, c2;

	l = s1->len;
	if(s2->len < l)
		l = s2->len;
	for(i=0; i<l; i++) {
		c1 = s1->str[i];
		c2 = s2->str[i];
		if(c1 < c2)
			return -1;
		if(c1 > c2)
			return +1;
	}
	if(s1->len < s2->len)
		return -1;
	if(s1->len > s2->len)
		return +1;
	return 0;
}

void
sys_mal(uint32 n, uint8 *ret)
{
	ret = mal(n);
	FLUSH(&ret);
}

void
sys_catstring(string s1, string s2, string s3)
{
	uint32 l;

	if(s1->len == 0) {
		s3 = s2;
		goto out;
	}
	if(s2->len == 0) {
		s3 = s1;
		goto out;
	}

	l = s1->len + s2->len;

	s3 = mal(sizeof(s3->len)+l);
	s3->len = l;
	mcpy(s3->str, s1->str, s1->len);
	mcpy(s3->str+s1->len, s2->str, s2->len);

out:
	FLUSH(&s3);
}

void
sys_cmpstring(string s1, string s2, int32 v)
{
	v = cmpstring(s1, s2);
	FLUSH(&v);
}

static int32
strcmp(byte *s1, byte *s2)
{
	uint32 i;
	byte c1, c2;

	for(i=0;; i++) {
		c1 = s1[i];
		c2 = s2[i];
		if(c1 < c2)
			return -1;
		if(c1 > c2)
			return +1;
		if(c1 == 0)
			return 0;
	}
}

static void
prbounds(int8* s, int32 a, int32 b, int32 c)
{
	int32 i;

	prints(s);
	prints(" ");
	sys_printint(a);
	prints("<");
	sys_printint(b);
	prints(">");
	sys_printint(c);
	prints("\n");
	throw("bounds");
}

void
sys_slicestring(string si, int32 lindex, int32 hindex, string so)
{
	string s, str;
	int32 l;

	if(lindex < 0 || lindex > si->len ||
	   hindex < lindex || hindex > si->len) {
		sys_printpc(&si);
		prints(" ");
		prbounds("slice", lindex, si->len, hindex);
	}

	l = hindex-lindex;
	so = mal(sizeof(so->len)+l);
	so->len = l;
	mcpy(so->str, si->str+lindex, l);
	FLUSH(&so);
}

void
sys_indexstring(string s, int32 i, byte b)
{
	if(i < 0 || i >= s->len) {
		sys_printpc(&s);
		prints(" ");
		prbounds("index", 0, i, s->len);
	}

	b = s->str[i];
	FLUSH(&b);
}

/*
 * this is the plan9 runetochar
 * extended for 36 bits in 7 bytes
 * note that it truncates to 32 bits
 * through the argument passing.
 */
static int32
runetochar(byte *str, uint32 c)
{
	int32 i, n;
	uint32 mask, mark;

	/*
	 * one character in 7 bits
	 */
	if(c <= 0x07FUL) {
		str[0] = c;
		return 1;
	}

	/*
	 * every new character picks up 5 bits
	 * one less in the first byte and
	 * six more in an extension byte
	 */
	mask = 0x7ffUL;
	mark = 0xC0UL;
	for(n=1;; n++) {
		if(c <= mask)
			break;
		mask = (mask<<5) | 0x1fUL;
		mark = (mark>>1) | 0x80UL;
	}

	/*
	 * lay down the bytes backwards
	 * n is the number of extension bytes
	 * mask is the max codepoint
	 * mark is the zeroth byte indicator
	 */
	for(i=n; i>0; i--) {
		str[i] = 0x80UL | (c&0x3fUL);
		c >>= 6;
	}

	str[0] = mark|c;
	return n+1;
}

void
sys_intstring(int64 v, string s)
{
	int32 l;

	s = mal(sizeof(s->len)+8);
	s->len = runetochar(s->str, v);
	FLUSH(&s);
}

void
sys_byteastring(byte *a, int32 l, string s)
{
	s = mal(sizeof(s->len)+l);
	s->len = l;
	mcpy(s->str, a, l);
	FLUSH(&s);
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
sys_ifaces2i(Sigi *si, Sigs *ss, Map *m, void *s)
{

	if(debug) {
		prints("s2i sigi=");
		sys_printpointer(si);
		prints(" sigs=");
		sys_printpointer(ss);
		prints(" s=");
		sys_printpointer(s);
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
		sys_printpointer(m);
		prints(" s=");
		sys_printpointer(s);
		prints("\n");
		dump((byte*)m, 64);
	}

	FLUSH(&m);
}

void
sys_ifacei2i(Sigi *si, Map *m, void *s)
{

	if(debug) {
		prints("i2i sigi=");
		sys_printpointer(si);
		prints(" m=");
		sys_printpointer(m);
		prints(" s=");
		sys_printpointer(s);
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
		sys_printpointer(m);
		prints(" s=");
		sys_printpointer(s);
		prints("\n");
		dump((byte*)m, 64);
	}
}

void
sys_ifacei2s(Sigs *ss, Map *m, void *s)
{

	if(debug) {
		prints("i2s m=");
		sys_printpointer(m);
		prints(" s=");
		sys_printpointer(s);
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
		x &= ~((uint64)1 << (64-11-e))-1;
	dd = *(float64*)&x;
	*ip = dd;
	return d - dd;
}

// func frexp(float64) (int32, float64); // break fp into exp,fract
void
sys_frexp(float64 din, int32 iou, float64 dou)
{
	dou = frexp(din, &iou);
	FLUSH(&dou);
}

//func	ldexp(int32, float64) float64;	// make fp from exp,fract
void
sys_ldexp(float64 din, int32 ein, float64 dou)
{
	dou = ldexp(din, ein);
	FLUSH(&dou);
}

//func	modf(float64) (float64, float64);	// break fp into double+double
float64
sys_modf(float64 din, float64 dou1, float64 dou2)
{
	dou1 = modf(din, &dou2);
	FLUSH(&dou2);
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
}
