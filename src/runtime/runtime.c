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

/*BUG: move traceback code to architecture-dependent runtime */
void
sys_panicl(int32 lno)
{
	uint8 *sp;

	prints("\npanic on line ");
	sys_printint(lno);
	prints(" ");
	sys_printpc(&lno);
	prints("\n");
	sp = (uint8*)&lno;
	traceback(sys_getcallerpc(&lno), sp);
	sys_breakpoint();
	sys_exit(2);
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

typedef	struct	Link	Link;
typedef	struct	Hmap	Hmap;
typedef	struct	Alg	Alg;

struct	Alg
{
	uint64	(*hash)(uint32, void*);
	uint32	(*equal)(uint32, void*, void*);
	void	(*print)(uint32, void*);
	void	(*copy)(uint32, void*, void*);
};

struct	Link
{
	Link*	link;
	byte	data[8];
};

struct	Hmap
{
	uint32	keysize;
	uint32	valsize;
	uint32	hint;
	Alg*	keyalg;
	Alg*	valalg;
	uint32	valoffset;
	uint32	ko;
	uint32	vo;
	uint32	po;
	Link*	link;
};

static uint64
memhash(uint32 s, void *a)
{
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
	sys_printint(v);
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
	prints("stringhash\n");
	return 0x12345;
}

static uint32
stringequal(uint32 s, string *a, string *b)
{
	return cmpstring(*a, *b) == 0;
}

static void
stringprint(uint32 s, string *a)
{
	sys_printstring(*a);
}

static void
stringcopy(uint32 s, string *a, string *b)
{
	if(b == nil) {
		*b = nil;
		return;
	}
	*a = *b;
}

static uint32
rnd(uint32 n, uint32 m)
{
	uint32 r;

	r = n % m;
	if(r)
		n += m-r;
	return n;
}

static	Alg
algarray[] =
{
	{	&memhash,	&memequal,	&memprint,	&memcopy	},
	{	&stringhash,	&stringequal,	&stringprint,	&stringcopy	},
};

// newmap(keysize uint32, valsize uint32,
//	keyalg uint32, valalg uint32,
//	hint uint32) (hmap *map[any]any);
void
sys_newmap(uint32 keysize, uint32 valsize,
	uint32 keyalg, uint32 valalg, uint32 hint,
	Hmap* ret)
{
	Hmap *m;

	if(keyalg >= nelem(algarray) ||
	   valalg >= nelem(algarray)) {
		prints("0<=");
		sys_printint(keyalg);
		prints("<");
		sys_printint(nelem(algarray));
		prints("\n0<=");
		sys_printint(valalg);
		prints("<");
		sys_printint(nelem(algarray));
		prints("\n");

		throw("sys_newmap: key/val algorithm out of range");
	}

	m = mal(sizeof(*m));

	m->keysize = keysize;
	m->valsize = valsize;
	m->keyalg = &algarray[keyalg];
	m->valalg = &algarray[valalg];
	m->hint = hint;

	// these calculations are compiler dependent
	m->valoffset = rnd(keysize, valsize);
	m->ko = rnd(sizeof(m), keysize);
	m->vo = rnd(m->ko+keysize, valsize);
	m->po = rnd(m->vo+valsize, 1);

	ret = m;
	FLUSH(&ret);

	if(debug) {
		prints("newmap: map=");
		sys_printpointer(m);
		prints("; keysize=");
		sys_printint(keysize);
		prints("; valsize=");
		sys_printint(valsize);
		prints("; keyalg=");
		sys_printint(keyalg);
		prints("; valalg=");
		sys_printint(valalg);
		prints("; valoffset=");
		sys_printint(m->valoffset);
		prints("; ko=");
		sys_printint(m->ko);
		prints("; vo=");
		sys_printint(m->vo);
		prints("; po=");
		sys_printint(m->po);
		prints("\n");
	}
}

// mapaccess1(hmap *map[any]any, key any) (val any);
void
sys_mapaccess1(Hmap *m, ...)
{
	Link *l;
	byte *ak, *av;

	ak = (byte*)&m + m->ko;
	av = (byte*)&m + m->vo;

	for(l=m->link; l!=nil; l=l->link) {
		if(m->keyalg->equal(m->keysize, ak, l->data)) {
			m->valalg->copy(m->valsize, av, l->data+m->valoffset);
			goto out;
		}
	}

	m->valalg->copy(m->valsize, av, 0);

out:
	if(1) {
		prints("sys_mapaccess1: map=");
		sys_printpointer(m);
		prints("; key=");
		m->keyalg->print(m->keysize, ak);
		prints("; val=");
		m->valalg->print(m->valsize, av);
		prints("\n");
	}
}

// mapaccess2(hmap *map[any]any, key any) (val any, pres bool);
void
sys_mapaccess2(Hmap *m, ...)
{
	Link *l;
	byte *ak, *av, *ap;

	ak = (byte*)&m + m->ko;
	av = (byte*)&m + m->vo;
	ap = (byte*)&m + m->po;

	for(l=m->link; l!=nil; l=l->link) {
		if(m->keyalg->equal(m->keysize, ak, l->data)) {
			*ap = true;
			m->valalg->copy(m->valsize, av, l->data+m->valoffset);
			goto out;
		}
	}

	*ap = false;
	m->valalg->copy(m->valsize, av, nil);

out:
	if(debug) {
		prints("sys_mapaccess2: map=");
		sys_printpointer(m);
		prints("; key=");
		m->keyalg->print(m->keysize, ak);
		prints("; val=");
		m->valalg->print(m->valsize, av);
		prints("; pres=");
		sys_printbool(*ap);
		prints("\n");
	}
}

static void
sys_mapassign(Hmap *m, byte *ak, byte *av)
{
	Link *l;

	// mapassign(hmap *map[any]any, key any, val any);

	for(l=m->link; l!=nil; l=l->link) {
		if(m->keyalg->equal(m->keysize, ak, l->data))
			goto out;
	}

	l = mal((sizeof(*l)-8) + m->keysize + m->valsize);
	l->link = m->link;
	m->link = l;
	m->keyalg->copy(m->keysize, l->data, ak);

out:
	m->valalg->copy(m->valsize, l->data+m->valoffset, av);

	if(debug) {
		prints("mapassign: map=");
		sys_printpointer(m);
		prints("; key=");
		m->keyalg->print(m->keysize, ak);
		prints("; val=");
		m->valalg->print(m->valsize, av);
		prints("\n");
	}
}

// mapassign1(hmap *map[any]any, key any, val any);
void
sys_mapassign1(Hmap *m, ...)
{
	Link **ll;
	byte *ak, *av;

	ak = (byte*)&m + m->ko;
	av = (byte*)&m + m->vo;

	sys_mapassign(m, ak, av);
}

// mapassign2(hmap *map[any]any, key any, val any, pres bool);
void
sys_mapassign2(Hmap *m, ...)
{
	Link **ll;
	byte *ak, *av, *ap;


	ak = (byte*)&m + m->ko;
	av = (byte*)&m + m->vo;
	ap = (byte*)&m + m->po;

	if(*ap == true) {
		// assign
		sys_mapassign(m, ak, av);
		return;
	}

	// delete
	for(ll=&m->link; (*ll)!=nil; ll=&(*ll)->link) {
		if(m->keyalg->equal(m->keysize, ak, (*ll)->data)) {
			m->valalg->copy(m->valsize, (*ll)->data+m->valoffset, nil);
			(*ll) = (*ll)->link;
			if(debug) {
				prints("mapdelete (found): map=");
				sys_printpointer(m);
				prints("; key=");
				m->keyalg->print(m->keysize, ak);
				prints("\n");
			}
			return;
		}
	}

	if(debug) {
		prints("mapdelete (not found): map=");
		sys_printpointer(m);
		prints("; key=");
		m->keyalg->print(m->keysize, ak);
		prints(" *** not found\n");
	}
}
