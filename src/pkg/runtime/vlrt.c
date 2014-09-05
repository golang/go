// Inferno's libkern/vlrt-386.c
// http://code.google.com/p/inferno-os/source/browse/libkern/vlrt-386.c
//
//         Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//         Revisions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).  All rights reserved.
//         Portions Copyright 2009 The Go Authors. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// +build arm 386

#include "textflag.h"

/*
 * C runtime for 64-bit divide, others.
 *
 * TODO(rsc): The simple functions are dregs--8c knows how
 * to generate the code directly now.  Find and remove.
 */

void	runtime·panicdivide(void);

typedef	unsigned long	ulong;
typedef	unsigned int	uint;
typedef	unsigned short	ushort;
typedef	unsigned char	uchar;
typedef	signed char	schar;

#define	SIGN(n)	(1UL<<(n-1))

typedef	struct	Vlong	Vlong;
struct	Vlong
{
	ulong	lo;
	ulong	hi;
};

typedef	union	Vlong64	Vlong64;
union	Vlong64
{
	long long	v;
	Vlong	v2;
};

void	runtime·abort(void);

#pragma textflag NOSPLIT
Vlong
_addv(Vlong a, Vlong b)
{
	Vlong r;

	r.lo = a.lo + b.lo;
	r.hi = a.hi + b.hi;
	if(r.lo < a.lo)
		r.hi++;
	return r;
}

#pragma textflag NOSPLIT
Vlong
_subv(Vlong a, Vlong b)
{
	Vlong r;

	r.lo = a.lo - b.lo;
	r.hi = a.hi - b.hi;
	if(r.lo > a.lo)
		r.hi--;
	return r;
}

Vlong
_d2v(double d)
{
	union { double d; Vlong vl; } x;
	ulong xhi, xlo, ylo, yhi;
	int sh;
	Vlong y;

	x.d = d;

	xhi = (x.vl.hi & 0xfffff) | 0x100000;
	xlo = x.vl.lo;
	sh = 1075 - ((x.vl.hi >> 20) & 0x7ff);

	ylo = 0;
	yhi = 0;
	if(sh >= 0) {
		/* v = (hi||lo) >> sh */
		if(sh < 32) {
			if(sh == 0) {
				ylo = xlo;
				yhi = xhi;
			} else {
				ylo = (xlo >> sh) | (xhi << (32-sh));
				yhi = xhi >> sh;
			}
		} else {
			if(sh == 32) {
				ylo = xhi;
			} else
			if(sh < 64) {
				ylo = xhi >> (sh-32);
			}
		}
	} else {
		/* v = (hi||lo) << -sh */
		sh = -sh;
		if(sh <= 10) { /* NOTE: sh <= 11 on ARM??? */
			ylo = xlo << sh;
			yhi = (xhi << sh) | (xlo >> (32-sh));
		} else {
			/* overflow */
			yhi = d;	/* causes something awful */
		}
	}
	if(x.vl.hi & SIGN(32)) {
		if(ylo != 0) {
			ylo = -ylo;
			yhi = ~yhi;
		} else
			yhi = -yhi;
	}

	y.hi = yhi;
	y.lo = ylo;
	return y;
}

Vlong
_f2v(float f)
{
	return _d2v(f);
}

double
_ul2d(ulong u)
{
	// compensate for bug in c
	if(u & SIGN(32)) {
		u ^= SIGN(32);
		return 2147483648. + u;
	}
	return u;
}

double
_v2d(Vlong x)
{
	if(x.hi & SIGN(32)) {
		if(x.lo) {
			x.lo = -x.lo;
			x.hi = ~x.hi;
		} else
			x.hi = -x.hi;
		return -(_ul2d(x.hi)*4294967296. + _ul2d(x.lo));
	}
	return (long)x.hi*4294967296. + x.lo;
}

float
_v2f(Vlong x)
{
	return _v2d(x);
}

ulong	runtime·_div64by32(Vlong, ulong, ulong*);
int	runtime·_mul64by32(Vlong*, Vlong, ulong);

static void
slowdodiv(Vlong num, Vlong den, Vlong *q, Vlong *r)
{
	ulong numlo, numhi, denhi, denlo, quohi, quolo, t;
	int i;

	numhi = num.hi;
	numlo = num.lo;
	denhi = den.hi;
	denlo = den.lo;

	/*
	 * get a divide by zero
	 */
	if(denlo==0 && denhi==0) {
		runtime·panicdivide();
	}

	/*
	 * set up the divisor and find the number of iterations needed
	 */
	if(numhi >= SIGN(32)) {
		quohi = SIGN(32);
		quolo = 0;
	} else {
		quohi = numhi;
		quolo = numlo;
	}
	i = 0;
	while(denhi < quohi || (denhi == quohi && denlo < quolo)) {
		denhi = (denhi<<1) | (denlo>>31);
		denlo <<= 1;
		i++;
	}

	quohi = 0;
	quolo = 0;
	for(; i >= 0; i--) {
		quohi = (quohi<<1) | (quolo>>31);
		quolo <<= 1;
		if(numhi > denhi || (numhi == denhi && numlo >= denlo)) {
			t = numlo;
			numlo -= denlo;
			if(numlo > t)
				numhi--;
			numhi -= denhi;
			quolo |= 1;
		}
		denlo = (denlo>>1) | (denhi<<31);
		denhi >>= 1;
	}

	if(q) {
		q->lo = quolo;
		q->hi = quohi;
	}
	if(r) {
		r->lo = numlo;
		r->hi = numhi;
	}
}

#ifdef GOARCH_arm
static void
dodiv(Vlong num, Vlong den, Vlong *qp, Vlong *rp)
{
	slowdodiv(num, den, qp, rp);
}
#endif

#ifdef GOARCH_386
static void
dodiv(Vlong num, Vlong den, Vlong *qp, Vlong *rp)
{
	ulong n;
	Vlong x, q, r;
	
	if(den.hi > num.hi || (den.hi == num.hi && den.lo > num.lo)){
		if(qp) {
			qp->hi = 0;
			qp->lo = 0;
		}
		if(rp) {
			rp->hi = num.hi;
			rp->lo = num.lo;
		}
		return;
	}

	if(den.hi != 0){
		q.hi = 0;
		n = num.hi/den.hi;
		if(runtime·_mul64by32(&x, den, n) || x.hi > num.hi || (x.hi == num.hi && x.lo > num.lo))
			slowdodiv(num, den, &q, &r);
		else {
			q.lo = n;
			*(long long*)&r = *(long long*)&num - *(long long*)&x;
		}
	} else {
		if(num.hi >= den.lo){
			if(den.lo == 0)
				runtime·panicdivide();
			q.hi = n = num.hi/den.lo;
			num.hi -= den.lo*n;
		} else {
			q.hi = 0;
		}
		q.lo = runtime·_div64by32(num, den.lo, &r.lo);
		r.hi = 0;
	}
	if(qp) {
		qp->lo = q.lo;
		qp->hi = q.hi;
	}
	if(rp) {
		rp->lo = r.lo;
		rp->hi = r.hi;
	}
}
#endif

Vlong
_divvu(Vlong n, Vlong d)
{
	Vlong q;

	if(n.hi == 0 && d.hi == 0) {
		if(d.lo == 0)
			runtime·panicdivide();
		q.hi = 0;
		q.lo = n.lo / d.lo;
		return q;
	}
	dodiv(n, d, &q, 0);
	return q;
}

Vlong
_modvu(Vlong n, Vlong d)
{
	Vlong r;

	if(n.hi == 0 && d.hi == 0) {
		if(d.lo == 0)
			runtime·panicdivide();
		r.hi = 0;
		r.lo = n.lo % d.lo;
		return r;
	}
	dodiv(n, d, 0, &r);
	return r;
}

static void
vneg(Vlong *v)
{

	if(v->lo == 0) {
		v->hi = -v->hi;
		return;
	}
	v->lo = -v->lo;
	v->hi = ~v->hi;
}

Vlong
_divv(Vlong n, Vlong d)
{
	long nneg, dneg;
	Vlong q;

	if(n.hi == (((long)n.lo)>>31) && d.hi == (((long)d.lo)>>31)) {
		if((long)n.lo == -0x80000000 && (long)d.lo == -1) {
			// special case: 32-bit -0x80000000 / -1 causes divide error,
			// but it's okay in this 64-bit context.
			q.lo = 0x80000000;
			q.hi = 0;
			return q;
		}
		if(d.lo == 0)
			runtime·panicdivide();
		q.lo = (long)n.lo / (long)d.lo;
		q.hi = ((long)q.lo) >> 31;
		return q;
	}
	nneg = n.hi >> 31;
	if(nneg)
		vneg(&n);
	dneg = d.hi >> 31;
	if(dneg)
		vneg(&d);
	dodiv(n, d, &q, 0);
	if(nneg != dneg)
		vneg(&q);
	return q;
}

Vlong
_modv(Vlong n, Vlong d)
{
	long nneg, dneg;
	Vlong r;

	if(n.hi == (((long)n.lo)>>31) && d.hi == (((long)d.lo)>>31)) {
		if((long)n.lo == -0x80000000 && (long)d.lo == -1) {
			// special case: 32-bit -0x80000000 % -1 causes divide error,
			// but it's okay in this 64-bit context.
			r.lo = 0;
			r.hi = 0;
			return r;
		}
		if(d.lo == 0)
			runtime·panicdivide();
		r.lo = (long)n.lo % (long)d.lo;
		r.hi = ((long)r.lo) >> 31;
		return r;
	}
	nneg = n.hi >> 31;
	if(nneg)
		vneg(&n);
	dneg = d.hi >> 31;
	if(dneg)
		vneg(&d);
	dodiv(n, d, 0, &r);
	if(nneg)
		vneg(&r);
	return r;
}

#pragma textflag NOSPLIT
Vlong
_rshav(Vlong a, int b)
{
	long t;
	Vlong r;

	t = a.hi;
	if(b >= 32) {
		r.hi = t>>31;
		if(b >= 64) {
			/* this is illegal re C standard */
			r.lo = t>>31;
			return r;
		}
		r.lo = t >> (b-32);
		return r;
	}
	if(b <= 0) {
		r.hi = t;
		r.lo = a.lo;
		return r;
	}
	r.hi = t >> b;
	r.lo = (t << (32-b)) | (a.lo >> b);
	return r;
}

#pragma textflag NOSPLIT
Vlong
_rshlv(Vlong a, int b)
{
	ulong t;
	Vlong r;

	t = a.hi;
	if(b >= 32) {
		r.hi = 0;
		if(b >= 64) {
			/* this is illegal re C standard */
			r.lo = 0;
			return r;
		}
		r.lo = t >> (b-32);
		return r;
	}
	if(b <= 0) {
		r.hi = t;
		r.lo = a.lo;
		return r;
	}
	r.hi = t >> b;
	r.lo = (t << (32-b)) | (a.lo >> b);
	return r;
}

#pragma textflag NOSPLIT
Vlong
_lshv(Vlong a, int b)
{
	ulong t;

	t = a.lo;
	if(b >= 32) {
		if(b >= 64) {
			/* this is illegal re C standard */
			return (Vlong){0, 0};
		}
		return (Vlong){0, t<<(b-32)};
	}
	if(b <= 0) {
		return (Vlong){t, a.hi};
	}
	return (Vlong){t<<b, (t >> (32-b)) | (a.hi << b)};
}

#pragma textflag NOSPLIT
Vlong
_andv(Vlong a, Vlong b)
{
	Vlong r;

	r.hi = a.hi & b.hi;
	r.lo = a.lo & b.lo;
	return r;
}

#pragma textflag NOSPLIT
Vlong
_orv(Vlong a, Vlong b)
{
	Vlong r;

	r.hi = a.hi | b.hi;
	r.lo = a.lo | b.lo;
	return r;
}

#pragma textflag NOSPLIT
Vlong
_xorv(Vlong a, Vlong b)
{
	Vlong r;

	r.hi = a.hi ^ b.hi;
	r.lo = a.lo ^ b.lo;
	return r;
}

Vlong
_vpp(Vlong *r)
{
	Vlong l;

	l = *r;
	r->lo++;
	if(r->lo == 0)
		r->hi++;
	return l;
}

#pragma textflag NOSPLIT
Vlong
_vmm(Vlong *r)
{
	Vlong l;

	l = *r;
	if(r->lo == 0)
		r->hi--;
	r->lo--;
	return l;
}

#pragma textflag NOSPLIT
Vlong
_ppv(Vlong *r)
{

	r->lo++;
	if(r->lo == 0)
		r->hi++;
	return *r;
}

#pragma textflag NOSPLIT
Vlong
_mmv(Vlong *r)
{

	if(r->lo == 0)
		r->hi--;
	r->lo--;
	return *r;
}

#pragma textflag NOSPLIT
Vlong
_vasop(void *lv, Vlong fn(Vlong, Vlong), int type, Vlong rv)
{
	Vlong t, u;

	u.lo = 0;
	u.hi = 0;
	switch(type) {
	default:
		runtime·abort();
		break;

	case 1:	/* schar */
		t.lo = *(schar*)lv;
		t.hi = t.lo >> 31;
		u = fn(t, rv);
		*(schar*)lv = u.lo;
		break;

	case 2:	/* uchar */
		t.lo = *(uchar*)lv;
		t.hi = 0;
		u = fn(t, rv);
		*(uchar*)lv = u.lo;
		break;

	case 3:	/* short */
		t.lo = *(short*)lv;
		t.hi = t.lo >> 31;
		u = fn(t, rv);
		*(short*)lv = u.lo;
		break;

	case 4:	/* ushort */
		t.lo = *(ushort*)lv;
		t.hi = 0;
		u = fn(t, rv);
		*(ushort*)lv = u.lo;
		break;

	case 9:	/* int */
		t.lo = *(int*)lv;
		t.hi = t.lo >> 31;
		u = fn(t, rv);
		*(int*)lv = u.lo;
		break;

	case 10:	/* uint */
		t.lo = *(uint*)lv;
		t.hi = 0;
		u = fn(t, rv);
		*(uint*)lv = u.lo;
		break;

	case 5:	/* long */
		t.lo = *(long*)lv;
		t.hi = t.lo >> 31;
		u = fn(t, rv);
		*(long*)lv = u.lo;
		break;

	case 6:	/* ulong */
		t.lo = *(ulong*)lv;
		t.hi = 0;
		u = fn(t, rv);
		*(ulong*)lv = u.lo;
		break;

	case 7:	/* vlong */
	case 8:	/* uvlong */
		if((void*)fn == _lshv || (void*)fn == _rshav || (void*)fn == _rshlv)
			u = ((Vlong(*)(Vlong,int))fn)(*(Vlong*)lv, *(int*)&rv);
		else
			u = fn(*(Vlong*)lv, rv);
		*(Vlong*)lv = u;
		break;
	}
	return u;
}

#pragma textflag NOSPLIT
Vlong
_p2v(void *p)
{
	long t;
	Vlong ret;

	t = (ulong)p;
	ret.lo = t;
	ret.hi = 0;
	return ret;
}

#pragma textflag NOSPLIT
Vlong
_sl2v(long sl)
{
	long t;
	Vlong ret;

	t = sl;
	ret.lo = t;
	ret.hi = t >> 31;
	return ret;
}

#pragma textflag NOSPLIT
Vlong
_ul2v(ulong ul)
{
	long t;
	Vlong ret;

	t = ul;
	ret.lo = t;
	ret.hi = 0;
	return ret;
}

#pragma textflag NOSPLIT
Vlong
_si2v(int si)
{
	return (Vlong){si, si>>31};
}

#pragma textflag NOSPLIT
Vlong
_ui2v(uint ui)
{
	long t;
	Vlong ret;

	t = ui;
	ret.lo = t;
	ret.hi = 0;
	return ret;
}

#pragma textflag NOSPLIT
Vlong
_sh2v(long sh)
{
	long t;
	Vlong ret;

	t = (sh << 16) >> 16;
	ret.lo = t;
	ret.hi = t >> 31;
	return ret;
}

#pragma textflag NOSPLIT
Vlong
_uh2v(ulong ul)
{
	long t;
	Vlong ret;

	t = ul & 0xffff;
	ret.lo = t;
	ret.hi = 0;
	return ret;
}

#pragma textflag NOSPLIT
Vlong
_sc2v(long uc)
{
	long t;
	Vlong ret;

	t = (uc << 24) >> 24;
	ret.lo = t;
	ret.hi = t >> 31;
	return ret;
}

#pragma textflag NOSPLIT
Vlong
_uc2v(ulong ul)
{
	long t;
	Vlong ret;

	t = ul & 0xff;
	ret.lo = t;
	ret.hi = 0;
	return ret;
}

#pragma textflag NOSPLIT
long
_v2sc(Vlong rv)
{
	long t;

	t = rv.lo & 0xff;
	return (t << 24) >> 24;
}

#pragma textflag NOSPLIT
long
_v2uc(Vlong rv)
{

	return rv.lo & 0xff;
}

#pragma textflag NOSPLIT
long
_v2sh(Vlong rv)
{
	long t;

	t = rv.lo & 0xffff;
	return (t << 16) >> 16;
}

#pragma textflag NOSPLIT
long
_v2uh(Vlong rv)
{

	return rv.lo & 0xffff;
}

#pragma textflag NOSPLIT
long
_v2sl(Vlong rv)
{

	return rv.lo;
}

#pragma textflag NOSPLIT
long
_v2ul(Vlong rv)
{

	return rv.lo;
}

#pragma textflag NOSPLIT
long
_v2si(Vlong rv)
{
	return rv.lo;
}

#pragma textflag NOSPLIT
long
_v2ui(Vlong rv)
{

	return rv.lo;
}

#pragma textflag NOSPLIT
int
_testv(Vlong rv)
{
	return rv.lo || rv.hi;
}

#pragma textflag NOSPLIT
int
_eqv(Vlong lv, Vlong rv)
{
	return lv.lo == rv.lo && lv.hi == rv.hi;
}

#pragma textflag NOSPLIT
int
_nev(Vlong lv, Vlong rv)
{
	return lv.lo != rv.lo || lv.hi != rv.hi;
}

#pragma textflag NOSPLIT
int
_ltv(Vlong lv, Vlong rv)
{
	return (long)lv.hi < (long)rv.hi ||
		(lv.hi == rv.hi && lv.lo < rv.lo);
}

#pragma textflag NOSPLIT
int
_lev(Vlong lv, Vlong rv)
{
	return (long)lv.hi < (long)rv.hi ||
		(lv.hi == rv.hi && lv.lo <= rv.lo);
}

#pragma textflag NOSPLIT
int
_gtv(Vlong lv, Vlong rv)
{
	return (long)lv.hi > (long)rv.hi ||
		(lv.hi == rv.hi && lv.lo > rv.lo);
}

#pragma textflag NOSPLIT
int
_gev(Vlong lv, Vlong rv)
{
	return (long)lv.hi > (long)rv.hi ||
		(lv.hi == rv.hi && lv.lo >= rv.lo);
}

#pragma textflag NOSPLIT
int
_lov(Vlong lv, Vlong rv)
{
	return lv.hi < rv.hi ||
		(lv.hi == rv.hi && lv.lo < rv.lo);
}

#pragma textflag NOSPLIT
int
_lsv(Vlong lv, Vlong rv)
{
	return lv.hi < rv.hi ||
		(lv.hi == rv.hi && lv.lo <= rv.lo);
}

#pragma textflag NOSPLIT
int
_hiv(Vlong lv, Vlong rv)
{
	return lv.hi > rv.hi ||
		(lv.hi == rv.hi && lv.lo > rv.lo);
}

#pragma textflag NOSPLIT
int
_hsv(Vlong lv, Vlong rv)
{
	return lv.hi > rv.hi ||
		(lv.hi == rv.hi && lv.lo >= rv.lo);
}
