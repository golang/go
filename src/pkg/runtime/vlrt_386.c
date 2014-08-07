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

#include "../../cmd/ld/textflag.h"

/*
 * C runtime for 64-bit divide, others.
 *
 * TODO(rsc): The simple functions are dregs--8c knows how
 * to generate the code directly now.  Find and remove.
 */

extern void runtime·panicdivide(void);

typedef	unsigned long	ulong;
typedef	unsigned int	uint;
typedef	unsigned short	ushort;
typedef	unsigned char	uchar;
typedef	signed char	schar;

#define	SIGN(n)	(1UL<<(n-1))

typedef	union	Vlong	Vlong;
union	Vlong
{
	long long	v;
	struct
	{
		ulong	lo;
		ulong	hi;
	}		v2;
};

void	runtime·abort(void);

void
_d2v(Vlong *y, double d)
{
	union { double d; Vlong vl; } x;
	ulong xhi, xlo, ylo, yhi;
	int sh;

	x.d = d;

	xhi = (x.vl.v2.hi & 0xfffff) | 0x100000;
	xlo = x.vl.v2.lo;
	sh = 1075 - ((x.vl.v2.hi >> 20) & 0x7ff);

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
		if(sh <= 10) {
			ylo = xlo << sh;
			yhi = (xhi << sh) | (xlo >> (32-sh));
		} else {
			/* overflow */
			yhi = d;	/* causes something awful */
		}
	}
	if(x.vl.v2.hi & SIGN(32)) {
		if(ylo != 0) {
			ylo = -ylo;
			yhi = ~yhi;
		} else
			yhi = -yhi;
	}

	y->v2.hi = yhi;
	y->v2.lo = ylo;
}

void
_f2v(Vlong *y, float f)
{

	_d2v(y, f);
}

double
_v2d(Vlong x)
{
	if(x.v2.hi & SIGN(32)) {
		if(x.v2.lo) {
			x.v2.lo = -x.v2.lo;
			x.v2.hi = ~x.v2.hi;
		} else
			x.v2.hi = -x.v2.hi;
		return -((long)x.v2.hi*4294967296. + x.v2.lo);
	}
	return (long)x.v2.hi*4294967296. + x.v2.lo;
}

float
_v2f(Vlong x)
{
	return _v2d(x);
}

ulong	_div64by32(Vlong, ulong, ulong*);
int	_mul64by32(Vlong*, Vlong, ulong);

static void
slowdodiv(Vlong num, Vlong den, Vlong *q, Vlong *r)
{
	ulong numlo, numhi, denhi, denlo, quohi, quolo, t;
	int i;

	numhi = num.v2.hi;
	numlo = num.v2.lo;
	denhi = den.v2.hi;
	denlo = den.v2.lo;

	/*
	 * get a divide by zero
	 */
	if(denlo==0 && denhi==0) {
		numlo = numlo / denlo;
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
		q->v2.lo = quolo;
		q->v2.hi = quohi;
	}
	if(r) {
		r->v2.lo = numlo;
		r->v2.hi = numhi;
	}
}

static void
dodiv(Vlong num, Vlong den, Vlong *qp, Vlong *rp)
{
	ulong n;
	Vlong x, q, r;

	if(den.v2.hi > num.v2.hi || (den.v2.hi == num.v2.hi && den.v2.lo > num.v2.lo)){
		if(qp) {
			qp->v2.hi = 0;
			qp->v2.lo = 0;
		}
		if(rp) {
			rp->v2.hi = num.v2.hi;
			rp->v2.lo = num.v2.lo;
		}
		return;
	}

	if(den.v2.hi != 0){
		q.v2.hi = 0;
		n = num.v2.hi/den.v2.hi;
		if(_mul64by32(&x, den, n) || x.v2.hi > num.v2.hi || (x.v2.hi == num.v2.hi && x.v2.lo > num.v2.lo))
			slowdodiv(num, den, &q, &r);
		else {
			q.v2.lo = n;
			r.v = num.v - x.v;
		}
	} else {
		if(num.v2.hi >= den.v2.lo){
			if(den.v2.lo == 0)
				runtime·panicdivide();
			q.v2.hi = n = num.v2.hi/den.v2.lo;
			num.v2.hi -= den.v2.lo*n;
		} else {
			q.v2.hi = 0;
		}
		q.v2.lo = _div64by32(num, den.v2.lo, &r.v2.lo);
		r.v2.hi = 0;
	}
	if(qp) {
		qp->v2.lo = q.v2.lo;
		qp->v2.hi = q.v2.hi;
	}
	if(rp) {
		rp->v2.lo = r.v2.lo;
		rp->v2.hi = r.v2.hi;
	}
}

void
_divvu(Vlong *q, Vlong n, Vlong d)
{

	if(n.v2.hi == 0 && d.v2.hi == 0) {
		if(d.v2.lo == 0)
			runtime·panicdivide();
		q->v2.hi = 0;
		q->v2.lo = n.v2.lo / d.v2.lo;
		return;
	}
	dodiv(n, d, q, 0);
}

void
runtime·uint64div(Vlong n, Vlong d, Vlong q)
{
	_divvu(&q, n, d);
}

void
_modvu(Vlong *r, Vlong n, Vlong d)
{

	if(n.v2.hi == 0 && d.v2.hi == 0) {
		if(d.v2.lo == 0)
			runtime·panicdivide();
		r->v2.hi = 0;
		r->v2.lo = n.v2.lo % d.v2.lo;
		return;
	}
	dodiv(n, d, 0, r);
}

void
runtime·uint64mod(Vlong n, Vlong d, Vlong q)
{
	_modvu(&q, n, d);
}

static void
vneg(Vlong *v)
{

	if(v->v2.lo == 0) {
		v->v2.hi = -v->v2.hi;
		return;
	}
	v->v2.lo = -v->v2.lo;
	v->v2.hi = ~v->v2.hi;
}

void
_divv(Vlong *q, Vlong n, Vlong d)
{
	long nneg, dneg;

	if(n.v2.hi == (((long)n.v2.lo)>>31) && d.v2.hi == (((long)d.v2.lo)>>31)) {
		if((long)n.v2.lo == -0x80000000 && (long)d.v2.lo == -1) {
			// special case: 32-bit -0x80000000 / -1 causes divide error,
			// but it's okay in this 64-bit context.
			q->v2.lo = 0x80000000;
			q->v2.hi = 0;
			return;
		}
		if(d.v2.lo == 0)
			runtime·panicdivide();
		q->v2.lo = (long)n.v2.lo / (long)d.v2.lo;
		q->v2.hi = ((long)q->v2.lo) >> 31;
		return;
	}
	nneg = n.v2.hi >> 31;
	if(nneg)
		vneg(&n);
	dneg = d.v2.hi >> 31;
	if(dneg)
		vneg(&d);
	dodiv(n, d, q, 0);
	if(nneg != dneg)
		vneg(q);
}

void
runtime·int64div(Vlong n, Vlong d, Vlong q)
{
	_divv(&q, n, d);
}

void
_modv(Vlong *r, Vlong n, Vlong d)
{
	long nneg, dneg;

	if(n.v2.hi == (((long)n.v2.lo)>>31) && d.v2.hi == (((long)d.v2.lo)>>31)) {
		if((long)n.v2.lo == -0x80000000 && (long)d.v2.lo == -1) {
			// special case: 32-bit -0x80000000 % -1 causes divide error,
			// but it's okay in this 64-bit context.
			r->v2.lo = 0;
			r->v2.hi = 0;
			return;
		}
		if(d.v2.lo == 0)
			runtime·panicdivide();
		r->v2.lo = (long)n.v2.lo % (long)d.v2.lo;
		r->v2.hi = ((long)r->v2.lo) >> 31;
		return;
	}
	nneg = n.v2.hi >> 31;
	if(nneg)
		vneg(&n);
	dneg = d.v2.hi >> 31;
	if(dneg)
		vneg(&d);
	dodiv(n, d, 0, r);
	if(nneg)
		vneg(r);
}

void
runtime·int64mod(Vlong n, Vlong d, Vlong q)
{
	_modv(&q, n, d);
}

void
_rshav(Vlong *r, Vlong a, int b)
{
	long t;

	t = a.v2.hi;
	if(b >= 32) {
		r->v2.hi = t>>31;
		if(b >= 64) {
			/* this is illegal re C standard */
			r->v2.lo = t>>31;
			return;
		}
		r->v2.lo = t >> (b-32);
		return;
	}
	if(b <= 0) {
		r->v2.hi = t;
		r->v2.lo = a.v2.lo;
		return;
	}
	r->v2.hi = t >> b;
	r->v2.lo = (t << (32-b)) | (a.v2.lo >> b);
}

void
_rshlv(Vlong *r, Vlong a, int b)
{
	ulong t;

	t = a.v2.hi;
	if(b >= 32) {
		r->v2.hi = 0;
		if(b >= 64) {
			/* this is illegal re C standard */
			r->v2.lo = 0;
			return;
		}
		r->v2.lo = t >> (b-32);
		return;
	}
	if(b <= 0) {
		r->v2.hi = t;
		r->v2.lo = a.v2.lo;
		return;
	}
	r->v2.hi = t >> b;
	r->v2.lo = (t << (32-b)) | (a.v2.lo >> b);
}

#pragma textflag NOSPLIT
void
_lshv(Vlong *r, Vlong a, int b)
{
	ulong t;

	t = a.v2.lo;
	if(b >= 32) {
		r->v2.lo = 0;
		if(b >= 64) {
			/* this is illegal re C standard */
			r->v2.hi = 0;
			return;
		}
		r->v2.hi = t << (b-32);
		return;
	}
	if(b <= 0) {
		r->v2.lo = t;
		r->v2.hi = a.v2.hi;
		return;
	}
	r->v2.lo = t << b;
	r->v2.hi = (t >> (32-b)) | (a.v2.hi << b);
}

void
_andv(Vlong *r, Vlong a, Vlong b)
{
	r->v2.hi = a.v2.hi & b.v2.hi;
	r->v2.lo = a.v2.lo & b.v2.lo;
}

void
_orv(Vlong *r, Vlong a, Vlong b)
{
	r->v2.hi = a.v2.hi | b.v2.hi;
	r->v2.lo = a.v2.lo | b.v2.lo;
}

void
_xorv(Vlong *r, Vlong a, Vlong b)
{
	r->v2.hi = a.v2.hi ^ b.v2.hi;
	r->v2.lo = a.v2.lo ^ b.v2.lo;
}

void
_vpp(Vlong *l, Vlong *r)
{

	l->v2.hi = r->v2.hi;
	l->v2.lo = r->v2.lo;
	r->v2.lo++;
	if(r->v2.lo == 0)
		r->v2.hi++;
}

void
_vmm(Vlong *l, Vlong *r)
{

	l->v2.hi = r->v2.hi;
	l->v2.lo = r->v2.lo;
	if(r->v2.lo == 0)
		r->v2.hi--;
	r->v2.lo--;
}

void
_ppv(Vlong *l, Vlong *r)
{

	r->v2.lo++;
	if(r->v2.lo == 0)
		r->v2.hi++;
	l->v2.hi = r->v2.hi;
	l->v2.lo = r->v2.lo;
}

void
_mmv(Vlong *l, Vlong *r)
{

	if(r->v2.lo == 0)
		r->v2.hi--;
	r->v2.lo--;
	l->v2.hi = r->v2.hi;
	l->v2.lo = r->v2.lo;
}

void
_vasop(Vlong *ret, void *lv, void fn(Vlong*, Vlong, Vlong), int type, Vlong rv)
{
	Vlong t, u;

	u.v2.lo = 0;
	u.v2.hi = 0;
	switch(type) {
	default:
		runtime·abort();
		break;

	case 1:	/* schar */
		t.v2.lo = *(schar*)lv;
		t.v2.hi = t.v2.lo >> 31;
		fn(&u, t, rv);
		*(schar*)lv = u.v2.lo;
		break;

	case 2:	/* uchar */
		t.v2.lo = *(uchar*)lv;
		t.v2.hi = 0;
		fn(&u, t, rv);
		*(uchar*)lv = u.v2.lo;
		break;

	case 3:	/* short */
		t.v2.lo = *(short*)lv;
		t.v2.hi = t.v2.lo >> 31;
		fn(&u, t, rv);
		*(short*)lv = u.v2.lo;
		break;

	case 4:	/* ushort */
		t.v2.lo = *(ushort*)lv;
		t.v2.hi = 0;
		fn(&u, t, rv);
		*(ushort*)lv = u.v2.lo;
		break;

	case 9:	/* int */
		t.v2.lo = *(int*)lv;
		t.v2.hi = t.v2.lo >> 31;
		fn(&u, t, rv);
		*(int*)lv = u.v2.lo;
		break;

	case 10:	/* uint */
		t.v2.lo = *(uint*)lv;
		t.v2.hi = 0;
		fn(&u, t, rv);
		*(uint*)lv = u.v2.lo;
		break;

	case 5:	/* long */
		t.v2.lo = *(long*)lv;
		t.v2.hi = t.v2.lo >> 31;
		fn(&u, t, rv);
		*(long*)lv = u.v2.lo;
		break;

	case 6:	/* ulong */
		t.v2.lo = *(ulong*)lv;
		t.v2.hi = 0;
		fn(&u, t, rv);
		*(ulong*)lv = u.v2.lo;
		break;

	case 7:	/* vlong */
	case 8:	/* uvlong */
		fn(&u, *(Vlong*)lv, rv);
		*(Vlong*)lv = u;
		break;
	}
	*ret = u;
}

void
_p2v(Vlong *ret, void *p)
{
	long t;

	t = (ulong)p;
	ret->v2.lo = t;
	ret->v2.hi = 0;
}

void
_sl2v(Vlong *ret, long sl)
{
	long t;

	t = sl;
	ret->v2.lo = t;
	ret->v2.hi = t >> 31;
}

void
_ul2v(Vlong *ret, ulong ul)
{
	long t;

	t = ul;
	ret->v2.lo = t;
	ret->v2.hi = 0;
}

void
_si2v(Vlong *ret, int si)
{
	long t;

	t = si;
	ret->v2.lo = t;
	ret->v2.hi = t >> 31;
}

void
_ui2v(Vlong *ret, uint ui)
{
	long t;

	t = ui;
	ret->v2.lo = t;
	ret->v2.hi = 0;
}

void
_sh2v(Vlong *ret, long sh)
{
	long t;

	t = (sh << 16) >> 16;
	ret->v2.lo = t;
	ret->v2.hi = t >> 31;
}

void
_uh2v(Vlong *ret, ulong ul)
{
	long t;

	t = ul & 0xffff;
	ret->v2.lo = t;
	ret->v2.hi = 0;
}

void
_sc2v(Vlong *ret, long uc)
{
	long t;

	t = (uc << 24) >> 24;
	ret->v2.lo = t;
	ret->v2.hi = t >> 31;
}

void
_uc2v(Vlong *ret, ulong ul)
{
	long t;

	t = ul & 0xff;
	ret->v2.lo = t;
	ret->v2.hi = 0;
}

long
_v2sc(Vlong rv)
{
	long t;

	t = rv.v2.lo & 0xff;
	return (t << 24) >> 24;
}

long
_v2uc(Vlong rv)
{

	return rv.v2.lo & 0xff;
}

long
_v2sh(Vlong rv)
{
	long t;

	t = rv.v2.lo & 0xffff;
	return (t << 16) >> 16;
}

long
_v2uh(Vlong rv)
{

	return rv.v2.lo & 0xffff;
}

long
_v2sl(Vlong rv)
{

	return rv.v2.lo;
}

long
_v2ul(Vlong rv)
{

	return rv.v2.lo;
}

long
_v2si(Vlong rv)
{

	return rv.v2.lo;
}

long
_v2ui(Vlong rv)
{

	return rv.v2.lo;
}

int
_testv(Vlong rv)
{
	return rv.v2.lo || rv.v2.hi;
}

int
_eqv(Vlong lv, Vlong rv)
{
	return lv.v2.lo == rv.v2.lo && lv.v2.hi == rv.v2.hi;
}

int
_nev(Vlong lv, Vlong rv)
{
	return lv.v2.lo != rv.v2.lo || lv.v2.hi != rv.v2.hi;
}

int
_ltv(Vlong lv, Vlong rv)
{
	return (long)lv.v2.hi < (long)rv.v2.hi ||
		(lv.v2.hi == rv.v2.hi && lv.v2.lo < rv.v2.lo);
}

int
_lev(Vlong lv, Vlong rv)
{
	return (long)lv.v2.hi < (long)rv.v2.hi ||
		(lv.v2.hi == rv.v2.hi && lv.v2.lo <= rv.v2.lo);
}

int
_gtv(Vlong lv, Vlong rv)
{
	return (long)lv.v2.hi > (long)rv.v2.hi ||
		(lv.v2.hi == rv.v2.hi && lv.v2.lo > rv.v2.lo);
}

int
_gev(Vlong lv, Vlong rv)
{
	return (long)lv.v2.hi > (long)rv.v2.hi ||
		(lv.v2.hi == rv.v2.hi && lv.v2.lo >= rv.v2.lo);
}

int
_lov(Vlong lv, Vlong rv)
{
	return lv.v2.hi < rv.v2.hi ||
		(lv.v2.hi == rv.v2.hi && lv.v2.lo < rv.v2.lo);
}

int
_lsv(Vlong lv, Vlong rv)
{
	return lv.v2.hi < rv.v2.hi ||
		(lv.v2.hi == rv.v2.hi && lv.v2.lo <= rv.v2.lo);
}

int
_hiv(Vlong lv, Vlong rv)
{
	return lv.v2.hi > rv.v2.hi ||
		(lv.v2.hi == rv.v2.hi && lv.v2.lo > rv.v2.lo);
}

int
_hsv(Vlong lv, Vlong rv)
{
	return lv.v2.hi > rv.v2.hi ||
		(lv.v2.hi == rv.v2.hi && lv.v2.lo >= rv.v2.lo);
}
