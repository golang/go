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

/*
 * C runtime for 64-bit divide, others.
 *
 * TODO(rsc): The simple functions are dregs--8c knows how
 * to generate the code directly now.  Find and remove.
 */

typedef	unsigned long	ulong;
typedef	unsigned int	uint;
typedef	unsigned short	ushort;
typedef	unsigned char	uchar;
typedef	signed char	schar;

#define	SIGN(n)	(1UL<<(n-1))

typedef	struct	Vlong	Vlong;
struct	Vlong
{
	union
	{
		long long	v;
		struct
		{
			ulong	lo;
			ulong	hi;
		};
		struct
		{
			ushort	lols;
			ushort	loms;
			ushort	hils;
			ushort	hims;
		};
	};
};

void	runtime·abort(void);

void
_d2v(Vlong *y, double d)
{
	union { double d; struct Vlong; } x;
	ulong xhi, xlo, ylo, yhi;
	int sh;

	x.d = d;

	xhi = (x.hi & 0xfffff) | 0x100000;
	xlo = x.lo;
	sh = 1075 - ((x.hi >> 20) & 0x7ff);

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
	if(x.hi & SIGN(32)) {
		if(ylo != 0) {
			ylo = -ylo;
			yhi = ~yhi;
		} else
			yhi = -yhi;
	}

	y->hi = yhi;
	y->lo = ylo;
}

void
_f2v(Vlong *y, float f)
{

	_d2v(y, f);
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
		return -((long)x.hi*4294967296. + x.lo);
	}
	return (long)x.hi*4294967296. + x.lo;
}

float
_v2f(Vlong x)
{
	return _v2d(x);
}

ulong	_div64by32(Vlong, ulong, ulong*);
void	_mul64by32(Vlong*, Vlong, ulong);

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
		q->lo = quolo;
		q->hi = quohi;
	}
	if(r) {
		r->lo = numlo;
		r->hi = numhi;
	}
}

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
		_mul64by32(&x, den, n);
		if(x.hi > num.hi || (x.hi == num.hi && x.lo > num.lo))
			slowdodiv(num, den, &q, &r);
		else {
			q.lo = n;
			r.v = num.v - x.v;
		}
	} else {
		if(num.hi >= den.lo){
			q.hi = n = num.hi/den.lo;
			num.hi -= den.lo*n;
		} else {
			q.hi = 0;
		}
		q.lo = _div64by32(num, den.lo, &r.lo);
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

void
_divvu(Vlong *q, Vlong n, Vlong d)
{

	if(n.hi == 0 && d.hi == 0) {
		q->hi = 0;
		q->lo = n.lo / d.lo;
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

	if(n.hi == 0 && d.hi == 0) {
		r->hi = 0;
		r->lo = n.lo % d.lo;
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

	if(v->lo == 0) {
		v->hi = -v->hi;
		return;
	}
	v->lo = -v->lo;
	v->hi = ~v->hi;
}

void
_divv(Vlong *q, Vlong n, Vlong d)
{
	long nneg, dneg;

	if(n.hi == (((long)n.lo)>>31) && d.hi == (((long)d.lo)>>31)) {
		if((long)n.lo == -0x80000000 && (long)d.lo == -1) {
			// special case: 32-bit -0x80000000 / -1 causes divide error,
			// but it's okay in this 64-bit context.
			q->lo = 0x80000000;
			q->hi = 0;
			return;
		}
		q->lo = (long)n.lo / (long)d.lo;
		q->hi = ((long)q->lo) >> 31;
		return;
	}
	nneg = n.hi >> 31;
	if(nneg)
		vneg(&n);
	dneg = d.hi >> 31;
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

	if(n.hi == (((long)n.lo)>>31) && d.hi == (((long)d.lo)>>31)) {
		if((long)n.lo == -0x80000000 && (long)d.lo == -1) {
			// special case: 32-bit -0x80000000 % -1 causes divide error,
			// but it's okay in this 64-bit context.
			r->lo = 0;
			r->hi = 0;
			return;
		}
		r->lo = (long)n.lo % (long)d.lo;
		r->hi = ((long)r->lo) >> 31;
		return;
	}
	nneg = n.hi >> 31;
	if(nneg)
		vneg(&n);
	dneg = d.hi >> 31;
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

	t = a.hi;
	if(b >= 32) {
		r->hi = t>>31;
		if(b >= 64) {
			/* this is illegal re C standard */
			r->lo = t>>31;
			return;
		}
		r->lo = t >> (b-32);
		return;
	}
	if(b <= 0) {
		r->hi = t;
		r->lo = a.lo;
		return;
	}
	r->hi = t >> b;
	r->lo = (t << (32-b)) | (a.lo >> b);
}

void
_rshlv(Vlong *r, Vlong a, int b)
{
	ulong t;

	t = a.hi;
	if(b >= 32) {
		r->hi = 0;
		if(b >= 64) {
			/* this is illegal re C standard */
			r->lo = 0;
			return;
		}
		r->lo = t >> (b-32);
		return;
	}
	if(b <= 0) {
		r->hi = t;
		r->lo = a.lo;
		return;
	}
	r->hi = t >> b;
	r->lo = (t << (32-b)) | (a.lo >> b);
}

void
_lshv(Vlong *r, Vlong a, int b)
{
	ulong t;

	t = a.lo;
	if(b >= 32) {
		r->lo = 0;
		if(b >= 64) {
			/* this is illegal re C standard */
			r->hi = 0;
			return;
		}
		r->hi = t << (b-32);
		return;
	}
	if(b <= 0) {
		r->lo = t;
		r->hi = a.hi;
		return;
	}
	r->lo = t << b;
	r->hi = (t >> (32-b)) | (a.hi << b);
}

void
_andv(Vlong *r, Vlong a, Vlong b)
{
	r->hi = a.hi & b.hi;
	r->lo = a.lo & b.lo;
}

void
_orv(Vlong *r, Vlong a, Vlong b)
{
	r->hi = a.hi | b.hi;
	r->lo = a.lo | b.lo;
}

void
_xorv(Vlong *r, Vlong a, Vlong b)
{
	r->hi = a.hi ^ b.hi;
	r->lo = a.lo ^ b.lo;
}

void
_vpp(Vlong *l, Vlong *r)
{

	l->hi = r->hi;
	l->lo = r->lo;
	r->lo++;
	if(r->lo == 0)
		r->hi++;
}

void
_vmm(Vlong *l, Vlong *r)
{

	l->hi = r->hi;
	l->lo = r->lo;
	if(r->lo == 0)
		r->hi--;
	r->lo--;
}

void
_ppv(Vlong *l, Vlong *r)
{

	r->lo++;
	if(r->lo == 0)
		r->hi++;
	l->hi = r->hi;
	l->lo = r->lo;
}

void
_mmv(Vlong *l, Vlong *r)
{

	if(r->lo == 0)
		r->hi--;
	r->lo--;
	l->hi = r->hi;
	l->lo = r->lo;
}

void
_vasop(Vlong *ret, void *lv, void fn(Vlong*, Vlong, Vlong), int type, Vlong rv)
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
		fn(&u, t, rv);
		*(schar*)lv = u.lo;
		break;

	case 2:	/* uchar */
		t.lo = *(uchar*)lv;
		t.hi = 0;
		fn(&u, t, rv);
		*(uchar*)lv = u.lo;
		break;

	case 3:	/* short */
		t.lo = *(short*)lv;
		t.hi = t.lo >> 31;
		fn(&u, t, rv);
		*(short*)lv = u.lo;
		break;

	case 4:	/* ushort */
		t.lo = *(ushort*)lv;
		t.hi = 0;
		fn(&u, t, rv);
		*(ushort*)lv = u.lo;
		break;

	case 9:	/* int */
		t.lo = *(int*)lv;
		t.hi = t.lo >> 31;
		fn(&u, t, rv);
		*(int*)lv = u.lo;
		break;

	case 10:	/* uint */
		t.lo = *(uint*)lv;
		t.hi = 0;
		fn(&u, t, rv);
		*(uint*)lv = u.lo;
		break;

	case 5:	/* long */
		t.lo = *(long*)lv;
		t.hi = t.lo >> 31;
		fn(&u, t, rv);
		*(long*)lv = u.lo;
		break;

	case 6:	/* ulong */
		t.lo = *(ulong*)lv;
		t.hi = 0;
		fn(&u, t, rv);
		*(ulong*)lv = u.lo;
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
	ret->lo = t;
	ret->hi = 0;
}

void
_sl2v(Vlong *ret, long sl)
{
	long t;

	t = sl;
	ret->lo = t;
	ret->hi = t >> 31;
}

void
_ul2v(Vlong *ret, ulong ul)
{
	long t;

	t = ul;
	ret->lo = t;
	ret->hi = 0;
}

void
_si2v(Vlong *ret, int si)
{
	long t;

	t = si;
	ret->lo = t;
	ret->hi = t >> 31;
}

void
_ui2v(Vlong *ret, uint ui)
{
	long t;

	t = ui;
	ret->lo = t;
	ret->hi = 0;
}

void
_sh2v(Vlong *ret, long sh)
{
	long t;

	t = (sh << 16) >> 16;
	ret->lo = t;
	ret->hi = t >> 31;
}

void
_uh2v(Vlong *ret, ulong ul)
{
	long t;

	t = ul & 0xffff;
	ret->lo = t;
	ret->hi = 0;
}

void
_sc2v(Vlong *ret, long uc)
{
	long t;

	t = (uc << 24) >> 24;
	ret->lo = t;
	ret->hi = t >> 31;
}

void
_uc2v(Vlong *ret, ulong ul)
{
	long t;

	t = ul & 0xff;
	ret->lo = t;
	ret->hi = 0;
}

long
_v2sc(Vlong rv)
{
	long t;

	t = rv.lo & 0xff;
	return (t << 24) >> 24;
}

long
_v2uc(Vlong rv)
{

	return rv.lo & 0xff;
}

long
_v2sh(Vlong rv)
{
	long t;

	t = rv.lo & 0xffff;
	return (t << 16) >> 16;
}

long
_v2uh(Vlong rv)
{

	return rv.lo & 0xffff;
}

long
_v2sl(Vlong rv)
{

	return rv.lo;
}

long
_v2ul(Vlong rv)
{

	return rv.lo;
}

long
_v2si(Vlong rv)
{

	return rv.lo;
}

long
_v2ui(Vlong rv)
{

	return rv.lo;
}

int
_testv(Vlong rv)
{
	return rv.lo || rv.hi;
}

int
_eqv(Vlong lv, Vlong rv)
{
	return lv.lo == rv.lo && lv.hi == rv.hi;
}

int
_nev(Vlong lv, Vlong rv)
{
	return lv.lo != rv.lo || lv.hi != rv.hi;
}

int
_ltv(Vlong lv, Vlong rv)
{
	return (long)lv.hi < (long)rv.hi ||
		(lv.hi == rv.hi && lv.lo < rv.lo);
}

int
_lev(Vlong lv, Vlong rv)
{
	return (long)lv.hi < (long)rv.hi ||
		(lv.hi == rv.hi && lv.lo <= rv.lo);
}

int
_gtv(Vlong lv, Vlong rv)
{
	return (long)lv.hi > (long)rv.hi ||
		(lv.hi == rv.hi && lv.lo > rv.lo);
}

int
_gev(Vlong lv, Vlong rv)
{
	return (long)lv.hi > (long)rv.hi ||
		(lv.hi == rv.hi && lv.lo >= rv.lo);
}

int
_lov(Vlong lv, Vlong rv)
{
	return lv.hi < rv.hi ||
		(lv.hi == rv.hi && lv.lo < rv.lo);
}

int
_lsv(Vlong lv, Vlong rv)
{
	return lv.hi < rv.hi ||
		(lv.hi == rv.hi && lv.lo <= rv.lo);
}

int
_hiv(Vlong lv, Vlong rv)
{
	return lv.hi > rv.hi ||
		(lv.hi == rv.hi && lv.lo > rv.lo);
}

int
_hsv(Vlong lv, Vlong rv)
{
	return lv.hi > rv.hi ||
		(lv.hi == rv.hi && lv.lo >= rv.lo);
}
