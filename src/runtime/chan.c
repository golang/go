// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

static	int32	debug	= 0;

typedef	struct	Hchan	Hchan;
typedef	struct	Link	Link;
typedef	struct	WaitQ	WaitQ;
typedef	struct	SudoG	SudoG;
typedef	struct	Select	Select;
typedef	struct	Scase	Scase;

struct	SudoG
{
	G*	g;		// g and selgen constitute
	byte	elem[8];	// synch data element
	int64	selgen;		// a weak pointer to g
	SudoG*	link;
};

struct	WaitQ
{
	SudoG*	first;
	SudoG*	last;
};

struct	Hchan
{
	uint32	elemsize;
	uint32	dataqsiz;		// size of the circular q
	uint32	qcount;			// total data in the q
	uint16	eo;			// vararg of element
	uint16	po;			// vararg of present bool
	Alg*	elemalg;		// interface for element type
	Link*	senddataq;		// pointer for sender
	Link*	recvdataq;		// pointer for receiver
	WaitQ	recvq;			// list of recv waiters
	WaitQ	sendq;			// list of send waiters
	SudoG*	free;			// freelist
};

struct	Link
{
	Link*	link;			// asynch queue circular linked list
	byte	elem[8];		// asynch queue data element
};

struct	Scase
{
	Hchan*	chan;			// chan
	byte*	pc;			// return pc
	uint16	send;			// 0-recv 1-send
	uint16	so;			// vararg of selected bool
	union {
		byte	elem[8];	// element (send)
		byte*	elemp;		// pointer to element (recv)
	} u;
};

struct	Select
{
	uint16	tcase;			// total count of scase[]
	uint16	ncase;			// currently filled scase[]
	Scase	scase[1];		// one per case
};

static	SudoG*	dequeue(WaitQ*, Hchan*);
static	void	enqueue(WaitQ*, SudoG*);
static	SudoG*	allocsg(Hchan*);
static	void	freesg(Hchan*, SudoG*);
static	uint32	gcd(uint32, uint32);
static	uint32	fastrand1(void);
static	uint32	fastrand2(void);

// newchan(elemsize uint32, elemalg uint32, hint uint32) (hchan *chan any);
void
sys·newchan(uint32 elemsize, uint32 elemalg, uint32 hint,
	Hchan* ret)
{
	Hchan *c;
	int32 i;

	if(elemalg >= nelem(algarray)) {
		prints("0<=");
		sys·printint(elemalg);
		prints("<");
		sys·printint(nelem(algarray));
		prints("\n");

		throw("sys·newchan: elem algorithm out of range");
	}

	c = mal(sizeof(*c));

	c->elemsize = elemsize;
	c->elemalg = &algarray[elemalg];

	if(hint > 0) {
		Link *d, *b, *e;

		// make a circular q
		b = nil;
		e = nil;
		for(i=0; i<hint; i++) {
			d = mal(sizeof(*d));
			if(e == nil)
				e = d;
			d->link = b;
			b = d;
		}
		e->link = b;
		c->recvdataq = b;
		c->senddataq = b;
		c->qcount = 0;
		c->dataqsiz = hint;
	}

	// these calculations are compiler dependent
	c->eo = rnd(sizeof(c), elemsize);
	c->po = rnd(c->eo+elemsize, 1);

	ret = c;
	FLUSH(&ret);

	if(debug) {
		prints("newchan: chan=");
		sys·printpointer(c);
		prints("; elemsize=");
		sys·printint(elemsize);
		prints("; elemalg=");
		sys·printint(elemalg);
		prints("; dataqsiz=");
		sys·printint(c->dataqsiz);
		prints("\n");
	}
}

// chansend1(hchan *chan any, elem any);
void
sys·chansend1(Hchan* c, ...)
{
	byte *ae;
	SudoG *sgr;
	G* gr;

	ae = (byte*)&c + c->eo;
	if(debug) {
		prints("chansend: chan=");
		sys·printpointer(c);
		prints("; elem=");
		c->elemalg->print(c->elemsize, ae);
		prints("\n");
	}
	if(c->dataqsiz > 0)
		goto asynch;

	sgr = dequeue(&c->recvq, c);
	if(sgr != nil) {
		c->elemalg->copy(c->elemsize, sgr->elem, ae);

		gr = sgr->g;
		gr->status = Grunnable;
		return;
	}

	sgr = allocsg(c);
	c->elemalg->copy(c->elemsize, sgr->elem, ae);
	g->status = Gwaiting;
	enqueue(&c->sendq, sgr);
	sys·gosched();
	return;

asynch:
	while(c->qcount >= c->dataqsiz) {
		sgr = allocsg(c);
		g->status = Gwaiting;
		enqueue(&c->sendq, sgr);
		sys·gosched();
	}
	c->elemalg->copy(c->elemsize, c->senddataq->elem, ae);
	c->senddataq = c->senddataq->link;
	c->qcount++;
	sgr = dequeue(&c->recvq, c);
	if(sgr != nil) {
		gr = sgr->g;
		freesg(c, sgr);
		gr->status = Grunnable;
	}
}

// chansend2(hchan *chan any, elem any) (pres bool);
void
sys·chansend2(Hchan* c, ...)
{
	byte *ae, *ap;
	SudoG *sgr;
	G *gr;

	ae = (byte*)&c + c->eo;
	ap = (byte*)&c + c->po;

	if(debug) {
		prints("chansend: chan=");
		sys·printpointer(c);
		prints("; elem=");
		c->elemalg->print(c->elemsize, ae);
		prints("\n");
	}
	if(c->dataqsiz > 0)
		goto asynch;

	sgr = dequeue(&c->recvq, c);
	if(sgr != nil) {
		gr = sgr->g;
		c->elemalg->copy(c->elemsize, sgr->elem, ae);

		gr->status = Grunnable;
		*ap = true;
		return;
	}
	*ap = false;
	return;

asynch:
	if(c->qcount >= c->dataqsiz) {
		*ap = false;
		return;
	}
	c->elemalg->copy(c->elemsize, c->senddataq->elem, ae);
	c->senddataq = c->senddataq->link;
	c->qcount++;
	sgr = dequeue(&c->recvq, c);
	if(gr != nil) {
		gr = sgr->g;
		freesg(c, sgr);
		gr->status = Grunnable;
	}
	*ap = true;
}

// chanrecv1(hchan *chan any) (elem any);
void
sys·chanrecv1(Hchan* c, ...)
{
	byte *ae;
	SudoG *sgs;
	G *gs;

	ae = (byte*)&c + c->eo;
	if(debug) {
		prints("chanrecv1: chan=");
		sys·printpointer(c);
		prints("\n");
	}
	if(c->dataqsiz > 0)
		goto asynch;

	sgs = dequeue(&c->sendq, c);
	if(sgs != nil) {
		c->elemalg->copy(c->elemsize, ae, sgs->elem);

		gs = sgs->g;
		gs->status = Grunnable;

		freesg(c, sgs);
		return;
	}
	sgs = allocsg(c);
	g->status = Gwaiting;
	enqueue(&c->recvq, sgs);
	sys·gosched();
	c->elemalg->copy(c->elemsize, ae, sgs->elem);
	freesg(c, sgs);
	return;

asynch:
	while(c->qcount <= 0) {
		sgs = allocsg(c);
		g->status = Gwaiting;
		enqueue(&c->recvq, sgs);
		sys·gosched();
	}
	c->elemalg->copy(c->elemsize, ae, c->recvdataq->elem);
	c->recvdataq = c->recvdataq->link;
	c->qcount--;
	sgs = dequeue(&c->sendq, c);
	if(gs != nil) {
		gs = sgs->g;
		freesg(c, sgs);

		gs->status = Grunnable;
	}
}

// chanrecv2(hchan *chan any) (elem any, pres bool);
void
sys·chanrecv2(Hchan* c, ...)
{
	byte *ae, *ap;
	SudoG *sgs;
	G *gs;

	ae = (byte*)&c + c->eo;
	ap = (byte*)&c + c->po;

	if(debug) {
		prints("chanrecv2: chan=");
		sys·printpointer(c);
		prints("\n");
	}
	if(c->dataqsiz > 0)
		goto asynch;

	sgs = dequeue(&c->sendq, c);
	if(sgs != nil) {
		c->elemalg->copy(c->elemsize, ae, sgs->elem);

		gs = sgs->g;
		gs->status = Grunnable;

		freesg(c, sgs);
		*ap = true;
		return;
	}
	*ap = false;
	return;

asynch:
	if(c->qcount <= 0) {
		*ap = false;
		return;
	}
	c->elemalg->copy(c->elemsize, ae, c->recvdataq->elem);
	c->recvdataq = c->recvdataq->link;
	c->qcount--;
	sgs = dequeue(&c->sendq, c);
	if(sgs != nil) {
		gs = sgs->g;
		freesg(c, sgs);

		gs->status = Grunnable;
	}
	*ap = true;
}

// newselect(size uint32) (sel *byte);
void
sys·newselect(int32 size, Select *sel)
{
	int32 n;

	n = 0;
	if(size > 1)
		n = size-1;
	sel = mal(sizeof(*sel) + n*sizeof(sel->scase[0]));
	sel->tcase = size;
	sel->ncase = 0;
	FLUSH(&sel);
	if(debug) {
		prints("newselect s=");
		sys·printpointer(sel);
		prints("\n");
	}
}

// selectsend(sel *byte, hchan *chan any, elem any) (selected bool);
void
sys·selectsend(Select *sel, Hchan *c, ...)
{
	int32 i, eo;
	Scase *cas;
	byte *as, *ae;

	// return val, selected, is preset to false
	if(c == nil)
		return;

	i = sel->ncase;
	if(i >= sel->tcase)
		throw("selectsend: too many cases");
	sel->ncase = i+1;
	cas = &sel->scase[i];

	cas->pc = sys·getcallerpc(&sel);
	cas->chan = c;

	eo = rnd(sizeof(sel), sizeof(c));
	eo = rnd(eo+sizeof(c), c->elemsize);
	cas->so = rnd(eo+c->elemsize, 1);
	cas->send = 1;

	ae = (byte*)&sel + eo;
	c->elemalg->copy(c->elemsize, cas->u.elem, ae);

	as = (byte*)&sel + cas->so;
	*as = false;

	if(debug) {
		prints("newselect s=");
		sys·printpointer(sel);
		prints(" pc=");
		sys·printpointer(cas->pc);
		prints(" chan=");
		sys·printpointer(cas->chan);
		prints(" po=");
		sys·printint(cas->so);
		prints(" send=");
		sys·printint(cas->send);
		prints("\n");
	}
}

// selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
void
sys·selectrecv(Select *sel, Hchan *c, ...)
{
	int32 i, epo;
	Scase *cas;
	byte *as;

	// return val, selected, is preset to false
	if(c == nil)
		return;

	i = sel->ncase;
	if(i >= sel->tcase)
		throw("selectsend: too many cases");
	sel->ncase = i+1;
	cas = &sel->scase[i];

	cas->pc = sys·getcallerpc(&sel);
	cas->chan = c;

	epo = rnd(sizeof(sel), sizeof(c));
	epo = rnd(epo+sizeof(c), sizeof(byte*));
	cas->so = rnd(epo+sizeof(byte*), 1);
	cas->send = 0;
	cas->u.elemp = *(byte**)((byte*)&sel + epo);

	as = (byte*)&sel + cas->so;
	*as = false;

	if(debug) {
		prints("newselect s=");
		sys·printpointer(sel);
		prints(" pc=");
		sys·printpointer(cas->pc);
		prints(" chan=");
		sys·printpointer(cas->chan);
		prints(" so=");
		sys·printint(cas->so);
		prints(" send=");
		sys·printint(cas->send);
		prints("\n");
	}
}

// selectgo(sel *byte);
void
sys·selectgo(Select *sel)
{
	uint32 p, o, i;
	Scase *cas;
	Hchan *c;

	byte *ae, *as;
	SudoG *sgr;
	G *gr;

	SudoG *sgs;
	G *gs;

	if(sel->ncase < 1) {
		throw("selectgo: no cases");
	}

	// select a (relative) prime
	for(i=0;; i++) {
		p = fastrand1();
		if(gcd(p, sel->ncase) == 1)
			break;
		if(i > 1000) {
			throw("selectgo: failed to select prime");
		}
	}
	o = fastrand2();

	p %= sel->ncase;
	o %= sel->ncase;

	// pass 1 - look for something that can go
	for(i=0; i<sel->ncase; i++) {
		cas = &sel->scase[o];
		c = cas->chan;
		if(cas->send) {
			if(c->dataqsiz > 0) {
				throw("selectgo: send asynch");
			}
			sgr = dequeue(&c->recvq, c);
			if(sgr == nil)
				continue;

			c->elemalg->copy(c->elemsize, sgr->elem, cas->u.elem);
			gr = sgr->g;
			gr->status = Grunnable;

			goto retc;
		} else {
			if(c->dataqsiz > 0) {
				throw("selectgo: recv asynch");
			}
			sgs = dequeue(&c->sendq, c);
			if(sgs == nil)
				continue;

			if(cas->u.elemp != nil)
				c->elemalg->copy(c->elemsize, cas->u.elemp, sgs->elem);

			gs = sgs->g;
			gs->status = Grunnable;

			freesg(c, sgs);

			goto retc;
		}

		o += p;
		if(o >= sel->ncase)
			o -= sel->ncase;
	}

	if(debug) {
		prints("selectgo s=");
		sys·printpointer(sel);
		prints(" p=");
		sys·printpointer((void*)p);
		prints("\n");
	}

	throw("selectgo");

retc:
	sys·setcallerpc(&sel, cas->pc);
	as = (byte*)&sel + cas->so;
	*as = true;
}

static SudoG*
dequeue(WaitQ *q, Hchan *c)
{
	SudoG *sgp;

loop:
	sgp = q->first;
	if(sgp == nil)
		return nil;
	q->first = sgp->link;

	// if sgp is stale, ignore it
	if(sgp->selgen != sgp->g->selgen) {
prints("INVALID PSEUDOG POINTER\n");
		freesg(c, sgp);
		goto loop;
	}

	// invalidate any others
	sgp->g->selgen++;
	return sgp;
}

static void
enqueue(WaitQ *q, SudoG *sgp)
{
	sgp->link = nil;
	if(q->first == nil) {
		q->first = sgp;
		q->last = sgp;
		return;
	}
	q->last->link = sgp;
	q->last = sgp;
}

static SudoG*
allocsg(Hchan *c)
{
	SudoG* sg;

	sg = c->free;
	if(sg != nil) {
		c->free = sg->link;
	} else
		sg = mal(sizeof(*sg));
	sg->selgen = g->selgen;
	sg->g = g;
	return sg;
}

static void
freesg(Hchan *c, SudoG *sg)
{
	sg->link = c->free;
	c->free = sg;
}

static uint32
gcd(uint32 u, uint32 v)
{
	for(;;) {
		if(u > v) {
			if(v == 0)
				return u;
			u = u%v;
			continue;
		}
		if(u == 0)
			return v;
		v = v%u;
	}
}

static uint32
fastrand1(void)
{
	static uint32 x = 0x49f6428aUL;

	x += x;
	if(x & 0x80000000L)
		x ^= 0x88888eefUL;
	return x;
}

static uint32
fastrand2(void)
{
	static uint32 x = 0x49f6428aUL;

	x += x;
	if(x & 0x80000000L)
		x ^= 0xfafd871bUL;
	return x;
}
