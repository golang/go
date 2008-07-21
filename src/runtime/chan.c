// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

static	int32	debug	= 0;

typedef	struct	Hchan	Hchan;
typedef	struct	Link	Link;
typedef	struct	WaitQ	WaitQ;
typedef	struct	SudoG	SudoG;

struct	SudoG
{
	G*	g;		// g and selgen constitute
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
	uint32	eo;			// vararg of element
	uint32	po;			// vararg of present bool
	Alg*	elemalg;		// interface for element type
	Link*	senddataq;		// pointer for sender
	Link*	recvdataq;		// pointer for receiver
	WaitQ	recvq;			// list of recv waiters
	WaitQ	sendq;			// list of send waiters
	SudoG*	free;			// freelist
};

struct	Link
{
	Link*	link;
	byte	elem[8];
};

static SudoG*	dequeue(WaitQ*, Hchan*);
static void	enqueue(WaitQ*, SudoG*);
static SudoG*	allocsg(Hchan*);
static void	freesg(Hchan*, SudoG*);

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
		gr = sgr->g;
		freesg(c, sgr);

		c->elemalg->copy(c->elemsize, gr->elem, ae);
		gr->status = Grunnable;
		return;
	}

	c->elemalg->copy(c->elemsize, g->elem, ae);
	sgr = allocsg(c);
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
		freesg(c, sgr);

		c->elemalg->copy(c->elemsize, gr->elem, ae);
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
		gs = sgs->g;
		freesg(c, sgs);

		c->elemalg->copy(c->elemsize, ae, gs->elem);
		gs->status = Grunnable;
		return;
	}
	sgs = allocsg(c);
	g->status = Gwaiting;
	enqueue(&c->recvq, sgs);
	sys·gosched();
	c->elemalg->copy(c->elemsize, ae, g->elem);
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
		gs = sgs->g;
		freesg(c, sgs);

		c->elemalg->copy(c->elemsize, ae, gs->elem);
		gs->status = Grunnable;
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
