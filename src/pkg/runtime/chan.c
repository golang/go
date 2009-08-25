// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

static	int32	debug	= 0;
static	Lock		chanlock;

enum
{
	Wclosed		= 0x0001,	// writer has closed
	Rclosed		= 0x0002,	// reader has seen close
	Eincr		= 0x0004,	// increment errors
	Emax		= 0x0800,	// error limit before throw
};

typedef	struct	Link	Link;
typedef	struct	WaitQ	WaitQ;
typedef	struct	SudoG	SudoG;
typedef	struct	Select	Select;
typedef	struct	Scase	Scase;

struct	SudoG
{
	G*	g;		// g and selgen constitute
	int32	selgen;		// a weak pointer to g
	int16	offset;		// offset of case number
	int8	isfree;		// offset of case number
	SudoG*	link;
	byte	elem[8];	// synch data element (+ more)
};

struct	WaitQ
{
	SudoG*	first;
	SudoG*	last;
};

struct	Hchan
{
	uint32	qcount;			// total data in the q
	uint32	dataqsiz;		// size of the circular q
	uint16	elemsize;
	uint16	closed;			// Wclosed Rclosed errorcount
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
	byte	elem[8];		// asynch queue data element (+ more)
};

struct	Scase
{
	Hchan*	chan;			// chan
	byte*	pc;			// return pc
	uint16	send;			// 0-recv 1-send 2-default
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
	Select*	link;			// for freelist
	Scase*	scase[1];		// one per case
};

static	Select*	selfree[20];

static	SudoG*	dequeue(WaitQ*, Hchan*);
static	void	enqueue(WaitQ*, SudoG*);
static	SudoG*	allocsg(Hchan*);
static	void	freesg(Hchan*, SudoG*);
static	uint32	gcd(uint32, uint32);
static	uint32	fastrand1(void);
static	uint32	fastrand2(void);

Hchan*
makechan(uint32 elemsize, uint32 elemalg, uint32 hint)
{
	Hchan *c;
	int32 i;

	if(elemalg >= nelem(algarray)) {
		printf("chan(alg=%d)\n", elemalg);
		throw("sys·makechan: unsupported elem type");
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
			d = mal(sizeof(*d) + c->elemsize - sizeof(d->elem));
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

	if(debug) {
		prints("makechan: chan=");
		sys·printpointer(c);
		prints("; elemsize=");
		sys·printint(elemsize);
		prints("; elemalg=");
		sys·printint(elemalg);
		prints("; dataqsiz=");
		sys·printint(c->dataqsiz);
		prints("\n");
	}

	return c;
}

// makechan(elemsize uint32, elemalg uint32, hint uint32) (hchan *chan any);
void
sys·makechan(uint32 elemsize, uint32 elemalg, uint32 hint, Hchan *ret)
{
	ret = makechan(elemsize, elemalg, hint);
	FLUSH(&ret);
}

static void
incerr(Hchan* c)
{
	c->closed += Eincr;
	if(c->closed & Emax) {
		unlock(&chanlock);
		throw("too many operations on a closed channel");
	}
}

/*
 * generic single channel send/recv
 * if the bool pointer is nil,
 * then the full exchange will
 * occur. if pres is not nil,
 * then the protocol will not
 * sleep but return if it could
 * not complete
 */
void
chansend(Hchan *c, byte *ep, bool *pres)
{
	SudoG *sg;
	G* gp;

	if(debug) {
		prints("chansend: chan=");
		sys·printpointer(c);
		prints("; elem=");
		c->elemalg->print(c->elemsize, ep);
		prints("\n");
	}

	lock(&chanlock);
loop:
	if(c->closed & Wclosed)
		goto closed;

	if(c->dataqsiz > 0)
		goto asynch;

	sg = dequeue(&c->recvq, c);
	if(sg != nil) {
		if(ep != nil)
			c->elemalg->copy(c->elemsize, sg->elem, ep);

		gp = sg->g;
		gp->param = sg;
		unlock(&chanlock);
		ready(gp);

		if(pres != nil)
			*pres = true;
		return;
	}

	if(pres != nil) {
		unlock(&chanlock);
		*pres = false;
		return;
	}

	sg = allocsg(c);
	if(ep != nil)
		c->elemalg->copy(c->elemsize, sg->elem, ep);
	g->param = nil;
	g->status = Gwaiting;
	enqueue(&c->sendq, sg);
	unlock(&chanlock);
	gosched();

	lock(&chanlock);
	sg = g->param;
	if(sg == nil)
		goto loop;
	freesg(c, sg);
	unlock(&chanlock);
	return;

asynch:
	if(c->closed & Wclosed)
		goto closed;

	if(c->qcount >= c->dataqsiz) {
		if(pres != nil) {
			unlock(&chanlock);
			*pres = false;
			return;
		}
		sg = allocsg(c);
		g->status = Gwaiting;
		enqueue(&c->sendq, sg);
		unlock(&chanlock);
		gosched();

		lock(&chanlock);
		goto asynch;
	}
	if(ep != nil)
		c->elemalg->copy(c->elemsize, c->senddataq->elem, ep);
	c->senddataq = c->senddataq->link;
	c->qcount++;

	sg = dequeue(&c->recvq, c);
	if(sg != nil) {
		gp = sg->g;
		freesg(c, sg);
		unlock(&chanlock);
		ready(gp);
	} else
		unlock(&chanlock);
	if(pres != nil)
		*pres = true;
	return;

closed:
	incerr(c);
	if(pres != nil)
		*pres = true;
	unlock(&chanlock);
}

void
chanrecv(Hchan* c, byte *ep, bool* pres)
{
	SudoG *sg;
	G *gp;

	if(debug) {
		prints("chanrecv: chan=");
		sys·printpointer(c);
		prints("\n");
	}

	lock(&chanlock);
loop:
	if(c->dataqsiz > 0)
		goto asynch;

	if(c->closed & Wclosed)
		goto closed;

	sg = dequeue(&c->sendq, c);
	if(sg != nil) {
		c->elemalg->copy(c->elemsize, ep, sg->elem);

		gp = sg->g;
		gp->param = sg;
		unlock(&chanlock);
		ready(gp);

		if(pres != nil)
			*pres = true;
		return;
	}

	if(pres != nil) {
		unlock(&chanlock);
		*pres = false;
		return;
	}

	sg = allocsg(c);
	g->param = nil;
	g->status = Gwaiting;
	enqueue(&c->recvq, sg);
	unlock(&chanlock);
	gosched();

	lock(&chanlock);
	sg = g->param;
	if(sg == nil)
		goto loop;

	c->elemalg->copy(c->elemsize, ep, sg->elem);
	freesg(c, sg);
	unlock(&chanlock);
	return;

asynch:
	if(c->qcount <= 0) {
		if(c->closed & Wclosed)
			goto closed;

		if(pres != nil) {
			unlock(&chanlock);
			*pres = false;
			return;
		}
		sg = allocsg(c);
		g->status = Gwaiting;
		enqueue(&c->recvq, sg);
		unlock(&chanlock);
		gosched();

		lock(&chanlock);
		goto asynch;
	}
	c->elemalg->copy(c->elemsize, ep, c->recvdataq->elem);
	c->recvdataq = c->recvdataq->link;
	c->qcount--;
	sg = dequeue(&c->sendq, c);
	if(sg != nil) {
		gp = sg->g;
		freesg(c, sg);
		unlock(&chanlock);
		ready(gp);
		if(pres != nil)
			*pres = true;
		return;
	}

	unlock(&chanlock);
	if(pres != nil)
		*pres = true;
	return;

closed:
	c->elemalg->copy(c->elemsize, ep, nil);
	c->closed |= Rclosed;
	incerr(c);
	if(pres != nil)
		*pres = true;
	unlock(&chanlock);
}

// chansend1(hchan *chan any, elem any);
void
sys·chansend1(Hchan* c, ...)
{
	int32 o;
	byte *ae;

	o = rnd(sizeof(c), c->elemsize);
	ae = (byte*)&c + o;
	chansend(c, ae, nil);
}

// chansend2(hchan *chan any, elem any) (pres bool);
void
sys·chansend2(Hchan* c, ...)
{
	int32 o;
	byte *ae, *ap;

	o = rnd(sizeof(c), c->elemsize);
	ae = (byte*)&c + o;
	o = rnd(o+c->elemsize, Structrnd);
	ap = (byte*)&c + o;

	chansend(c, ae, ap);
}

// chanrecv1(hchan *chan any) (elem any);
void
sys·chanrecv1(Hchan* c, ...)
{
	int32 o;
	byte *ae;

	o = rnd(sizeof(c), Structrnd);
	ae = (byte*)&c + o;

	chanrecv(c, ae, nil);
}

// chanrecv2(hchan *chan any) (elem any, pres bool);
void
sys·chanrecv2(Hchan* c, ...)
{
	int32 o;
	byte *ae, *ap;

	o = rnd(sizeof(c), Structrnd);
	ae = (byte*)&c + o;
	o = rnd(o+c->elemsize, 1);
	ap = (byte*)&c + o;

	chanrecv(c, ae, ap);
}

// newselect(size uint32) (sel *byte);
void
sys·newselect(int32 size, ...)
{
	int32 n, o;
	Select **selp;
	Select *sel;

	o = rnd(sizeof(size), Structrnd);
	selp = (Select**)((byte*)&size + o);
	n = 0;
	if(size > 1)
		n = size-1;

	lock(&chanlock);
	sel = nil;
	if(size >= 1 && size < nelem(selfree)) {
		sel = selfree[size];
		if(sel != nil)
			selfree[size] = sel->link;
	}
	unlock(&chanlock);
	if(sel == nil)
		sel = mal(sizeof(*sel) + n*sizeof(sel->scase[0]));

	sel->tcase = size;
	sel->ncase = 0;
	*selp = sel;
	if(debug) {
		prints("newselect s=");
		sys·printpointer(sel);
		prints(" size=");
		sys·printint(size);
		prints("\n");
	}
}

// selectsend(sel *byte, hchan *chan any, elem any) (selected bool);
void
sys·selectsend(Select *sel, Hchan *c, ...)
{
	int32 i, eo;
	Scase *cas;
	byte *ae;

	// nil cases do not compete
	if(c == nil)
		return;

	i = sel->ncase;
	if(i >= sel->tcase)
		throw("selectsend: too many cases");
	sel->ncase = i+1;
	cas = sel->scase[i];
	if(cas == nil) {
		cas = mal(sizeof *cas + c->elemsize - sizeof(cas->u.elem));
		sel->scase[i] = cas;
	}

	cas->pc = sys·getcallerpc(&sel);
	cas->chan = c;

	eo = rnd(sizeof(sel), sizeof(c));
	eo = rnd(eo+sizeof(c), c->elemsize);
	cas->so = rnd(eo+c->elemsize, Structrnd);
	cas->send = 1;

	ae = (byte*)&sel + eo;
	c->elemalg->copy(c->elemsize, cas->u.elem, ae);

	if(debug) {
		prints("selectsend s=");
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
	int32 i, eo;
	Scase *cas;

	// nil cases do not compete
	if(c == nil)
		return;

	i = sel->ncase;
	if(i >= sel->tcase)
		throw("selectrecv: too many cases");
	sel->ncase = i+1;
	cas = sel->scase[i];
	if(cas == nil) {
		cas = mal(sizeof *cas);
		sel->scase[i] = cas;
	}
	cas->pc = sys·getcallerpc(&sel);
	cas->chan = c;

	eo = rnd(sizeof(sel), sizeof(c));
	eo = rnd(eo+sizeof(c), sizeof(byte*));
	cas->so = rnd(eo+sizeof(byte*), Structrnd);
	cas->send = 0;
	cas->u.elemp = *(byte**)((byte*)&sel + eo);

	if(debug) {
		prints("selectrecv s=");
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


// selectdefaul(sel *byte) (selected bool);
void
sys·selectdefault(Select *sel, ...)
{
	int32 i;
	Scase *cas;

	i = sel->ncase;
	if(i >= sel->tcase)
		throw("selectdefault: too many cases");
	sel->ncase = i+1;
	cas = sel->scase[i];
	if(cas == nil) {
		cas = mal(sizeof *cas);
		sel->scase[i] = cas;
	}
	cas->pc = sys·getcallerpc(&sel);
	cas->chan = nil;

	cas->so = rnd(sizeof(sel), Structrnd);
	cas->send = 2;
	cas->u.elemp = nil;

	if(debug) {
		prints("selectdefault s=");
		sys·printpointer(sel);
		prints(" pc=");
		sys·printpointer(cas->pc);
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
	Scase *cas, *dfl;
	Hchan *c;
	SudoG *sg;
	G *gp;
	byte *as;

	if(debug) {
		prints("selectgo: sel=");
		sys·printpointer(sel);
		prints("\n");
	}

	if(sel->ncase < 2) {
		if(sel->ncase < 1)
			throw("selectgo: no cases");
		// make special case of one.
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

	// select an initial offset
	o = fastrand2();

	p %= sel->ncase;
	o %= sel->ncase;

	lock(&chanlock);

loop:
	// pass 1 - look for something already waiting
	dfl = nil;
	for(i=0; i<sel->ncase; i++) {
		cas = sel->scase[o];

		if(cas->send == 2) {	// default
			dfl = cas;
			goto next1;
		}

		c = cas->chan;
		if(c->dataqsiz > 0) {
			if(cas->send) {
				if(c->closed & Wclosed)
					goto sclose;
				if(c->qcount < c->dataqsiz)
					goto asyns;
				goto next1;
			}
			if(c->qcount > 0)
				goto asynr;
			if(c->closed & Wclosed)
				goto rclose;
			goto next1;
		}

		if(cas->send) {
			if(c->closed & Wclosed)
				goto sclose;
			sg = dequeue(&c->recvq, c);
			if(sg != nil)
				goto gots;
			goto next1;
		}
		sg = dequeue(&c->sendq, c);
		if(sg != nil)
			goto gotr;
		if(c->closed & Wclosed)
			goto rclose;

	next1:
		o += p;
		if(o >= sel->ncase)
			o -= sel->ncase;
	}

	if(dfl != nil) {
		cas = dfl;
		goto retc;
	}


	// pass 2 - enqueue on all chans
	for(i=0; i<sel->ncase; i++) {
		cas = sel->scase[o];
		c = cas->chan;

		if(c->dataqsiz > 0) {
			if(cas->send) {
				if(c->qcount < c->dataqsiz) {
					prints("selectgo: pass 2 async send\n");
					goto asyns;
				}
				sg = allocsg(c);
				sg->offset = o;
				enqueue(&c->sendq, sg);
				goto next2;
			}
			if(c->qcount > 0) {
				prints("selectgo: pass 2 async recv\n");
				goto asynr;
			}
			sg = allocsg(c);
			sg->offset = o;
			enqueue(&c->recvq, sg);
			goto next2;
		}

		if(cas->send) {
			sg = dequeue(&c->recvq, c);
			if(sg != nil) {
				prints("selectgo: pass 2 sync send\n");
				g->selgen++;
				goto gots;
			}
			sg = allocsg(c);
			sg->offset = o;
			c->elemalg->copy(c->elemsize, sg->elem, cas->u.elem);
			enqueue(&c->sendq, sg);
			goto next2;
		}
		sg = dequeue(&c->sendq, c);
		if(sg != nil) {
			prints("selectgo: pass 2 sync recv\n");
			g->selgen++;
			goto gotr;
		}
		sg = allocsg(c);
		sg->offset = o;
		enqueue(&c->recvq, sg);

	next2:
		o += p;
		if(o >= sel->ncase)
			o -= sel->ncase;
	}

	g->param = nil;
	g->status = Gwaiting;
	unlock(&chanlock);
	gosched();

	lock(&chanlock);
	sg = g->param;
	if(sg == nil)
		goto loop;

	o = sg->offset;
	cas = sel->scase[o];
	c = cas->chan;

	if(c->dataqsiz > 0) {
//		prints("shouldnt happen\n");
		goto loop;
	}

	if(debug) {
		prints("wait-return: sel=");
		sys·printpointer(sel);
		prints(" c=");
		sys·printpointer(c);
		prints(" cas=");
		sys·printpointer(cas);
		prints(" send=");
		sys·printint(cas->send);
		prints(" o=");
		sys·printint(o);
		prints("\n");
	}

	if(!cas->send) {
		if(cas->u.elemp != nil)
			c->elemalg->copy(c->elemsize, cas->u.elemp, sg->elem);
	}

	freesg(c, sg);
	goto retc;

asynr:
	if(cas->u.elemp != nil)
		c->elemalg->copy(c->elemsize, cas->u.elemp, c->recvdataq->elem);
	c->recvdataq = c->recvdataq->link;
	c->qcount--;
	sg = dequeue(&c->sendq, c);
	if(sg != nil) {
		gp = sg->g;
		freesg(c, sg);
		ready(gp);
	}
	goto retc;

asyns:
	if(cas->u.elem != nil)
		c->elemalg->copy(c->elemsize, c->senddataq->elem, cas->u.elem);
	c->senddataq = c->senddataq->link;
	c->qcount++;
	sg = dequeue(&c->recvq, c);
	if(sg != nil) {
		gp = sg->g;
		freesg(c, sg);
		ready(gp);
	}
	goto retc;

gotr:
	// recv path to wakeup the sender (sg)
	if(debug) {
		prints("gotr: sel=");
		sys·printpointer(sel);
		prints(" c=");
		sys·printpointer(c);
		prints(" o=");
		sys·printint(o);
		prints("\n");
	}
	if(cas->u.elemp != nil)
		c->elemalg->copy(c->elemsize, cas->u.elemp, sg->elem);
	gp = sg->g;
	gp->param = sg;
	ready(gp);
	goto retc;

rclose:
	if(cas->u.elemp != nil)
		c->elemalg->copy(c->elemsize, cas->u.elemp, nil);
	c->closed |= Rclosed;
	incerr(c);
	goto retc;

gots:
	// send path to wakeup the receiver (sg)
	if(debug) {
		prints("gots: sel=");
		sys·printpointer(sel);
		prints(" c=");
		sys·printpointer(c);
		prints(" o=");
		sys·printint(o);
		prints("\n");
	}
	if(c->closed & Wclosed)
		goto sclose;
	c->elemalg->copy(c->elemsize, sg->elem, cas->u.elem);
	gp = sg->g;
	gp->param = sg;
	ready(gp);
	goto retc;

sclose:
	incerr(c);
	goto retc;

retc:
	if(sel->ncase >= 1 && sel->ncase < nelem(selfree)) {
		sel->link = selfree[sel->ncase];
		selfree[sel->ncase] = sel;
	}
	unlock(&chanlock);

	sys·setcallerpc(&sel, cas->pc);
	as = (byte*)&sel + cas->so;
	*as = true;
}

// closechan(sel *byte);
void
sys·closechan(Hchan *c)
{
	SudoG *sg;
	G* gp;

	lock(&chanlock);
	incerr(c);
	c->closed |= Wclosed;

	// release all readers
	for(;;) {
		sg = dequeue(&c->recvq, c);
		if(sg == nil)
			break;
		gp = sg->g;
		gp->param = nil;
		freesg(c, sg);
		ready(gp);
	}

	// release all writers
	for(;;) {
		sg = dequeue(&c->sendq, c);
		if(sg == nil)
			break;
		gp = sg->g;
		gp->param = nil;
		freesg(c, sg);
		ready(gp);
	}

	unlock(&chanlock);
}

// closedchan(sel *byte) bool;
void
sys·closedchan(Hchan *c, bool closed)
{
	// test Rclosed
	closed = 0;
	if(c->closed & Rclosed)
		closed = 1;
	FLUSH(&closed);
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
		//prints("INVALID PSEUDOG POINTER\n");
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
		sg = mal(sizeof(*sg) + c->elemsize - sizeof(sg->elem));
	sg->selgen = g->selgen;
	sg->g = g;
	sg->offset = 0;
	sg->isfree = 0;

	return sg;
}

static void
freesg(Hchan *c, SudoG *sg)
{
	if(sg != nil) {
		if(sg->isfree)
			throw("chan.freesg: already free");
		sg->isfree = 1;
		sg->link = c->free;
		c->free = sg;
	}
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
