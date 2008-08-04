// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

// TODO locking of select

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
	int16	offset;		// offset of case number
	int32	selgen;		// a weak pointer to g
	SudoG*	link;
};

struct	WaitQ
{
	SudoG*	first;
	SudoG*	last;
};

struct	Hchan
{
	Lock;
	uint32	elemsize;
	uint32	dataqsiz;		// size of the circular q
	uint32	qcount;			// total data in the q
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
	Select*	link;			// for freelist
	Scase	scase[1];		// one per case
};

static	Select*	selfree[20];

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
sendchan(Hchan *c, byte *ep, bool *pres)
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

	lock(c);
	if(c->dataqsiz > 0)
		goto asynch;

	sg = dequeue(&c->recvq, c);
	if(sg != nil) {
		if(ep != nil)
			c->elemalg->copy(c->elemsize, sg->elem, ep);

		gp = sg->g;
		gp->param = sg;
		unlock(c);
		ready(gp);

		if(pres != nil)
			*pres = true;
		return;
	}

	if(pres != nil) {
		unlock(c);
		*pres = false;
		return;
	}

	sg = allocsg(c);
	if(ep != nil)
		c->elemalg->copy(c->elemsize, sg->elem, ep);
	g->param = nil;
	g->status = Gwaiting;
	enqueue(&c->sendq, sg);
	unlock(c);
	sys·gosched();

	lock(c);
	sg = g->param;
	freesg(c, sg);
	unlock(c);
	return;

asynch:
	while(c->qcount >= c->dataqsiz) {
		// (rsc) should check for pres != nil
		sg = allocsg(c);
		g->status = Gwaiting;
		enqueue(&c->sendq, sg);
		unlock(c);
		sys·gosched();
		lock(c);
	}
	if(ep != nil)
		c->elemalg->copy(c->elemsize, c->senddataq->elem, ep);
	c->senddataq = c->senddataq->link;
	c->qcount++;

	sg = dequeue(&c->recvq, c);
	if(sg != nil) {
		gp = sg->g;
		freesg(c, sg);
		unlock(c);
		ready(gp);
	}else
		unlock(c);
}

static void
chanrecv(Hchan* c, byte *ep, bool* pres)
{
	SudoG *sg;
	G *gp;

	if(debug) {
		prints("chanrecv: chan=");
		sys·printpointer(c);
		prints("\n");
	}

	lock(c);
	if(c->dataqsiz > 0)
		goto asynch;

	sg = dequeue(&c->sendq, c);
	if(sg != nil) {
		c->elemalg->copy(c->elemsize, ep, sg->elem);

		gp = sg->g;
		gp->param = sg;
		unlock(c);
		ready(gp);

		if(pres != nil)
			*pres = true;
		return;
	}

	if(pres != nil) {
		unlock(c);
		*pres = false;
		return;
	}
		
	sg = allocsg(c);
	g->param = nil;
	g->status = Gwaiting;
	enqueue(&c->recvq, sg);
	unlock(c);
	sys·gosched();

	lock(c);
	sg = g->param;
	c->elemalg->copy(c->elemsize, ep, sg->elem);
	freesg(c, sg);
	unlock(c);
	return;

asynch:
	while(c->qcount <= 0) {
		sg = allocsg(c);
		g->status = Gwaiting;
		enqueue(&c->recvq, sg);
		unlock(c);
		sys·gosched();
		lock(c);
	}
	c->elemalg->copy(c->elemsize, ep, c->recvdataq->elem);
	c->recvdataq = c->recvdataq->link;
	c->qcount--;
	sg = dequeue(&c->sendq, c);
	if(sg != nil) {
		gp = sg->g;
		freesg(c, sg);
		unlock(c);
		ready(gp);
	}else
		unlock(c);
}

// chansend1(hchan *chan any, elem any);
void
sys·chansend1(Hchan* c, ...)
{
	int32 o;
	byte *ae;

	o = rnd(sizeof(c), c->elemsize);
	ae = (byte*)&c + o;
	sendchan(c, ae, nil);
}

// chansend2(hchan *chan any, elem any) (pres bool);
void
sys·chansend2(Hchan* c, ...)
{
	int32 o;
	byte *ae, *ap;

	o = rnd(sizeof(c), c->elemsize);
	ae = (byte*)&c + o;
	o = rnd(o+c->elemsize, 1);
	ap = (byte*)&c + o;

	sendchan(c, ae, ap);
}

// chanrecv1(hchan *chan any) (elem any);
void
sys·chanrecv1(Hchan* c, ...)
{
	int32 o;
	byte *ae;

	o = rnd(sizeof(c), c->elemsize);
	ae = (byte*)&c + o;

	chanrecv(c, ae, nil);
}

// chanrecv2(hchan *chan any) (elem any, pres bool);
void
sys·chanrecv2(Hchan* c, ...)
{
	int32 o;
	byte *ae, *ap;

	o = rnd(sizeof(c), c->elemsize);
	ae = (byte*)&c + o;
	o = rnd(o+c->elemsize, 1);
	ap = (byte*)&c + o;

	chanrecv(c, ae, ap);
}

// chanrecv3(hchan *chan any, elem *any) (pres bool);
void
sys·chanrecv3(Hchan* c, byte* ep, byte pres)
{
	chanrecv(c, ep, &pres);
}

// newselect(size uint32) (sel *byte);
void
sys·newselect(int32 size, Select *sel)
{
	int32 n;

	n = 0;
	if(size > 1)
		n = size-1;

	sel = nil;
	if(size >= 1 && size < nelem(selfree)) {
		sel = selfree[size];
		if(sel != nil)
			selfree[size] = sel->link;
	}
	if(sel == nil)
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
	byte *ae;

	// nil cases do not compete
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
	int32 i, eo;
	Scase *cas;

	// nil cases do not compete
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
	eo = rnd(eo+sizeof(c), sizeof(byte*));
	cas->so = rnd(eo+sizeof(byte*), 1);
	cas->send = 0;
	cas->u.elemp = *(byte**)((byte*)&sel + eo);

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

uint32	xxx	= 0;

// selectgo(sel *byte);
void
sys·selectgo(Select *sel)
{
	uint32 p, o, i;
	Scase *cas;
	Hchan *c;
	SudoG *sg;
	G *gp;

	byte *ae, *as;

	if(xxx) {
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

	// pass 1 - look for something already waiting
	for(i=0; i<sel->ncase; i++) {
		cas = &sel->scase[o];
		c = cas->chan;

		if(c->dataqsiz > 0) {
			if(cas->send) {
				if(c->qcount < c->dataqsiz)
					goto asyns;
			} else {
				if(c->qcount > 0)
					goto asynr;
			}
		}

		if(cas->send) {
			sg = dequeue(&c->recvq, c);
			if(sg != nil)
				goto gots;
		} else {
			sg = dequeue(&c->sendq, c);
			if(sg != nil)
				goto gotr;
		}

		o += p;
		if(o >= sel->ncase)
			o -= sel->ncase;
	}

	// pass 2 - enqueue on all chans
	for(i=0; i<sel->ncase; i++) {
		cas = &sel->scase[o];
		c = cas->chan;

		if(c->dataqsiz > 0) {
			if(cas->send) {
				if(c->qcount < c->dataqsiz) {
					prints("second pass asyn send\n");
					goto asyns;
				}
			} else {
				if(c->qcount > 0) {
					prints("second pass asyn recv\n");
					goto asynr;
				}
			}
		}

		if(cas->send) {
			sg = dequeue(&c->recvq, c);
			if(sg != nil) {
				prints("second pass syn send\n");
				g->selgen++;
				goto gots;	// probably an error
			}
			sg = allocsg(c);
			sg->offset = o;
			c->elemalg->copy(c->elemsize, sg->elem, cas->u.elem);
			enqueue(&c->sendq, sg);
		} else {
			sg = dequeue(&c->sendq, c);
			if(sg != nil) {
				prints("second pass syn recv\n");
				g->selgen++;
				goto gotr;	// probably an error
			}
			sg = allocsg(c);
			sg->offset = o;
			enqueue(&c->recvq, sg);
		}

		o += p;
		if(o >= sel->ncase)
			o -= sel->ncase;
	}

	// send and recv paths to sleep for a rendezvous
	// (rsc) not correct to set Gwaiting after queueing;
	// might already have been readied.
	g->status = Gwaiting;
	sys·gosched();

	sg = g->param;
	o = sg->offset;
	cas = &sel->scase[o];
	c = cas->chan;

	if(xxx) {
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
asyns:
	throw("asyn");
gotr:
	// recv path to wakeup the sender (sg)
	if(xxx) {
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

gots:
	// send path to wakeup the receiver (sg)
	if(xxx) {
		prints("gots: sel=");
		sys·printpointer(sel);
		prints(" c=");
		sys·printpointer(c);
		prints(" o=");
		sys·printint(o);
		prints("\n");
	}
	c->elemalg->copy(c->elemsize, sg->elem, cas->u.elem);
	gp = sg->g;
	gp->param = sg;
	ready(gp);

retc:
	if(sel->ncase >= 1 && sel->ncase < nelem(selfree)) {
		sel->link = selfree[sel->ncase];
		selfree[sel->ncase] = sel;
	}

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
		sg = mal(sizeof(*sg));
	sg->selgen = g->selgen;
	sg->g = g;
	sg->offset = 0;

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
