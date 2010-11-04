// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "type.h"

static	int32	debug	= 0;

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
	uint32	selgen;		// a weak pointer to g
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
	uint8	elemalign;
	Alg*	elemalg;		// interface for element type
	Link*	senddataq;		// pointer for sender
	Link*	recvdataq;		// pointer for receiver
	WaitQ	recvq;			// list of recv waiters
	WaitQ	sendq;			// list of send waiters
	SudoG*	free;			// freelist
	Lock;
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

static	void	dequeueg(WaitQ*, Hchan*);
static	SudoG*	dequeue(WaitQ*, Hchan*);
static	void	enqueue(WaitQ*, SudoG*);
static	SudoG*	allocsg(Hchan*);
static	void	freesg(Hchan*, SudoG*);
static	uint32	gcd(uint32, uint32);
static	uint32	fastrand1(void);
static	uint32	fastrand2(void);
static	void	destroychan(Hchan*);

Hchan*
runtime·makechan_c(Type *elem, int64 hint)
{
	Hchan *c;
	int32 i;

	if(hint < 0 || (int32)hint != hint || hint > ((uintptr)-1) / elem->size)
		runtime·panicstring("makechan: size out of range");

	if(elem->alg >= nelem(runtime·algarray)) {
		runtime·printf("chan(alg=%d)\n", elem->alg);
		runtime·throw("runtime.makechan: unsupported elem type");
	}

	c = runtime·mal(sizeof(*c));
	runtime·addfinalizer(c, destroychan, 0);

	c->elemsize = elem->size;
	c->elemalg = &runtime·algarray[elem->alg];
	c->elemalign = elem->align;

	if(hint > 0) {
		Link *d, *b, *e;

		// make a circular q
		b = nil;
		e = nil;
		for(i=0; i<hint; i++) {
			d = runtime·mal(sizeof(*d) + c->elemsize - sizeof(d->elem));
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

	if(debug)
		runtime·printf("makechan: chan=%p; elemsize=%D; elemalg=%d; elemalign=%d; dataqsiz=%d\n",
			c, (int64)elem->size, elem->alg, elem->align, c->dataqsiz);

	return c;
}

static void
destroychan(Hchan *c)
{
	runtime·destroylock(&c->Lock);
}


// makechan(elem *Type, hint int64) (hchan *chan any);
void
runtime·makechan(Type *elem, int64 hint, Hchan *ret)
{
	ret = runtime·makechan_c(elem, hint);
	FLUSH(&ret);
}

static void
incerr(Hchan* c)
{
	c->closed += Eincr;
	if(c->closed & Emax) {
		// Note that channel locks may still be held at this point.
		runtime·throw("too many operations on a closed channel");
	}
}

/*
 * generic single channel send/recv
 * if the bool pointer is nil,
 * then the full exchange will
 * occur. if pres is not nil,
 * then the protocol will not
 * sleep but return if it could
 * not complete.
 *
 * sleep can wake up with g->param == nil
 * when a channel involved in the sleep has
 * been closed.  it is easiest to loop and re-run
 * the operation; we'll see that it's now closed.
 */
void
runtime·chansend(Hchan *c, byte *ep, bool *pres)
{
	SudoG *sg;
	G* gp;

	if(c == nil)
		runtime·panicstring("send to nil channel");

	if(runtime·gcwaiting)
		runtime·gosched();

	if(debug) {
		runtime·printf("chansend: chan=%p; elem=", c);
		c->elemalg->print(c->elemsize, ep);
		runtime·prints("\n");
	}

	runtime·lock(c);
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
		runtime·unlock(c);
		runtime·ready(gp);

		if(pres != nil)
			*pres = true;
		return;
	}

	if(pres != nil) {
		runtime·unlock(c);
		*pres = false;
		return;
	}

	sg = allocsg(c);
	if(ep != nil)
		c->elemalg->copy(c->elemsize, sg->elem, ep);
	g->param = nil;
	g->status = Gwaiting;
	enqueue(&c->sendq, sg);
	runtime·unlock(c);
	runtime·gosched();

	runtime·lock(c);
	sg = g->param;
	if(sg == nil)
		goto loop;
	freesg(c, sg);
	runtime·unlock(c);
	return;

asynch:
	if(c->closed & Wclosed)
		goto closed;

	if(c->qcount >= c->dataqsiz) {
		if(pres != nil) {
			runtime·unlock(c);
			*pres = false;
			return;
		}
		sg = allocsg(c);
		g->status = Gwaiting;
		enqueue(&c->sendq, sg);
		runtime·unlock(c);
		runtime·gosched();

		runtime·lock(c);
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
		runtime·unlock(c);
		runtime·ready(gp);
	} else
		runtime·unlock(c);
	if(pres != nil)
		*pres = true;
	return;

closed:
	incerr(c);
	if(pres != nil)
		*pres = true;
	runtime·unlock(c);
}

void
runtime·chanrecv(Hchan* c, byte *ep, bool* pres)
{
	SudoG *sg;
	G *gp;

	if(c == nil)
		runtime·panicstring("receive from nil channel");

	if(runtime·gcwaiting)
		runtime·gosched();

	if(debug)
		runtime·printf("chanrecv: chan=%p\n", c);

	runtime·lock(c);
loop:
	if(c->dataqsiz > 0)
		goto asynch;

	if(c->closed & Wclosed)
		goto closed;

	sg = dequeue(&c->sendq, c);
	if(sg != nil) {
		c->elemalg->copy(c->elemsize, ep, sg->elem);
		c->elemalg->copy(c->elemsize, sg->elem, nil);

		gp = sg->g;
		gp->param = sg;
		runtime·unlock(c);
		runtime·ready(gp);

		if(pres != nil)
			*pres = true;
		return;
	}

	if(pres != nil) {
		runtime·unlock(c);
		c->elemalg->copy(c->elemsize, ep, nil);
		*pres = false;
		return;
	}

	sg = allocsg(c);
	g->param = nil;
	g->status = Gwaiting;
	enqueue(&c->recvq, sg);
	runtime·unlock(c);
	runtime·gosched();

	runtime·lock(c);
	sg = g->param;
	if(sg == nil)
		goto loop;

	c->elemalg->copy(c->elemsize, ep, sg->elem);
	c->elemalg->copy(c->elemsize, sg->elem, nil);
	freesg(c, sg);
	runtime·unlock(c);
	return;

asynch:
	if(c->qcount <= 0) {
		if(c->closed & Wclosed)
			goto closed;

		if(pres != nil) {
			runtime·unlock(c);
			c->elemalg->copy(c->elemsize, ep, nil);
			*pres = false;
			return;
		}
		sg = allocsg(c);
		g->status = Gwaiting;
		enqueue(&c->recvq, sg);
		runtime·unlock(c);
		runtime·gosched();

		runtime·lock(c);
		goto asynch;
	}
	c->elemalg->copy(c->elemsize, ep, c->recvdataq->elem);
	c->elemalg->copy(c->elemsize, c->recvdataq->elem, nil);
	c->recvdataq = c->recvdataq->link;
	c->qcount--;
	sg = dequeue(&c->sendq, c);
	if(sg != nil) {
		gp = sg->g;
		freesg(c, sg);
		runtime·unlock(c);
		runtime·ready(gp);
		if(pres != nil)
			*pres = true;
		return;
	}

	runtime·unlock(c);
	if(pres != nil)
		*pres = true;
	return;

closed:
	c->elemalg->copy(c->elemsize, ep, nil);
	c->closed |= Rclosed;
	incerr(c);
	if(pres != nil)
		*pres = true;
	runtime·unlock(c);
}

// chansend1(hchan *chan any, elem any);
#pragma textflag 7
void
runtime·chansend1(Hchan* c, ...)
{
	int32 o;
	byte *ae;

	if(c == nil)
		runtime·panicstring("send to nil channel");

	o = runtime·rnd(sizeof(c), c->elemalign);
	ae = (byte*)&c + o;
	runtime·chansend(c, ae, nil);
}

// chansend2(hchan *chan any, elem any) (pres bool);
#pragma textflag 7
void
runtime·chansend2(Hchan* c, ...)
{
	int32 o;
	byte *ae, *ap;

	if(c == nil)
		runtime·panicstring("send to nil channel");

	o = runtime·rnd(sizeof(c), c->elemalign);
	ae = (byte*)&c + o;
	o = runtime·rnd(o+c->elemsize, Structrnd);
	ap = (byte*)&c + o;

	runtime·chansend(c, ae, ap);
}

// chanrecv1(hchan *chan any) (elem any);
#pragma textflag 7
void
runtime·chanrecv1(Hchan* c, ...)
{
	int32 o;
	byte *ae;

	o = runtime·rnd(sizeof(c), Structrnd);
	ae = (byte*)&c + o;

	runtime·chanrecv(c, ae, nil);
}

// chanrecv2(hchan *chan any) (elem any, pres bool);
#pragma textflag 7
void
runtime·chanrecv2(Hchan* c, ...)
{
	int32 o;
	byte *ae, *ap;

	o = runtime·rnd(sizeof(c), Structrnd);
	ae = (byte*)&c + o;
	o = runtime·rnd(o+c->elemsize, 1);
	ap = (byte*)&c + o;

	runtime·chanrecv(c, ae, ap);
}

// newselect(size uint32) (sel *byte);
#pragma textflag 7
void
runtime·newselect(int32 size, ...)
{
	int32 n, o;
	Select **selp;
	Select *sel;

	o = runtime·rnd(sizeof(size), Structrnd);
	selp = (Select**)((byte*)&size + o);
	n = 0;
	if(size > 1)
		n = size-1;

	sel = runtime·mal(sizeof(*sel) + n*sizeof(sel->scase[0]));

	sel->tcase = size;
	sel->ncase = 0;
	*selp = sel;
	if(debug)
		runtime·printf("newselect s=%p size=%d\n", sel, size);
}

// selectsend(sel *byte, hchan *chan any, elem any) (selected bool);
#pragma textflag 7
void
runtime·selectsend(Select *sel, Hchan *c, ...)
{
	int32 i, eo;
	Scase *cas;
	byte *ae;

	// nil cases do not compete
	if(c == nil)
		return;

	i = sel->ncase;
	if(i >= sel->tcase)
		runtime·throw("selectsend: too many cases");
	sel->ncase = i+1;
	cas = runtime·mal(sizeof *cas + c->elemsize - sizeof(cas->u.elem));
	sel->scase[i] = cas;

	cas->pc = runtime·getcallerpc(&sel);
	cas->chan = c;

	eo = runtime·rnd(sizeof(sel), sizeof(c));
	eo = runtime·rnd(eo+sizeof(c), c->elemsize);
	cas->so = runtime·rnd(eo+c->elemsize, Structrnd);
	cas->send = 1;

	ae = (byte*)&sel + eo;
	c->elemalg->copy(c->elemsize, cas->u.elem, ae);

	if(debug)
		runtime·printf("selectsend s=%p pc=%p chan=%p so=%d send=%d\n",
			sel, cas->pc, cas->chan, cas->so, cas->send);
}

// selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
#pragma textflag 7
void
runtime·selectrecv(Select *sel, Hchan *c, ...)
{
	int32 i, eo;
	Scase *cas;

	// nil cases do not compete
	if(c == nil)
		return;

	i = sel->ncase;
	if(i >= sel->tcase)
		runtime·throw("selectrecv: too many cases");
	sel->ncase = i+1;
	cas = runtime·mal(sizeof *cas);
	sel->scase[i] = cas;
	cas->pc = runtime·getcallerpc(&sel);
	cas->chan = c;

	eo = runtime·rnd(sizeof(sel), sizeof(c));
	eo = runtime·rnd(eo+sizeof(c), sizeof(byte*));
	cas->so = runtime·rnd(eo+sizeof(byte*), Structrnd);
	cas->send = 0;
	cas->u.elemp = *(byte**)((byte*)&sel + eo);

	if(debug)
		runtime·printf("selectrecv s=%p pc=%p chan=%p so=%d send=%d\n",
			sel, cas->pc, cas->chan, cas->so, cas->send);
}


// selectdefaul(sel *byte) (selected bool);
#pragma textflag 7
void
runtime·selectdefault(Select *sel, ...)
{
	int32 i;
	Scase *cas;

	i = sel->ncase;
	if(i >= sel->tcase)
		runtime·throw("selectdefault: too many cases");
	sel->ncase = i+1;
	cas = runtime·mal(sizeof *cas);
	sel->scase[i] = cas;
	cas->pc = runtime·getcallerpc(&sel);
	cas->chan = nil;

	cas->so = runtime·rnd(sizeof(sel), Structrnd);
	cas->send = 2;
	cas->u.elemp = nil;

	if(debug)
		runtime·printf("selectdefault s=%p pc=%p so=%d send=%d\n",
			sel, cas->pc, cas->so, cas->send);
}

static void
freesel(Select *sel)
{
	uint32 i;

	for(i=0; i<sel->ncase; i++)
		runtime·free(sel->scase[i]);
	runtime·free(sel);
}

static void
sellock(Select *sel)
{
	uint32 i;
	Hchan *c;

	c = nil;
	for(i=0; i<sel->ncase; i++) {
		if(sel->scase[i]->chan != c) {
			c = sel->scase[i]->chan;
			runtime·lock(c);
		}
	}
}

static void
selunlock(Select *sel)
{
	uint32 i;
	Hchan *c;

	c = nil;
	for(i=sel->ncase; i>0; i--) {
		if(sel->scase[i-1]->chan && sel->scase[i-1]->chan != c) {
			c = sel->scase[i-1]->chan;
			runtime·unlock(c);
		}
	}
}

// selectgo(sel *byte);
//
// overwrites return pc on stack to signal which case of the select
// to run, so cannot appear at the top of a split stack.
// frame has 6 pointers and 4 int32 so 64 bytes max.
// that's less than StackGuard-StackSmall, so okay.
#pragma textflag 7
void
runtime·selectgo(Select *sel)
{
	uint32 p, o, i, j;
	Scase *cas, *dfl;
	Hchan *c;
	SudoG *sg;
	G *gp;
	byte *as;

	if(runtime·gcwaiting)
		runtime·gosched();

	if(debug)
		runtime·printf("select: sel=%p\n", sel);

	if(sel->ncase < 2) {
		if(sel->ncase < 1) {
			g->status = Gwaiting;	// forever
			runtime·gosched();
		}
		// TODO: make special case of one.
	}

	// select a (relative) prime
	for(i=0;; i++) {
		p = fastrand1();
		if(gcd(p, sel->ncase) == 1)
			break;
		if(i > 1000)
			runtime·throw("select: failed to select prime");
	}

	// select an initial offset
	o = fastrand2();

	p %= sel->ncase;
	o %= sel->ncase;

	// sort the cases by Hchan address to get the locking order.
	for(i=1; i<sel->ncase; i++) {
		cas = sel->scase[i];
		for(j=i; j>0 && sel->scase[j-1]->chan >= cas->chan; j--)
			sel->scase[j] = sel->scase[j-1];
		sel->scase[j] = cas;
	}

	sellock(sel);

loop:
	// pass 1 - look for something already waiting
	dfl = nil;
	for(i=0; i<sel->ncase; i++) {
		cas = sel->scase[o];
		c = cas->chan;

		switch(cas->send) {
		case 0:	// recv
			if(c->dataqsiz > 0) {
				if(c->qcount > 0)
					goto asyncrecv;
			} else {
				sg = dequeue(&c->sendq, c);
				if(sg != nil)
					goto syncrecv;
			}
			if(c->closed & Wclosed)
				goto rclose;
			break;

		case 1:	// send
			if(c->closed & Wclosed)
				goto sclose;
			if(c->dataqsiz > 0) {
				if(c->qcount < c->dataqsiz)
					goto asyncsend;
			} else {
				sg = dequeue(&c->recvq, c);
				if(sg != nil)
					goto syncsend;
			}
			break;

		case 2:	// default
			dfl = cas;
			break;
		}

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
		sg = allocsg(c);
		sg->offset = o;

		switch(cas->send) {
		case 0:	// recv
			if(c->dataqsiz > 0) {
				if(c->qcount > 0)
					runtime·throw("select: pass 2 async recv");
			} else {
				if(dequeue(&c->sendq, c))
					runtime·throw("select: pass 2 sync recv");
			}
			enqueue(&c->recvq, sg);
			break;
		
		case 1:	// send
			if(c->dataqsiz > 0) {
				if(c->qcount < c->dataqsiz)
					runtime·throw("select: pass 2 async send");
			} else {
				if(dequeue(&c->recvq, c))
					runtime·throw("select: pass 2 sync send");
				c->elemalg->copy(c->elemsize, sg->elem, cas->u.elem);
			}
			enqueue(&c->sendq, sg);
			break;
		}

		o += p;
		if(o >= sel->ncase)
			o -= sel->ncase;
	}

	g->param = nil;
	g->status = Gwaiting;
	selunlock(sel);
	runtime·gosched();

	sellock(sel);
	sg = g->param;

	// pass 3 - dequeue from unsuccessful chans
	// otherwise they stack up on quiet channels
	for(i=0; i<sel->ncase; i++) {
		if(sg == nil || o != sg->offset) {
			cas = sel->scase[o];
			c = cas->chan;
			if(cas->send)
				dequeueg(&c->sendq, c);
			else
				dequeueg(&c->recvq, c);
		}
		
		o += p;
		if(o >= sel->ncase)
			o -= sel->ncase;
	}

	if(sg == nil)
		goto loop;

	o = sg->offset;
	cas = sel->scase[o];
	c = cas->chan;

	if(c->dataqsiz > 0) {
//		prints("shouldnt happen\n");
		goto loop;
	}

	if(debug)
		runtime·printf("wait-return: sel=%p c=%p cas=%p send=%d o=%d\n",
			sel, c, cas, cas->send, o);

	if(!cas->send) {
		if(cas->u.elemp != nil)
			c->elemalg->copy(c->elemsize, cas->u.elemp, sg->elem);
		c->elemalg->copy(c->elemsize, sg->elem, nil);
	}

	freesg(c, sg);
	goto retc;

asyncrecv:
	// can receive from buffer
	if(cas->u.elemp != nil)
		c->elemalg->copy(c->elemsize, cas->u.elemp, c->recvdataq->elem);
	c->elemalg->copy(c->elemsize, c->recvdataq->elem, nil);
	c->recvdataq = c->recvdataq->link;
	c->qcount--;
	sg = dequeue(&c->sendq, c);
	if(sg != nil) {
		gp = sg->g;
		freesg(c, sg);
		runtime·ready(gp);
	}
	goto retc;

asyncsend:
	// can send to buffer
	if(cas->u.elem != nil)
		c->elemalg->copy(c->elemsize, c->senddataq->elem, cas->u.elem);
	c->senddataq = c->senddataq->link;
	c->qcount++;
	sg = dequeue(&c->recvq, c);
	if(sg != nil) {
		gp = sg->g;
		freesg(c, sg);
		runtime·ready(gp);
	}
	goto retc;

syncrecv:
	// can receive from sleeping sender (sg)
	if(debug)
		runtime·printf("syncrecv: sel=%p c=%p o=%d\n", sel, c, o);
	if(cas->u.elemp != nil)
		c->elemalg->copy(c->elemsize, cas->u.elemp, sg->elem);
	c->elemalg->copy(c->elemsize, sg->elem, nil);
	gp = sg->g;
	gp->param = sg;
	runtime·ready(gp);
	goto retc;

rclose:
	// read at end of closed channel
	if(cas->u.elemp != nil)
		c->elemalg->copy(c->elemsize, cas->u.elemp, nil);
	c->closed |= Rclosed;
	incerr(c);
	goto retc;

syncsend:
	// can send to sleeping receiver (sg)
	if(debug)
		runtime·printf("syncsend: sel=%p c=%p o=%d\n", sel, c, o);
	if(c->closed & Wclosed)
		goto sclose;
	c->elemalg->copy(c->elemsize, sg->elem, cas->u.elem);
	gp = sg->g;
	gp->param = sg;
	runtime·ready(gp);
	goto retc;

sclose:
	// send on closed channel
	incerr(c);
	goto retc;

retc:
	selunlock(sel);

	// return to pc corresponding to chosen case
	runtime·setcallerpc(&sel, cas->pc);
	as = (byte*)&sel + cas->so;
	freesel(sel);
	*as = true;
}

// closechan(sel *byte);
void
runtime·closechan(Hchan *c)
{
	SudoG *sg;
	G* gp;

	if(runtime·gcwaiting)
		runtime·gosched();

	runtime·lock(c);
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
		runtime·ready(gp);
	}

	// release all writers
	for(;;) {
		sg = dequeue(&c->sendq, c);
		if(sg == nil)
			break;
		gp = sg->g;
		gp->param = nil;
		freesg(c, sg);
		runtime·ready(gp);
	}

	runtime·unlock(c);
}

void
runtime·chanclose(Hchan *c)
{
	runtime·closechan(c);
}

bool
runtime·chanclosed(Hchan *c)
{
	return (c->closed & Rclosed) != 0;
}

int32
runtime·chanlen(Hchan *c)
{
	return c->qcount;
}

int32
runtime·chancap(Hchan *c)
{
	return c->dataqsiz;
}


// closedchan(sel *byte) bool;
void
runtime·closedchan(Hchan *c, bool closed)
{
	closed = runtime·chanclosed(c);
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
	if(!runtime·cas(&sgp->g->selgen, sgp->selgen, sgp->selgen + 1)) {
		//prints("INVALID PSEUDOG POINTER\n");
		freesg(c, sgp);
		goto loop;
	}

	return sgp;
}

static void
dequeueg(WaitQ *q, Hchan *c)
{
	SudoG **l, *sgp;
	
	for(l=&q->first; (sgp=*l) != nil; l=&sgp->link) {
		if(sgp->g == g) {
			*l = sgp->link;
			freesg(c, sgp);
			break;
		}
	}
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
		sg = runtime·mal(sizeof(*sg) + c->elemsize - sizeof(sg->elem));
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
			runtime·throw("chan.freesg: already free");
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
