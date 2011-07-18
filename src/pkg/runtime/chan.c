// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "type.h"

#define	MAXALIGN	7

static	int32	debug	= 0;

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
	bool	closed;
	uint8	elemalign;
	Alg*	elemalg;		// interface for element type
	uint32	sendx;	// send index
	uint32	recvx;	// receive index
	WaitQ	recvq;			// list of recv waiters
	WaitQ	sendq;			// list of send waiters
	SudoG*	free;			// freelist
	Lock;
};

// Buffer follows Hchan immediately in memory.
// chanbuf(c, i) is pointer to the i'th slot in the buffer.
#define chanbuf(c, i) ((byte*)((c)+1)+(uintptr)(c)->elemsize*(i))

enum
{
	// Scase.kind
	CaseRecv,
	CaseSend,
	CaseDefault,
};

struct	Scase
{
	Hchan*	chan;			// chan
	byte*	pc;			// return pc
	uint16	kind;
	uint16	so;			// vararg of selected bool
	union {
		byte	elem[2*sizeof(void*)];	// element (send)
		struct {
			byte*	elemp;		// pointer to element (recv)
			bool*	receivedp;	// pointer to received bool (recv2)
		} recv;
	} u;
};

struct	Select
{
	uint16	tcase;			// total count of scase[]
	uint16	ncase;			// currently filled scase[]
	Select*	link;			// for freelist
	uint16*	order;
	Scase*	scase[1];		// one per case
};

static	void	dequeueg(WaitQ*, Hchan*);
static	SudoG*	dequeue(WaitQ*, Hchan*);
static	void	enqueue(WaitQ*, SudoG*);
static	SudoG*	allocsg(Hchan*);
static	void	freesg(Hchan*, SudoG*);
static	uint32	fastrandn(uint32);
static	void	destroychan(Hchan*);

Hchan*
runtime·makechan_c(Type *elem, int64 hint)
{
	Hchan *c;
	int32 n;
	byte *by;

	if(hint < 0 || (int32)hint != hint || (elem->size > 0 && hint > ((uintptr)-1) / elem->size))
		runtime·panicstring("makechan: size out of range");

	if(elem->alg >= nelem(runtime·algarray)) {
		runtime·printf("chan(alg=%d)\n", elem->alg);
		runtime·throw("runtime.makechan: unsupported elem type");
	}

	// calculate rounded size of Hchan
	n = sizeof(*c);
	while(n & MAXALIGN)
		n++;

	// allocate memory in one call
	by = runtime·mal(n + hint*elem->size);

	c = (Hchan*)by;
	runtime·addfinalizer(c, destroychan, 0);

	c->elemsize = elem->size;
	c->elemalg = &runtime·algarray[elem->alg];
	c->elemalign = elem->align;
	c->dataqsiz = hint;

	if(debug)
		runtime·printf("makechan: chan=%p; elemsize=%D; elemalg=%d; elemalign=%d; dataqsiz=%d\n",
			c, (int64)elem->size, elem->alg, elem->align, c->dataqsiz);

	return c;
}

// For reflect
//	func makechan(typ *ChanType, size uint32) (chan)
void
reflect·makechan(ChanType *t, uint32 size, Hchan *c)
{
	c = runtime·makechan_c(t->elem, size);
	FLUSH(&c);
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
	if(c->closed)
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
	if(c->closed)
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
		c->elemalg->copy(c->elemsize, chanbuf(c, c->sendx), ep);
	if(++c->sendx == c->dataqsiz)
		c->sendx = 0;
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
	runtime·unlock(c);
	runtime·panicstring("send on closed channel");
}


void
runtime·chanrecv(Hchan* c, byte *ep, bool *selected, bool *received)
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

	if(c->closed)
		goto closed;

	sg = dequeue(&c->sendq, c);
	if(sg != nil) {
		if(ep != nil)
			c->elemalg->copy(c->elemsize, ep, sg->elem);
		c->elemalg->copy(c->elemsize, sg->elem, nil);

		gp = sg->g;
		gp->param = sg;
		runtime·unlock(c);
		runtime·ready(gp);

		if(selected != nil)
			*selected = true;
		if(received != nil)
			*received = true;
		return;
	}

	if(selected != nil) {
		runtime·unlock(c);
		*selected = false;
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

	if(ep != nil)
		c->elemalg->copy(c->elemsize, ep, sg->elem);
	c->elemalg->copy(c->elemsize, sg->elem, nil);
	if(received != nil)
		*received = true;
	freesg(c, sg);
	runtime·unlock(c);
	return;

asynch:
	if(c->qcount <= 0) {
		if(c->closed)
			goto closed;

		if(selected != nil) {
			runtime·unlock(c);
			*selected = false;
			if(received != nil)
				*received = false;
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
	if(ep != nil)
		c->elemalg->copy(c->elemsize, ep, chanbuf(c, c->recvx));
	c->elemalg->copy(c->elemsize, chanbuf(c, c->recvx), nil);
	if(++c->recvx == c->dataqsiz)
		c->recvx = 0;
	c->qcount--;
	sg = dequeue(&c->sendq, c);
	if(sg != nil) {
		gp = sg->g;
		freesg(c, sg);
		runtime·unlock(c);
		runtime·ready(gp);
	} else
		runtime·unlock(c);

	if(selected != nil)
		*selected = true;
	if(received != nil)
		*received = true;
	return;

closed:
	if(ep != nil)
		c->elemalg->copy(c->elemsize, ep, nil);
	if(selected != nil)
		*selected = true;
	if(received != nil)
		*received = false;
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

// chanrecv1(hchan *chan any) (elem any);
#pragma textflag 7
void
runtime·chanrecv1(Hchan* c, ...)
{
	int32 o;
	byte *ae;

	o = runtime·rnd(sizeof(c), Structrnd);
	ae = (byte*)&c + o;

	runtime·chanrecv(c, ae, nil, nil);
}

// chanrecv2(hchan *chan any) (elem any, received bool);
#pragma textflag 7
void
runtime·chanrecv2(Hchan* c, ...)
{
	int32 o;
	byte *ae, *ac;
	
	if(c == nil)
		runtime·panicstring("receive from nil channel");

	o = runtime·rnd(sizeof(c), Structrnd);
	ae = (byte*)&c + o;
	o = runtime·rnd(o+c->elemsize, 1);
	ac = (byte*)&c + o;

	runtime·chanrecv(c, ae, nil, ac);
}

// func selectnbsend(c chan any, elem any) bool
//
// compiler implements
//
//	select {
//	case c <- v:
//		... foo
//	default:
//		... bar
//	}
//
// as
//
//	if c != nil && selectnbsend(c, v) {
//		... foo
//	} else {
//		... bar
//	}
//
#pragma textflag 7
void
runtime·selectnbsend(Hchan *c, ...)
{
	int32 o;
	byte *ae, *ap;

	o = runtime·rnd(sizeof(c), c->elemalign);
	ae = (byte*)&c + o;
	o = runtime·rnd(o+c->elemsize, Structrnd);
	ap = (byte*)&c + o;

	runtime·chansend(c, ae, ap);
}

// func selectnbrecv(elem *any, c chan any) bool
//
// compiler implements
//
//	select {
//	case v = <-c:
//		... foo
//	default:
//		... bar
//	}
//
// as
//
//	if c != nil && selectnbrecv(&v, c) {
//		... foo
//	} else {
//		... bar
//	}
//
#pragma textflag 7
void
runtime·selectnbrecv(byte *v, Hchan *c, bool selected)
{
	runtime·chanrecv(c, v, &selected, nil);
}	

// func selectnbrecv2(elem *any, ok *bool, c chan any) bool
//
// compiler implements
//
//	select {
//	case v, ok = <-c:
//		... foo
//	default:
//		... bar
//	}
//
// as
//
//	if c != nil && selectnbrecv2(&v, &ok, c) {
//		... foo
//	} else {
//		... bar
//	}
//
#pragma textflag 7
void
runtime·selectnbrecv2(byte *v, bool *received, Hchan *c, bool selected)
{
	runtime·chanrecv(c, v, &selected, received);
}	

// For reflect:
//	func chansend(c chan, val iword, nb bool) (selected bool)
// where an iword is the same word an interface value would use:
// the actual data if it fits, or else a pointer to the data.
//
// The "uintptr selected" is really "bool selected" but saying
// uintptr gets us the right alignment for the output parameter block.
void
reflect·chansend(Hchan *c, uintptr val, bool nb, uintptr selected)
{
	bool *sp;
	byte *vp;
	
	if(c == nil)
		runtime·panicstring("send to nil channel");

	if(nb) {
		selected = false;
		sp = (bool*)&selected;
	} else {
		*(bool*)&selected = true;
		FLUSH(&selected);
		sp = nil;
	}
	if(c->elemsize <= sizeof(val))
		vp = (byte*)&val;
	else
		vp = (byte*)val;
	runtime·chansend(c, vp, sp);
}

// For reflect:
//	func chanrecv(c chan, nb bool) (val iword, selected, received bool)
// where an iword is the same word an interface value would use:
// the actual data if it fits, or else a pointer to the data.
void
reflect·chanrecv(Hchan *c, bool nb, uintptr val, bool selected, bool received)
{
	byte *vp;
	bool *sp;
	
	if(c == nil)
		runtime·panicstring("receive from nil channel");

	if(nb) {
		selected = false;
		sp = &selected;
	} else {
		selected = true;
		FLUSH(&selected);
		sp = nil;
	}
	received = false;
	FLUSH(&received);
	if(c->elemsize <= sizeof(val)) {
		val = 0;
		vp = (byte*)&val;
	} else {
		vp = runtime·mal(c->elemsize);
		val = (uintptr)vp;
		FLUSH(&val);
	}
	runtime·chanrecv(c, vp, sp, &received);
}

static void newselect(int32, Select**);

// newselect(size uint32) (sel *byte);
#pragma textflag 7
void
runtime·newselect(int32 size, ...)
{
	int32 o;
	Select **selp;

	o = runtime·rnd(sizeof(size), Structrnd);
	selp = (Select**)((byte*)&size + o);
	newselect(size, selp);
}

static void
newselect(int32 size, Select **selp)
{
	int32 n;
	Select *sel;

	n = 0;
	if(size > 1)
		n = size-1;

	sel = runtime·mal(sizeof(*sel) + n*sizeof(sel->scase[0]) + size*sizeof(sel->order[0]));

	sel->tcase = size;
	sel->ncase = 0;
	sel->order = (void*)(sel->scase + size);
	*selp = sel;
	if(debug)
		runtime·printf("newselect s=%p size=%d\n", sel, size);
}

// cut in half to give stack a chance to split
static void selectsend(Select **selp, Hchan *c, void *pc);

// selectsend(sel *byte, hchan *chan any, elem any) (selected bool);
#pragma textflag 7
void
runtime·selectsend(Select *sel, Hchan *c, ...)
{
	// nil cases do not compete
	if(c == nil)
		return;
	
	selectsend(&sel, c, runtime·getcallerpc(&sel));
}

static void
selectsend(Select **selp, Hchan *c, void *pc)
{
	int32 i, eo;
	Scase *cas;
	byte *ae;
	Select *sel;
	
	sel = *selp;
	i = sel->ncase;
	if(i >= sel->tcase)
		runtime·throw("selectsend: too many cases");
	sel->ncase = i+1;
	cas = runtime·mal(sizeof *cas + c->elemsize - sizeof(cas->u.elem));
	sel->scase[i] = cas;

	cas->pc = pc;
	cas->chan = c;

	eo = runtime·rnd(sizeof(sel), sizeof(c));
	eo = runtime·rnd(eo+sizeof(c), c->elemsize);
	cas->so = runtime·rnd(eo+c->elemsize, Structrnd);
	cas->kind = CaseSend;

	ae = (byte*)selp + eo;
	c->elemalg->copy(c->elemsize, cas->u.elem, ae);

	if(debug)
		runtime·printf("selectsend s=%p pc=%p chan=%p so=%d\n",
			sel, cas->pc, cas->chan, cas->so);
}

// cut in half to give stack a chance to split
static void selectrecv(Select *sel, Hchan *c, void *pc, void *elem, bool*, int32 so);

// selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
#pragma textflag 7
void
runtime·selectrecv(Select *sel, Hchan *c, void *elem, bool selected)
{
	// nil cases do not compete
	if(c == nil)
		return;

	selectrecv(sel, c, runtime·getcallerpc(&sel), elem, nil, (byte*)&selected - (byte*)&sel);
}

// selectrecv2(sel *byte, hchan *chan any, elem *any, received *bool) (selected bool);
#pragma textflag 7
void
runtime·selectrecv2(Select *sel, Hchan *c, void *elem, bool *received, bool selected)
{
	// nil cases do not compete
	if(c == nil)
		return;

	selectrecv(sel, c, runtime·getcallerpc(&sel), elem, received, (byte*)&selected - (byte*)&sel);
}

static void
selectrecv(Select *sel, Hchan *c, void *pc, void *elem, bool *received, int32 so)
{
	int32 i;
	Scase *cas;

	i = sel->ncase;
	if(i >= sel->tcase)
		runtime·throw("selectrecv: too many cases");
	sel->ncase = i+1;
	cas = runtime·mal(sizeof *cas);
	sel->scase[i] = cas;
	cas->pc = pc;
	cas->chan = c;

	cas->so = so;
	cas->kind = CaseRecv;
	cas->u.recv.elemp = elem;
	cas->u.recv.receivedp = nil;
	cas->u.recv.receivedp = received;

	if(debug)
		runtime·printf("selectrecv s=%p pc=%p chan=%p so=%d\n",
			sel, cas->pc, cas->chan, cas->so);
}

// cut in half to give stack a chance to split
static void selectdefault(Select*, void*, int32);

// selectdefault(sel *byte) (selected bool);
#pragma textflag 7
void
runtime·selectdefault(Select *sel, bool selected)
{
	selectdefault(sel, runtime·getcallerpc(&sel), (byte*)&selected - (byte*)&sel);
}

static void
selectdefault(Select *sel, void *callerpc, int32 so)
{
	int32 i;
	Scase *cas;

	i = sel->ncase;
	if(i >= sel->tcase)
		runtime·throw("selectdefault: too many cases");
	sel->ncase = i+1;
	cas = runtime·mal(sizeof *cas);
	sel->scase[i] = cas;
	cas->pc = callerpc;
	cas->chan = nil;

	cas->so = so;
	cas->kind = CaseDefault;

	if(debug)
		runtime·printf("selectdefault s=%p pc=%p so=%d\n",
			sel, cas->pc, cas->so);
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

void
runtime·block(void)
{
	g->status = Gwaiting;	// forever
	runtime·gosched();
}

static void* selectgo(Select**);

// selectgo(sel *byte);
//
// overwrites return pc on stack to signal which case of the select
// to run, so cannot appear at the top of a split stack.
#pragma textflag 7
void
runtime·selectgo(Select *sel)
{
	runtime·setcallerpc(&sel, selectgo(&sel));
}

static void*
selectgo(Select **selp)
{
	Select *sel;
	uint32 o, i, j;
	Scase *cas, *dfl;
	Hchan *c;
	SudoG *sg;
	G *gp;
	byte *as;
	void *pc;

	sel = *selp;
	if(runtime·gcwaiting)
		runtime·gosched();

	if(debug)
		runtime·printf("select: sel=%p\n", sel);

	// The compiler rewrites selects that statically have
	// only 0 or 1 cases plus default into simpler constructs.
	// The only way we can end up with such small sel->ncase
	// values here is for a larger select in which most channels
	// have been nilled out.  The general code handles those
	// cases correctly, and they are rare enough not to bother
	// optimizing (and needing to test).

	// generate permuted order
	for(i=0; i<sel->ncase; i++)
		sel->order[i] = i;
	for(i=1; i<sel->ncase; i++) {
		o = sel->order[i];
		j = fastrandn(i+1);
		sel->order[i] = sel->order[j];
		sel->order[j] = o;
	}

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
		o = sel->order[i];
		cas = sel->scase[o];
		c = cas->chan;

		switch(cas->kind) {
		case CaseRecv:
			if(c->dataqsiz > 0) {
				if(c->qcount > 0)
					goto asyncrecv;
			} else {
				sg = dequeue(&c->sendq, c);
				if(sg != nil)
					goto syncrecv;
			}
			if(c->closed)
				goto rclose;
			break;

		case CaseSend:
			if(c->closed)
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

		case CaseDefault:
			dfl = cas;
			break;
		}
	}

	if(dfl != nil) {
		cas = dfl;
		goto retc;
	}


	// pass 2 - enqueue on all chans
	for(i=0; i<sel->ncase; i++) {
		o = sel->order[i];
		cas = sel->scase[o];
		c = cas->chan;
		sg = allocsg(c);
		sg->offset = o;

		switch(cas->kind) {
		case CaseRecv:
			enqueue(&c->recvq, sg);
			break;
		
		case CaseSend:
			if(c->dataqsiz == 0)
				c->elemalg->copy(c->elemsize, sg->elem, cas->u.elem);
			enqueue(&c->sendq, sg);
			break;
		}
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
		if(sg == nil || i != sg->offset) {
			cas = sel->scase[i];
			c = cas->chan;
			if(cas->kind == CaseSend)
				dequeueg(&c->sendq, c);
			else
				dequeueg(&c->recvq, c);
		}
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
		runtime·printf("wait-return: sel=%p c=%p cas=%p kind=%d o=%d\n",
			sel, c, cas, cas->kind, o);

	if(cas->kind == CaseRecv) {
		if(cas->u.recv.receivedp != nil)
			*cas->u.recv.receivedp = true;
		if(cas->u.recv.elemp != nil)
			c->elemalg->copy(c->elemsize, cas->u.recv.elemp, sg->elem);
		c->elemalg->copy(c->elemsize, sg->elem, nil);
	}

	freesg(c, sg);
	goto retc;

asyncrecv:
	// can receive from buffer
	if(cas->u.recv.receivedp != nil)
		*cas->u.recv.receivedp = true;
	if(cas->u.recv.elemp != nil)
		c->elemalg->copy(c->elemsize, cas->u.recv.elemp, chanbuf(c, c->recvx));
	c->elemalg->copy(c->elemsize, chanbuf(c, c->recvx), nil);
	if(++c->recvx == c->dataqsiz)
		c->recvx = 0;
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
		c->elemalg->copy(c->elemsize, chanbuf(c, c->sendx), cas->u.elem);
	if(++c->sendx == c->dataqsiz)
		c->sendx = 0;
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
	if(cas->u.recv.receivedp != nil)
		*cas->u.recv.receivedp = true;
	if(cas->u.recv.elemp != nil)
		c->elemalg->copy(c->elemsize, cas->u.recv.elemp, sg->elem);
	c->elemalg->copy(c->elemsize, sg->elem, nil);
	gp = sg->g;
	gp->param = sg;
	runtime·ready(gp);
	goto retc;

rclose:
	// read at end of closed channel
	if(cas->u.recv.receivedp != nil)
		*cas->u.recv.receivedp = false;
	if(cas->u.recv.elemp != nil)
		c->elemalg->copy(c->elemsize, cas->u.recv.elemp, nil);
	goto retc;

syncsend:
	// can send to sleeping receiver (sg)
	if(debug)
		runtime·printf("syncsend: sel=%p c=%p o=%d\n", sel, c, o);
	if(c->closed)
		goto sclose;
	c->elemalg->copy(c->elemsize, sg->elem, cas->u.elem);
	gp = sg->g;
	gp->param = sg;
	runtime·ready(gp);

retc:
	selunlock(sel);

	// return to pc corresponding to chosen case
	pc = cas->pc;
	as = (byte*)selp + cas->so;
	freesel(sel);
	*as = true;
	return pc;

sclose:
	// send on closed channel
	selunlock(sel);
	runtime·panicstring("send on closed channel");
	return nil;  // not reached
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
	if(c->closed) {
		runtime·unlock(c);
		runtime·panicstring("close of closed channel");
	}

	c->closed = true;

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

// For reflect
//	func chanclose(c chan)
void
reflect·chanclose(Hchan *c)
{
	runtime·closechan(c);
}

// For reflect
//	func chanlen(c chan) (len int32)
void
reflect·chanlen(Hchan *c, int32 len)
{
	if(c == nil)
		len = 0;
	else
		len = c->qcount;
	FLUSH(&len);
}

// For reflect
//	func chancap(c chan) (cap int32)
void
reflect·chancap(Hchan *c, int32 cap)
{
	if(c == nil)
		cap = 0;
	else
		cap = c->dataqsiz;
	FLUSH(&cap);
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
	SudoG **l, *sgp, *prevsgp;

	prevsgp = nil;
	for(l=&q->first; (sgp=*l) != nil; l=&sgp->link, prevsgp=sgp) {
		if(sgp->g == g) {
			*l = sgp->link;
			freesg(c, sgp);
			if(q->last == sgp)
				q->last = prevsgp;
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
fastrandn(uint32 n)
{
	uint32 max, r;

	if(n <= 1)
		return 0;

	r = runtime·fastrand1();
	if(r < (1ULL<<31)-n)  // avoid computing max in common case
		return r%n;

	max = (1ULL<<31)/n * n;
	while(r >= max)
		r = runtime·fastrand1();
	return r%n;
}
