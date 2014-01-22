// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "type.h"
#include "race.h"
#include "malloc.h"
#include "../../cmd/ld/textflag.h"

#define	MAXALIGN	8
#define	NOSELGEN	1

typedef	struct	WaitQ	WaitQ;
typedef	struct	SudoG	SudoG;
typedef	struct	Select	Select;
typedef	struct	Scase	Scase;

struct	SudoG
{
	G*	g;		// g and selgen constitute
	uint32	selgen;		// a weak pointer to g
	SudoG*	link;
	int64	releasetime;
	byte*	elem;		// data element
};

struct	WaitQ
{
	SudoG*	first;
	SudoG*	last;
};

// The garbage collector is assuming that Hchan can only contain pointers into the stack
// and cannot contain pointers into the heap.
struct	Hchan
{
	uintgo	qcount;			// total data in the q
	uintgo	dataqsiz;		// size of the circular q
	uint16	elemsize;
	uint16	pad;			// ensures proper alignment of the buffer that follows Hchan in memory
	bool	closed;
	Type*	elemtype;		// element type
	uintgo	sendx;			// send index
	uintgo	recvx;			// receive index
	WaitQ	recvq;			// list of recv waiters
	WaitQ	sendq;			// list of send waiters
	Lock;
};

uint32 runtime·Hchansize = sizeof(Hchan);

// Buffer follows Hchan immediately in memory.
// chanbuf(c, i) is pointer to the i'th slot in the buffer.
#define chanbuf(c, i) ((byte*)((c)+1)+(uintptr)(c)->elemsize*(i))

enum
{
	debug = 0,

	// Scase.kind
	CaseRecv,
	CaseSend,
	CaseDefault,
};

struct	Scase
{
	SudoG	sg;			// must be first member (cast to Scase)
	Hchan*	chan;			// chan
	byte*	pc;			// return pc
	uint16	kind;
	uint16	so;			// vararg of selected bool
	bool*	receivedp;		// pointer to received bool (recv2)
};

struct	Select
{
	uint16	tcase;			// total count of scase[]
	uint16	ncase;			// currently filled scase[]
	uint16*	pollorder;		// case poll order
	Hchan**	lockorder;		// channel lock order
	Scase	scase[1];		// one per case (in order of appearance)
};

static	void	dequeueg(WaitQ*);
static	SudoG*	dequeue(WaitQ*);
static	void	enqueue(WaitQ*, SudoG*);
static	void	destroychan(Hchan*);
static	void	racesync(Hchan*, SudoG*);

Hchan*
runtime·makechan_c(ChanType *t, int64 hint)
{
	Hchan *c;
	Type *elem;

	elem = t->elem;

	// compiler checks this but be safe.
	if(elem->size >= (1<<16))
		runtime·throw("makechan: invalid channel element type");
	if((sizeof(*c)%MAXALIGN) != 0 || elem->align > MAXALIGN)
		runtime·throw("makechan: bad alignment");

	if(hint < 0 || (intgo)hint != hint || (elem->size > 0 && hint > MaxMem / elem->size))
		runtime·panicstring("makechan: size out of range");

	// allocate memory in one call
	c = (Hchan*)runtime·mallocgc(sizeof(*c) + hint*elem->size, (uintptr)t | TypeInfo_Chan, 0);
	c->elemsize = elem->size;
	c->elemtype = elem;
	c->dataqsiz = hint;

	if(debug)
		runtime·printf("makechan: chan=%p; elemsize=%D; elemalg=%p; dataqsiz=%D\n",
			c, (int64)elem->size, elem->alg, (int64)c->dataqsiz);

	return c;
}

// For reflect
//	func makechan(typ *ChanType, size uint64) (chan)
void
reflect·makechan(ChanType *t, uint64 size, Hchan *c)
{
	c = runtime·makechan_c(t, size);
	FLUSH(&c);
}

// makechan(t *ChanType, hint int64) (hchan *chan any);
void
runtime·makechan(ChanType *t, int64 hint, Hchan *ret)
{
	ret = runtime·makechan_c(t, hint);
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
runtime·chansend(ChanType *t, Hchan *c, byte *ep, bool *pres, void *pc)
{
	SudoG *sg;
	SudoG mysg;
	G* gp;
	int64 t0;

	if(raceenabled)
		runtime·racereadobjectpc(ep, t->elem, runtime·getcallerpc(&t), runtime·chansend);

	if(c == nil) {
		USED(t);
		if(pres != nil) {
			*pres = false;
			return;
		}
		runtime·park(nil, nil, "chan send (nil chan)");
		return;  // not reached
	}

	if(debug) {
		runtime·printf("chansend: chan=%p; elem=", c);
		c->elemtype->alg->print(c->elemsize, ep);
		runtime·prints("\n");
	}

	t0 = 0;
	mysg.releasetime = 0;
	if(runtime·blockprofilerate > 0) {
		t0 = runtime·cputicks();
		mysg.releasetime = -1;
	}

	runtime·lock(c);
	if(raceenabled)
		runtime·racereadpc(c, pc, runtime·chansend);
	if(c->closed)
		goto closed;

	if(c->dataqsiz > 0)
		goto asynch;

	sg = dequeue(&c->recvq);
	if(sg != nil) {
		if(raceenabled)
			racesync(c, sg);
		runtime·unlock(c);

		gp = sg->g;
		gp->param = sg;
		if(sg->elem != nil)
			c->elemtype->alg->copy(c->elemsize, sg->elem, ep);
		if(sg->releasetime)
			sg->releasetime = runtime·cputicks();
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

	mysg.elem = ep;
	mysg.g = g;
	mysg.selgen = NOSELGEN;
	g->param = nil;
	enqueue(&c->sendq, &mysg);
	runtime·parkunlock(c, "chan send");

	if(g->param == nil) {
		runtime·lock(c);
		if(!c->closed)
			runtime·throw("chansend: spurious wakeup");
		goto closed;
	}

	if(mysg.releasetime > 0)
		runtime·blockevent(mysg.releasetime - t0, 2);

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
		mysg.g = g;
		mysg.elem = nil;
		mysg.selgen = NOSELGEN;
		enqueue(&c->sendq, &mysg);
		runtime·parkunlock(c, "chan send");

		runtime·lock(c);
		goto asynch;
	}

	if(raceenabled)
		runtime·racerelease(chanbuf(c, c->sendx));

	c->elemtype->alg->copy(c->elemsize, chanbuf(c, c->sendx), ep);
	if(++c->sendx == c->dataqsiz)
		c->sendx = 0;
	c->qcount++;

	sg = dequeue(&c->recvq);
	if(sg != nil) {
		gp = sg->g;
		runtime·unlock(c);
		if(sg->releasetime)
			sg->releasetime = runtime·cputicks();
		runtime·ready(gp);
	} else
		runtime·unlock(c);
	if(pres != nil)
		*pres = true;
	if(mysg.releasetime > 0)
		runtime·blockevent(mysg.releasetime - t0, 2);
	return;

closed:
	runtime·unlock(c);
	runtime·panicstring("send on closed channel");
}


void
runtime·chanrecv(ChanType *t, Hchan* c, byte *ep, bool *selected, bool *received)
{
	SudoG *sg;
	SudoG mysg;
	G *gp;
	int64 t0;

	// raceenabled: don't need to check ep, as it is always on the stack.

	if(debug)
		runtime·printf("chanrecv: chan=%p\n", c);

	if(c == nil) {
		USED(t);
		if(selected != nil) {
			*selected = false;
			return;
		}
		runtime·park(nil, nil, "chan receive (nil chan)");
		return;  // not reached
	}

	t0 = 0;
	mysg.releasetime = 0;
	if(runtime·blockprofilerate > 0) {
		t0 = runtime·cputicks();
		mysg.releasetime = -1;
	}

	runtime·lock(c);
	if(c->dataqsiz > 0)
		goto asynch;

	if(c->closed)
		goto closed;

	sg = dequeue(&c->sendq);
	if(sg != nil) {
		if(raceenabled)
			racesync(c, sg);
		runtime·unlock(c);

		if(ep != nil)
			c->elemtype->alg->copy(c->elemsize, ep, sg->elem);
		gp = sg->g;
		gp->param = sg;
		if(sg->releasetime)
			sg->releasetime = runtime·cputicks();
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

	mysg.elem = ep;
	mysg.g = g;
	mysg.selgen = NOSELGEN;
	g->param = nil;
	enqueue(&c->recvq, &mysg);
	runtime·parkunlock(c, "chan receive");

	if(g->param == nil) {
		runtime·lock(c);
		if(!c->closed)
			runtime·throw("chanrecv: spurious wakeup");
		goto closed;
	}

	if(received != nil)
		*received = true;
	if(mysg.releasetime > 0)
		runtime·blockevent(mysg.releasetime - t0, 2);
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
		mysg.g = g;
		mysg.elem = nil;
		mysg.selgen = NOSELGEN;
		enqueue(&c->recvq, &mysg);
		runtime·parkunlock(c, "chan receive");

		runtime·lock(c);
		goto asynch;
	}

	if(raceenabled)
		runtime·raceacquire(chanbuf(c, c->recvx));

	if(ep != nil)
		c->elemtype->alg->copy(c->elemsize, ep, chanbuf(c, c->recvx));
	c->elemtype->alg->copy(c->elemsize, chanbuf(c, c->recvx), nil);
	if(++c->recvx == c->dataqsiz)
		c->recvx = 0;
	c->qcount--;

	sg = dequeue(&c->sendq);
	if(sg != nil) {
		gp = sg->g;
		runtime·unlock(c);
		if(sg->releasetime)
			sg->releasetime = runtime·cputicks();
		runtime·ready(gp);
	} else
		runtime·unlock(c);

	if(selected != nil)
		*selected = true;
	if(received != nil)
		*received = true;
	if(mysg.releasetime > 0)
		runtime·blockevent(mysg.releasetime - t0, 2);
	return;

closed:
	if(ep != nil)
		c->elemtype->alg->copy(c->elemsize, ep, nil);
	if(selected != nil)
		*selected = true;
	if(received != nil)
		*received = false;
	if(raceenabled)
		runtime·raceacquire(c);
	runtime·unlock(c);
	if(mysg.releasetime > 0)
		runtime·blockevent(mysg.releasetime - t0, 2);
}

// chansend1(hchan *chan any, elem *any);
#pragma textflag NOSPLIT
void
runtime·chansend1(ChanType *t, Hchan* c, byte *v)
{
	runtime·chansend(t, c, v, nil, runtime·getcallerpc(&t));
}

// chanrecv1(hchan *chan any, elem *any);
#pragma textflag NOSPLIT
void
runtime·chanrecv1(ChanType *t, Hchan* c, byte *v)
{
	runtime·chanrecv(t, c, v, nil, nil);
}

// chanrecv2(hchan *chan any, elem *any) (received bool);
#pragma textflag NOSPLIT
void
runtime·chanrecv2(ChanType *t, Hchan* c, byte *v, bool received)
{
	runtime·chanrecv(t, c, v, nil, &received);
}

// func selectnbsend(c chan any, elem *any) bool
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
//	if selectnbsend(c, v) {
//		... foo
//	} else {
//		... bar
//	}
//
#pragma textflag NOSPLIT
void
runtime·selectnbsend(ChanType *t, Hchan *c, byte *val, bool pres)
{
	runtime·chansend(t, c, val, &pres, runtime·getcallerpc(&t));
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
//	if selectnbrecv(&v, c) {
//		... foo
//	} else {
//		... bar
//	}
//
#pragma textflag NOSPLIT
void
runtime·selectnbrecv(ChanType *t, byte *v, Hchan *c, bool selected)
{
	runtime·chanrecv(t, c, v, &selected, nil);
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
#pragma textflag NOSPLIT
void
runtime·selectnbrecv2(ChanType *t, byte *v, bool *received, Hchan *c, bool selected)
{
	runtime·chanrecv(t, c, v, &selected, received);
}

// For reflect:
//	func chansend(c chan, val *any, nb bool) (selected bool)
// where val points to the data to be sent.
//
// The "uintptr selected" is really "bool selected" but saying
// uintptr gets us the right alignment for the output parameter block.
#pragma textflag NOSPLIT
void
reflect·chansend(ChanType *t, Hchan *c, byte *val, bool nb, uintptr selected)
{
	bool *sp;

	if(nb) {
		selected = false;
		sp = (bool*)&selected;
	} else {
		*(bool*)&selected = true;
		FLUSH(&selected);
		sp = nil;
	}
	runtime·chansend(t, c, val, sp, runtime·getcallerpc(&t));
}

// For reflect:
//	func chanrecv(c chan, nb bool, val *any) (selected, received bool)
// where val points to a data area that will be filled in with the
// received value.  val must have the size and type of the channel element type.
void
reflect·chanrecv(ChanType *t, Hchan *c, bool nb, byte *val, bool selected, bool received)
{
	bool *sp;

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
	runtime·chanrecv(t, c, val, sp, &received);
}

static Select* newselect(int32);

// newselect(size uint32) (sel *byte);
#pragma textflag NOSPLIT
void
runtime·newselect(int32 size, byte *sel)
{
	sel = (byte*)newselect(size);
	FLUSH(&sel);
}

static Select*
newselect(int32 size)
{
	int32 n;
	Select *sel;

	n = 0;
	if(size > 1)
		n = size-1;

	// allocate all the memory we need in a single allocation
	// start with Select with size cases
	// then lockorder with size entries
	// then pollorder with size entries
	sel = runtime·mal(sizeof(*sel) +
		n*sizeof(sel->scase[0]) +
		size*sizeof(sel->lockorder[0]) +
		size*sizeof(sel->pollorder[0]));

	sel->tcase = size;
	sel->ncase = 0;
	sel->lockorder = (void*)(sel->scase + size);
	sel->pollorder = (void*)(sel->lockorder + size);

	if(debug)
		runtime·printf("newselect s=%p size=%d\n", sel, size);
	return sel;
}

// cut in half to give stack a chance to split
static void selectsend(Select *sel, Hchan *c, void *pc, void *elem, int32 so);

// selectsend(sel *byte, hchan *chan any, elem *any) (selected bool);
#pragma textflag NOSPLIT
void
runtime·selectsend(Select *sel, Hchan *c, void *elem, bool selected)
{
	selected = false;
	FLUSH(&selected);

	// nil cases do not compete
	if(c == nil)
		return;

	selectsend(sel, c, runtime·getcallerpc(&sel), elem, (byte*)&selected - (byte*)&sel);
}

static void
selectsend(Select *sel, Hchan *c, void *pc, void *elem, int32 so)
{
	int32 i;
	Scase *cas;

	i = sel->ncase;
	if(i >= sel->tcase)
		runtime·throw("selectsend: too many cases");
	sel->ncase = i+1;
	cas = &sel->scase[i];

	cas->pc = pc;
	cas->chan = c;
	cas->so = so;
	cas->kind = CaseSend;
	cas->sg.elem = elem;

	if(debug)
		runtime·printf("selectsend s=%p pc=%p chan=%p so=%d\n",
			sel, cas->pc, cas->chan, cas->so);
}

// cut in half to give stack a chance to split
static void selectrecv(Select *sel, Hchan *c, void *pc, void *elem, bool*, int32 so);

// selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
#pragma textflag NOSPLIT
void
runtime·selectrecv(Select *sel, Hchan *c, void *elem, bool selected)
{
	selected = false;
	FLUSH(&selected);

	// nil cases do not compete
	if(c == nil)
		return;

	selectrecv(sel, c, runtime·getcallerpc(&sel), elem, nil, (byte*)&selected - (byte*)&sel);
}

// selectrecv2(sel *byte, hchan *chan any, elem *any, received *bool) (selected bool);
#pragma textflag NOSPLIT
void
runtime·selectrecv2(Select *sel, Hchan *c, void *elem, bool *received, bool selected)
{
	selected = false;
	FLUSH(&selected);

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
	cas = &sel->scase[i];
	cas->pc = pc;
	cas->chan = c;

	cas->so = so;
	cas->kind = CaseRecv;
	cas->sg.elem = elem;
	cas->receivedp = received;

	if(debug)
		runtime·printf("selectrecv s=%p pc=%p chan=%p so=%d\n",
			sel, cas->pc, cas->chan, cas->so);
}

// cut in half to give stack a chance to split
static void selectdefault(Select*, void*, int32);

// selectdefault(sel *byte) (selected bool);
#pragma textflag NOSPLIT
void
runtime·selectdefault(Select *sel, bool selected)
{
	selected = false;
	FLUSH(&selected);

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
	cas = &sel->scase[i];
	cas->pc = callerpc;
	cas->chan = nil;

	cas->so = so;
	cas->kind = CaseDefault;

	if(debug)
		runtime·printf("selectdefault s=%p pc=%p so=%d\n",
			sel, cas->pc, cas->so);
}

static void
sellock(Select *sel)
{
	uint32 i;
	Hchan *c, *c0;

	c = nil;
	for(i=0; i<sel->ncase; i++) {
		c0 = sel->lockorder[i];
		if(c0 && c0 != c) {
			c = sel->lockorder[i];
			runtime·lock(c);
		}
	}
}

static void
selunlock(Select *sel)
{
	int32 i, n, r;
	Hchan *c;

	// We must be very careful here to not touch sel after we have unlocked
	// the last lock, because sel can be freed right after the last unlock.
	// Consider the following situation.
	// First M calls runtime·park() in runtime·selectgo() passing the sel.
	// Once runtime·park() has unlocked the last lock, another M makes
	// the G that calls select runnable again and schedules it for execution.
	// When the G runs on another M, it locks all the locks and frees sel.
	// Now if the first M touches sel, it will access freed memory.
	n = (int32)sel->ncase;
	r = 0;
	// skip the default case
	if(n>0 && sel->lockorder[0] == nil)
		r = 1;
	for(i = n-1; i >= r; i--) {
		c = sel->lockorder[i];
		if(i>0 && sel->lockorder[i-1] == c)
			continue;  // will unlock it on the next iteration
		runtime·unlock(c);
	}
}

static bool
selparkcommit(G *gp, void *sel)
{
	USED(gp);
	selunlock(sel);
	return true;
}

void
runtime·block(void)
{
	runtime·park(nil, nil, "select (no cases)");	// forever
}

static void* selectgo(Select**);

// selectgo(sel *byte);
//
// overwrites return pc on stack to signal which case of the select
// to run, so cannot appear at the top of a split stack.
#pragma textflag NOSPLIT
void
runtime·selectgo(Select *sel)
{
	runtime·setcallerpc(&sel, selectgo(&sel));
}

static void*
selectgo(Select **selp)
{
	Select *sel;
	uint32 o, i, j, k;
	int64 t0;
	Scase *cas, *dfl;
	Hchan *c;
	SudoG *sg;
	G *gp;
	byte *as;
	void *pc;

	sel = *selp;

	if(debug)
		runtime·printf("select: sel=%p\n", sel);

	t0 = 0;
	if(runtime·blockprofilerate > 0) {
		t0 = runtime·cputicks();
		for(i=0; i<sel->ncase; i++)
			sel->scase[i].sg.releasetime = -1;
	}

	// The compiler rewrites selects that statically have
	// only 0 or 1 cases plus default into simpler constructs.
	// The only way we can end up with such small sel->ncase
	// values here is for a larger select in which most channels
	// have been nilled out.  The general code handles those
	// cases correctly, and they are rare enough not to bother
	// optimizing (and needing to test).

	// generate permuted order
	for(i=0; i<sel->ncase; i++)
		sel->pollorder[i] = i;
	for(i=1; i<sel->ncase; i++) {
		o = sel->pollorder[i];
		j = runtime·fastrand1()%(i+1);
		sel->pollorder[i] = sel->pollorder[j];
		sel->pollorder[j] = o;
	}

	// sort the cases by Hchan address to get the locking order.
	// simple heap sort, to guarantee n log n time and constant stack footprint.
	for(i=0; i<sel->ncase; i++) {
		j = i;
		c = sel->scase[j].chan;
		while(j > 0 && sel->lockorder[k=(j-1)/2] < c) {
			sel->lockorder[j] = sel->lockorder[k];
			j = k;
		}
		sel->lockorder[j] = c;
	}
	for(i=sel->ncase; i-->0; ) {
		c = sel->lockorder[i];
		sel->lockorder[i] = sel->lockorder[0];
		j = 0;
		for(;;) {
			k = j*2+1;
			if(k >= i)
				break;
			if(k+1 < i && sel->lockorder[k] < sel->lockorder[k+1])
				k++;
			if(c < sel->lockorder[k]) {
				sel->lockorder[j] = sel->lockorder[k];
				j = k;
				continue;
			}
			break;
		}
		sel->lockorder[j] = c;
	}
	/*
	for(i=0; i+1<sel->ncase; i++)
		if(sel->lockorder[i] > sel->lockorder[i+1]) {
			runtime·printf("i=%d %p %p\n", i, sel->lockorder[i], sel->lockorder[i+1]);
			runtime·throw("select: broken sort");
		}
	*/
	sellock(sel);

loop:
	// pass 1 - look for something already waiting
	dfl = nil;
	for(i=0; i<sel->ncase; i++) {
		o = sel->pollorder[i];
		cas = &sel->scase[o];
		c = cas->chan;

		switch(cas->kind) {
		case CaseRecv:
			if(c->dataqsiz > 0) {
				if(c->qcount > 0)
					goto asyncrecv;
			} else {
				sg = dequeue(&c->sendq);
				if(sg != nil)
					goto syncrecv;
			}
			if(c->closed)
				goto rclose;
			break;

		case CaseSend:
			if(raceenabled)
				runtime·racereadpc(c, cas->pc, runtime·chansend);
			if(c->closed)
				goto sclose;
			if(c->dataqsiz > 0) {
				if(c->qcount < c->dataqsiz)
					goto asyncsend;
			} else {
				sg = dequeue(&c->recvq);
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
		selunlock(sel);
		cas = dfl;
		goto retc;
	}


	// pass 2 - enqueue on all chans
	for(i=0; i<sel->ncase; i++) {
		o = sel->pollorder[i];
		cas = &sel->scase[o];
		c = cas->chan;
		sg = &cas->sg;
		sg->g = g;
		sg->selgen = g->selgen;

		switch(cas->kind) {
		case CaseRecv:
			enqueue(&c->recvq, sg);
			break;

		case CaseSend:
			enqueue(&c->sendq, sg);
			break;
		}
	}

	g->param = nil;
	runtime·park(selparkcommit, sel, "select");

	sellock(sel);
	sg = g->param;

	// pass 3 - dequeue from unsuccessful chans
	// otherwise they stack up on quiet channels
	for(i=0; i<sel->ncase; i++) {
		cas = &sel->scase[i];
		if(cas != (Scase*)sg) {
			c = cas->chan;
			if(cas->kind == CaseSend)
				dequeueg(&c->sendq);
			else
				dequeueg(&c->recvq);
		}
	}

	if(sg == nil)
		goto loop;

	cas = (Scase*)sg;
	c = cas->chan;

	if(c->dataqsiz > 0)
		runtime·throw("selectgo: shouldn't happen");

	if(debug)
		runtime·printf("wait-return: sel=%p c=%p cas=%p kind=%d\n",
			sel, c, cas, cas->kind);

	if(cas->kind == CaseRecv) {
		if(cas->receivedp != nil)
			*cas->receivedp = true;
	}

	if(raceenabled) {
		if(cas->kind == CaseRecv && cas->sg.elem != nil)
			runtime·racewriteobjectpc(cas->sg.elem, c->elemtype, cas->pc, runtime·chanrecv);
		else if(cas->kind == CaseSend)
			runtime·racereadobjectpc(cas->sg.elem, c->elemtype, cas->pc, runtime·chansend);
	}

	selunlock(sel);
	goto retc;

asyncrecv:
	// can receive from buffer
	if(raceenabled) {
		if(cas->sg.elem != nil)
			runtime·racewriteobjectpc(cas->sg.elem, c->elemtype, cas->pc, runtime·chanrecv);
		runtime·raceacquire(chanbuf(c, c->recvx));
	}
	if(cas->receivedp != nil)
		*cas->receivedp = true;
	if(cas->sg.elem != nil)
		c->elemtype->alg->copy(c->elemsize, cas->sg.elem, chanbuf(c, c->recvx));
	c->elemtype->alg->copy(c->elemsize, chanbuf(c, c->recvx), nil);
	if(++c->recvx == c->dataqsiz)
		c->recvx = 0;
	c->qcount--;
	sg = dequeue(&c->sendq);
	if(sg != nil) {
		gp = sg->g;
		selunlock(sel);
		if(sg->releasetime)
			sg->releasetime = runtime·cputicks();
		runtime·ready(gp);
	} else {
		selunlock(sel);
	}
	goto retc;

asyncsend:
	// can send to buffer
	if(raceenabled) {
		runtime·racerelease(chanbuf(c, c->sendx));
		runtime·racereadobjectpc(cas->sg.elem, c->elemtype, cas->pc, runtime·chansend);
	}
	c->elemtype->alg->copy(c->elemsize, chanbuf(c, c->sendx), cas->sg.elem);
	if(++c->sendx == c->dataqsiz)
		c->sendx = 0;
	c->qcount++;
	sg = dequeue(&c->recvq);
	if(sg != nil) {
		gp = sg->g;
		selunlock(sel);
		if(sg->releasetime)
			sg->releasetime = runtime·cputicks();
		runtime·ready(gp);
	} else {
		selunlock(sel);
	}
	goto retc;

syncrecv:
	// can receive from sleeping sender (sg)
	if(raceenabled) {
		if(cas->sg.elem != nil)
			runtime·racewriteobjectpc(cas->sg.elem, c->elemtype, cas->pc, runtime·chanrecv);
		racesync(c, sg);
	}
	selunlock(sel);
	if(debug)
		runtime·printf("syncrecv: sel=%p c=%p o=%d\n", sel, c, o);
	if(cas->receivedp != nil)
		*cas->receivedp = true;
	if(cas->sg.elem != nil)
		c->elemtype->alg->copy(c->elemsize, cas->sg.elem, sg->elem);
	gp = sg->g;
	gp->param = sg;
	if(sg->releasetime)
		sg->releasetime = runtime·cputicks();
	runtime·ready(gp);
	goto retc;

rclose:
	// read at end of closed channel
	selunlock(sel);
	if(cas->receivedp != nil)
		*cas->receivedp = false;
	if(cas->sg.elem != nil)
		c->elemtype->alg->copy(c->elemsize, cas->sg.elem, nil);
	if(raceenabled)
		runtime·raceacquire(c);
	goto retc;

syncsend:
	// can send to sleeping receiver (sg)
	if(raceenabled) {
		runtime·racereadobjectpc(cas->sg.elem, c->elemtype, cas->pc, runtime·chansend);
		racesync(c, sg);
	}
	selunlock(sel);
	if(debug)
		runtime·printf("syncsend: sel=%p c=%p o=%d\n", sel, c, o);
	if(sg->elem != nil)
		c->elemtype->alg->copy(c->elemsize, sg->elem, cas->sg.elem);
	gp = sg->g;
	gp->param = sg;
	if(sg->releasetime)
		sg->releasetime = runtime·cputicks();
	runtime·ready(gp);

retc:
	// return pc corresponding to chosen case.
	// Set boolean passed during select creation
	// (at offset selp + cas->so) to true.
	// If cas->so == 0, this is a reflect-driven select and we
	// don't need to update the boolean.
	pc = cas->pc;
	if(cas->so > 0) {
		as = (byte*)selp + cas->so;
		*as = true;
	}
	if(cas->sg.releasetime > 0)
		runtime·blockevent(cas->sg.releasetime - t0, 2);
	runtime·free(sel);
	return pc;

sclose:
	// send on closed channel
	selunlock(sel);
	runtime·panicstring("send on closed channel");
	return nil;  // not reached
}

// This struct must match ../reflect/value.go:/runtimeSelect.
typedef struct runtimeSelect runtimeSelect;
struct runtimeSelect
{
	uintptr dir;
	ChanType *typ;
	Hchan *ch;
	byte *val;
};

// This enum must match ../reflect/value.go:/SelectDir.
enum SelectDir {
	SelectSend = 1,
	SelectRecv,
	SelectDefault,
};

// func rselect(cases []runtimeSelect) (chosen int, recvOK bool)
void
reflect·rselect(Slice cases, intgo chosen, bool recvOK)
{
	int32 i;
	Select *sel;
	runtimeSelect* rcase, *rc;

	chosen = -1;
	recvOK = false;

	rcase = (runtimeSelect*)cases.array;

	sel = newselect(cases.len);
	for(i=0; i<cases.len; i++) {
		rc = &rcase[i];
		switch(rc->dir) {
		case SelectDefault:
			selectdefault(sel, (void*)i, 0);
			break;
		case SelectSend:
			if(rc->ch == nil)
				break;
			selectsend(sel, rc->ch, (void*)i, rc->val, 0);
			break;
		case SelectRecv:
			if(rc->ch == nil)
				break;
			selectrecv(sel, rc->ch, (void*)i, rc->val, &recvOK, 0);
			break;
		}
	}

	chosen = (intgo)(uintptr)selectgo(&sel);

	FLUSH(&chosen);
	FLUSH(&recvOK);
}

static void closechan(Hchan *c, void *pc);

// closechan(sel *byte);
#pragma textflag NOSPLIT
void
runtime·closechan(Hchan *c)
{
	closechan(c, runtime·getcallerpc(&c));
}

// For reflect
//	func chanclose(c chan)
#pragma textflag NOSPLIT
void
reflect·chanclose(Hchan *c)
{
	closechan(c, runtime·getcallerpc(&c));
}

static void
closechan(Hchan *c, void *pc)
{
	SudoG *sg;
	G* gp;

	if(c == nil)
		runtime·panicstring("close of nil channel");

	runtime·lock(c);
	if(c->closed) {
		runtime·unlock(c);
		runtime·panicstring("close of closed channel");
	}

	if(raceenabled) {
		runtime·racewritepc(c, pc, runtime·closechan);
		runtime·racerelease(c);
	}

	c->closed = true;

	// release all readers
	for(;;) {
		sg = dequeue(&c->recvq);
		if(sg == nil)
			break;
		gp = sg->g;
		gp->param = nil;
		if(sg->releasetime)
			sg->releasetime = runtime·cputicks();
		runtime·ready(gp);
	}

	// release all writers
	for(;;) {
		sg = dequeue(&c->sendq);
		if(sg == nil)
			break;
		gp = sg->g;
		gp->param = nil;
		if(sg->releasetime)
			sg->releasetime = runtime·cputicks();
		runtime·ready(gp);
	}

	runtime·unlock(c);
}

// For reflect
//	func chanlen(c chan) (len int)
void
reflect·chanlen(Hchan *c, intgo len)
{
	if(c == nil)
		len = 0;
	else
		len = c->qcount;
	FLUSH(&len);
}

// For reflect
//	func chancap(c chan) int
void
reflect·chancap(Hchan *c, intgo cap)
{
	if(c == nil)
		cap = 0;
	else
		cap = c->dataqsiz;
	FLUSH(&cap);
}

static SudoG*
dequeue(WaitQ *q)
{
	SudoG *sgp;

loop:
	sgp = q->first;
	if(sgp == nil)
		return nil;
	q->first = sgp->link;

	// if sgp is stale, ignore it
	if(sgp->selgen != NOSELGEN &&
		(sgp->selgen != sgp->g->selgen ||
		!runtime·cas(&sgp->g->selgen, sgp->selgen, sgp->selgen + 2))) {
		//prints("INVALID PSEUDOG POINTER\n");
		goto loop;
	}

	return sgp;
}

static void
dequeueg(WaitQ *q)
{
	SudoG **l, *sgp, *prevsgp;

	prevsgp = nil;
	for(l=&q->first; (sgp=*l) != nil; l=&sgp->link, prevsgp=sgp) {
		if(sgp->g == g) {
			*l = sgp->link;
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

static void
racesync(Hchan *c, SudoG *sg)
{
	runtime·racerelease(chanbuf(c, 0));
	runtime·raceacquireg(sg->g, chanbuf(c, 0));
	runtime·racereleaseg(sg->g, chanbuf(c, 0));
	runtime·raceacquire(chanbuf(c, 0));
}
