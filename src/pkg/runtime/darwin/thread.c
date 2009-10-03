// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs.h"
#include "os.h"

static void
unimplemented(int8 *name)
{
	prints(name);
	prints(" not implemented\n");
	*(int32*)1231 = 1231;
}

// Thread-safe allocation of a semaphore.
// Psema points at a kernel semaphore key.
// It starts out zero, meaning no semaphore.
// Fill it in, being careful of others calling initsema
// simultaneously.
static void
initsema(uint32 *psema)
{
	uint32 sema;

	if(*psema != 0)	// already have one
		return;

	sema = mach_semcreate();
	if(!cas(psema, 0, sema)){
		// Someone else filled it in.  Use theirs.
		mach_semdestroy(sema);
		return;
	}
}


// Atomic add and return new value.
static uint32
xadd(uint32 volatile *val, int32 delta)
{
	uint32 oval, nval;

	for(;;){
		oval = *val;
		nval = oval + delta;
		if(cas(val, oval, nval))
			return nval;
	}
}


// Blocking locks.

// Implement Locks, using semaphores.
// l->key is the number of threads who want the lock.
// In a race, one thread increments l->key from 0 to 1
// and the others increment it from >0 to >1.  The thread
// who does the 0->1 increment gets the lock, and the
// others wait on the semaphore.  When the 0->1 thread
// releases the lock by decrementing l->key, l->key will
// be >0, so it will increment the semaphore to wake up
// one of the others.  This is the same algorithm used
// in Plan 9's user-level locks.
//
// Note that semaphores are never destroyed (the kernel
// will clean up when the process exits).  We assume for now
// that Locks are only used for long-lived structures like M and G.

void
lock(Lock *l)
{
	if(m->locks < 0)
		throw("lock count");
	m->locks++;

	// Allocate semaphore if needed.
	if(l->sema == 0)
		initsema(&l->sema);

	if(xadd(&l->key, 1) > 1)	// someone else has it; wait
		mach_semacquire(l->sema);
}

void
unlock(Lock *l)
{
	m->locks--;
	if(m->locks < 0)
		throw("lock count");

	if(xadd(&l->key, -1) > 0)	// someone else is waiting
		mach_semrelease(l->sema);
}


// User-level semaphore implementation:
// try to do the operations in user space on u,
// but when it's time to block, fall back on the kernel semaphore k.
// This is the same algorithm used in Plan 9.
void
usemacquire(Usema *s)
{
	if((int32)xadd(&s->u, -1) < 0)
		mach_semacquire(s->k);
}

void
usemrelease(Usema *s)
{
	if((int32)xadd(&s->u, 1) <= 0)
		mach_semrelease(s->k);
}


// Event notifications.
void
noteclear(Note *n)
{
	n->wakeup = 0;
}

void
notesleep(Note *n)
{
	if(n->sema.k == 0)
		initsema(&n->sema.k);
	while(!n->wakeup)
		usemacquire(&n->sema);
}

void
notewakeup(Note *n)
{
	if(n->sema.k == 0)
		initsema(&n->sema.k);
	n->wakeup = 1;
	usemrelease(&n->sema);
}


// BSD interface for threading.
void
osinit(void)
{
	// Register our thread-creation callback (see {amd64,386}/sys.s)
	// but only if we're not using cgo.  If we are using cgo we need
	// to let the C pthread libary install its own thread-creation callback.
	extern void (*libcgo_thread_start)(void*);
	if(libcgo_thread_start == nil)
		bsdthread_register();
}

void
newosproc(M *m, G *g, void *stk, void (*fn)(void))
{
	m->tls[0] = m->id;	// so 386 asm can find it
	if(0){
		printf("newosproc stk=%p m=%p g=%p fn=%p id=%d/%d ostk=%p\n",
			stk, m, g, fn, m->id, m->tls[0], &m);
	}
	bsdthread_create(stk, m, g, fn);
}

// Called to initialize a new m (including the bootstrap m).
void
minit(void)
{
	// Initialize signal handling.
	m->gsignal = malg(32*1024);	// OS X wants >=8K, Linux >=2K
	signalstack(m->gsignal->stackguard, 32*1024);
}

// Mach IPC, to get at semaphores
// Definitions are in /usr/include/mach on a Mac.

static void
macherror(int32 r, int8 *fn)
{
	printf("mach error %s: %d\n", fn, r);
	throw("mach error");
}

enum
{
	DebugMach = 0
};

static MachNDR zerondr;

#define MACH_MSGH_BITS(a, b) ((a) | ((b)<<8))

static int32
mach_msg(MachHeader *h,
	int32 op,
	uint32 send_size,
	uint32 rcv_size,
	uint32 rcv_name,
	uint32 timeout,
	uint32 notify)
{
	// TODO: Loop on interrupt.
	return mach_msg_trap(h, op, send_size, rcv_size, rcv_name, timeout, notify);
}

// Mach RPC (MIG)

enum
{
	MinMachMsg = 48,
	Reply = 100,
};

#pragma pack on
typedef struct CodeMsg CodeMsg;
struct CodeMsg
{
	MachHeader h;
	MachNDR NDR;
	int32 code;
};
#pragma pack off

static int32
machcall(MachHeader *h, int32 maxsize, int32 rxsize)
{
	uint32 *p;
	int32 i, ret, id;
	uint32 port;
	CodeMsg *c;

	if((port = m->machport) == 0){
		port = mach_reply_port();
		m->machport = port;
	}

	h->msgh_bits |= MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, MACH_MSG_TYPE_MAKE_SEND_ONCE);
	h->msgh_local_port = port;
	h->msgh_reserved = 0;
	id = h->msgh_id;

	if(DebugMach){
		p = (uint32*)h;
		prints("send:\t");
		for(i=0; i<h->msgh_size/sizeof(p[0]); i++){
			prints(" ");
			sys·printpointer((void*)p[i]);
			if(i%8 == 7)
				prints("\n\t");
		}
		if(i%8)
			prints("\n");
	}

	ret = mach_msg(h, MACH_SEND_MSG|MACH_RCV_MSG,
		h->msgh_size, maxsize, port, 0, 0);
	if(ret != 0){
		if(DebugMach){
			prints("mach_msg error ");
			sys·printint(ret);
			prints("\n");
		}
		return ret;
	}

	if(DebugMach){
		p = (uint32*)h;
		prints("recv:\t");
		for(i=0; i<h->msgh_size/sizeof(p[0]); i++){
			prints(" ");
			sys·printpointer((void*)p[i]);
			if(i%8 == 7)
				prints("\n\t");
		}
		if(i%8)
			prints("\n");
	}

	if(h->msgh_id != id+Reply){
		if(DebugMach){
			prints("mach_msg reply id mismatch ");
			sys·printint(h->msgh_id);
			prints(" != ");
			sys·printint(id+Reply);
			prints("\n");
		}
		return -303;	// MIG_REPLY_MISMATCH
	}

	// Look for a response giving the return value.
	// Any call can send this back with an error,
	// and some calls only have return values so they
	// send it back on success too.  I don't quite see how
	// you know it's one of these and not the full response
	// format, so just look if the message is right.
	c = (CodeMsg*)h;
	if(h->msgh_size == sizeof(CodeMsg)
	&& !(h->msgh_bits & MACH_MSGH_BITS_COMPLEX)){
		if(DebugMach){
			prints("mig result ");
			sys·printint(c->code);
			prints("\n");
		}
		return c->code;
	}

	if(h->msgh_size != rxsize){
		if(DebugMach){
			prints("mach_msg reply size mismatch ");
			sys·printint(h->msgh_size);
			prints(" != ");
			sys·printint(rxsize);
			prints("\n");
		}
		return -307;	// MIG_ARRAY_TOO_LARGE
	}

	return 0;
}


// Semaphores!

enum
{
	Tmach_semcreate = 3418,
	Rmach_semcreate = Tmach_semcreate + Reply,

	Tmach_semdestroy = 3419,
	Rmach_semdestroy = Tmach_semdestroy + Reply,

	// Mach calls that get interrupted by Unix signals
	// return this error code.  We retry them.
	KERN_ABORTED = 14,
};

typedef struct Tmach_semcreateMsg Tmach_semcreateMsg;
typedef struct Rmach_semcreateMsg Rmach_semcreateMsg;
typedef struct Tmach_semdestroyMsg Tmach_semdestroyMsg;
// Rmach_semdestroyMsg = CodeMsg

#pragma pack on
struct Tmach_semcreateMsg
{
	MachHeader h;
	MachNDR ndr;
	int32 policy;
	int32 value;
};

struct Rmach_semcreateMsg
{
	MachHeader h;
	MachBody body;
	MachPort semaphore;
};

struct Tmach_semdestroyMsg
{
	MachHeader h;
	MachBody body;
	MachPort semaphore;
};
#pragma pack off

uint32
mach_semcreate(void)
{
	union {
		Tmach_semcreateMsg tx;
		Rmach_semcreateMsg rx;
		uint8 pad[MinMachMsg];
	} m;
	int32 r;

	m.tx.h.msgh_bits = 0;
	m.tx.h.msgh_size = sizeof(m.tx);
	m.tx.h.msgh_remote_port = mach_task_self();
	m.tx.h.msgh_id = Tmach_semcreate;
	m.tx.ndr = zerondr;

	m.tx.policy = 0;	// 0 = SYNC_POLICY_FIFO
	m.tx.value = 0;

	while((r = machcall(&m.tx.h, sizeof m, sizeof(m.rx))) != 0){
		if(r == KERN_ABORTED)	// interrupted
			continue;
		macherror(r, "semaphore_create");
	}
	if(m.rx.body.msgh_descriptor_count != 1)
		unimplemented("mach_semcreate desc count");
	return m.rx.semaphore.name;
}

void
mach_semdestroy(uint32 sem)
{
	union {
		Tmach_semdestroyMsg tx;
		uint8 pad[MinMachMsg];
	} m;
	int32 r;

	m.tx.h.msgh_bits = MACH_MSGH_BITS_COMPLEX;
	m.tx.h.msgh_size = sizeof(m.tx);
	m.tx.h.msgh_remote_port = mach_task_self();
	m.tx.h.msgh_id = Tmach_semdestroy;
	m.tx.body.msgh_descriptor_count = 1;
	m.tx.semaphore.name = sem;
	m.tx.semaphore.disposition = MACH_MSG_TYPE_MOVE_SEND;
	m.tx.semaphore.type = 0;

	while((r = machcall(&m.tx.h, sizeof m, 0)) != 0){
		if(r == KERN_ABORTED)	// interrupted
			continue;
		macherror(r, "semaphore_destroy");
	}
}

// The other calls have simple system call traps in sys.s
int32 mach_semaphore_wait(uint32 sema);
int32 mach_semaphore_timedwait(uint32 sema, uint32 sec, uint32 nsec);
int32 mach_semaphore_signal(uint32 sema);
int32 mach_semaphore_signal_all(uint32 sema);

void
mach_semacquire(uint32 sem)
{
	int32 r;

	while((r = mach_semaphore_wait(sem)) != 0) {
		if(r == KERN_ABORTED)	// interrupted
			continue;
		macherror(r, "semaphore_wait");
	}
}

void
mach_semrelease(uint32 sem)
{
	int32 r;

	while((r = mach_semaphore_signal(sem)) != 0) {
		if(r == KERN_ABORTED)	// interrupted
			continue;
		macherror(r, "semaphore_signal");
	}
}

