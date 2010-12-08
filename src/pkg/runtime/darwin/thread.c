// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs.h"
#include "os.h"

extern SigTab runtime·sigtab[];

static void
unimplemented(int8 *name)
{
	runtime·prints(name);
	runtime·prints(" not implemented\n");
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

	sema = runtime·mach_semcreate();
	if(!runtime·cas(psema, 0, sema)){
		// Someone else filled it in.  Use theirs.
		runtime·mach_semdestroy(sema);
		return;
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

void
runtime·lock(Lock *l)
{
	if(m->locks < 0)
		runtime·throw("lock count");
	m->locks++;

	if(runtime·xadd(&l->key, 1) > 1) {	// someone else has it; wait
		// Allocate semaphore if needed.
		if(l->sema == 0)
			initsema(&l->sema);
		runtime·mach_semacquire(l->sema);
	}
}

void
runtime·unlock(Lock *l)
{
	m->locks--;
	if(m->locks < 0)
		runtime·throw("lock count");

	if(runtime·xadd(&l->key, -1) > 0) {	// someone else is waiting
		// Allocate semaphore if needed.
		if(l->sema == 0)
			initsema(&l->sema);
		runtime·mach_semrelease(l->sema);
	}
}

void
runtime·destroylock(Lock *l)
{
	if(l->sema != 0) {
		runtime·mach_semdestroy(l->sema);
		l->sema = 0;
	}
}

// User-level semaphore implementation:
// try to do the operations in user space on u,
// but when it's time to block, fall back on the kernel semaphore k.
// This is the same algorithm used in Plan 9.
void
runtime·usemacquire(Usema *s)
{
	if((int32)runtime·xadd(&s->u, -1) < 0) {
		if(s->k == 0)
			initsema(&s->k);
		runtime·mach_semacquire(s->k);
	}
}

void
runtime·usemrelease(Usema *s)
{
	if((int32)runtime·xadd(&s->u, 1) <= 0) {
		if(s->k == 0)
			initsema(&s->k);
		runtime·mach_semrelease(s->k);
	}
}


// Event notifications.
void
runtime·noteclear(Note *n)
{
	n->wakeup = 0;
}

void
runtime·notesleep(Note *n)
{
	while(!n->wakeup)
		runtime·usemacquire(&n->sema);
}

void
runtime·notewakeup(Note *n)
{
	n->wakeup = 1;
	runtime·usemrelease(&n->sema);
}


// BSD interface for threading.
void
runtime·osinit(void)
{
	// Register our thread-creation callback (see {amd64,386}/sys.s)
	// but only if we're not using cgo.  If we are using cgo we need
	// to let the C pthread libary install its own thread-creation callback.
	if(!runtime·iscgo)
		runtime·bsdthread_register();
}

void
runtime·newosproc(M *m, G *g, void *stk, void (*fn)(void))
{
	m->tls[0] = m->id;	// so 386 asm can find it
	if(0){
		runtime·printf("newosproc stk=%p m=%p g=%p fn=%p id=%d/%d ostk=%p\n",
			stk, m, g, fn, m->id, m->tls[0], &m);
	}
	if(runtime·bsdthread_create(stk, m, g, fn) < 0)
		runtime·throw("cannot create new OS thread");
}

// Called to initialize a new m (including the bootstrap m).
void
runtime·minit(void)
{
	// Initialize signal handling.
	m->gsignal = runtime·malg(32*1024);	// OS X wants >=8K, Linux >=2K
	runtime·signalstack(m->gsignal->stackguard, 32*1024);
}

// Mach IPC, to get at semaphores
// Definitions are in /usr/include/mach on a Mac.

static void
macherror(int32 r, int8 *fn)
{
	runtime·printf("mach error %s: %d\n", fn, r);
	runtime·throw("mach error");
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
	return runtime·mach_msg_trap(h, op, send_size, rcv_size, rcv_name, timeout, notify);
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
		port = runtime·mach_reply_port();
		m->machport = port;
	}

	h->msgh_bits |= MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, MACH_MSG_TYPE_MAKE_SEND_ONCE);
	h->msgh_local_port = port;
	h->msgh_reserved = 0;
	id = h->msgh_id;

	if(DebugMach){
		p = (uint32*)h;
		runtime·prints("send:\t");
		for(i=0; i<h->msgh_size/sizeof(p[0]); i++){
			runtime·prints(" ");
			runtime·printpointer((void*)p[i]);
			if(i%8 == 7)
				runtime·prints("\n\t");
		}
		if(i%8)
			runtime·prints("\n");
	}

	ret = mach_msg(h, MACH_SEND_MSG|MACH_RCV_MSG,
		h->msgh_size, maxsize, port, 0, 0);
	if(ret != 0){
		if(DebugMach){
			runtime·prints("mach_msg error ");
			runtime·printint(ret);
			runtime·prints("\n");
		}
		return ret;
	}

	if(DebugMach){
		p = (uint32*)h;
		runtime·prints("recv:\t");
		for(i=0; i<h->msgh_size/sizeof(p[0]); i++){
			runtime·prints(" ");
			runtime·printpointer((void*)p[i]);
			if(i%8 == 7)
				runtime·prints("\n\t");
		}
		if(i%8)
			runtime·prints("\n");
	}

	if(h->msgh_id != id+Reply){
		if(DebugMach){
			runtime·prints("mach_msg reply id mismatch ");
			runtime·printint(h->msgh_id);
			runtime·prints(" != ");
			runtime·printint(id+Reply);
			runtime·prints("\n");
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
			runtime·prints("mig result ");
			runtime·printint(c->code);
			runtime·prints("\n");
		}
		return c->code;
	}

	if(h->msgh_size != rxsize){
		if(DebugMach){
			runtime·prints("mach_msg reply size mismatch ");
			runtime·printint(h->msgh_size);
			runtime·prints(" != ");
			runtime·printint(rxsize);
			runtime·prints("\n");
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
runtime·mach_semcreate(void)
{
	union {
		Tmach_semcreateMsg tx;
		Rmach_semcreateMsg rx;
		uint8 pad[MinMachMsg];
	} m;
	int32 r;

	m.tx.h.msgh_bits = 0;
	m.tx.h.msgh_size = sizeof(m.tx);
	m.tx.h.msgh_remote_port = runtime·mach_task_self();
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
runtime·mach_semdestroy(uint32 sem)
{
	union {
		Tmach_semdestroyMsg tx;
		uint8 pad[MinMachMsg];
	} m;
	int32 r;

	m.tx.h.msgh_bits = MACH_MSGH_BITS_COMPLEX;
	m.tx.h.msgh_size = sizeof(m.tx);
	m.tx.h.msgh_remote_port = runtime·mach_task_self();
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
int32 runtime·mach_semaphore_wait(uint32 sema);
int32 runtime·mach_semaphore_timedwait(uint32 sema, uint32 sec, uint32 nsec);
int32 runtime·mach_semaphore_signal(uint32 sema);
int32 runtime·mach_semaphore_signal_all(uint32 sema);

void
runtime·mach_semacquire(uint32 sem)
{
	int32 r;

	while((r = runtime·mach_semaphore_wait(sem)) != 0) {
		if(r == KERN_ABORTED)	// interrupted
			continue;
		macherror(r, "semaphore_wait");
	}
}

void
runtime·mach_semrelease(uint32 sem)
{
	int32 r;

	while((r = runtime·mach_semaphore_signal(sem)) != 0) {
		if(r == KERN_ABORTED)	// interrupted
			continue;
		macherror(r, "semaphore_signal");
	}
}

void
runtime·sigpanic(void)
{
	switch(g->sig) {
	case SIGBUS:
		if(g->sigcode0 == BUS_ADRERR && g->sigcode1 < 0x1000)
			runtime·panicstring("invalid memory address or nil pointer dereference");
		runtime·printf("unexpected fault address %p\n", g->sigcode1);
		runtime·throw("fault");
	case SIGSEGV:
		if((g->sigcode0 == 0 || g->sigcode0 == SEGV_MAPERR || g->sigcode0 == SEGV_ACCERR) && g->sigcode1 < 0x1000)
			runtime·panicstring("invalid memory address or nil pointer dereference");
		runtime·printf("unexpected fault address %p\n", g->sigcode1);
		runtime·throw("fault");
	case SIGFPE:
		switch(g->sigcode0) {
		case FPE_INTDIV:
			runtime·panicstring("integer divide by zero");
		case FPE_INTOVF:
			runtime·panicstring("integer overflow");
		}
		runtime·panicstring("floating point error");
	}
	runtime·panicstring(runtime·sigtab[g->sig].name);
}
