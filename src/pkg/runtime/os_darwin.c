// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "signal_unix.h"
#include "stack.h"

extern SigTab runtime·sigtab[];

static Sigset sigset_none;
static Sigset sigset_all = ~(Sigset)0;
static Sigset sigset_prof = 1<<(SIGPROF-1);

static void
unimplemented(int8 *name)
{
	runtime·prints(name);
	runtime·prints(" not implemented\n");
	*(int32*)1231 = 1231;
}

int32
runtime·semasleep(int64 ns)
{
	int32 v;

	if(m->profilehz > 0)
		runtime·setprof(false);
	v = runtime·mach_semacquire(m->waitsema, ns);
	if(m->profilehz > 0)
		runtime·setprof(true);
	return v;
}

void
runtime·semawakeup(M *mp)
{
	runtime·mach_semrelease(mp->waitsema);
}

uintptr
runtime·semacreate(void)
{
	return runtime·mach_semcreate();
}

// BSD interface for threading.
void
runtime·osinit(void)
{
	// bsdthread_register delayed until end of goenvs so that we
	// can look at the environment first.

	// Use sysctl to fetch hw.ncpu.
	uint32 mib[2];
	uint32 out;
	int32 ret;
	uintptr nout;

	mib[0] = 6;
	mib[1] = 3;
	nout = sizeof out;
	out = 0;
	ret = runtime·sysctl(mib, 2, (byte*)&out, &nout, nil, 0);
	if(ret >= 0)
		runtime·ncpu = out;
}

void
runtime·get_random_data(byte **rnd, int32 *rnd_len)
{
	static byte urandom_data[HashRandomBytes];
	int32 fd;
	fd = runtime·open("/dev/urandom", 0 /* O_RDONLY */, 0);
	if(runtime·read(fd, urandom_data, HashRandomBytes) == HashRandomBytes) {
		*rnd = urandom_data;
		*rnd_len = HashRandomBytes;
	} else {
		*rnd = nil;
		*rnd_len = 0;
	}
	runtime·close(fd);
}

void
runtime·goenvs(void)
{
	runtime·goenvs_unix();

	// Register our thread-creation callback (see sys_darwin_{amd64,386}.s)
	// but only if we're not using cgo.  If we are using cgo we need
	// to let the C pthread library install its own thread-creation callback.
	if(!runtime·iscgo) {
		if(runtime·bsdthread_register() != 0) {
			if(runtime·getenv("DYLD_INSERT_LIBRARIES"))
				runtime·throw("runtime: bsdthread_register error (unset DYLD_INSERT_LIBRARIES)");
			runtime·throw("runtime: bsdthread_register error");
		}
	}

}

void
runtime·newosproc(M *mp, void *stk)
{
	int32 errno;
	Sigset oset;

	mp->tls[0] = mp->id;	// so 386 asm can find it
	if(0){
		runtime·printf("newosproc stk=%p m=%p g=%p id=%d/%d ostk=%p\n",
			stk, mp, mp->g0, mp->id, (int32)mp->tls[0], &mp);
	}

	runtime·sigprocmask(SIG_SETMASK, &sigset_all, &oset);
	errno = runtime·bsdthread_create(stk, mp, mp->g0, runtime·mstart);
	runtime·sigprocmask(SIG_SETMASK, &oset, nil);

	if(errno < 0) {
		runtime·printf("runtime: failed to create new OS thread (have %d already; errno=%d)\n", runtime·mcount(), -errno);
		runtime·throw("runtime.newosproc");
	}
}

// Called to initialize a new m (including the bootstrap m).
// Called on the parent thread (main thread in case of bootstrap), can allocate memory.
void
runtime·mpreinit(M *mp)
{
	mp->gsignal = runtime·malg(32*1024);	// OS X wants >=8K, Linux >=2K
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, can not allocate memory.
void
runtime·minit(void)
{
	// Initialize signal handling.
	runtime·signalstack((byte*)m->gsignal->stackguard - StackGuard, 32*1024);

	runtime·sigprocmask(SIG_SETMASK, &sigset_none, nil);
	runtime·setprof(m->profilehz > 0);
}

// Called from dropm to undo the effect of an minit.
void
runtime·unminit(void)
{
	runtime·signalstack(nil, 0);
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
	KERN_OPERATION_TIMED_OUT = 49,
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

// The other calls have simple system call traps in sys_darwin_{amd64,386}.s
int32 runtime·mach_semaphore_wait(uint32 sema);
int32 runtime·mach_semaphore_timedwait(uint32 sema, uint32 sec, uint32 nsec);
int32 runtime·mach_semaphore_signal(uint32 sema);
int32 runtime·mach_semaphore_signal_all(uint32 sema);

int32
runtime·mach_semacquire(uint32 sem, int64 ns)
{
	int32 r;
	int64 secs;

	if(ns >= 0) {
		secs = ns/1000000000LL;
		// Avoid overflow
		if(secs > 1LL<<30)
			secs = 1LL<<30;
		r = runtime·mach_semaphore_timedwait(sem, secs, ns%1000000000LL);
		if(r == KERN_ABORTED || r == KERN_OPERATION_TIMED_OUT)
			return -1;
		if(r != 0)
			macherror(r, "semaphore_wait");
		return 0;
	}
	while((r = runtime·mach_semaphore_wait(sem)) != 0) {
		if(r == KERN_ABORTED)	// interrupted
			continue;
		macherror(r, "semaphore_wait");
	}
	return 0;
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
		if(g->sigcode0 == BUS_ADRERR && g->sigcode1 < 0x1000) {
			if(g->sigpc == 0)
				runtime·panicstring("call of nil func value");
			runtime·panicstring("invalid memory address or nil pointer dereference");
		}
		runtime·printf("unexpected fault address %p\n", g->sigcode1);
		runtime·throw("fault");
	case SIGSEGV:
		if((g->sigcode0 == 0 || g->sigcode0 == SEGV_MAPERR || g->sigcode0 == SEGV_ACCERR) && g->sigcode1 < 0x1000) {
			if(g->sigpc == 0)
				runtime·panicstring("call of nil func value");
			runtime·panicstring("invalid memory address or nil pointer dereference");
		}
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

#pragma textflag 7
void
runtime·osyield(void)
{
	runtime·usleep(1);
}

uintptr
runtime·memlimit(void)
{
	// NOTE(rsc): Could use getrlimit here,
	// like on FreeBSD or Linux, but Darwin doesn't enforce
	// ulimit -v, so it's unclear why we'd try to stay within
	// the limit.
	return 0;
}

// NOTE(rsc): On OS X, when the CPU profiling timer expires, the SIGPROF
// signal is not guaranteed to be sent to the thread that was executing to
// cause it to expire.  It can and often does go to a sleeping thread, which is
// not interesting for our profile.  This is filed Apple Bug Report #9177434,
// copied to http://code.google.com/p/go/source/detail?r=35b716c94225.
// To work around this bug, we disable receipt of the profiling signal on
// a thread while in blocking system calls.  This forces the kernel to deliver
// the profiling signal to an executing thread.
//
// The workaround fails on OS X machines using a 64-bit Snow Leopard kernel.
// In that configuration, the kernel appears to want to deliver SIGPROF to the
// sleeping threads regardless of signal mask and, worse, does not deliver
// the signal until the thread wakes up on its own.
//
// If necessary, we can switch to using ITIMER_REAL for OS X and handle
// the kernel-generated SIGALRM by generating our own SIGALRMs to deliver
// to all the running threads.  SIGALRM does not appear to be affected by
// the 64-bit Snow Leopard bug.  However, as of this writing Mountain Lion
// is in preview, making Snow Leopard two versions old, so it is unclear how
// much effort we need to spend on one buggy kernel.

// Control whether profiling signal can be delivered to this thread.
void
runtime·setprof(bool on)
{
	if(on)
		runtime·sigprocmask(SIG_UNBLOCK, &sigset_prof, nil);
	else
		runtime·sigprocmask(SIG_BLOCK, &sigset_prof, nil);
}

static int8 badcallback[] = "runtime: cgo callback on thread not created by Go.\n";

// This runs on a foreign stack, without an m or a g.  No stack split.
#pragma textflag 7
void
runtime·badcallback(void)
{
	runtime·write(2, badcallback, sizeof badcallback - 1);
}

static int8 badsignal[] = "runtime: signal received on thread not created by Go: ";

// This runs on a foreign stack, without an m or a g.  No stack split.
#pragma textflag 7
void
runtime·badsignal(int32 sig)
{
	int32 len;

	if (sig == SIGPROF) {
		return;  // Ignore SIGPROFs intended for a non-Go thread.
	}
	runtime·write(2, badsignal, sizeof badsignal - 1);
	if (0 <= sig && sig < NSIG) {
		// Can't call findnull() because it will split stack.
		for(len = 0; runtime·sigtab[sig].name[len]; len++)
			;
		runtime·write(2, runtime·sigtab[sig].name, len);
	}
	runtime·write(2, "\n", 1);
	runtime·exit(1);
}

void
runtime·setsig(int32 i, GoSighandler *fn, bool restart)
{
	Sigaction sa;
		
	runtime·memclr((byte*)&sa, sizeof sa);
	sa.sa_flags = SA_SIGINFO|SA_ONSTACK;
	if(restart)
		sa.sa_flags |= SA_RESTART;
	sa.sa_mask = ~(uintptr)0;
	sa.sa_tramp = (void*)runtime·sigtramp;	// runtime·sigtramp's job is to call into real handler
	*(uintptr*)sa.__sigaction_u = (uintptr)fn;
	runtime·sigaction(i, &sa, nil);
}

GoSighandler*
runtime·getsig(int32 i)
{
	Sigaction sa;

	runtime·memclr((byte*)&sa, sizeof sa);
	runtime·sigaction(i, nil, &sa);
	return *(void**)sa.__sigaction_u;
}

void
runtime·signalstack(byte *p, int32 n)
{
	StackT st;

	st.ss_sp = (void*)p;
	st.ss_size = n;
	st.ss_flags = 0;
	if(p == nil)
		st.ss_flags = SS_DISABLE;
	runtime·sigaltstack(&st, nil);
}
