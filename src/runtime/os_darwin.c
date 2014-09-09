// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "signal_unix.h"
#include "stack.h"
#include "textflag.h"

extern SigTab runtime·sigtab[];

static Sigset sigset_none;
static Sigset sigset_all = ~(Sigset)0;

static void
unimplemented(int8 *name)
{
	runtime·prints(name);
	runtime·prints(" not implemented\n");
	*(int32*)1231 = 1231;
}

#pragma textflag NOSPLIT
void
runtime·semawakeup(M *mp)
{
	runtime·mach_semrelease(mp->waitsema);
}

static void
semacreate(void)
{
	g->m->scalararg[0] = runtime·mach_semcreate();
}

#pragma textflag NOSPLIT
uintptr
runtime·semacreate(void)
{
	uintptr x;
	void (*fn)(void);
	
	fn = semacreate;
	runtime·onM(&fn);
	x = g->m->scalararg[0];
	g->m->scalararg[0] = 0;
	return x;
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

#pragma textflag NOSPLIT
void
runtime·get_random_data(byte **rnd, int32 *rnd_len)
{
	#pragma dataflag NOPTR
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
	mp->gsignal->m = mp;
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, can not allocate memory.
void
runtime·minit(void)
{
	// Initialize signal handling.
	runtime·signalstack((byte*)g->m->gsignal->stack.lo, 32*1024);

	runtime·sigprocmask(SIG_SETMASK, &sigset_none, nil);
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
	runtime·prints("mach error ");
	runtime·prints(fn);
	runtime·prints(": ");
	runtime·printint(r);
	runtime·prints("\n");
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

	if((port = g->m->machport) == 0){
		port = runtime·mach_reply_port();
		g->m->machport = port;
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

static void
semasleep(void)
{
	int32 r, secs, nsecs;
	int64 ns;
	
	ns = (int64)(uint32)g->m->scalararg[0] | (int64)(uint32)g->m->scalararg[1]<<32;
	g->m->scalararg[0] = 0;
	g->m->scalararg[1] = 0;

	if(ns >= 0) {
		secs = runtime·timediv(ns, 1000000000, &nsecs);
		r = runtime·mach_semaphore_timedwait(g->m->waitsema, secs, nsecs);
		if(r == KERN_ABORTED || r == KERN_OPERATION_TIMED_OUT) {
			g->m->scalararg[0] = -1;
			return;
		}
		if(r != 0)
			macherror(r, "semaphore_wait");
		g->m->scalararg[0] = 0;
		return;
	}
	while((r = runtime·mach_semaphore_wait(g->m->waitsema)) != 0) {
		if(r == KERN_ABORTED)	// interrupted
			continue;
		macherror(r, "semaphore_wait");
	}
	g->m->scalararg[0] = 0;
	return;
}

#pragma textflag NOSPLIT
int32
runtime·semasleep(int64 ns)
{
	int32 r;
	void (*fn)(void);

	g->m->scalararg[0] = (uint32)ns;
	g->m->scalararg[1] = (uint32)(ns>>32);
	fn = semasleep;
	runtime·onM(&fn);
	r = g->m->scalararg[0];
	g->m->scalararg[0] = 0;
	return r;
}

static int32 mach_semrelease_errno;

static void
mach_semrelease_fail(void)
{
	macherror(mach_semrelease_errno, "semaphore_signal");
}

#pragma textflag NOSPLIT
void
runtime·mach_semrelease(uint32 sem)
{
	int32 r;
	void (*fn)(void);

	while((r = runtime·mach_semaphore_signal(sem)) != 0) {
		if(r == KERN_ABORTED)	// interrupted
			continue;
		
		// mach_semrelease must be completely nosplit,
		// because it is called from Go code.
		// If we're going to die, start that process on the m stack
		// to avoid a Go stack split.
		// Only do that if we're actually running on the g stack.
		// We might be on the gsignal stack, and if so, onM will abort.
		// We use the global variable instead of scalararg because
		// we might be on the gsignal stack, having interrupted a
		// normal call to onM. It doesn't quite matter, since the
		// program is about to die, but better to be clean.
		mach_semrelease_errno = r;
		fn = mach_semrelease_fail;
		if(g == g->m->curg)
			runtime·onM(&fn);
		else
			fn();
	}
}

#pragma textflag NOSPLIT
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

void
runtime·setsig(int32 i, GoSighandler *fn, bool restart)
{
	SigactionT sa;
		
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
	SigactionT sa;

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

void
runtime·unblocksignals(void)
{
	runtime·sigprocmask(SIG_SETMASK, &sigset_none, nil);
}

#pragma textflag NOSPLIT
int8*
runtime·signame(int32 sig)
{
	return runtime·sigtab[sig].name;
}
