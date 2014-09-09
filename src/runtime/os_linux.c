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
static Sigset sigset_all = { ~(uint32)0, ~(uint32)0 };

// Linux futex.
//
//	futexsleep(uint32 *addr, uint32 val)
//	futexwakeup(uint32 *addr)
//
// Futexsleep atomically checks if *addr == val and if so, sleeps on addr.
// Futexwakeup wakes up threads sleeping on addr.
// Futexsleep is allowed to wake up spuriously.

enum
{
	FUTEX_WAIT = 0,
	FUTEX_WAKE = 1,
};

// Atomically,
//	if(*addr == val) sleep
// Might be woken up spuriously; that's allowed.
// Don't sleep longer than ns; ns < 0 means forever.
#pragma textflag NOSPLIT
void
runtime·futexsleep(uint32 *addr, uint32 val, int64 ns)
{
	Timespec ts;

	// Some Linux kernels have a bug where futex of
	// FUTEX_WAIT returns an internal error code
	// as an errno.  Libpthread ignores the return value
	// here, and so can we: as it says a few lines up,
	// spurious wakeups are allowed.

	if(ns < 0) {
		runtime·futex(addr, FUTEX_WAIT, val, nil, nil, 0);
		return;
	}
	// NOTE: tv_nsec is int64 on amd64, so this assumes a little-endian system.
	ts.tv_nsec = 0;
	ts.tv_sec = runtime·timediv(ns, 1000000000LL, (int32*)&ts.tv_nsec);
	runtime·futex(addr, FUTEX_WAIT, val, &ts, nil, 0);
}

static void badfutexwakeup(void);

// If any procs are sleeping on addr, wake up at most cnt.
#pragma textflag NOSPLIT
void
runtime·futexwakeup(uint32 *addr, uint32 cnt)
{
	int64 ret;
	void (*fn)(void);

	ret = runtime·futex(addr, FUTEX_WAKE, cnt, nil, nil, 0);
	if(ret >= 0)
		return;

	// I don't know that futex wakeup can return
	// EAGAIN or EINTR, but if it does, it would be
	// safe to loop and call futex again.
	g->m->ptrarg[0] = addr;
	g->m->scalararg[0] = (int32)ret; // truncated but fine
	fn = badfutexwakeup;
	if(g == g->m->gsignal)
		fn();
	else
		runtime·onM(&fn);
	*(int32*)0x1006 = 0x1006;
}

static void
badfutexwakeup(void)
{
	void *addr;
	int64 ret;
	
	addr = g->m->ptrarg[0];
	ret = (int32)g->m->scalararg[0];
	runtime·printf("futexwakeup addr=%p returned %D\n", addr, ret);
}

extern runtime·sched_getaffinity(uintptr pid, uintptr len, uintptr *buf);
static int32
getproccount(void)
{
	uintptr buf[16], t;
	int32 r, cnt, i;

	cnt = 0;
	r = runtime·sched_getaffinity(0, sizeof(buf), buf);
	if(r > 0)
	for(i = 0; i < r/sizeof(buf[0]); i++) {
		t = buf[i];
		t = t - ((t >> 1) & 0x5555555555555555ULL);
		t = (t & 0x3333333333333333ULL) + ((t >> 2) & 0x3333333333333333ULL);
		cnt += (int32)((((t + (t >> 4)) & 0xF0F0F0F0F0F0F0FULL) * 0x101010101010101ULL) >> 56);
	}

	return cnt ? cnt : 1;
}

// Clone, the Linux rfork.
enum
{
	CLONE_VM = 0x100,
	CLONE_FS = 0x200,
	CLONE_FILES = 0x400,
	CLONE_SIGHAND = 0x800,
	CLONE_PTRACE = 0x2000,
	CLONE_VFORK = 0x4000,
	CLONE_PARENT = 0x8000,
	CLONE_THREAD = 0x10000,
	CLONE_NEWNS = 0x20000,
	CLONE_SYSVSEM = 0x40000,
	CLONE_SETTLS = 0x80000,
	CLONE_PARENT_SETTID = 0x100000,
	CLONE_CHILD_CLEARTID = 0x200000,
	CLONE_UNTRACED = 0x800000,
	CLONE_CHILD_SETTID = 0x1000000,
	CLONE_STOPPED = 0x2000000,
	CLONE_NEWUTS = 0x4000000,
	CLONE_NEWIPC = 0x8000000,
};

void
runtime·newosproc(M *mp, void *stk)
{
	int32 ret;
	int32 flags;
	Sigset oset;

	/*
	 * note: strace gets confused if we use CLONE_PTRACE here.
	 */
	flags = CLONE_VM	/* share memory */
		| CLONE_FS	/* share cwd, etc */
		| CLONE_FILES	/* share fd table */
		| CLONE_SIGHAND	/* share sig handler table */
		| CLONE_THREAD	/* revisit - okay for now */
		;

	mp->tls[0] = mp->id;	// so 386 asm can find it
	if(0){
		runtime·printf("newosproc stk=%p m=%p g=%p clone=%p id=%d/%d ostk=%p\n",
			stk, mp, mp->g0, runtime·clone, mp->id, (int32)mp->tls[0], &mp);
	}

	// Disable signals during clone, so that the new thread starts
	// with signals disabled.  It will enable them in minit.
	runtime·rtsigprocmask(SIG_SETMASK, &sigset_all, &oset, sizeof oset);
	ret = runtime·clone(flags, stk, mp, mp->g0, runtime·mstart);
	runtime·rtsigprocmask(SIG_SETMASK, &oset, nil, sizeof oset);

	if(ret < 0) {
		runtime·printf("runtime: failed to create new OS thread (have %d already; errno=%d)\n", runtime·mcount(), -ret);
		runtime·throw("runtime.newosproc");
	}
}

void
runtime·osinit(void)
{
	runtime·ncpu = getproccount();
}

// Random bytes initialized at startup.  These come
// from the ELF AT_RANDOM auxiliary vector (vdso_linux_amd64.c).
byte*	runtime·startup_random_data;
uint32	runtime·startup_random_data_len;

#pragma textflag NOSPLIT
void
runtime·get_random_data(byte **rnd, int32 *rnd_len)
{
	if(runtime·startup_random_data != nil) {
		*rnd = runtime·startup_random_data;
		*rnd_len = runtime·startup_random_data_len;
	} else {
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
}

void
runtime·goenvs(void)
{
	runtime·goenvs_unix();
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
	runtime·rtsigprocmask(SIG_SETMASK, &sigset_none, nil, sizeof(Sigset));
}

// Called from dropm to undo the effect of an minit.
void
runtime·unminit(void)
{
	runtime·signalstack(nil, 0);
}

uintptr
runtime·memlimit(void)
{
	Rlimit rl;
	extern byte runtime·text[], runtime·end[];
	uintptr used;

	if(runtime·getrlimit(RLIMIT_AS, &rl) != 0)
		return 0;
	if(rl.rlim_cur >= 0x7fffffff)
		return 0;

	// Estimate our VM footprint excluding the heap.
	// Not an exact science: use size of binary plus
	// some room for thread stacks.
	used = runtime·end - runtime·text + (64<<20);
	if(used >= rl.rlim_cur)
		return 0;

	// If there's not at least 16 MB left, we're probably
	// not going to be able to do much.  Treat as no limit.
	rl.rlim_cur -= used;
	if(rl.rlim_cur < (16<<20))
		return 0;

	return rl.rlim_cur - used;
}

#ifdef GOARCH_386
#define sa_handler k_sa_handler
#endif

/*
 * This assembler routine takes the args from registers, puts them on the stack,
 * and calls sighandler().
 */
extern void runtime·sigtramp(void);
extern void runtime·sigreturn(void);	// calls rt_sigreturn, only used with SA_RESTORER

void
runtime·setsig(int32 i, GoSighandler *fn, bool restart)
{
	SigactionT sa;

	runtime·memclr((byte*)&sa, sizeof sa);
	sa.sa_flags = SA_ONSTACK | SA_SIGINFO | SA_RESTORER;
	if(restart)
		sa.sa_flags |= SA_RESTART;
	sa.sa_mask = ~0ULL;
	// Although Linux manpage says "sa_restorer element is obsolete and
	// should not be used". x86_64 kernel requires it. Only use it on
	// x86.
#ifdef GOARCH_386
	sa.sa_restorer = (void*)runtime·sigreturn;
#endif
#ifdef GOARCH_amd64
	sa.sa_restorer = (void*)runtime·sigreturn;
#endif
	if(fn == runtime·sighandler)
		fn = (void*)runtime·sigtramp;
	sa.sa_handler = fn;
	if(runtime·rt_sigaction(i, &sa, nil, sizeof(sa.sa_mask)) != 0)
		runtime·throw("rt_sigaction failure");
}

GoSighandler*
runtime·getsig(int32 i)
{
	SigactionT sa;

	runtime·memclr((byte*)&sa, sizeof sa);
	if(runtime·rt_sigaction(i, nil, &sa, sizeof(sa.sa_mask)) != 0)
		runtime·throw("rt_sigaction read failure");
	if((void*)sa.sa_handler == runtime·sigtramp)
		return runtime·sighandler;
	return (void*)sa.sa_handler;
}

void
runtime·signalstack(byte *p, int32 n)
{
	SigaltstackT st;

	st.ss_sp = p;
	st.ss_size = n;
	st.ss_flags = 0;
	if(p == nil)
		st.ss_flags = SS_DISABLE;
	runtime·sigaltstack(&st, nil);
}

void
runtime·unblocksignals(void)
{
	runtime·rtsigprocmask(SIG_SETMASK, &sigset_none, nil, sizeof sigset_none);
}

#pragma textflag NOSPLIT
int8*
runtime·signame(int32 sig)
{
	return runtime·sigtab[sig].name;
}
