// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "signal_unix.h"
#include "stack.h"
#include "textflag.h"

#pragma dynexport runtime·end _end
#pragma dynexport runtime·etext _etext
#pragma dynexport runtime·edata _edata

#pragma dynimport libc·___errno ___errno "libc.so"
#pragma dynimport libc·clock_gettime clock_gettime "libc.so"
#pragma dynimport libc·close close "libc.so"
#pragma dynimport libc·exit exit "libc.so"
#pragma dynimport libc·fstat fstat "libc.so"
#pragma dynimport libc·getcontext getcontext "libc.so"
#pragma dynimport libc·getrlimit getrlimit "libc.so"
#pragma dynimport libc·malloc malloc "libc.so"
#pragma dynimport libc·mmap mmap "libc.so"
#pragma dynimport libc·munmap munmap "libc.so"
#pragma dynimport libc·open open "libc.so"
#pragma dynimport libc·pthread_attr_destroy pthread_attr_destroy "libc.so"
#pragma dynimport libc·pthread_attr_getstack pthread_attr_getstack "libc.so"
#pragma dynimport libc·pthread_attr_init pthread_attr_init "libc.so"
#pragma dynimport libc·pthread_attr_setdetachstate pthread_attr_setdetachstate "libc.so"
#pragma dynimport libc·pthread_attr_setstack pthread_attr_setstack "libc.so"
#pragma dynimport libc·pthread_create pthread_create "libc.so"
#pragma dynimport libc·raise raise "libc.so"
#pragma dynimport libc·read read "libc.so"
#pragma dynimport libc·select select "libc.so"
#pragma dynimport libc·sched_yield sched_yield "libc.so"
#pragma dynimport libc·sem_init sem_init "libc.so"
#pragma dynimport libc·sem_post sem_post "libc.so"
#pragma dynimport libc·sem_reltimedwait_np sem_reltimedwait_np "libc.so"
#pragma dynimport libc·sem_wait sem_wait "libc.so"
#pragma dynimport libc·setitimer setitimer "libc.so"
#pragma dynimport libc·sigaction sigaction "libc.so"
#pragma dynimport libc·sigaltstack sigaltstack "libc.so"
#pragma dynimport libc·sigprocmask sigprocmask "libc.so"
#pragma dynimport libc·sysconf sysconf "libc.so"
#pragma dynimport libc·usleep usleep "libc.so"
#pragma dynimport libc·write write "libc.so"

extern uintptr libc·___errno;
extern uintptr libc·clock_gettime;
extern uintptr libc·close;
extern uintptr libc·exit;
extern uintptr libc·fstat;
extern uintptr libc·getcontext;
extern uintptr libc·getrlimit;
extern uintptr libc·malloc;
extern uintptr libc·mmap;
extern uintptr libc·munmap;
extern uintptr libc·open;
extern uintptr libc·pthread_attr_destroy;
extern uintptr libc·pthread_attr_getstack;
extern uintptr libc·pthread_attr_init;
extern uintptr libc·pthread_attr_setdetachstate;
extern uintptr libc·pthread_attr_setstack;
extern uintptr libc·pthread_create;
extern uintptr libc·raise;
extern uintptr libc·read;
extern uintptr libc·sched_yield;
extern uintptr libc·select;
extern uintptr libc·sem_init;
extern uintptr libc·sem_post;
extern uintptr libc·sem_reltimedwait_np;
extern uintptr libc·sem_wait;
extern uintptr libc·setitimer;
extern uintptr libc·sigaction;
extern uintptr libc·sigaltstack;
extern uintptr libc·sigprocmask;
extern uintptr libc·sysconf;
extern uintptr libc·usleep;
extern uintptr libc·write;

void	runtime·getcontext(Ucontext *context);
int32	runtime·pthread_attr_destroy(PthreadAttr* attr);
int32	runtime·pthread_attr_init(PthreadAttr* attr);
int32	runtime·pthread_attr_getstack(PthreadAttr* attr, void** addr, uint64* size);
int32	runtime·pthread_attr_setdetachstate(PthreadAttr* attr, int32 state);
int32	runtime·pthread_attr_setstack(PthreadAttr* attr, void* addr, uint64 size);
int32	runtime·pthread_create(Pthread* thread, PthreadAttr* attr, void(*fn)(void), void *arg);
uint32	runtime·tstart_sysvicall(M *newm);
int32	runtime·sem_init(SemT* sem, int32 pshared, uint32 value);
int32	runtime·sem_post(SemT* sem);
int32	runtime·sem_reltimedwait_np(SemT* sem, Timespec* timeout);
int32	runtime·sem_wait(SemT* sem);
int64	runtime·sysconf(int32 name);

extern SigTab runtime·sigtab[];
static Sigset sigset_none;
static Sigset sigset_all = { ~(uint32)0, ~(uint32)0, ~(uint32)0, ~(uint32)0, };

static int32
getncpu(void) 
{
	int32 n;
	
	n = (int32)runtime·sysconf(_SC_NPROCESSORS_ONLN);
	if(n < 1)
		return 1;
	return n;
}

void
runtime·osinit(void)
{
	runtime·ncpu = getncpu(); 
}

void
runtime·newosproc(M *mp, void *stk)
{
	PthreadAttr attr;
	Sigset oset;
	Pthread tid;
	int32 ret;
	uint64 size;

	USED(stk);
	if(runtime·pthread_attr_init(&attr) != 0)
		runtime·throw("pthread_attr_init");
	if(runtime·pthread_attr_setstack(&attr, 0, 0x200000) != 0)
		runtime·throw("pthread_attr_setstack");
	size = 0;
	if(runtime·pthread_attr_getstack(&attr, (void**)&mp->g0->stack.hi, &size) != 0)
		runtime·throw("pthread_attr_getstack");	
	mp->g0->stack.lo = mp->g0->stack.hi - size;
	if(runtime·pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED) != 0)
		runtime·throw("pthread_attr_setdetachstate");

	// Disable signals during create, so that the new thread starts
	// with signals disabled.  It will enable them in minit.
	runtime·sigprocmask(SIG_SETMASK, &sigset_all, &oset);
	ret = runtime·pthread_create(&tid, &attr, (void (*)(void))runtime·tstart_sysvicall, mp);
	runtime·sigprocmask(SIG_SETMASK, &oset, nil);
	if(ret != 0) {
		runtime·printf("runtime: failed to create new OS thread (have %d already; errno=%d)\n", runtime·mcount(), ret);
		runtime·throw("runtime.newosproc");
	}
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
}

// Called to initialize a new m (including the bootstrap m).
// Called on the parent thread (main thread in case of bootstrap), can allocate memory.
void
runtime·mpreinit(M *mp)
{
	mp->gsignal = runtime·malg(32*1024);
	mp->gsignal->m = mp;
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, can not allocate memory.
void
runtime·minit(void)
{
	runtime·asmcgocall(runtime·miniterrno, (void *)libc·___errno);
	// Initialize signal handling
	runtime·signalstack((byte*)g->m->gsignal->stack.lo, 32*1024);
	runtime·sigprocmask(SIG_SETMASK, &sigset_none, nil);
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

void
runtime·setprof(bool on)
{
	USED(on);
}

extern void runtime·sigtramp(void);

void
runtime·setsig(int32 i, GoSighandler *fn, bool restart)
{
	SigactionT sa;

	runtime·memclr((byte*)&sa, sizeof sa);
	sa.sa_flags = SA_SIGINFO|SA_ONSTACK;
	if(restart)
		sa.sa_flags |= SA_RESTART;
	sa.sa_mask.__sigbits[0] = ~(uint32)0;
	sa.sa_mask.__sigbits[1] = ~(uint32)0;
	sa.sa_mask.__sigbits[2] = ~(uint32)0;
	sa.sa_mask.__sigbits[3] = ~(uint32)0;
	if(fn == runtime·sighandler)
		fn = (void*)runtime·sigtramp;
	*((void**)&sa._funcptr[0]) = (void*)fn;
	runtime·sigaction(i, &sa, nil);
}

GoSighandler*
runtime·getsig(int32 i)
{
	SigactionT sa;

	runtime·memclr((byte*)&sa, sizeof sa);
	runtime·sigaction(i, nil, &sa);
	if(*((void**)&sa._funcptr[0]) == runtime·sigtramp)
		return runtime·sighandler;
	return *((void**)&sa._funcptr[0]);
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
uintptr
runtime·semacreate(void)
{
	SemT* sem;

	// Call libc's malloc rather than runtime·malloc.  This will
	// allocate space on the C heap.  We can't call runtime·malloc
	// here because it could cause a deadlock.
	g->m->libcall.fn = (uintptr)(void*)libc·malloc;
	g->m->libcall.n = 1;
	runtime·memclr((byte*)&g->m->scratch, sizeof(g->m->scratch));
	g->m->scratch.v[0] = (uintptr)sizeof(*sem);
	g->m->libcall.args = (uintptr)(uintptr*)&g->m->scratch;
	runtime·asmcgocall(runtime·asmsysvicall6, &g->m->libcall);
	sem = (void*)g->m->libcall.r1;
	if(runtime·sem_init(sem, 0, 0) != 0)
		runtime·throw("sem_init");
	return (uintptr)sem;
}

#pragma textflag NOSPLIT
int32
runtime·semasleep(int64 ns)
{
	M *m;

	m = g->m;
	if(ns >= 0) {
		m->ts.tv_sec = ns / 1000000000LL;
		m->ts.tv_nsec = ns % 1000000000LL;

		m->libcall.fn = (uintptr)(void*)libc·sem_reltimedwait_np;
		m->libcall.n = 2;
		runtime·memclr((byte*)&m->scratch, sizeof(m->scratch));
		m->scratch.v[0] = m->waitsema;
		m->scratch.v[1] = (uintptr)&m->ts;
		m->libcall.args = (uintptr)(uintptr*)&m->scratch;
		runtime·asmcgocall(runtime·asmsysvicall6, &m->libcall);
		if(*m->perrno != 0) {
			if(*m->perrno == ETIMEDOUT || *m->perrno == EAGAIN || *m->perrno == EINTR)
				return -1;
			runtime·throw("sem_reltimedwait_np");
		}
		return 0;
	}
	for(;;) {
		m->libcall.fn = (uintptr)(void*)libc·sem_wait;
		m->libcall.n = 1;
		runtime·memclr((byte*)&m->scratch, sizeof(m->scratch));
		m->scratch.v[0] = m->waitsema;
		m->libcall.args = (uintptr)(uintptr*)&m->scratch;
		runtime·asmcgocall(runtime·asmsysvicall6, &m->libcall);
		if(m->libcall.r1 == 0)
			break;
		if(*m->perrno == EINTR) 
			continue;
		runtime·throw("sem_wait");
	}
	return 0;
}

#pragma textflag NOSPLIT
void
runtime·semawakeup(M *mp)
{
	SemT* sem = (SemT*)mp->waitsema;
	if(runtime·sem_post(sem) != 0)
		runtime·throw("sem_post");
}

#pragma textflag NOSPLIT
int32
runtime·close(int32 fd)
{
	return runtime·sysvicall1(libc·close, (uintptr)fd);
}

#pragma textflag NOSPLIT
void
runtime·exit(int32 r)
{
	runtime·sysvicall1(libc·exit, (uintptr)r);
}

#pragma textflag NOSPLIT
/* int32 */ void
runtime·getcontext(Ucontext* context)
{
	runtime·sysvicall1(libc·getcontext, (uintptr)context);
}

#pragma textflag NOSPLIT
int32
runtime·getrlimit(int32 res, Rlimit* rlp)
{
	return runtime·sysvicall2(libc·getrlimit, (uintptr)res, (uintptr)rlp);
}

#pragma textflag NOSPLIT
uint8*
runtime·mmap(byte* addr, uintptr len, int32 prot, int32 flags, int32 fildes, uint32 off)
{
	return (uint8*)runtime·sysvicall6(libc·mmap, (uintptr)addr, (uintptr)len, (uintptr)prot, (uintptr)flags, (uintptr)fildes, (uintptr)off);
}

#pragma textflag NOSPLIT
void
runtime·munmap(byte* addr, uintptr len)
{
	runtime·sysvicall2(libc·munmap, (uintptr)addr, (uintptr)len);
}

extern int64 runtime·nanotime1(void);
#pragma textflag NOSPLIT
int64
runtime·nanotime(void)
{
	return runtime·sysvicall0((uintptr)runtime·nanotime1);
}

#pragma textflag NOSPLIT
int32
runtime·open(int8* path, int32 oflag, int32 mode)
{
	return runtime·sysvicall3(libc·open, (uintptr)path, (uintptr)oflag, (uintptr)mode);
}

int32
runtime·pthread_attr_destroy(PthreadAttr* attr)
{
	return runtime·sysvicall1(libc·pthread_attr_destroy, (uintptr)attr);
}

int32
runtime·pthread_attr_getstack(PthreadAttr* attr, void** addr, uint64* size)
{
	return runtime·sysvicall3(libc·pthread_attr_getstack, (uintptr)attr, (uintptr)addr, (uintptr)size);
}

int32
runtime·pthread_attr_init(PthreadAttr* attr)
{
	return runtime·sysvicall1(libc·pthread_attr_init, (uintptr)attr);
}

int32
runtime·pthread_attr_setdetachstate(PthreadAttr* attr, int32 state)
{
	return runtime·sysvicall2(libc·pthread_attr_setdetachstate, (uintptr)attr, (uintptr)state);
}

int32
runtime·pthread_attr_setstack(PthreadAttr* attr, void* addr, uint64 size)
{
	return runtime·sysvicall3(libc·pthread_attr_setstack, (uintptr)attr, (uintptr)addr, (uintptr)size);
}

int32
runtime·pthread_create(Pthread* thread, PthreadAttr* attr, void(*fn)(void), void *arg)
{
	return runtime·sysvicall4(libc·pthread_create, (uintptr)thread, (uintptr)attr, (uintptr)fn, (uintptr)arg);
}

/* int32 */ void
runtime·raise(int32 sig)
{
	runtime·sysvicall1(libc·raise, (uintptr)sig);
}

#pragma textflag NOSPLIT
int32
runtime·read(int32 fd, void* buf, int32 nbyte)
{
	return runtime·sysvicall3(libc·read, (uintptr)fd, (uintptr)buf, (uintptr)nbyte);
}

#pragma textflag NOSPLIT
int32
runtime·sem_init(SemT* sem, int32 pshared, uint32 value)
{
	return runtime·sysvicall3(libc·sem_init, (uintptr)sem, (uintptr)pshared, (uintptr)value);
}

#pragma textflag NOSPLIT
int32
runtime·sem_post(SemT* sem)
{
	return runtime·sysvicall1(libc·sem_post, (uintptr)sem);
}

#pragma textflag NOSPLIT
int32
runtime·sem_reltimedwait_np(SemT* sem, Timespec* timeout)
{
	return runtime·sysvicall2(libc·sem_reltimedwait_np, (uintptr)sem, (uintptr)timeout);
}

#pragma textflag NOSPLIT
int32
runtime·sem_wait(SemT* sem)
{
	return runtime·sysvicall1(libc·sem_wait, (uintptr)sem);
}

/* int32 */ void
runtime·setitimer(int32 which, Itimerval* value, Itimerval* ovalue)
{
	runtime·sysvicall3(libc·setitimer, (uintptr)which, (uintptr)value, (uintptr)ovalue);
}

/* int32 */ void
runtime·sigaction(int32 sig, struct SigactionT* act, struct SigactionT* oact)
{
	runtime·sysvicall3(libc·sigaction, (uintptr)sig, (uintptr)act, (uintptr)oact);
}

/* int32 */ void
runtime·sigaltstack(SigaltstackT* ss, SigaltstackT* oss)
{
	runtime·sysvicall2(libc·sigaltstack, (uintptr)ss, (uintptr)oss);
}

/* int32 */ void
runtime·sigprocmask(int32 how, Sigset* set, Sigset* oset)
{
	runtime·sysvicall3(libc·sigprocmask, (uintptr)how, (uintptr)set, (uintptr)oset);
}

int64
runtime·sysconf(int32 name)
{
	return runtime·sysvicall1(libc·sysconf, (uintptr)name);
}

extern void runtime·usleep1(uint32);

#pragma textflag NOSPLIT
void
runtime·usleep(uint32 µs)
{
	runtime·usleep1(µs);
}

#pragma textflag NOSPLIT
int32
runtime·write(uintptr fd, void* buf, int32 nbyte)
{
	return runtime·sysvicall3(libc·write, (uintptr)fd, (uintptr)buf, (uintptr)nbyte);
}

extern void runtime·osyield1(void);

#pragma textflag NOSPLIT
void
runtime·osyield(void)
{
	// Check the validity of m because we might be called in cgo callback
	// path early enough where there isn't a m available yet.
	if(g && g->m != nil) {
		runtime·sysvicall0(libc·sched_yield);
		return;
	}
	runtime·osyield1();
}

#pragma textflag NOSPLIT
int8*
runtime·signame(int32 sig)
{
	return runtime·sigtab[sig].name;
}
