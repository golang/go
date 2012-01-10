// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "stack.h"

extern SigTab runtime·sigtab[];

int32 runtime·open(uint8*, int32, int32);
int32 runtime·close(int32);
int32 runtime·read(int32, void*, int32);

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

	EINTR = 4,
	EAGAIN = 11,
};

// Atomically,
//	if(*addr == val) sleep
// Might be woken up spuriously; that's allowed.
// Don't sleep longer than ns; ns < 0 means forever.
void
runtime·futexsleep(uint32 *addr, uint32 val, int64 ns)
{
	Timespec ts, *tsp;

	if(ns < 0)
		tsp = nil;
	else {
		ts.tv_sec = ns/1000000000LL;
		ts.tv_nsec = ns%1000000000LL;
		// Avoid overflow
		if(ts.tv_sec > 1<<30)
			ts.tv_sec = 1<<30;
		tsp = &ts;
	}

	// Some Linux kernels have a bug where futex of
	// FUTEX_WAIT returns an internal error code
	// as an errno.  Libpthread ignores the return value
	// here, and so can we: as it says a few lines up,
	// spurious wakeups are allowed.
	runtime·futex(addr, FUTEX_WAIT, val, tsp, nil, 0);
}

// If any procs are sleeping on addr, wake up at most cnt.
void
runtime·futexwakeup(uint32 *addr, uint32 cnt)
{
	int64 ret;

	ret = runtime·futex(addr, FUTEX_WAKE, cnt, nil, nil, 0);

	if(ret >= 0)
		return;

	// I don't know that futex wakeup can return
	// EAGAIN or EINTR, but if it does, it would be
	// safe to loop and call futex again.
	runtime·printf("futexwakeup addr=%p returned %D\n", addr, ret);
	*(int32*)0x1006 = 0x1006;
}

static int32
getproccount(void)
{
	int32 fd, rd, cnt, cpustrlen;
	byte *cpustr, *pos, *bufpos;
	byte buf[256];

	fd = runtime·open((byte*)"/proc/stat", O_RDONLY|O_CLOEXEC, 0);
	if(fd == -1)
		return 1;
	cnt = 0;
	bufpos = buf;
	cpustr = (byte*)"\ncpu";
	cpustrlen = runtime·findnull(cpustr);
	for(;;) {
		rd = runtime·read(fd, bufpos, sizeof(buf)-cpustrlen);
		if(rd == -1)
			break;
		bufpos[rd] = 0;
		for(pos=buf; pos=runtime·strstr(pos, cpustr); cnt++, pos++) {
		}
		if(rd < cpustrlen)
			break;
		runtime·memmove(buf, bufpos+rd-cpustrlen+1, cpustrlen-1);
		bufpos = buf+cpustrlen-1;
	}
	runtime·close(fd);
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
runtime·newosproc(M *m, G *g, void *stk, void (*fn)(void))
{
	int32 ret;
	int32 flags;

	/*
	 * note: strace gets confused if we use CLONE_PTRACE here.
	 */
	flags = CLONE_VM	/* share memory */
		| CLONE_FS	/* share cwd, etc */
		| CLONE_FILES	/* share fd table */
		| CLONE_SIGHAND	/* share sig handler table */
		| CLONE_THREAD	/* revisit - okay for now */
		;

	m->tls[0] = m->id;	// so 386 asm can find it
	if(0){
		runtime·printf("newosproc stk=%p m=%p g=%p fn=%p clone=%p id=%d/%d ostk=%p\n",
			stk, m, g, fn, runtime·clone, m->id, m->tls[0], &m);
	}

	if((ret = runtime·clone(flags, stk, m, g, fn)) < 0) {
		runtime·printf("runtime: failed to create new OS thread (have %d already; errno=%d)\n", runtime·mcount(), -ret);
		runtime·throw("runtime.newosproc");
	}
}

void
runtime·osinit(void)
{
	runtime·ncpu = getproccount();
}

void
runtime·goenvs(void)
{
	runtime·goenvs_unix();
}

// Called to initialize a new m (including the bootstrap m).
void
runtime·minit(void)
{
	// Initialize signal handling.
	m->gsignal = runtime·malg(32*1024);	// OS X wants >=8K, Linux >=2K
	runtime·signalstack(m->gsignal->stackguard - StackGuard, 32*1024);
}

void
runtime·sigpanic(void)
{
	switch(g->sig) {
	case SIGBUS:
		if(g->sigcode0 == BUS_ADRERR && g->sigcode1 < 0x1000) {
			if(g->sigpc == 0)
				runtime·panicstring("call of nil func value");
			}
			runtime·panicstring("invalid memory address or nil pointer dereference");
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
