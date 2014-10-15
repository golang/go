// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "type.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "textflag.h"
#include "arch_GOARCH.h"
#include "malloc.h"

#pragma dynimport runtime·AddVectoredExceptionHandler AddVectoredExceptionHandler "kernel32.dll"
#pragma dynimport runtime·CloseHandle CloseHandle "kernel32.dll"
#pragma dynimport runtime·CreateEvent CreateEventA "kernel32.dll"
#pragma dynimport runtime·CreateThread CreateThread "kernel32.dll"
#pragma dynimport runtime·CreateWaitableTimer CreateWaitableTimerA "kernel32.dll"
#pragma dynimport runtime·CryptAcquireContextW CryptAcquireContextW "advapi32.dll"
#pragma dynimport runtime·CryptGenRandom CryptGenRandom "advapi32.dll"
#pragma dynimport runtime·CryptReleaseContext CryptReleaseContext "advapi32.dll"
#pragma dynimport runtime·DuplicateHandle DuplicateHandle "kernel32.dll"
#pragma dynimport runtime·ExitProcess ExitProcess "kernel32.dll"
#pragma dynimport runtime·FreeEnvironmentStringsW FreeEnvironmentStringsW "kernel32.dll"
#pragma dynimport runtime·GetEnvironmentStringsW GetEnvironmentStringsW "kernel32.dll"
#pragma dynimport runtime·GetProcAddress GetProcAddress "kernel32.dll"
#pragma dynimport runtime·GetStdHandle GetStdHandle "kernel32.dll"
#pragma dynimport runtime·GetSystemInfo GetSystemInfo "kernel32.dll"
#pragma dynimport runtime·GetThreadContext GetThreadContext "kernel32.dll"
#pragma dynimport runtime·LoadLibrary LoadLibraryW "kernel32.dll"
#pragma dynimport runtime·LoadLibraryA LoadLibraryA "kernel32.dll"
#pragma dynimport runtime·NtWaitForSingleObject NtWaitForSingleObject "ntdll.dll"
#pragma dynimport runtime·ResumeThread ResumeThread "kernel32.dll"
#pragma dynimport runtime·SetConsoleCtrlHandler SetConsoleCtrlHandler "kernel32.dll"
#pragma dynimport runtime·SetEvent SetEvent "kernel32.dll"
#pragma dynimport runtime·SetProcessPriorityBoost SetProcessPriorityBoost "kernel32.dll"
#pragma dynimport runtime·SetThreadPriority SetThreadPriority "kernel32.dll"
#pragma dynimport runtime·SetUnhandledExceptionFilter SetUnhandledExceptionFilter "kernel32.dll"
#pragma dynimport runtime·SetWaitableTimer SetWaitableTimer "kernel32.dll"
#pragma dynimport runtime·Sleep Sleep "kernel32.dll"
#pragma dynimport runtime·SuspendThread SuspendThread "kernel32.dll"
#pragma dynimport runtime·WaitForSingleObject WaitForSingleObject "kernel32.dll"
#pragma dynimport runtime·WriteFile WriteFile "kernel32.dll"
#pragma dynimport runtime·timeBeginPeriod timeBeginPeriod "winmm.dll"

extern void *runtime·AddVectoredExceptionHandler;
extern void *runtime·CloseHandle;
extern void *runtime·CreateEvent;
extern void *runtime·CreateThread;
extern void *runtime·CreateWaitableTimer;
extern void *runtime·CryptAcquireContextW;
extern void *runtime·CryptGenRandom;
extern void *runtime·CryptReleaseContext;
extern void *runtime·DuplicateHandle;
extern void *runtime·ExitProcess;
extern void *runtime·FreeEnvironmentStringsW;
extern void *runtime·GetEnvironmentStringsW;
extern void *runtime·GetProcAddress;
extern void *runtime·GetStdHandle;
extern void *runtime·GetSystemInfo;
extern void *runtime·GetThreadContext;
extern void *runtime·LoadLibrary;
extern void *runtime·LoadLibraryA;
extern void *runtime·NtWaitForSingleObject;
extern void *runtime·ResumeThread;
extern void *runtime·SetConsoleCtrlHandler;
extern void *runtime·SetEvent;
extern void *runtime·SetProcessPriorityBoost;
extern void *runtime·SetThreadPriority;
extern void *runtime·SetUnhandledExceptionFilter;
extern void *runtime·SetWaitableTimer;
extern void *runtime·Sleep;
extern void *runtime·SuspendThread;
extern void *runtime·WaitForSingleObject;
extern void *runtime·WriteFile;
extern void *runtime·timeBeginPeriod;

#pragma dataflag NOPTR
void *runtime·GetQueuedCompletionStatusEx;

extern uintptr runtime·externalthreadhandlerp;
void runtime·externalthreadhandler(void);
void runtime·exceptiontramp(void);
void runtime·firstcontinuetramp(void);
void runtime·lastcontinuetramp(void);

#pragma textflag NOSPLIT
uintptr
runtime·getLoadLibrary(void)
{
	return (uintptr)runtime·LoadLibrary;
}

#pragma textflag NOSPLIT
uintptr
runtime·getGetProcAddress(void)
{
	return (uintptr)runtime·GetProcAddress;
}

static int32
getproccount(void)
{
	SystemInfo info;

	runtime·stdcall1(runtime·GetSystemInfo, (uintptr)&info);
	return info.dwNumberOfProcessors;
}

void
runtime·osinit(void)
{
	void *kernel32;
	void *addVectoredContinueHandler;

	kernel32 = runtime·stdcall1(runtime·LoadLibraryA, (uintptr)"kernel32.dll");

	runtime·externalthreadhandlerp = (uintptr)runtime·externalthreadhandler;

	runtime·stdcall2(runtime·AddVectoredExceptionHandler, 1, (uintptr)runtime·exceptiontramp);
	addVectoredContinueHandler = nil;
	if(kernel32 != nil)
		addVectoredContinueHandler = runtime·stdcall2(runtime·GetProcAddress, (uintptr)kernel32, (uintptr)"AddVectoredContinueHandler");
	if(addVectoredContinueHandler == nil || sizeof(void*) == 4) {
		// use SetUnhandledExceptionFilter for windows-386 or
		// if VectoredContinueHandler is unavailable.
		// note: SetUnhandledExceptionFilter handler won't be called, if debugging.
		runtime·stdcall1(runtime·SetUnhandledExceptionFilter, (uintptr)runtime·lastcontinuetramp);
	} else {
		runtime·stdcall2(addVectoredContinueHandler, 1, (uintptr)runtime·firstcontinuetramp);
		runtime·stdcall2(addVectoredContinueHandler, 0, (uintptr)runtime·lastcontinuetramp);
	}

	runtime·stdcall2(runtime·SetConsoleCtrlHandler, (uintptr)runtime·ctrlhandler, 1);

	runtime·stdcall1(runtime·timeBeginPeriod, 1);

	runtime·ncpu = getproccount();
	
	// Windows dynamic priority boosting assumes that a process has different types
	// of dedicated threads -- GUI, IO, computational, etc. Go processes use
	// equivalent threads that all do a mix of GUI, IO, computations, etc.
	// In such context dynamic priority boosting does nothing but harm, so we turn it off.
	runtime·stdcall2(runtime·SetProcessPriorityBoost, -1, 1);

	if(kernel32 != nil) {
		runtime·GetQueuedCompletionStatusEx = runtime·stdcall2(runtime·GetProcAddress, (uintptr)kernel32, (uintptr)"GetQueuedCompletionStatusEx");
	}
}

#pragma textflag NOSPLIT
void
runtime·get_random_data(byte **rnd, int32 *rnd_len)
{
	uintptr handle;
	*rnd = nil;
	*rnd_len = 0;
	if(runtime·stdcall5(runtime·CryptAcquireContextW, (uintptr)&handle, (uintptr)nil, (uintptr)nil,
			   1 /* PROV_RSA_FULL */,
			   0xf0000000U /* CRYPT_VERIFYCONTEXT */) != 0) {
		static byte random_data[HashRandomBytes];
		if(runtime·stdcall3(runtime·CryptGenRandom, handle, HashRandomBytes, (uintptr)&random_data[0])) {
			*rnd = random_data;
			*rnd_len = HashRandomBytes;
		}
		runtime·stdcall2(runtime·CryptReleaseContext, handle, 0);
	}
}

void
runtime·goenvs(void)
{
	extern Slice runtime·envs;

	uint16 *env;
	String *s;
	int32 i, n;
	uint16 *p;

	env = runtime·stdcall0(runtime·GetEnvironmentStringsW);

	n = 0;
	for(p=env; *p; n++)
		p += runtime·findnullw(p)+1;

	runtime·envs = runtime·makeStringSlice(n);
	s = (String*)runtime·envs.array;

	p = env;
	for(i=0; i<n; i++) {
		s[i] = runtime·gostringw(p);
		p += runtime·findnullw(p)+1;
	}

	runtime·stdcall1(runtime·FreeEnvironmentStringsW, (uintptr)env);
}

#pragma textflag NOSPLIT
void
runtime·exit(int32 code)
{
	runtime·stdcall1(runtime·ExitProcess, code);
}

#pragma textflag NOSPLIT
int32
runtime·write(uintptr fd, void *buf, int32 n)
{
	void *handle;
	uint32 written;

	written = 0;
	switch(fd) {
	case 1:
		handle = runtime·stdcall1(runtime·GetStdHandle, -11);
		break;
	case 2:
		handle = runtime·stdcall1(runtime·GetStdHandle, -12);
		break;
	default:
		// assume fd is real windows handle.
		handle = (void*)fd;
		break;
	}
	runtime·stdcall5(runtime·WriteFile, (uintptr)handle, (uintptr)buf, n, (uintptr)&written, 0);
	return written;
}

#define INFINITE ((uintptr)0xFFFFFFFF)

#pragma textflag NOSPLIT
int32
runtime·semasleep(int64 ns)
{
	// store ms in ns to save stack space
	if(ns < 0)
		ns = INFINITE;
	else {
		ns = runtime·timediv(ns, 1000000, nil);
		if(ns == 0)
			ns = 1;
	}
	if(runtime·stdcall2(runtime·WaitForSingleObject, (uintptr)g->m->waitsema, ns) != 0)
		return -1;  // timeout
	return 0;
}

#pragma textflag NOSPLIT
void
runtime·semawakeup(M *mp)
{
	runtime·stdcall1(runtime·SetEvent, mp->waitsema);
}

#pragma textflag NOSPLIT
uintptr
runtime·semacreate(void)
{
	return (uintptr)runtime·stdcall4(runtime·CreateEvent, 0, 0, 0, 0);
}

#define STACK_SIZE_PARAM_IS_A_RESERVATION ((uintptr)0x00010000)

void
runtime·newosproc(M *mp, void *stk)
{
	void *thandle;

	USED(stk);

	thandle = runtime·stdcall6(runtime·CreateThread,
		(uintptr)nil, 0x20000, (uintptr)runtime·tstart_stdcall, (uintptr)mp,
		STACK_SIZE_PARAM_IS_A_RESERVATION, (uintptr)nil);
	if(thandle == nil) {
		runtime·printf("runtime: failed to create new OS thread (have %d already; errno=%d)\n", runtime·mcount(), runtime·getlasterror());
		runtime·throw("runtime.newosproc");
	}
}

// Called to initialize a new m (including the bootstrap m).
// Called on the parent thread (main thread in case of bootstrap), can allocate memory.
void
runtime·mpreinit(M *mp)
{
	USED(mp);
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, can not allocate memory.
void
runtime·minit(void)
{
	uintptr thandle;

	// -1 = current process, -2 = current thread
	runtime·stdcall7(runtime·DuplicateHandle, -1, -2, -1, (uintptr)&thandle, 0, 0, DUPLICATE_SAME_ACCESS);
	runtime·atomicstoreuintptr(&g->m->thread, thandle);
}

// Called from dropm to undo the effect of an minit.
void
runtime·unminit(void)
{
	runtime·stdcall1(runtime·CloseHandle, g->m->thread);
	g->m->thread = 0;
}

// Described in http://www.dcl.hpi.uni-potsdam.de/research/WRK/2007/08/getting-os-information-the-kuser_shared_data-structure/
typedef struct KSYSTEM_TIME {
	uint32	LowPart;
	int32	High1Time;
	int32	High2Time;
} KSYSTEM_TIME;

#pragma dataflag NOPTR
const KSYSTEM_TIME* INTERRUPT_TIME	= (KSYSTEM_TIME*)0x7ffe0008;
#pragma dataflag NOPTR
const KSYSTEM_TIME* SYSTEM_TIME		= (KSYSTEM_TIME*)0x7ffe0014;

static void badsystime(void);

#pragma textflag NOSPLIT
int64
runtime·systime(KSYSTEM_TIME *timeaddr)
{
	KSYSTEM_TIME t;
	int32 i;
	void (*fn)(void);

	for(i = 1; i < 10000; i++) {
		// these fields must be read in that order (see URL above)
		t.High1Time = timeaddr->High1Time;
		t.LowPart = timeaddr->LowPart;
		t.High2Time = timeaddr->High2Time;
		if(t.High1Time == t.High2Time)
			return (int64)t.High1Time<<32 | t.LowPart;
		if((i%100) == 0)
			runtime·osyield();
	}
	fn = badsystime;
	runtime·onM(&fn);
	return 0;
}

#pragma textflag NOSPLIT
int64
runtime·unixnano(void)
{
	return (runtime·systime(SYSTEM_TIME) - 116444736000000000LL) * 100LL;
}

static void
badsystime(void)
{
	runtime·throw("interrupt/system time is changing too fast");
}

#pragma textflag NOSPLIT
int64
runtime·nanotime(void)
{
	return runtime·systime(INTERRUPT_TIME) * 100LL;
}

// Calling stdcall on os stack.
#pragma textflag NOSPLIT
static void*
stdcall(void *fn)
{
	g->m->libcall.fn = (uintptr)fn;
	if(g->m->profilehz != 0) {
		// leave pc/sp for cpu profiler
		g->m->libcallg = g;
		g->m->libcallpc = (uintptr)runtime·getcallerpc(&fn);
		// sp must be the last, because once async cpu profiler finds
		// all three values to be non-zero, it will use them
		g->m->libcallsp = (uintptr)runtime·getcallersp(&fn);
	}
	runtime·asmcgocall(runtime·asmstdcall, &g->m->libcall);
	g->m->libcallsp = 0;
	return (void*)g->m->libcall.r1;
}

#pragma textflag NOSPLIT
void*
runtime·stdcall0(void *fn)
{
	g->m->libcall.n = 0;
	g->m->libcall.args = (uintptr)&fn;  // it's unused but must be non-nil, otherwise crashes
	return stdcall(fn);
}

#pragma textflag NOSPLIT
void*
runtime·stdcall1(void *fn, uintptr a0)
{
	USED(a0);
	g->m->libcall.n = 1;
	g->m->libcall.args = (uintptr)&a0;
	return stdcall(fn);
}

#pragma textflag NOSPLIT
void*
runtime·stdcall2(void *fn, uintptr a0, uintptr a1)
{
	USED(a0, a1);
	g->m->libcall.n = 2;
	g->m->libcall.args = (uintptr)&a0;
	return stdcall(fn);
}

#pragma textflag NOSPLIT
void*
runtime·stdcall3(void *fn, uintptr a0, uintptr a1, uintptr a2)
{
	USED(a0, a1, a2);
	g->m->libcall.n = 3;
	g->m->libcall.args = (uintptr)&a0;
	return stdcall(fn);
}

#pragma textflag NOSPLIT
void*
runtime·stdcall4(void *fn, uintptr a0, uintptr a1, uintptr a2, uintptr a3)
{
	USED(a0, a1, a2, a3);
	g->m->libcall.n = 4;
	g->m->libcall.args = (uintptr)&a0;
	return stdcall(fn);
}

#pragma textflag NOSPLIT
void*
runtime·stdcall5(void *fn, uintptr a0, uintptr a1, uintptr a2, uintptr a3, uintptr a4)
{
	USED(a0, a1, a2, a3, a4);
	g->m->libcall.n = 5;
	g->m->libcall.args = (uintptr)&a0;
	return stdcall(fn);
}

#pragma textflag NOSPLIT
void*
runtime·stdcall6(void *fn, uintptr a0, uintptr a1, uintptr a2, uintptr a3, uintptr a4, uintptr a5)
{
	USED(a0, a1, a2, a3, a4, a5);
	g->m->libcall.n = 6;
	g->m->libcall.args = (uintptr)&a0;
	return stdcall(fn);
}

#pragma textflag NOSPLIT
void*
runtime·stdcall7(void *fn, uintptr a0, uintptr a1, uintptr a2, uintptr a3, uintptr a4, uintptr a5, uintptr a6)
{
	USED(a0, a1, a2, a3, a4, a5, a6);
	g->m->libcall.n = 7;
	g->m->libcall.args = (uintptr)&a0;
	return stdcall(fn);
}

extern void runtime·usleep1(uint32);

#pragma textflag NOSPLIT
void
runtime·osyield(void)
{
	runtime·usleep1(1);
}

#pragma textflag NOSPLIT
void
runtime·usleep(uint32 us)
{
	// Have 1us units; want 100ns units.
	runtime·usleep1(10*us);
}

uint32
runtime·issigpanic(uint32 code)
{
	switch(code) {
	case EXCEPTION_ACCESS_VIOLATION:
	case EXCEPTION_INT_DIVIDE_BY_ZERO:
	case EXCEPTION_INT_OVERFLOW:
	case EXCEPTION_FLT_DENORMAL_OPERAND:
	case EXCEPTION_FLT_DIVIDE_BY_ZERO:
	case EXCEPTION_FLT_INEXACT_RESULT:
	case EXCEPTION_FLT_OVERFLOW:
	case EXCEPTION_FLT_UNDERFLOW:
	case EXCEPTION_BREAKPOINT:
		return 1;
	}
	return 0;
}

void
runtime·initsig(void)
{
	// following line keeps these functions alive at link stage
	// if there's a better way please write it here
	void *e = runtime·exceptiontramp;
	void *f = runtime·firstcontinuetramp;
	void *l = runtime·lastcontinuetramp;
	USED(e);
	USED(f);
	USED(l);
}

uint32
runtime·ctrlhandler1(uint32 type)
{
	int32 s;

	switch(type) {
	case CTRL_C_EVENT:
	case CTRL_BREAK_EVENT:
		s = SIGINT;
		break;
	default:
		return 0;
	}

	if(runtime·sigsend(s))
		return 1;
	runtime·exit(2);	// SIGINT, SIGTERM, etc
	return 0;
}

extern void runtime·dosigprof(Context *r, G *gp, M *mp);
extern void runtime·profileloop(void);
#pragma dataflag NOPTR
static void *profiletimer;

static void
profilem(M *mp)
{
	extern M runtime·m0;
	extern uint32 runtime·tls0[];
	byte rbuf[sizeof(Context)+15];
	Context *r;
	void *tls;
	G *gp;

	tls = mp->tls;
	if(mp == &runtime·m0)
		tls = runtime·tls0;
	gp = *(G**)tls;

	// align Context to 16 bytes
	r = (Context*)((uintptr)(&rbuf[15]) & ~15);
	r->ContextFlags = CONTEXT_CONTROL;
	runtime·stdcall2(runtime·GetThreadContext, (uintptr)mp->thread, (uintptr)r);
	runtime·dosigprof(r, gp, mp);
}

void
runtime·profileloop1(void)
{
	M *mp, *allm;
	uintptr thread;

	runtime·stdcall2(runtime·SetThreadPriority, -2, THREAD_PRIORITY_HIGHEST);

	for(;;) {
		runtime·stdcall2(runtime·WaitForSingleObject, (uintptr)profiletimer, -1);
		allm = runtime·atomicloadp(&runtime·allm);
		for(mp = allm; mp != nil; mp = mp->alllink) {
			thread = runtime·atomicloaduintptr(&mp->thread);
			// Do not profile threads blocked on Notes,
			// this includes idle worker threads,
			// idle timer thread, idle heap scavenger, etc.
			if(thread == 0 || mp->profilehz == 0 || mp->blocked)
				continue;
			runtime·stdcall1(runtime·SuspendThread, (uintptr)thread);
			if(mp->profilehz != 0 && !mp->blocked)
				profilem(mp);
			runtime·stdcall1(runtime·ResumeThread, (uintptr)thread);
		}
	}
}

void
runtime·resetcpuprofiler(int32 hz)
{
	static Mutex lock;
	void *timer, *thread;
	int32 ms;
	int64 due;

	runtime·lock(&lock);
	if(profiletimer == nil) {
		timer = runtime·stdcall3(runtime·CreateWaitableTimer, (uintptr)nil, (uintptr)nil, (uintptr)nil);
		runtime·atomicstorep(&profiletimer, timer);
		thread = runtime·stdcall6(runtime·CreateThread,
			(uintptr)nil, (uintptr)nil, (uintptr)runtime·profileloop, (uintptr)nil, (uintptr)nil, (uintptr)nil);
		runtime·stdcall2(runtime·SetThreadPriority, (uintptr)thread, THREAD_PRIORITY_HIGHEST);
		runtime·stdcall1(runtime·CloseHandle, (uintptr)thread);
	}
	runtime·unlock(&lock);

	ms = 0;
	due = 1LL<<63;
	if(hz > 0) {
		ms = 1000 / hz;
		if(ms == 0)
			ms = 1;
		due = ms * -10000;
	}
	runtime·stdcall6(runtime·SetWaitableTimer,
		(uintptr)profiletimer, (uintptr)&due, ms, (uintptr)nil, (uintptr)nil, (uintptr)nil);
	runtime·atomicstore((uint32*)&g->m->profilehz, hz);
}

uintptr
runtime·memlimit(void)
{
	return 0;
}

#pragma dataflag NOPTR
int8 runtime·badsignalmsg[] = "runtime: signal received on thread not created by Go.\n";
int32 runtime·badsignallen = sizeof runtime·badsignalmsg - 1;

void
runtime·crash(void)
{
	// TODO: This routine should do whatever is needed
	// to make the Windows program abort/crash as it
	// would if Go was not intercepting signals.
	// On Unix the routine would remove the custom signal
	// handler and then raise a signal (like SIGABRT).
	// Something like that should happen here.
	// It's okay to leave this empty for now: if crash returns
	// the ordinary exit-after-panic happens.
}
