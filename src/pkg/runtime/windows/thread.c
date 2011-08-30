// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "type.h"
#include "defs.h"
#include "os.h"
#include "cgocall.h"

#pragma dynimport runtime·CloseHandle CloseHandle "kernel32.dll"
#pragma dynimport runtime·CreateEvent CreateEventA "kernel32.dll"
#pragma dynimport runtime·CreateThread CreateThread "kernel32.dll"
#pragma dynimport runtime·ExitProcess ExitProcess "kernel32.dll"
#pragma dynimport runtime·FreeEnvironmentStringsW FreeEnvironmentStringsW "kernel32.dll"
#pragma dynimport runtime·GetEnvironmentStringsW GetEnvironmentStringsW "kernel32.dll"
#pragma dynimport runtime·GetProcAddress GetProcAddress "kernel32.dll"
#pragma dynimport runtime·GetStdHandle GetStdHandle "kernel32.dll"
#pragma dynimport runtime·LoadLibraryEx LoadLibraryExA "kernel32.dll"
#pragma dynimport runtime·QueryPerformanceCounter QueryPerformanceCounter "kernel32.dll"
#pragma dynimport runtime·QueryPerformanceFrequency QueryPerformanceFrequency "kernel32.dll"
#pragma dynimport runtime·SetConsoleCtrlHandler SetConsoleCtrlHandler "kernel32.dll"
#pragma dynimport runtime·SetEvent SetEvent "kernel32.dll"
#pragma dynimport runtime·WaitForSingleObject WaitForSingleObject "kernel32.dll"
#pragma dynimport runtime·WriteFile WriteFile "kernel32.dll"

extern void *runtime·CloseHandle;
extern void *runtime·CreateEvent;
extern void *runtime·CreateThread;
extern void *runtime·ExitProcess;
extern void *runtime·FreeEnvironmentStringsW;
extern void *runtime·GetEnvironmentStringsW;
extern void *runtime·GetProcAddress;
extern void *runtime·GetStdHandle;
extern void *runtime·LoadLibraryEx;
extern void *runtime·QueryPerformanceCounter;
extern void *runtime·QueryPerformanceFrequency;
extern void *runtime·SetConsoleCtrlHandler;
extern void *runtime·SetEvent;
extern void *runtime·WaitForSingleObject;
extern void *runtime·WriteFile;

static int64 timerfreq;
static void destroylock(Lock *l);

void
runtime·osinit(void)
{
	runtime·stdcall(runtime·QueryPerformanceFrequency, 1, &timerfreq);
	runtime·stdcall(runtime·SetConsoleCtrlHandler, 2, runtime·ctrlhandler, (uintptr)1);
	runtime·destroylock = destroylock;
}

void
runtime·goenvs(void)
{
	extern Slice os·Envs;

	uint16 *env;
	String *s;
	int32 i, n;
	uint16 *p;

	env = runtime·stdcall(runtime·GetEnvironmentStringsW, 0);

	n = 0;
	for(p=env; *p; n++)
		p += runtime·findnullw(p)+1;

	s = runtime·malloc(n*sizeof s[0]);

	p = env;
	for(i=0; i<n; i++) {
		s[i] = runtime·gostringw(p);
		p += runtime·findnullw(p)+1;
	}
	os·Envs.array = (byte*)s;
	os·Envs.len = n;
	os·Envs.cap = n;

	runtime·stdcall(runtime·FreeEnvironmentStringsW, 1, env);
}

void
runtime·exit(int32 code)
{
	runtime·stdcall(runtime·ExitProcess, 1, (uintptr)code);
}

int32
runtime·write(int32 fd, void *buf, int32 n)
{
	void *handle;
	uint32 written;

	written = 0;
	switch(fd) {
	case 1:
		handle = runtime·stdcall(runtime·GetStdHandle, 1, (uintptr)-11);
		break;
	case 2:
		handle = runtime·stdcall(runtime·GetStdHandle, 1, (uintptr)-12);
		break;
	default:
		return -1;
	}
	runtime·stdcall(runtime·WriteFile, 5, handle, buf, (uintptr)n, &written, (uintptr)0);
	return written;
}

// Thread-safe allocation of an event.
static void
initevent(void **pevent)
{
	void *event;

	event = runtime·stdcall(runtime·CreateEvent, 4, (uintptr)0, (uintptr)0, (uintptr)0, (uintptr)0);
	if(!runtime·casp(pevent, 0, event)) {
		// Someone else filled it in.  Use theirs.
		runtime·stdcall(runtime·CloseHandle, 1, event);
	}
}

static void
eventlock(Lock *l)
{
	// Allocate event if needed.
	if(l->event == 0)
		initevent(&l->event);

	if(runtime·xadd(&l->key, 1) > 1)	// someone else has it; wait
		runtime·stdcall(runtime·WaitForSingleObject, 2, l->event, (uintptr)-1);
}

static void
eventunlock(Lock *l)
{
	if(runtime·xadd(&l->key, -1) > 0)	// someone else is waiting
		runtime·stdcall(runtime·SetEvent, 1, l->event);
}

void
runtime·lock(Lock *l)
{
	if(m->locks < 0)
		runtime·throw("lock count");
	m->locks++;
	eventlock(l);
}

void
runtime·unlock(Lock *l)
{
	m->locks--;
	if(m->locks < 0)
		runtime·throw("lock count");
	eventunlock(l);
}

static void
destroylock(Lock *l)
{
	if(l->event != 0)
		runtime·stdcall(runtime·CloseHandle, 1, l->event);
}

void
runtime·noteclear(Note *n)
{
	n->lock.key = 0;	// memset(n, 0, sizeof *n)
	eventlock(&n->lock);
}

void
runtime·notewakeup(Note *n)
{
	eventunlock(&n->lock);
}

void
runtime·notesleep(Note *n)
{
	eventlock(&n->lock);
	eventunlock(&n->lock);	// Let other sleepers find out too.
}

void
runtime·newosproc(M *m, G *g, void *stk, void (*fn)(void))
{
	void *thandle;

	USED(stk);
	USED(g);	// assuming g = m->g0
	USED(fn);	// assuming fn = mstart

	thandle = runtime·stdcall(runtime·CreateThread, 6, (uintptr)0, (uintptr)0, runtime·tstart_stdcall, m, (uintptr)0, (uintptr)0);
	if(thandle == 0) {
		runtime·printf("runtime: failed to create new OS thread (have %d already; errno=%d)\n", runtime·mcount(), runtime·getlasterror());
		runtime·throw("runtime.newosproc");
	}
}

// Called to initialize a new m (including the bootstrap m).
void
runtime·minit(void)
{
}

void
runtime·gettime(int64 *sec, int32 *usec)
{
	int64 count;

	runtime·stdcall(runtime·QueryPerformanceCounter, 1, &count);
	*sec = count / timerfreq;
	count %= timerfreq;
	*usec = count*1000000 / timerfreq;
}

// Calling stdcall on os stack.
#pragma textflag 7
void *
runtime·stdcall(void *fn, int32 count, ...)
{
	WinCall c;

	c.fn = fn;
	c.n = count;
	c.args = (uintptr*)&count + 1;
	runtime·asmcgocall(runtime·asmstdcall, &c);
	return (void*)c.r;
}

uintptr
runtime·syscall(void *fn, uintptr nargs, void *args, uintptr *err)
{
	WinCall c;

	c.fn = fn;
	c.n = nargs;
	c.args = args;
	runtime·cgocall(runtime·asmstdcall, &c);
	if(err)
		*err = c.err;
	return c.r;
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
		return 1;
	}
	return 0;
}

void
runtime·sigpanic(void)
{
	switch(g->sig) {
	case EXCEPTION_ACCESS_VIOLATION:
		if(g->sigcode1 < 0x1000)
			runtime·panicstring("invalid memory address or nil pointer dereference");
		runtime·printf("unexpected fault address %p\n", g->sigcode1);
		runtime·throw("fault");
	case EXCEPTION_INT_DIVIDE_BY_ZERO:
		runtime·panicstring("integer divide by zero");
	case EXCEPTION_INT_OVERFLOW:
		runtime·panicstring("integer overflow");
	case EXCEPTION_FLT_DENORMAL_OPERAND:
	case EXCEPTION_FLT_DIVIDE_BY_ZERO:
	case EXCEPTION_FLT_INEXACT_RESULT:
	case EXCEPTION_FLT_OVERFLOW:
	case EXCEPTION_FLT_UNDERFLOW:
		runtime·panicstring("floating point error");
	}
	runtime·throw("fault");
}

String
runtime·signame(int32 sig)
{
	int8 *s;

	switch(sig) {
	case SIGINT:
		s = "SIGINT: interrupt";
		break;
	default:
		return runtime·emptystring;
	}
	return runtime·gostringnocopy((byte*)s);
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

void
os·sigpipe(void)
{
	runtime·throw("too many writes on closed pipe");
}
