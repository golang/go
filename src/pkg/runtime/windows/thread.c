// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "type.h"
#include "defs.h"
#include "os.h"

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

void
runtime·osinit(void)
{
	runtime·stdcall(runtime·QueryPerformanceFrequency, 1, &timerfreq);
	runtime·stdcall(runtime·SetConsoleCtrlHandler, 2, runtime·ctrlhandler, 1);
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
	runtime·stdcall(runtime·ExitProcess, 1, code);
}

int32
runtime·write(int32 fd, void *buf, int32 n)
{
	void *handle;
	uint32 written;

	written = 0;
	switch(fd) {
	case 1:
		handle = runtime·stdcall(runtime·GetStdHandle, 1, -11);
		break;
	case 2:
		handle = runtime·stdcall(runtime·GetStdHandle, 1, -12);
		break;
	default:
		return -1;
	}
	runtime·stdcall(runtime·WriteFile, 5, handle, buf, n, &written, 0);
	return written;
}

// Thread-safe allocation of an event.
static void
initevent(void **pevent)
{
	void *event;

	event = runtime·stdcall(runtime·CreateEvent, 4, 0, 0, 0, 0);
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
		runtime·stdcall(runtime·WaitForSingleObject, 2, l->event, -1);
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

void
runtime·destroylock(Lock *l)
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

	thandle = runtime·stdcall(runtime·CreateThread, 6, 0, 0, runtime·tstart_stdcall, m, 0, 0);
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
	return runtime·stdcall_raw(fn, count, (uintptr*)(&count + 1));
}

uintptr
runtime·syscall(void *fn, uintptr nargs, void *args, uintptr *err)
{
	G *oldlock;
	uintptr ret;

	/*
	 * Lock g to m to ensure we stay on the same stack if we do a callback.
	 */
	oldlock = m->lockedg;
	m->lockedg = g;
	g->lockedm = m;

	runtime·entersyscall();
	runtime·setlasterror(0);
	ret = (uintptr)runtime·stdcall_raw(fn, nargs, args);
	if(err)
		*err = runtime·getlasterror();
	runtime·exitsyscall();

	m->lockedg = oldlock;
	if(oldlock == nil)
		g->lockedm = nil;

	return ret;
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

// Will keep all callbacks in a linked list, so they don't get garbage collected.
typedef	struct	Callback	Callback;
struct	Callback {
	Callback*	link;
	void*		gobody;
	byte		asmbody;
};

typedef	struct	Callbacks	Callbacks;
struct	Callbacks {
	Lock;
	Callback*	link;
	int32		n;
};

static	Callbacks	cbs;

// Call back from windows dll into go.
byte *
runtime·compilecallback(Eface fn, bool cleanstack)
{
	Func *f;
	int32 argsize, n;
	byte *p;
	Callback *c;

	if(fn.type->kind != KindFunc)
		runtime·panicstring("not a function");
	if((f = runtime·findfunc((uintptr)fn.data)) == nil)
		runtime·throw("cannot find function");
	argsize = (f->args-2) * 4;

	// compute size of new fn.
	// must match code laid out below.
	n = 1+4;		// MOVL fn, AX
	n += 1+4;		// MOVL argsize, DX
	n += 1+4;		// MOVL callbackasm, CX
	n += 2;			// CALL CX
	n += 1;			// RET
	if(cleanstack)
		n += 2;		// ... argsize

	runtime·lock(&cbs);
	for(c = cbs.link; c != nil; c = c->link) {
		if(c->gobody == fn.data) {
			runtime·unlock(&cbs);
			return &c->asmbody;
		}
	}
	if(cbs.n >= 20)
		runtime·throw("too many callback functions");
	c = runtime·mal(sizeof *c + n);
	c->gobody = fn.data;
	c->link = cbs.link;
	cbs.link = c;
	cbs.n++;
	runtime·unlock(&cbs);

	p = &c->asmbody;

	// MOVL fn, AX
	*p++ = 0xb8;
	*(uint32*)p = (uint32)fn.data;
	p += 4;

	// MOVL argsize, DX
	*p++ = 0xba;
	*(uint32*)p = argsize;
	p += 4;

	// MOVL callbackasm, CX
	*p++ = 0xb9;
	*(uint32*)p = (uint32)runtime·callbackasm;
	p += 4;

	// CALL CX
	*p++ = 0xff;
	*p++ = 0xd1;

	// RET argsize?
	if(cleanstack) {
		*p++ = 0xc2;
		*(uint16*)p = argsize;
	} else
		*p = 0xc3;

	return &c->asmbody;
}

void
os·sigpipe(void)
{
	runtime·throw("too many writes on closed pipe");
}
