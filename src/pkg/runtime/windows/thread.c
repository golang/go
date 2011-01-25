// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs.h"
#include "os.h"

#pragma dynimport runtime·LoadLibraryEx LoadLibraryExA "kernel32.dll"
#pragma dynimport runtime·GetProcAddress GetProcAddress "kernel32.dll"
#pragma dynimport runtime·CloseHandle CloseHandle "kernel32.dll"
#pragma dynimport runtime·ExitProcess ExitProcess "kernel32.dll"
#pragma dynimport runtime·GetStdHandle GetStdHandle "kernel32.dll"
#pragma dynimport runtime·SetEvent SetEvent "kernel32.dll"
#pragma dynimport runtime·WriteFile WriteFile "kernel32.dll"
#pragma dynimport runtime·GetLastError GetLastError "kernel32.dll"
#pragma dynimport runtime·SetLastError SetLastError "kernel32.dll"

// Also referenced by external packages
extern void *runtime·CloseHandle;
extern void *runtime·ExitProcess;
extern void *runtime·GetStdHandle;
extern void *runtime·SetEvent;
extern void *runtime·WriteFile;
extern void *runtime·LoadLibraryEx;
extern void *runtime·GetProcAddress;
extern void *runtime·GetLastError;
extern void *runtime·SetLastError;

#pragma dynimport runtime·CreateEvent CreateEventA "kernel32.dll"
#pragma dynimport runtime·CreateThread CreateThread "kernel32.dll"
#pragma dynimport runtime·WaitForSingleObject WaitForSingleObject "kernel32.dll"

extern void *runtime·CreateEvent;
extern void *runtime·CreateThread;
extern void *runtime·WaitForSingleObject;

void
runtime·osinit(void)
{
}

#pragma dynimport runtime·GetEnvironmentStringsW GetEnvironmentStringsW  "kernel32.dll"
#pragma dynimport runtime·FreeEnvironmentStringsW FreeEnvironmentStringsW  "kernel32.dll"

extern void *runtime·GetEnvironmentStringsW;
extern void *runtime·FreeEnvironmentStringsW;

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
	USED(stk);
	USED(g);	// assuming g = m->g0
	USED(fn);	// assuming fn = mstart

	runtime·stdcall(runtime·CreateThread, 6, 0, 0, runtime·tstart_stdcall, m, 0, 0);
}

// Called to initialize a new m (including the bootstrap m).
void
runtime·minit(void)
{
}

// Calling stdcall on os stack.
#pragma textflag 7
void *
runtime·stdcall(void *fn, int32 count, ...)
{
	return runtime·stdcall_raw(fn, count, (uintptr*)(&count + 1));
}

void
runtime·syscall(StdcallParams *p)
{
	G *oldlock;
	uintptr a;

	/*
	 * Lock g to m to ensure we stay on the same stack if we do a callback.
	 */
	oldlock = m->lockedg;
	m->lockedg = g;
	g->lockedm = m;

	runtime·entersyscall();
	// TODO(brainman): Move calls to SetLastError and GetLastError
	// to stdcall_raw to speed up syscall.
	a = 0;
	runtime·stdcall_raw(runtime·SetLastError, 1, &a);
	p->r = (uintptr)runtime·stdcall_raw((void*)p->fn, p->n, p->args);
	p->err = (uintptr)runtime·stdcall_raw(runtime·GetLastError, 0, &a);
	runtime·exitsyscall();

	m->lockedg = oldlock;
	if(oldlock == nil)
		g->lockedm = nil;
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

// Call back from windows dll into go.
void
runtime·compilecallback(byte *code, void *fn, uint32 argsize)
{
	byte *p;

	p = code;
	// SUBL $16, SP
	*p++ = 0x83;
	*p++ = 0xec;
	*p++ = 0x10;
	// PUSH argsize * 4
	*p++ = 0x68;
	*(uint32*)p = argsize << 2;
	p += 4;
	// PUSH fn
	*p++ = 0x68;
	*(uint32*)p = (uint32)fn;
	p += 4;
	// MOV callbackasm, AX
	void* (*x)(void) = runtime·callbackasm;
	*p++ = 0xb8;
	*(uint32*)p = (uint32)x;
	p += 4;
	// CALL AX
	*p++ = 0xff;
	*p = 0xd0;
}

enum { StackGuard = 2048 }; // defined in proc.c

#pragma textflag 7
void*
runtime·callback(void *arg, void (*fn)(void), int32 argsize)
{
	Gobuf msched, g1sched;
	G *g1;
	void *sp, *gostack;
	void **p;

	USED(argsize);

	if(g != m->g0)
		runtime·throw("bad g in callback");

	g1 = m->curg;

	gostack = m->gostack;		// preserve previous call stack parameters
	msched = m->sched;
	g1sched = g1->sched;

	runtime·startcgocallback(g1);

	sp = g1->sched.sp - 4 - 4;	// one input, one output

	if(sp < g1->stackguard - StackGuard + 4) // +4 for return address
		runtime·throw("g stack overflow in callback");

	p = sp;
	p[0] = arg;

	runtime·runcgocallback(g1, sp, fn);

	runtime·endcgocallback(g1);

	g1->sched = g1sched; 
	m->sched = msched;
	m->gostack = gostack;		// restore previous call stack parameters

	return p[1];
}
