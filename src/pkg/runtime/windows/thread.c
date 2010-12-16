// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
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
	uintptr a;

	runtime·entersyscall();
	// TODO(brainman): Move calls to SetLastError and GetLastError
	// to stdcall_raw to speed up syscall.
	a = 0;
	runtime·stdcall_raw(runtime·SetLastError, 1, &a);
	p->r = (uintptr)runtime·stdcall_raw((void*)p->fn, p->n, p->args);
	p->err = (uintptr)runtime·stdcall_raw(runtime·GetLastError, 0, &a);
	runtime·exitsyscall();
}
