// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "os.h"

extern void *runtime·get_kernel_module(void);

// Also referenced by external packages
void *runtime·CloseHandle;
void *runtime·ExitProcess;
void *runtime·GetStdHandle;
void *runtime·SetEvent;
void *runtime·WriteFile;
void *runtime·VirtualAlloc;
void *runtime·VirtualFree;
void *runtime·LoadLibraryEx;
void *runtime·GetProcAddress;
void *runtime·GetLastError;
void *runtime·SetLastError;

static void *CreateEvent;
static void *CreateThread;
static void *WaitForSingleObject;

static void*
get_proc_addr2(byte *base, byte *name)
{
	byte *pe_header, *exports;
	uint32 entries, *addr, *names, i;
	uint16 *ordinals;

	pe_header = base+*(uint32*)(base+0x3c);
	exports = base+*(uint32*)(pe_header+0x78);
	entries = *(uint32*)(exports+0x18);
	addr = (uint32*)(base+*(uint32*)(exports+0x1c));
	names = (uint32*)(base+*(uint32*)(exports+0x20));
	ordinals = (uint16*)(base+*(uint32*)(exports+0x24));
	for(i=0; i<entries; i++) {
		byte *s = base+names[i];
		if(runtime·strcmp(name, s) == 0)
			break;
	}
	if(i == entries)
		return 0;
	return base+addr[ordinals[i]];
}

void
runtime·osinit(void)
{
	void *base;

	base = runtime·get_kernel_module();
	runtime·GetProcAddress = get_proc_addr2(base, (byte*)"GetProcAddress");
	runtime·LoadLibraryEx = get_proc_addr2(base, (byte*)"LoadLibraryExA");
	runtime·CloseHandle = runtime·get_proc_addr("kernel32.dll", "CloseHandle");
	CreateEvent = runtime·get_proc_addr("kernel32.dll", "CreateEventA");
	CreateThread = runtime·get_proc_addr("kernel32.dll", "CreateThread");
	runtime·ExitProcess = runtime·get_proc_addr("kernel32.dll", "ExitProcess");
	runtime·GetStdHandle = runtime·get_proc_addr("kernel32.dll", "GetStdHandle");
	runtime·SetEvent = runtime·get_proc_addr("kernel32.dll", "SetEvent");
	runtime·VirtualAlloc = runtime·get_proc_addr("kernel32.dll", "VirtualAlloc");
	runtime·VirtualFree = runtime·get_proc_addr("kernel32.dll", "VirtualFree");
	WaitForSingleObject = runtime·get_proc_addr("kernel32.dll", "WaitForSingleObject");
	runtime·WriteFile = runtime·get_proc_addr("kernel32.dll", "WriteFile");
	runtime·GetLastError = runtime·get_proc_addr("kernel32.dll", "GetLastError");
	runtime·SetLastError = runtime·get_proc_addr("kernel32.dll", "SetLastError");
}

// The arguments are strings.
void*
runtime·get_proc_addr(void *library, void *name)
{
	void *base;

	base = runtime·stdcall(runtime·LoadLibraryEx, 3, library, 0, 0);
	return runtime·stdcall(runtime·GetProcAddress, 2, base, name);
}

void
runtime·windows_goargs(void)
{
	extern Slice os·Args;
	extern Slice os·Envs;

	void *gcl, *clta, *ges, *fes;
	uint16 *cmd, *env, **argv;
	String *gargv;
	String *genvv;
	int32 i, argc, envc;
	uint16 *envp;

	gcl = runtime·get_proc_addr("kernel32.dll", "GetCommandLineW");
	clta = runtime·get_proc_addr("shell32.dll", "CommandLineToArgvW");
	ges = runtime·get_proc_addr("kernel32.dll", "GetEnvironmentStringsW");
	fes = runtime·get_proc_addr("kernel32.dll", "FreeEnvironmentStringsW");

	cmd = runtime·stdcall(gcl, 0);
	env = runtime·stdcall(ges, 0);
	argv = runtime·stdcall(clta, 2, cmd, &argc);

	envc = 0;
	for(envp=env; *envp; envc++)
		envp += runtime·findnullw(envp)+1;

	gargv = runtime·malloc(argc*sizeof gargv[0]);
	genvv = runtime·malloc(envc*sizeof genvv[0]);

	for(i=0; i<argc; i++)
		gargv[i] = runtime·gostringw(argv[i]);
	os·Args.array = (byte*)gargv;
	os·Args.len = argc;
	os·Args.cap = argc;

	envp = env;
	for(i=0; i<envc; i++) {
		genvv[i] = runtime·gostringw(envp);
		envp += runtime·findnullw(envp)+1;
	}
	os·Envs.array = (byte*)genvv;
	os·Envs.len = envc;
	os·Envs.cap = envc;

	runtime·stdcall(fes, 1, env);
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

	event = runtime·stdcall(CreateEvent, 4, 0, 0, 0, 0);
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
		runtime·stdcall(WaitForSingleObject, 2, l->event, -1);
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

	runtime·stdcall(CreateThread, 6, 0, 0, runtime·tstart_stdcall, m, 0, 0);
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
