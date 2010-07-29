// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "os.h"

extern void *get_kernel_module(void);

// Also referenced by external packages
void *CloseHandle;
void *ExitProcess;
void *GetStdHandle;
void *SetEvent;
void *WriteFile;
void *VirtualAlloc;
void *LoadLibraryEx;
void *GetProcAddress;
void *GetLastError;
void *SetLastError;

static void *CreateEvent;
static void *CreateThread;
static void *GetModuleHandle;
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
		if(!strcmp(name, s))
			break;
	}
	if(i == entries)
		return 0;
	return base+addr[ordinals[i]];
}

void
osinit(void)
{
	void *base;

	base = get_kernel_module();
	GetProcAddress = get_proc_addr2(base, (byte*)"GetProcAddress");
	LoadLibraryEx = get_proc_addr2(base, (byte*)"LoadLibraryExA");
	CloseHandle = get_proc_addr("kernel32.dll", "CloseHandle");
	CreateEvent = get_proc_addr("kernel32.dll", "CreateEventA");
	CreateThread = get_proc_addr("kernel32.dll", "CreateThread");
	ExitProcess = get_proc_addr("kernel32.dll", "ExitProcess");
	GetModuleHandle = get_proc_addr("kernel32.dll", "GetModuleHandleA");
	GetStdHandle = get_proc_addr("kernel32.dll", "GetStdHandle");
	SetEvent = get_proc_addr("kernel32.dll", "SetEvent");
	VirtualAlloc = get_proc_addr("kernel32.dll", "VirtualAlloc");
	WaitForSingleObject = get_proc_addr("kernel32.dll", "WaitForSingleObject");
	WriteFile = get_proc_addr("kernel32.dll", "WriteFile");
	GetLastError = get_proc_addr("kernel32.dll", "GetLastError");
	SetLastError = get_proc_addr("kernel32.dll", "SetLastError");
}

// The arguments are strings.
void*
get_proc_addr(void *library, void *name)
{
	void *base;

	base = stdcall_raw(LoadLibraryEx, library, 0, 0);
	return stdcall_raw(GetProcAddress, base, name);
}

void
windows_goargs(void)
{
	extern Slice os·Args;
	extern Slice os·Envs;

	void *gcl, *clta, *ges;
	uint16 *cmd, *env, **argv;
	String *gargv;
	String *genvv;
	int32 i, argc, envc;
	uint16 *envp;

	gcl = get_proc_addr("kernel32.dll", "GetCommandLineW");
	clta = get_proc_addr("shell32.dll", "CommandLineToArgvW");
	ges = get_proc_addr("kernel32.dll", "GetEnvironmentStringsW");

	cmd = stdcall(gcl, 0);
	env = stdcall(ges, 0);
	argv = stdcall(clta, 2, cmd, &argc);

	envc = 0;
	for(envp=env; *envp; envc++)
		envp += findnullw(envp)+1;

	gargv = malloc(argc*sizeof gargv[0]);
	genvv = malloc(envc*sizeof genvv[0]);

	for(i=0; i<argc; i++)
		gargv[i] = gostringw(argv[i]);
	os·Args.array = (byte*)gargv;
	os·Args.len = argc;
	os·Args.cap = argc;

	envp = env;
	for(i=0; i<envc; i++) {
		genvv[i] = gostringw(envp);
		envp += findnullw(envp)+1;
	}
	os·Envs.array = (byte*)genvv;
	os·Envs.len = envc;
	os·Envs.cap = envc;
}

void
exit(int32 code)
{
	stdcall(ExitProcess, 1, code);
}

int32
write(int32 fd, void *buf, int32 n)
{
	void *handle;
	uint32 written;

	written = 0;
	switch(fd) {
	case 1:
		handle = stdcall(GetStdHandle, 1, -11);
		break;
	case 2:
		handle = stdcall(GetStdHandle, 1, -12);
		break;
	default:
		return -1;
	}
	stdcall(WriteFile, 5, handle, buf, n, &written, 0);
	return written;
}

void*
get_symdat_addr(void)
{
	byte *mod, *p;
	uint32 peh, add;
	uint16 oph;

	mod = stdcall(GetModuleHandle, 1, 0);
	peh = *(uint32*)(mod+0x3c);
	p = mod+peh+4;
	oph = *(uint16*)(p+0x10);
	p += 0x14+oph;
	while(strcmp(p, (byte*)".symdat"))
		p += 40;
	add = *(uint32*)(p+0x0c);
	return mod+add;
}

// Thread-safe allocation of an event.
static void
initevent(void **pevent)
{
	void *event;

	event = stdcall(CreateEvent, 4, 0, 0, 0, 0);
	if(!casp(pevent, 0, event)) {
		// Someone else filled it in.  Use theirs.
		stdcall(CloseHandle, 1, event);
	}
}

static void
eventlock(Lock *l)
{
	// Allocate event if needed.
	if(l->event == 0)
		initevent(&l->event);

	if(xadd(&l->key, 1) > 1)	// someone else has it; wait
		stdcall(WaitForSingleObject, 2, l->event, -1);
}

static void
eventunlock(Lock *l)
{
	if(xadd(&l->key, -1) > 0)	// someone else is waiting
		stdcall(SetEvent, 1, l->event);
}

void
lock(Lock *l)
{
	if(m->locks < 0)
		throw("lock count");
	m->locks++;
	eventlock(l);
}

void
unlock(Lock *l)
{
	m->locks--;
	if(m->locks < 0)
		throw("lock count");
	eventunlock(l);
}

void
destroylock(Lock *l)
{
	if(l->event != 0)
		stdcall(CloseHandle, 1, l->event);
}

void
noteclear(Note *n)
{
	eventlock(&n->lock);
}

void
notewakeup(Note *n)
{
	eventunlock(&n->lock);
}

void
notesleep(Note *n)
{
	eventlock(&n->lock);
	eventunlock(&n->lock);	// Let other sleepers find out too.
}

void
newosproc(M *m, G *g, void *stk, void (*fn)(void))
{
	struct {
		void *args;
		void *event_handle;
	} param = { &m };
	extern uint32 threadstart(void *p);

	USED(g, stk, fn);
	param.event_handle = stdcall(CreateEvent, 4, 0, 0, 0, 0);
	stdcall(CreateThread, 6, 0, 0, threadstart, &param, 0, 0);
	stdcall(WaitForSingleObject, 2, param.event_handle, -1);
	stdcall(CloseHandle, 1, param.event_handle);
}

// Called to initialize a new m (including the bootstrap m).
void
minit(void)
{
}

// Calling stdcall on os stack.
#pragma textflag 7
void *
stdcall(void *fn, int32 count, ...)
{
	uintptr *a;
	StdcallParams p;

	p.fn = fn;
	a = (uintptr*)(&count + 1);
	while(count > 0) {
		count--;
		p.args[count] = a[count];
	}
	syscall(&p);
	return (void*)(p.r);
}

void
call_syscall(void *args)
{
	StdcallParams *p = (StdcallParams*)args;
	stdcall_raw(SetLastError, 0);
	p->r = (uintptr)stdcall_raw((void*)p->fn, p->args[0], p->args[1], p->args[2], p->args[3], p->args[4], p->args[5], p->args[6], p->args[7], p->args[8], p->args[9], p->args[10], p->args[11]);
	p->err = (uintptr)stdcall_raw(GetLastError);
	return;
}
