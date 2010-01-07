// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "os.h"

#define stdcall stdcall_raw

extern void *get_kernel_module(void);

// Also referenced by external packages
void *CloseHandle;
void *ExitProcess;
void *GetStdHandle;
void *SetEvent;
void *WriteFile;

static void *CreateEvent;
static void *CreateThread;
static void *GetModuleHandle;
static void *GetProcAddress;
static void *LoadLibraryEx;
static void *VirtualAlloc;
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
}

// The arguments are strings.
void*
get_proc_addr(void *library, void *name)
{
	void *base;

	base = stdcall(LoadLibraryEx, library, 0, 0);
	return stdcall(GetProcAddress, base, name);
}

void
mingw_goargs(void)
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

	cmd = stdcall(gcl);
	env = stdcall(ges);
	argv = stdcall(clta, cmd, &argc);

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
	stdcall(ExitProcess, code);
}

int32
write(int32 fd, void *buf, int32 n)
{
	void *handle;
	uint32 written;

	written = 0;
	switch(fd) {
	case 1:
		handle = stdcall(GetStdHandle, -11);
		break;
	case 2:
		handle = stdcall(GetStdHandle, -12);
		break;
	default:
		return -1;
	}
	stdcall(WriteFile, handle, buf, n, &written, 0);
	return written;
}

uint8*
runtime_mmap(byte *addr, uint32 len, int32 prot,
	int32 flags, int32 fd, uint32 off)
{
	USED(prot, flags, fd, off);
	return stdcall(VirtualAlloc, addr, len, 0x3000, 0x40);
}

void*
get_symdat_addr(void)
{
	byte *mod, *p;
	uint32 peh, add;
	uint16 oph;

	mod = stdcall(GetModuleHandle, 0);
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

	event = stdcall(CreateEvent, 0, 0, 0, 0);
	if(!casp(pevent, 0, event)) {
		// Someone else filled it in.  Use theirs.
		stdcall(CloseHandle, event);
	}
}

static void
eventlock(Lock *l)
{
	// Allocate event if needed.
	if(l->event == 0)
		initevent(&l->event);

	if(xadd(&l->key, 1) > 1)	// someone else has it; wait
		stdcall(WaitForSingleObject, l->event, -1);
}

static void
eventunlock(Lock *l)
{
	if(xadd(&l->key, -1) > 0)	// someone else is waiting
		stdcall(SetEvent, l->event);
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
	param.event_handle = stdcall(CreateEvent, 0, 0, 0, 0);
	stdcall(CreateThread, 0, 0, threadstart, &param, 0, 0);
	stdcall(WaitForSingleObject, param.event_handle, -1);
	stdcall(CloseHandle, param.event_handle);
}

// Called to initialize a new m (including the bootstrap m).
void
minit(void)
{
}
