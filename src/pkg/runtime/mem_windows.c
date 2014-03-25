// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "os_GOOS.h"
#include "defs_GOOS_GOARCH.h"
#include "malloc.h"

enum {
	MEM_COMMIT = 0x1000,
	MEM_RESERVE = 0x2000,
	MEM_DECOMMIT = 0x4000,
	MEM_RELEASE = 0x8000,
	
	PAGE_READWRITE = 0x0004,
	PAGE_NOACCESS = 0x0001,
};

#pragma dynimport runtime·VirtualAlloc VirtualAlloc "kernel32.dll"
#pragma dynimport runtime·VirtualFree VirtualFree "kernel32.dll"
#pragma dynimport runtime·VirtualProtect VirtualProtect "kernel32.dll"
extern void *runtime·VirtualAlloc;
extern void *runtime·VirtualFree;
extern void *runtime·VirtualProtect;

void*
runtime·SysAlloc(uintptr n, uint64 *stat)
{
	runtime·xadd64(stat, n);
	return runtime·stdcall(runtime·VirtualAlloc, 4, nil, n, (uintptr)(MEM_COMMIT|MEM_RESERVE), (uintptr)PAGE_READWRITE);
}

void
runtime·SysUnused(void *v, uintptr n)
{
	void *r;

	r = runtime·stdcall(runtime·VirtualFree, 3, v, n, (uintptr)MEM_DECOMMIT);
	if(r == nil)
		runtime·throw("runtime: failed to decommit pages");
}

void
runtime·SysUsed(void *v, uintptr n)
{
	void *r;

	r = runtime·stdcall(runtime·VirtualAlloc, 4, v, n, (uintptr)MEM_COMMIT, (uintptr)PAGE_READWRITE);
	if(r != v)
		runtime·throw("runtime: failed to commit pages");
}

void
runtime·SysFree(void *v, uintptr n, uint64 *stat)
{
	uintptr r;

	runtime·xadd64(stat, -(uint64)n);
	r = (uintptr)runtime·stdcall(runtime·VirtualFree, 3, v, (uintptr)0, (uintptr)MEM_RELEASE);
	if(r == 0)
		runtime·throw("runtime: failed to release pages");
}

void
runtime·SysFault(void *v, uintptr n)
{
	// SysUnused makes the memory inaccessible and prevents its reuse
	runtime·SysUnused(v, n);
}

void*
runtime·SysReserve(void *v, uintptr n)
{
	*reserved = true;
	// v is just a hint.
	// First try at v.
	v = runtime·stdcall(runtime·VirtualAlloc, 4, v, n, (uintptr)MEM_RESERVE, (uintptr)PAGE_READWRITE);
	if(v != nil)
		return v;
	
	// Next let the kernel choose the address.
	return runtime·stdcall(runtime·VirtualAlloc, 4, nil, n, (uintptr)MEM_RESERVE, (uintptr)PAGE_READWRITE);
}

void
runtime·SysMap(void *v, uintptr n, bool reserved, uint64 *stat)
{
	void *p;

	USED(reserved);

	runtime·xadd64(stat, n);
	p = runtime·stdcall(runtime·VirtualAlloc, 4, v, n, (uintptr)MEM_COMMIT, (uintptr)PAGE_READWRITE);
	if(p != v)
		runtime·throw("runtime: cannot map pages in arena address space");
}
