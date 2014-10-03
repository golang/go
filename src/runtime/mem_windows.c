// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "os_GOOS.h"
#include "defs_GOOS_GOARCH.h"
#include "malloc.h"
#include "textflag.h"

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

#pragma textflag NOSPLIT
void*
runtime·sysAlloc(uintptr n, uint64 *stat)
{
	runtime·xadd64(stat, n);
	return runtime·stdcall4(runtime·VirtualAlloc, 0, n, MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE);
}

void
runtime·SysUnused(void *v, uintptr n)
{
	void *r;
	uintptr small;

	r = runtime·stdcall3(runtime·VirtualFree, (uintptr)v, n, MEM_DECOMMIT);
	if(r != nil)
		return;

	// Decommit failed. Usual reason is that we've merged memory from two different
	// VirtualAlloc calls, and Windows will only let each VirtualFree handle pages from
	// a single VirtualAlloc. It is okay to specify a subset of the pages from a single alloc,
	// just not pages from multiple allocs. This is a rare case, arising only when we're
	// trying to give memory back to the operating system, which happens on a time
	// scale of minutes. It doesn't have to be terribly fast. Instead of extra bookkeeping
	// on all our VirtualAlloc calls, try freeing successively smaller pieces until
	// we manage to free something, and then repeat. This ends up being O(n log n)
	// in the worst case, but that's fast enough.
	while(n > 0) {
		small = n;
		while(small >= 4096 && runtime·stdcall3(runtime·VirtualFree, (uintptr)v, small, MEM_DECOMMIT) == nil)
			small = (small / 2) & ~(4096-1);
		if(small < 4096)
			runtime·throw("runtime: failed to decommit pages");
		v = (byte*)v + small;
		n -= small;
	}
}

void
runtime·SysUsed(void *v, uintptr n)
{
	void *r;
	uintptr small;

	r = runtime·stdcall4(runtime·VirtualAlloc, (uintptr)v, n, MEM_COMMIT, PAGE_READWRITE);
	if(r != v)
		runtime·throw("runtime: failed to commit pages");

	// Commit failed. See SysUnused.
	while(n > 0) {
		small = n;
		while(small >= 4096 && runtime·stdcall4(runtime·VirtualAlloc, (uintptr)v, small, MEM_COMMIT, PAGE_READWRITE) == nil)
			small = (small / 2) & ~(4096-1);
		if(small < 4096)
			runtime·throw("runtime: failed to decommit pages");
		v = (byte*)v + small;
		n -= small;
	}
}

void
runtime·SysFree(void *v, uintptr n, uint64 *stat)
{
	uintptr r;

	runtime·xadd64(stat, -(uint64)n);
	r = (uintptr)runtime·stdcall3(runtime·VirtualFree, (uintptr)v, 0, MEM_RELEASE);
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
runtime·SysReserve(void *v, uintptr n, bool *reserved)
{
	*reserved = true;
	// v is just a hint.
	// First try at v.
	v = runtime·stdcall4(runtime·VirtualAlloc, (uintptr)v, n, MEM_RESERVE, PAGE_READWRITE);
	if(v != nil)
		return v;
	
	// Next let the kernel choose the address.
	return runtime·stdcall4(runtime·VirtualAlloc, 0, n, MEM_RESERVE, PAGE_READWRITE);
}

void
runtime·SysMap(void *v, uintptr n, bool reserved, uint64 *stat)
{
	void *p;

	USED(reserved);

	runtime·xadd64(stat, n);
	p = runtime·stdcall4(runtime·VirtualAlloc, (uintptr)v, n, MEM_COMMIT, PAGE_READWRITE);
	if(p != v)
		runtime·throw("runtime: cannot map pages in arena address space");
}
