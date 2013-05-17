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
	MEM_RELEASE = 0x8000,
	
	PAGE_EXECUTE_READWRITE = 0x40,
};

#pragma dynimport runtime·VirtualAlloc VirtualAlloc "kernel32.dll"
#pragma dynimport runtime·VirtualFree VirtualFree "kernel32.dll"
extern void *runtime·VirtualAlloc;
extern void *runtime·VirtualFree;

void*
runtime·SysAlloc(uintptr n)
{
	mstats.sys += n;
	return runtime·stdcall(runtime·VirtualAlloc, 4, nil, n, (uintptr)(MEM_COMMIT|MEM_RESERVE), (uintptr)PAGE_EXECUTE_READWRITE);
}

void
runtime·SysUnused(void *v, uintptr n)
{
	USED(v);
	USED(n);
}

void
runtime·SysFree(void *v, uintptr n)
{
	uintptr r;

	mstats.sys -= n;
	r = (uintptr)runtime·stdcall(runtime·VirtualFree, 3, v, (uintptr)0, (uintptr)MEM_RELEASE);
	if(r == 0)
		runtime·throw("runtime: failed to release pages");
}

void*
runtime·SysReserve(void *v, uintptr n)
{
	// v is just a hint.
	// First try at v.
	v = runtime·stdcall(runtime·VirtualAlloc, 4, v, n, (uintptr)MEM_RESERVE, (uintptr)PAGE_EXECUTE_READWRITE);
	if(v != nil)
		return v;
	
	// Next let the kernel choose the address.
	return runtime·stdcall(runtime·VirtualAlloc, 4, nil, n, (uintptr)MEM_RESERVE, (uintptr)PAGE_EXECUTE_READWRITE);
}

void
runtime·SysMap(void *v, uintptr n)
{
	void *p;
	
	mstats.sys += n;
	p = runtime·stdcall(runtime·VirtualAlloc, 4, v, n, (uintptr)MEM_COMMIT, (uintptr)PAGE_EXECUTE_READWRITE);
	if(p != v)
		runtime·throw("runtime: cannot map pages in arena address space");
}
