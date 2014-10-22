// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "malloc.h"
#include "textflag.h"

enum
{
	Debug = 0,
};

#pragma textflag NOSPLIT
void*
runtime·sysAlloc(uintptr n, uint64 *stat)
{
	void *v;

	v = runtime·mmap(nil, n, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, -1, 0);
	if(v < (void*)4096) {
		if(Debug)
			runtime·printf("sysAlloc(%p): %p\n", n, v);
		return nil;
	}
	runtime·xadd64(stat, n);
	if(Debug)
		runtime·printf("sysAlloc(%p) = %p\n", n, v);
	return v;
}

void
runtime·SysUnused(void *v, uintptr n)
{
	if(Debug)
		runtime·printf("SysUnused(%p, %p)\n", v, n);
}

void
runtime·SysUsed(void *v, uintptr n)
{
	USED(v);
	USED(n);
}

void
runtime·SysFree(void *v, uintptr n, uint64 *stat)
{
	if(Debug)
		runtime·printf("SysFree(%p, %p)\n", v, n);
	runtime·xadd64(stat, -(uint64)n);
	runtime·munmap(v, n);
}

void
runtime·SysFault(void *v, uintptr n)
{
	runtime·mmap(v, n, PROT_NONE, MAP_ANON|MAP_PRIVATE|MAP_FIXED, -1, 0);
}

void*
runtime·SysReserve(void *v, uintptr n, bool *reserved)
{
	void *p;

	// On 64-bit, people with ulimit -v set complain if we reserve too
	// much address space.  Instead, assume that the reservation is okay
	// and check the assumption in SysMap.
	if(NaCl || sizeof(void*) == 8) {
		*reserved = false;
		return v;
	}
	
	p = runtime·mmap(v, n, PROT_NONE, MAP_ANON|MAP_PRIVATE, -1, 0);
	if(p < (void*)4096)
		return nil;
	*reserved = true;
	return p;
}

void
runtime·SysMap(void *v, uintptr n, bool reserved, uint64 *stat)
{
	void *p;
	
	runtime·xadd64(stat, n);

	// On 64-bit, we don't actually have v reserved, so tread carefully.
	if(!reserved) {
		p = runtime·mmap(v, n, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, -1, 0);
		if(p == (void*)ENOMEM) {
			runtime·printf("SysMap(%p, %p): %p\n", v, n, p);
			runtime·throw("runtime: out of memory");
		}
		if(p != v) {
			runtime·printf("SysMap(%p, %p): %p\n", v, n, p);
			runtime·printf("runtime: address space conflict: map(%p) = %p\n", v, p);
			runtime·throw("runtime: address space conflict");
		}
		if(Debug)
			runtime·printf("SysMap(%p, %p) = %p\n", v, n, p);
		return;
	}

	p = runtime·mmap(v, n, PROT_READ|PROT_WRITE, MAP_ANON|MAP_FIXED|MAP_PRIVATE, -1, 0);
	if(p == (void*)ENOMEM) {
		runtime·printf("SysMap(%p, %p): %p\n", v, n, p);
		runtime·throw("runtime: out of memory");
	}
	if(p != v) {
		runtime·printf("SysMap(%p, %p): %p\n", v, n, p);
		runtime·printf("mmap MAP_FIXED %p returned %p\n", v, p);
		runtime·throw("runtime: cannot map pages in arena address space");
	}
	if(Debug)
		runtime·printf("SysMap(%p, %p) = %p\n", v, n, p);
}
