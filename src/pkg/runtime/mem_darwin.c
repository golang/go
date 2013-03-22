// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "malloc.h"

void*
runtime·SysAlloc(uintptr n)
{
	void *v;

	mstats.sys += n;
	v = runtime·mmap(nil, n, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, -1, 0);
	if(v < (void*)4096)
		return nil;
	return v;
}

void
runtime·SysUnused(void *v, uintptr n)
{
	// Linux's MADV_DONTNEED is like BSD's MADV_FREE.
	runtime·madvise(v, n, MADV_FREE);
}

void
runtime·SysFree(void *v, uintptr n)
{
	mstats.sys -= n;
	runtime·munmap(v, n);
}

void*
runtime·SysReserve(void *v, uintptr n)
{
	void *p;

	p = runtime·mmap(v, n, PROT_NONE, MAP_ANON|MAP_PRIVATE, -1, 0);
	if(p < (void*)4096)
		return nil;
	return p;
}

enum
{
	ENOMEM = 12,
};

void
runtime·SysMap(void *v, uintptr n)
{
	void *p;
	
	mstats.sys += n;
	p = runtime·mmap(v, n, PROT_READ|PROT_WRITE, MAP_ANON|MAP_FIXED|MAP_PRIVATE, -1, 0);
	if(p == (void*)ENOMEM)
		runtime·throw("runtime: out of memory");
	if(p != v)
		runtime·throw("runtime: cannot map pages in arena address space");
}
