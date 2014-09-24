// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "os_GOOS.h"
#include "textflag.h"

extern byte runtime·end[];
#pragma dataflag NOPTR
static byte *bloc = { runtime·end };
static Mutex memlock;

enum
{
	Round = PAGESIZE-1
};

static void*
brk(uintptr nbytes)
{
	uintptr bl;

	runtime·lock(&memlock);
	// Plan 9 sbrk from /sys/src/libc/9sys/sbrk.c
	bl = ((uintptr)bloc + Round) & ~Round;
	if(runtime·brk_((void*)(bl + nbytes)) < 0) {
		runtime·unlock(&memlock);
		return nil;
	}
	bloc = (byte*)bl + nbytes;
	runtime·unlock(&memlock);
	return (void*)bl;	
}

static void
sysalloc(void)
{
	uintptr nbytes;
	uint64 *stat;
	void *p;

	nbytes = g->m->scalararg[0];
	stat = g->m->ptrarg[0];
	g->m->scalararg[0] = 0;
	g->m->ptrarg[0] = nil;

	p = brk(nbytes);
	if(p != nil)
		runtime·xadd64(stat, nbytes);

	g->m->ptrarg[0] = p;
}

#pragma textflag NOSPLIT
void*
runtime·sysAlloc(uintptr nbytes, uint64 *stat)
{
	void (*fn)(void);
	void *p;

	g->m->scalararg[0] = nbytes;
	g->m->ptrarg[0] = stat;
	fn = sysalloc;
	runtime·onM(&fn);
	p = g->m->ptrarg[0];
	g->m->ptrarg[0] = nil;
	return p;
}

void
runtime·SysFree(void *v, uintptr nbytes, uint64 *stat)
{
	runtime·xadd64(stat, -(uint64)nbytes);
	runtime·lock(&memlock);
	// from tiny/mem.c
	// Push pointer back if this is a free
	// of the most recent sysAlloc.
	nbytes += (nbytes + Round) & ~Round;
	if(bloc == (byte*)v+nbytes)
		bloc -= nbytes;
	runtime·unlock(&memlock);
}

void
runtime·SysUnused(void *v, uintptr nbytes)
{
	USED(v, nbytes);
}

void
runtime·SysUsed(void *v, uintptr nbytes)
{
	USED(v, nbytes);
}

void
runtime·SysMap(void *v, uintptr nbytes, bool reserved, uint64 *stat)
{
	// SysReserve has already allocated all heap memory,
	// but has not adjusted stats.
	USED(v, reserved);
	runtime·xadd64(stat, nbytes);
}

void
runtime·SysFault(void *v, uintptr nbytes)
{
	USED(v, nbytes);
}

void*
runtime·SysReserve(void *v, uintptr nbytes, bool *reserved)
{
	USED(v);
	*reserved = true;
	return brk(nbytes);
}
