// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "os_GOOS.h"

extern byte end[];
static byte *bloc = { end };
static Lock memlock;

enum
{
	Round = PAGESIZE-1
};

void*
runtime·SysAlloc(uintptr nbytes)
{
	uintptr bl;

	runtime·lock(&memlock);
	mstats.sys += nbytes;
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

void
runtime·SysFree(void *v, uintptr nbytes)
{
	runtime·lock(&memlock);
	mstats.sys -= nbytes;
	// from tiny/mem.c
	// Push pointer back if this is a free
	// of the most recent SysAlloc.
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
runtime·SysMap(void *v, uintptr nbytes)
{
	USED(v, nbytes);
}

void*
runtime·SysReserve(void *v, uintptr nbytes)
{
	USED(v);
	return runtime·SysAlloc(nbytes);
}
