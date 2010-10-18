// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "malloc.h"

extern byte end[];
static byte *bloc = { end };

enum
{
	Round = 7
};

void*
SysAlloc(uintptr ask)
{
	uintptr bl;
	
	// Plan 9 sbrk from /sys/src/libc/9sys/sbrk.c
	bl = ((uintptr)bloc + Round) & ~Round;
	if(brk_((void*)(bl + ask)) < 0)
		return (void*)-1;
	bloc = (byte*)bl + ask;
	return (void*)bl;
}

void
SysFree(void *v, uintptr n)
{
	// from tiny/mem.c
	// Push pointer back if this is a free
	// of the most recent SysAlloc.
	n += (n + Round) & ~Round;
	if(bloc == (byte*)v+n)
		bloc -= n;	
}

void
SysUnused(void *v, uintptr n)
{
	USED(v, n);
}

void
SysMemInit(void)
{
}
