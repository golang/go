// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define WIN64_LEAN_AND_MEAN
#include <windows.h>
#include "libcgo.h"

static void *threadentry(void*);

/* From what I've read 2MB is default for 64-bit Linux. 
   Allocation granularity on Windows is typically 64 KB. */
#define STACKSIZE (2*1024*1024)

static void
xinitcgo(G *g)
{
	int tmp;
	g->stackguard = (uintptr)&tmp - STACKSIZE + 4096;
}

void (*initcgo)(G*) = xinitcgo;

void
libcgo_sys_thread_start(ThreadStart *ts)
{
	ts->g->stackguard = STACKSIZE;
	_beginthread(threadentry, STACKSIZE, ts);
}

static void*
threadentry(void *v)
{
	ThreadStart ts;
	void *tls0;

	ts = *(ThreadStart*)v;
	free(v);

	ts.g->stackbase = (uintptr)&ts;

	/*
	 * libcgo_sys_thread_start set stackguard to stack size;
	 * change to actual guard pointer.
	 */
	ts.g->stackguard = (uintptr)&ts - ts.g->stackguard + 4096;

	/*
	 * Set specific keys in thread local storage.
	 */
	tls0 = (void*)LocalAlloc(LPTR, 64);
	asm volatile (
	  "movq %0, %%gs:0x58\n"	// MOVL tls0, 0x58(GS)
	  "movq %%gs:0x58, %%rax\n" // MOVQ 0x58(GS), tmp
	  "movq %1, 0(%%rax)\n" // MOVQ g, 0(GS)
	  "movq %2, 8(%%rax)\n" // MOVQ m, 8(GS)
	  :: "r"(tls0), "r"(ts.g), "r"(ts.m) : "%rax"
	);

	crosscall_amd64(ts.fn);
	return nil;
}
