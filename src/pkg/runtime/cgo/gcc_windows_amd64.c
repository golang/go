// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define WIN64_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#include "libcgo.h"

static void threadentry(void*);

/* 2MB is default stack size for 64-bit Windows.
   Allocation granularity on Windows is typically 64 KB.
   The constant is also hardcoded in cmd/ld/pe.c (keep synchronized). */
#define STACKSIZE (2*1024*1024)

void
x_cgo_init(G *g)
{
	int tmp;
	g->stackguard = (uintptr)&tmp - STACKSIZE + 8*1024;
}


void
_cgo_sys_thread_start(ThreadStart *ts)
{
	_beginthread(threadentry, 0, ts);
}

static void
threadentry(void *v)
{
	ThreadStart ts;
	void *tls0;

	ts = *(ThreadStart*)v;
	free(v);

	ts.g->stackbase = (uintptr)&ts;
	ts.g->stackguard = (uintptr)&ts - STACKSIZE + 8*1024;

	/*
	 * Set specific keys in thread local storage.
	 */
	tls0 = (void*)LocalAlloc(LPTR, 64);
	asm volatile (
	  "movq %0, %%gs:0x28\n"	// MOVL tls0, 0x28(GS)
	  "movq %%gs:0x28, %%rax\n" // MOVQ 0x28(GS), tmp
	  "movq %1, 0(%%rax)\n" // MOVQ g, 0(GS)
	  "movq %2, 8(%%rax)\n" // MOVQ m, 8(GS)
	  :: "r"(tls0), "r"(ts.g), "r"(ts.m) : "%rax"
	);

	crosscall_amd64(ts.fn);
}
