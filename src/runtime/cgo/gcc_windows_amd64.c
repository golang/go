// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define WIN64_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include "libcgo.h"
#include "libcgo_windows.h"

static void threadentry(void*);

void
x_cgo_init(G *g)
{
}


void
_cgo_sys_thread_start(ThreadStart *ts)
{
	uintptr_t thandle;

	thandle = _beginthread(threadentry, 0, ts);
	if(thandle == -1) {
		fprintf(stderr, "runtime: failed to create new OS thread (%d)\n", errno);
		abort();
	}
}

static void
threadentry(void *v)
{
	ThreadStart ts;

	ts = *(ThreadStart*)v;
	free(v);

	// minit queries stack bounds from the OS.

	/*
	 * Set specific keys in thread local storage.
	 */
	asm volatile (
	  "movq %0, %%gs:0x28\n"	// MOVL tls0, 0x28(GS)
	  "movq %%gs:0x28, %%rax\n" // MOVQ 0x28(GS), tmp
	  "movq %1, 0(%%rax)\n" // MOVQ g, 0(GS)
	  :: "r"(ts.tls), "r"(ts.g) : "%rax"
	);

	crosscall_amd64(ts.fn);
}
