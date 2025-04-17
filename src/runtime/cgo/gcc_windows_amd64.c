// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include "libcgo.h"
#include "libcgo_windows.h"

static unsigned long __stdcall threadentry(void*);
static void (*setg_gcc)(void*);
static DWORD *tls_g;

void
x_cgo_init(G *g, void (*setg)(void*), void **tlsg, void **tlsbase)
{
	setg_gcc = setg;
	tls_g = (DWORD *)tlsg;
}


void
_cgo_sys_thread_start(ThreadStart *ts)
{
	_cgo_beginthread(threadentry, ts);
}

extern void crosscall1(void (*fn)(void), void (*setg_gcc)(void*), void *g);
static unsigned long
__stdcall
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
	  "movq %0, %%gs:0(%1)\n"	// MOVL tls0, 0(tls_g)(GS)
	  :: "r"(ts.tls), "r"(*tls_g)
	);

	crosscall1(ts.fn, setg_gcc, (void*)ts.g);
	return 0;
}
