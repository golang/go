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

void
initcgo(void)
{
}

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

    ts = *(ThreadStart*)v;
    free(v);

    ts.g->stackbase = (uintptr)&ts;

    /*
     * libcgo_sys_thread_start set stackguard to stack size;
     * change to actual guard pointer.
     */
    ts.g->stackguard = (uintptr)&ts - ts.g->stackguard + 4096;

    crosscall_386(ts.fn);
    return nil;
}
