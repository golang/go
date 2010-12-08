// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <pthread.h>
#include "libcgo.h"

static void* threadentry(void*);

char *environ[] = { 0 };
char *__progname;

static void
inittls(void)
{
}

void
initcgo(void)
{
}

void
libcgo_sys_thread_start(ThreadStart *ts)
{
	pthread_attr_t attr;
	pthread_t p;
	size_t size;

	pthread_attr_init(&attr);
	pthread_attr_getstacksize(&attr, &size);
	ts->g->stackguard = size;
	pthread_create(&p, &attr, threadentry, ts);
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

	/*
	 * Set specific keys.  On FreeBSD/ELF, the thread local storage
	 * is just before %gs:0.  Our dynamic 8.out's reserve 8 bytes
	 * for the two words g and m at %gs:-8 and %gs:-4.
	 */
	asm volatile (
		"movl %0, %%gs:-8\n"	// MOVL g, -8(GS)
		"movl %1, %%gs:-4\n"	// MOVL m, -4(GS)
		:: "r"(ts.g), "r"(ts.m)
	);

	crosscall_386(ts.fn);
	return nil;
}
