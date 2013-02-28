// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <pthread.h>
#include <string.h>
#include <signal.h>
#include "libcgo.h"

static void *threadentry(void*);

void
x_cgo_init(G *g)
{
	pthread_attr_t attr;
	size_t size;

	pthread_attr_init(&attr);
	pthread_attr_getstacksize(&attr, &size);
	g->stackguard = (uintptr)&attr - size + 4096;
	pthread_attr_destroy(&attr);
}

void (*_cgo_init)(G*) = x_cgo_init;

void
_cgo_sys_thread_start(ThreadStart *ts)
{
	pthread_attr_t attr;
	sigset_t ign, oset;
	pthread_t p;
	size_t size;
	int err;

	sigfillset(&ign);
	sigprocmask(SIG_SETMASK, &ign, &oset);

	// Not sure why the memset is necessary here,
	// but without it, we get a bogus stack size
	// out of pthread_attr_getstacksize.  C'est la Linux.
	memset(&attr, 0, sizeof attr);
	pthread_attr_init(&attr);
	size = 0;
	pthread_attr_getstacksize(&attr, &size);
	ts->g->stackguard = size;
	err = pthread_create(&p, &attr, threadentry, ts);

	sigprocmask(SIG_SETMASK, &oset, nil);

	if (err != 0) {
		fprintf(stderr, "runtime/cgo: pthread_create failed: %s\n", strerror(err));
		abort();
	}
}

static void*
threadentry(void *v)
{
	ThreadStart ts;

	ts = *(ThreadStart*)v;
	free(v);

	ts.g->stackbase = (uintptr)&ts;

	/*
	 * _cgo_sys_thread_start set stackguard to stack size;
	 * change to actual guard pointer.
	 */
	ts.g->stackguard = (uintptr)&ts - ts.g->stackguard + 4096;

	/*
	 * Set specific keys.  On Linux/ELF, the thread local storage
	 * is just before %gs:0.  Our dynamic 8.out's reserve 8 bytes
	 * for the two words g and m at %gs:-8 and %gs:-4.
	 * Xen requires us to access those words indirect from %gs:0
	 * which points at itself.
	 */
	asm volatile (
		"movl %%gs:0, %%eax\n"		// MOVL 0(GS), tmp
		"movl %0, -8(%%eax)\n"	// MOVL g, -8(GS)
		"movl %1, -4(%%eax)\n"	// MOVL m, -4(GS)
		:: "r"(ts.g), "r"(ts.m) : "%eax"
	);

	crosscall_386(ts.fn);
	return nil;
}
