// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <pthread.h>
#include <errno.h>
#include <string.h> // strerror
#include <signal.h>
#include <stdlib.h>
#include "libcgo.h"

static void* threadentry(void*);
static void (*setg_gcc)(void*);

// These will be set in gcc_android_amd64.c for android-specific customization.
void (*x_cgo_inittls)(void);
void* (*x_cgo_threadentry)(void*);

void
x_cgo_init(G* g, void (*setg)(void*))
{
	pthread_attr_t *attr;
	size_t size;

	/* The memory sanitizer distributed with versions of clang
	   before 3.8 has a bug: if you call mmap before malloc, mmap
	   may return an address that is later overwritten by the msan
	   library.  Avoid this problem by forcing a call to malloc
	   here, before we ever call malloc.

	   This is only required for the memory sanitizer, so it's
	   unfortunate that we always run it.  It should be possible
	   to remove this when we no longer care about versions of
	   clang before 3.8.  The test for this is
	   misc/cgo/testsanitizers.

	   GCC works hard to eliminate a seemingly unnecessary call to
	   malloc, so we actually use the memory we allocate.  */

	setg_gcc = setg;
	attr = (pthread_attr_t*)malloc(sizeof *attr);
	if (attr == NULL) {
		fatalf("malloc failed: %s", strerror(errno));
	}
	pthread_attr_init(attr);
	pthread_attr_getstacksize(attr, &size);
	g->stacklo = (uintptr)&size - size + 4096;
	pthread_attr_destroy(attr);
	free(attr);

	if (x_cgo_inittls) {
		x_cgo_inittls();
	}
}


void
_cgo_sys_thread_start(ThreadStart *ts)
{
	pthread_attr_t attr;
	sigset_t ign, oset;
	pthread_t p;
	size_t size;
	int err;

	sigfillset(&ign);
	pthread_sigmask(SIG_SETMASK, &ign, &oset);

	pthread_attr_init(&attr);
	pthread_attr_getstacksize(&attr, &size);
	// Leave stacklo=0 and set stackhi=size; mstack will do the rest.
	ts->g->stackhi = size;
	err = pthread_create(&p, &attr, threadentry, ts);

	pthread_sigmask(SIG_SETMASK, &oset, nil);

	if (err != 0) {
		fatalf("pthread_create failed: %s", strerror(err));
	}
}

static void*
threadentry(void *v)
{
	if (x_cgo_threadentry) {
		return x_cgo_threadentry(v);
	}

	ThreadStart ts;

	ts = *(ThreadStart*)v;
	_cgo_tsan_acquire();
	free(v);
	_cgo_tsan_release();

	/*
	 * Set specific keys.
	 */
	setg_gcc((void*)ts.g);

	crosscall_amd64(ts.fn);
	return nil;
}
