// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <pthread.h>
#include <string.h>
#include <signal.h>
#include "libcgo.h"

static void *threadentry(void*);
static void (*setg_gcc)(void*);

// These will be set in gcc_android_386.c for android-specific customization.
void (*x_cgo_inittls)(void);
void* (*x_cgo_threadentry)(void*);

void
x_cgo_init(G *g, void (*setg)(void*))
{
	pthread_attr_t attr;
	size_t size;

	setg_gcc = setg;
	pthread_attr_init(&attr);
	pthread_attr_getstacksize(&attr, &size);
	g->stacklo = (uintptr)&attr - size + 4096;
	pthread_attr_destroy(&attr);

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

	// Not sure why the memset is necessary here,
	// but without it, we get a bogus stack size
	// out of pthread_attr_getstacksize. C'est la Linux.
	memset(&attr, 0, sizeof attr);
	pthread_attr_init(&attr);
	size = 0;
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
	free(v);

	/*
	 * Set specific keys.
	 */
	setg_gcc((void*)ts.g);

	crosscall_386(ts.fn);
	return nil;
}
