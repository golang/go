// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <sys/types.h>
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include "libcgo.h"
#include "libcgo_unix.h"

static void* threadentry(void*);
static void (*setg_gcc)(void*);

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
	// Leave stacklo=0 and set stackhi=size; mstart will do the rest.
	ts->g->stackhi = size;
	err = _cgo_try_pthread_create(&p, &attr, threadentry, ts);

	pthread_sigmask(SIG_SETMASK, &oset, nil);

	if (err != 0) {
		fprintf(stderr, "runtime/cgo: pthread_create failed: %s\n", strerror(err));
		abort();
	}
}

static void*
threadentry(void *v)
{
	ThreadStart ts;
	stack_t ss;

	ts = *(ThreadStart*)v;
	free(v);

	/*
	 * Set specific keys.
	 */
	setg_gcc((void*)ts.g);

	// On NetBSD, a new thread inherits the signal stack of the
	// creating thread. That confuses minit, so we remove that
	// signal stack here before calling the regular mstart. It's
	// a bit baroque to remove a signal stack here only to add one
	// in minit, but it's a simple change that keeps NetBSD
	// working like other OS's. At this point all signals are
	// blocked, so there is no race.
	memset(&ss, 0, sizeof ss);
	ss.ss_flags = SS_DISABLE;
	sigaltstack(&ss, nil);

	crosscall_386(ts.fn);
	return nil;
}
