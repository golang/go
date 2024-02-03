// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

#include <pthread.h>
#include <string.h>
#include <signal.h>
#include "libcgo.h"
#include "libcgo_unix.h"

static void *threadentry(void*);

void (*x_cgo_inittls)(void **tlsg, void **tlsbase);
static void (*setg_gcc)(void*);

void
x_cgo_init(G *g, void (*setg)(void*), void **tlsbase)
{
	setg_gcc = setg;
	_cgo_set_stacklo(g, NULL);
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
		fatalf("pthread_create failed: %s", strerror(err));
	}
}

extern void crosscall_ppc64(void (*fn)(void), void *g);

static void*
threadentry(void *v)
{
	ThreadStart ts;

	ts = *(ThreadStart*)v;
	_cgo_tsan_acquire();
	free(v);
	_cgo_tsan_release();

	// Save g for this thread in C TLS
	setg_gcc((void*)ts.g);

	crosscall_ppc64(ts.fn, (void*)ts.g);
	return nil;
}
