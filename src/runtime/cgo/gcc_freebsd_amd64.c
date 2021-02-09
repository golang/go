// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <sys/types.h>
#include <errno.h>
#include <sys/signalvar.h>
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
	pthread_attr_t *attr;
	size_t size;

	// Deal with memory sanitizer/clang interaction.
	// See gcc_linux_amd64.c for details.
	setg_gcc = setg;
	attr = (pthread_attr_t*)malloc(sizeof *attr);
	if (attr == NULL) {
		fatalf("malloc failed: %s", strerror(errno));
	}
	pthread_attr_init(attr);
	pthread_attr_getstacksize(attr, &size);
	g->stacklo = (uintptr)&attr - size + 4096;
	pthread_attr_destroy(attr);
	free(attr);
}

void
_cgo_sys_thread_start(ThreadStart *ts)
{
	pthread_attr_t attr;
	sigset_t ign, oset;
	pthread_t p;
	size_t size;
	int err;

	SIGFILLSET(ign);
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

static void*
threadentry(void *v)
{
	ThreadStart ts;

	ts = *(ThreadStart*)v;
	_cgo_tsan_acquire();
	free(v);
	_cgo_tsan_release();

	crosscall_amd64(ts.fn, setg_gcc, (void*)ts.g);
	return nil;
}
