// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <pthread.h>
#include <string.h>
#include <signal.h>
#include <ucontext.h>
#include "libcgo.h"
#include "libcgo_unix.h"

static void* threadentry(void*);
static void (*setg_gcc)(void*);

void
x_cgo_init(G *g, void (*setg)(void*))
{
	ucontext_t ctx;

	setg_gcc = setg;
	if (getcontext(&ctx) != 0)
		perror("runtime/cgo: getcontext failed");
	g->stacklo = (uintptr_t)ctx.uc_stack.ss_sp;

	// Solaris processes report a tiny stack when run with "ulimit -s unlimited".
	// Correct that as best we can: assume it's at least 1 MB.
	// See golang.org/issue/12210.
	if(ctx.uc_stack.ss_size < 1024*1024)
		g->stacklo -= 1024*1024 - ctx.uc_stack.ss_size;
}

void
_cgo_sys_thread_start(ThreadStart *ts)
{
	pthread_attr_t attr;
	sigset_t ign, oset;
	pthread_t p;
	void *base;
	size_t size;
	int err;

	sigfillset(&ign);
	pthread_sigmask(SIG_SETMASK, &ign, &oset);

	pthread_attr_init(&attr);

	if (pthread_attr_getstack(&attr, &base, &size) != 0)
		perror("runtime/cgo: pthread_attr_getstack failed");
	if (size == 0) {
		ts->g->stackhi = 2 << 20;
		if (pthread_attr_setstack(&attr, NULL, ts->g->stackhi) != 0)
			perror("runtime/cgo: pthread_attr_setstack failed");
	} else {
		ts->g->stackhi = size;
	}
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
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

	ts = *(ThreadStart*)v;
	free(v);

	crosscall_amd64(ts.fn, setg_gcc, (void*)ts.g);
	return nil;
}
