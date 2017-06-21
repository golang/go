// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <sys/types.h>
#include <machine/sysarch.h>
#include <sys/signalvar.h>
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include "libcgo.h"
#include "libcgo_unix.h"

#ifdef ARM_TP_ADDRESS
// ARM_TP_ADDRESS is (ARM_VECTORS_HIGH + 0x1000) or 0xffff1000
// and is known to runtime.read_tls_fallback. Verify it with
// cpp.
#if ARM_TP_ADDRESS != 0xffff1000
#error Wrong ARM_TP_ADDRESS!
#endif
#endif

static void *threadentry(void*);

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

	SIGFILLSET(ign);
	pthread_sigmask(SIG_SETMASK, &ign, &oset);

	// Not sure why the memset is necessary here,
	// but without it, we get a bogus stack size
	// out of pthread_attr_getstacksize. C'est la Linux.
	memset(&attr, 0, sizeof attr);
	pthread_attr_init(&attr);
	size = 0;
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

extern void crosscall_arm1(void (*fn)(void), void (*setg_gcc)(void*), void *g);
static void*
threadentry(void *v)
{
	ThreadStart ts;

	ts = *(ThreadStart*)v;
	free(v);

	crosscall_arm1(ts.fn, setg_gcc, (void*)ts.g);
	return nil;
}
