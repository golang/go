// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

// When cross-compiling with clang to linux/armv5, atomics are emulated
// and cause a compiler warning. This results in a build failure since
// cgo uses -Werror. See #65290.
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Watomic-alignment"
#endif

#include <pthread.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // strerror
#include <time.h>
#include "libcgo.h"
#include "libcgo_unix.h"

static pthread_cond_t runtime_init_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t runtime_init_mu = PTHREAD_MUTEX_INITIALIZER;
static int runtime_init_done;

// pthread_g is a pthread specific key, for storing the g that binded to the C thread.
// The registered pthread_key_destructor will dropm, when the pthread-specified value g is not NULL,
// while a C thread is exiting.
static pthread_key_t pthread_g;
static void pthread_key_destructor(void* g);
uintptr_t x_cgo_pthread_key_created;
void (*x_crosscall2_ptr)(void (*fn)(void *), void *, int, size_t);

// The context function, used when tracing back C calls into Go.
static void (*cgo_context_function)(struct context_arg*);

void
x_cgo_sys_thread_create(void* (*func)(void*), void* arg) {
	pthread_t p;
	int err = _cgo_try_pthread_create(&p, NULL, func, arg);
	if (err != 0) {
		fprintf(stderr, "pthread_create failed: %s", strerror(err));
		abort();
	}
}

uintptr_t
_cgo_wait_runtime_init_done(void) {
	void (*pfn)(struct context_arg*);
	pfn = __atomic_load_n(&cgo_context_function, __ATOMIC_CONSUME);

	int done = 2;
	if (__atomic_load_n(&runtime_init_done, __ATOMIC_CONSUME) != done) {
		pthread_mutex_lock(&runtime_init_mu);
		while (__atomic_load_n(&runtime_init_done, __ATOMIC_CONSUME) == 0) {
			pthread_cond_wait(&runtime_init_cond, &runtime_init_mu);
		}

		// The key and x_cgo_pthread_key_created are for the whole program,
		// whereas the specific and destructor is per thread.
		if (x_cgo_pthread_key_created == 0 && pthread_key_create(&pthread_g, pthread_key_destructor) == 0) {
			x_cgo_pthread_key_created = 1;
		}


		// TODO(iant): For the case of a new C thread calling into Go, such
		// as when using -buildmode=c-archive, we know that Go runtime
		// initialization is complete but we do not know that all Go init
		// functions have been run. We should not fetch cgo_context_function
		// until they have been, because that is where a call to
		// SetCgoTraceback is likely to occur. We are going to wait for Go
		// initialization to be complete anyhow, later, by waiting for
		// main_init_done to be closed in cgocallbackg1. We should wait here
		// instead. See also issue #15943.
		pfn = __atomic_load_n(&cgo_context_function, __ATOMIC_CONSUME);

		__atomic_store_n(&runtime_init_done, done, __ATOMIC_RELEASE);
		pthread_mutex_unlock(&runtime_init_mu);
	}

	if (pfn != nil) {
		struct context_arg arg;

		arg.Context = 0;
		(*pfn)(&arg);
		return arg.Context;
	}
	return 0;
}

// _cgo_set_stacklo sets g->stacklo based on the stack size.
// This is common code called from x_cgo_init, which is itself
// called by rt0_go in the runtime package.
void _cgo_set_stacklo(G *g, uintptr *pbounds)
{
	uintptr bounds[2];

	// pbounds can be passed in by the caller; see gcc_linux_amd64.c.
	if (pbounds == NULL) {
		pbounds = &bounds[0];
	}

	x_cgo_getstackbound(pbounds);

	g->stacklo = *pbounds;

	// Sanity check the results now, rather than getting a
	// morestack on g0 crash.
	if (g->stacklo >= g->stackhi) {
		fprintf(stderr, "runtime/cgo: bad stack bounds: lo=%p hi=%p\n", (void*)(g->stacklo), (void*)(g->stackhi));
		abort();
	}
}

// Store the g into a thread-specific value associated with the pthread key pthread_g.
// And pthread_key_destructor will dropm when the thread is exiting.
void x_cgo_bindm(void* g) {
	// We assume this will always succeed, otherwise, there might be extra M leaking,
	// when a C thread exits after a cgo call.
	// We only invoke this function once per thread in runtime.needAndBindM,
	// and the next calls just reuse the bound m.
	pthread_setspecific(pthread_g, g);
}

void
x_cgo_notify_runtime_init_done(void* dummy __attribute__ ((unused))) {
	pthread_mutex_lock(&runtime_init_mu);
	__atomic_store_n(&runtime_init_done, 1, __ATOMIC_RELEASE);
	pthread_cond_broadcast(&runtime_init_cond);
	pthread_mutex_unlock(&runtime_init_mu);
}

// Sets the context function to call to record the traceback context
// when calling a Go function from C code. Called from runtime.SetCgoTraceback.
void x_cgo_set_context_function(void (*context)(struct context_arg*)) {
	__atomic_store_n(&cgo_context_function, context, __ATOMIC_RELEASE);
}

// Gets the context function.
void (*(_cgo_get_context_function(void)))(struct context_arg*) {
	return __atomic_load_n(&cgo_context_function, __ATOMIC_CONSUME);
}

// _cgo_try_pthread_create retries pthread_create if it fails with
// EAGAIN.
int
_cgo_try_pthread_create(pthread_t* thread, const pthread_attr_t* attr, void* (*pfn)(void*), void* arg) {
	int tries;
	int err;
	struct timespec ts;

	for (tries = 0; tries < 20; tries++) {
		err = pthread_create(thread, attr, pfn, arg);
		if (err == 0) {
			pthread_detach(*thread);
			return 0;
		}
		if (err != EAGAIN) {
			return err;
		}
		ts.tv_sec = 0;
		ts.tv_nsec = (tries + 1) * 1000 * 1000; // Milliseconds.
		nanosleep(&ts, nil);
	}
	return EAGAIN;
}

static void
pthread_key_destructor(void* g) {
	if (x_crosscall2_ptr != NULL) {
		// fn == NULL means dropm.
		// We restore g by using the stored g, before dropm in runtime.cgocallback,
		// since the g stored in the TLS by Go might be cleared in some platforms,
		// before this destructor invoked.
		x_crosscall2_ptr(NULL, g, 0, 0);
	}
}
