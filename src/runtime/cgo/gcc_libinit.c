// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo
// +build darwin dragonfly freebsd linux netbsd openbsd solaris

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
_cgo_wait_runtime_init_done() {
	void (*pfn)(struct context_arg*);

	pthread_mutex_lock(&runtime_init_mu);
	while (runtime_init_done == 0) {
		pthread_cond_wait(&runtime_init_cond, &runtime_init_mu);
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
	pfn = cgo_context_function;

	pthread_mutex_unlock(&runtime_init_mu);
	if (pfn != nil) {
		struct context_arg arg;

		arg.Context = 0;
		(*pfn)(&arg);
		return arg.Context;
	}
	return 0;
}

void
x_cgo_notify_runtime_init_done(void* dummy) {
	pthread_mutex_lock(&runtime_init_mu);
	runtime_init_done = 1;
	pthread_cond_broadcast(&runtime_init_cond);
	pthread_mutex_unlock(&runtime_init_mu);
}

// Sets the context function to call to record the traceback context
// when calling a Go function from C code. Called from runtime.SetCgoTraceback.
void x_cgo_set_context_function(void (*context)(struct context_arg*)) {
	pthread_mutex_lock(&runtime_init_mu);
	cgo_context_function = context;
	pthread_mutex_unlock(&runtime_init_mu);
}

// Gets the context function.
void (*(_cgo_get_context_function(void)))(struct context_arg*) {
	void (*ret)(struct context_arg*);

	pthread_mutex_lock(&runtime_init_mu);
	ret = cgo_context_function;
	pthread_mutex_unlock(&runtime_init_mu);
	return ret;
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
		if (err != EAGAIN) {
			return err;
		}
		ts.tv_sec = 0;
		ts.tv_nsec = (tries + 1) * 1000 * 1000; // Milliseconds.
		nanosleep(&ts, nil);
	}
	return EAGAIN;
}
