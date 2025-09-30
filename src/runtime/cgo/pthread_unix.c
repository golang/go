// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

#ifndef _GNU_SOURCE // pthread_getattr_np
#define _GNU_SOURCE
#endif

#include <pthread.h>
#include <string.h>
#include <signal.h>
#include <errno.h>
#include "libcgo.h"
#include "libcgo_unix.h"

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
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
#if defined(__APPLE__)
	// Copy stack size from parent thread instead of using the
	// non-main thread default stack size.
	size = pthread_get_stacksize_np(pthread_self());
	pthread_attr_setstacksize(&attr, size);
#else
	pthread_attr_getstacksize(&attr, &size);
#endif

#if defined(__sun)
	// Solaris can report 0 stack size, fix it.
	if (size == 0) {
		size = 2 << 20;
		if (pthread_attr_setstacksize(&attr, size) != 0) {
			perror("runtime/cgo: pthread_attr_setstacksize failed");
		}
	}
#endif

	// Leave stacklo=0 and set stackhi=size; mstart will do the rest.
	ts->g->stackhi = size;
	err = _cgo_try_pthread_create(&p, &attr, threadentry, ts);

	pthread_sigmask(SIG_SETMASK, &oset, nil);

	if (err != 0) {
		fatalf("pthread_create failed: %s", strerror(err));
	}
}

void
x_cgo_sys_thread_create(void* (*func)(void*), void* arg) {
	pthread_attr_t attr;
	pthread_t p;
	int err;

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	err = _cgo_try_pthread_create(&p, &attr, func, arg);
	if (err != 0) {
		fatalf("pthread_create failed: %s", strerror(err));
	}
}

void
x_cgo_getstackbound(uintptr bounds[2])
{
	pthread_attr_t attr;
	void *addr;
	size_t size;

	// Needed before pthread_getattr_np, too, since before glibc 2.32
	// it did not call pthread_attr_init in all cases (see #65625).
	pthread_attr_init(&attr);
#if defined(__APPLE__)
	// On macOS/iOS, use the non-portable pthread_get_stackaddr_np
	// and pthread_get_stacksize_np APIs (high address + size).
	addr = pthread_get_stackaddr_np(pthread_self());
	size = pthread_get_stacksize_np(pthread_self());
	addr = (void*)((uintptr)addr - size); // convert to low address
#elif defined(__GLIBC__) || defined(__BIONIC__) || (defined(__sun) && !defined(__illumos__))
	// pthread_getattr_np is a GNU extension supported in glibc.
	// Solaris is not glibc but does support pthread_getattr_np
	// (and the fallback doesn't work...). Illumos does not.
	pthread_getattr_np(pthread_self(), &attr);  // GNU extension
	pthread_attr_getstack(&attr, &addr, &size); // low address
#elif defined(__illumos__)
	pthread_attr_get_np(pthread_self(), &attr);
	pthread_attr_getstack(&attr, &addr, &size); // low address
#else
	// We don't know how to get the current stacks, leave it as
	// 0 and the caller will use an estimate based on the current
	// SP.
	addr = 0;
	size = 0;
#endif
	pthread_attr_destroy(&attr);

	// bounds points into the Go stack. TSAN can't see the synchronization
	// in Go around stack reuse.
	_cgo_tsan_acquire();
	bounds[0] = (uintptr)addr;
	bounds[1] = (uintptr)addr + size;
	_cgo_tsan_release();
}

// _cgo_try_pthread_create retries pthread_create if it fails with EAGAIN.
int
_cgo_try_pthread_create(pthread_t* thread, const pthread_attr_t* attr, void* (*pfn)(void*), void* arg) {
	int tries;
	int err;
	struct timespec ts;

	for (tries = 0; tries < 20; tries++) {
		err = pthread_create(thread, attr, pfn, arg);
		if (err == 0) {
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
