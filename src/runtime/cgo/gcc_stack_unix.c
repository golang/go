// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix && !darwin

#ifndef _GNU_SOURCE // pthread_getattr_np
#define _GNU_SOURCE
#endif

#include <pthread.h>
#include "libcgo.h"

void
x_cgo_getstackbound(uintptr bounds[2])
{
	pthread_attr_t attr;
	void *addr;
	size_t size;
	int err;

	// Needed before pthread_getattr_np, too, since before glibc 2.32
	// it did not call pthread_attr_init in all cases (see #65625).
	err = pthread_attr_init(&attr);
	if (err != 0)
		fatalf("pthread_attr_init failed: %d", err);
#if defined(__GLIBC__) || defined(__BIONIC__) || (defined(__sun) && !defined(__illumos__))
	// pthread_getattr_np is a GNU extension supported in glibc.
	// Solaris is not glibc but does support pthread_getattr_np
	// (and the fallback doesn't work...). Illumos does not.
	err = pthread_getattr_np(pthread_self(), &attr);
	if (err != 0)
		fatalf("pthread_getattr_np failed: %d", err);
	err = pthread_attr_getstack(&attr, &addr, &size);
	if (err != 0)
		fatalf("pthread_attr_getstack failed: %d", err);
#elif defined(__illumos__)
	err = pthread_attr_get_np(pthread_self(), &attr);
	if (err != 0)
		fatalf("pthread_attr_get_np failed: %d", err);
	err = pthread_attr_getstack(&attr, &addr, &size);
	if (err != 0)
		fatalf("pthread_attr_getstack failed: %d", err);
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
