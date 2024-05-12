// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix && !darwin

#ifndef _GNU_SOURCE // pthread_getattr_np
#define _GNU_SOURCE
#endif

#include <pthread.h>
#include <string.h>
#include "libcgo.h"

void
x_cgo_getstackbound(uintptr bounds[2])
{
	pthread_attr_t attr;
	void *addr;
	size_t size;
	int err;

#if defined(__GLIBC__) || (defined(__sun) && !defined(__illumos__))
	// pthread_getattr_np is a GNU extension supported in glibc.
	// Solaris is not glibc but does support pthread_getattr_np
	// (and the fallback doesn't work...). Illumos does not.

	// After glibc 2.31, there is a `__pthread_attr_init` call in
	// `pthread_getattr_np`, but if there is no init for `attr`, it
	// will cause `pthread_attr_destroy` free some unknown memory,
	// which will cause golang crash, so we need to call
	// `pthread_attr_init` firstly to have a backward compatibility
	//  with glibc(<= 2.31).
	pthread_attr_init(&attr);

	err = pthread_getattr_np(pthread_self(), &attr);  // GNU extension

	// As we all know, when using clone(2), there is a tid dirty cache
	// bug in all versions of glibc(>=2.25), which is introduced by:
	// https://sourceware.org/git/?p=glibc.git;a=commitdiff;h=c579f48e
	// But we can ignore this bug because we only need the stack's addr
	// and size here, and the error is from `__pthread_getaffinity_np`,
	// which is unrelated to the stack info.
	if (err != 0 && err != 3) {
		fatalf("pthread_getattr_np failed: %s", strerror(err));
	}

	pthread_attr_getstack(&attr, &addr, &size); // low address
#elif defined(__illumos__)
	pthread_attr_init(&attr);
	pthread_attr_get_np(pthread_self(), &attr);
	pthread_attr_getstack(&attr, &addr, &size); // low address
#else
	// We don't know how to get the current stacks, so assume they are the
	// same as the default stack bounds.
	pthread_attr_init(&attr);
	pthread_attr_getstacksize(&attr, &size);
	addr = __builtin_frame_address(0) + 4096 - size;
#endif
	pthread_attr_destroy(&attr);

	// bounds points into the Go stack. TSAN can't see the synchronization
	// in Go around stack reuse.
	_cgo_tsan_acquire();
	bounds[0] = (uintptr)addr;
	bounds[1] = (uintptr)addr + size;
	_cgo_tsan_release();
}
