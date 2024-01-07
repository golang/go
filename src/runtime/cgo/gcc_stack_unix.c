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

#if defined(__GLIBC__) || (defined(__sun) && !defined(__illumos__))
	// pthread_getattr_np is a GNU extension supported in glibc.
	// Solaris is not glibc but does support pthread_getattr_np
	// (and the fallback doesn't work...). Illumos does not.
	pthread_getattr_np(pthread_self(), &attr);  // GNU extension
	pthread_attr_getstack(&attr, &addr, &size); // low address
#elif defined(__illumos__)
	pthread_attr_init(&attr);
	pthread_attr_get_np(pthread_self(), &attr);
	pthread_attr_getstack(&attr, &addr, &size); // low address
#else
	pthread_attr_init(&attr);
	pthread_attr_getstacksize(&attr, &size);
	addr = __builtin_frame_address(0) + 4096 - size;
#endif
	pthread_attr_destroy(&attr);

	bounds[0] = (uintptr)addr;
	bounds[1] = (uintptr)addr + size;
}
