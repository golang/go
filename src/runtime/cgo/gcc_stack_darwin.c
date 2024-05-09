// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <pthread.h>
#include "libcgo.h"

void
x_cgo_getstackbound(uintptr bounds[2])
{
	void* addr;
	size_t size;
	pthread_t p;

	p = pthread_self();
	addr = pthread_get_stackaddr_np(p); // high address (!)
	size = pthread_get_stacksize_np(p);

	// bounds points into the Go stack. TSAN can't see the synchronization
	// in Go around stack reuse.
	_cgo_tsan_acquire();
	bounds[0] = (uintptr)addr - size;
	bounds[1] = (uintptr)addr;
	_cgo_tsan_release();
}
