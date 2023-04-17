// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <pthread.h>
#include "libcgo.h"

void
x_cgo_getstackbound(uintptr *low)
{
	void* addr;
	size_t size;
	pthread_t p;

	p = pthread_self();
	addr = pthread_get_stackaddr_np(p); // high address (!)
	size = pthread_get_stacksize_np(p);
	*low = (uintptr)addr - size;
}
