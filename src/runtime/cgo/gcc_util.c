// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "libcgo.h"

/* Stub for creating a new thread */
void
x_cgo_thread_start(ThreadStart *arg)
{
	ThreadStart *ts;

	/* Make our own copy that can persist after we return. */
	_cgo_tsan_acquire();
	ts = malloc(sizeof *ts);
	_cgo_tsan_release();
	if(ts == nil) {
		fprintf(stderr, "runtime/cgo: out of memory in thread_start\n");
		abort();
	}
	*ts = *arg;

	_cgo_sys_thread_start(ts);	/* OS-dependent half */
}
