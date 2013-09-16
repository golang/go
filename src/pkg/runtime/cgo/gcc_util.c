// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "libcgo.h"

/* Stub for calling malloc from Go */
void
x_cgo_malloc(void *p)
{
	struct a {
		long long n;
		void *ret;
	} *a = p;

	a->ret = malloc(a->n);
	if(a->ret == NULL && a->n == 0)
		a->ret = malloc(1);
}

/* Stub for calling free from Go */
void
x_cgo_free(void *p)
{
	struct a {
		void *arg;
	} *a = p;

	free(a->arg);
}

/* Stub for creating a new thread */
void
x_cgo_thread_start(ThreadStart *arg)
{
	ThreadStart *ts;

	/* Make our own copy that can persist after we return. */
	ts = malloc(sizeof *ts);
	if(ts == nil) {
		fprintf(stderr, "runtime/cgo: out of memory in thread_start\n");
		abort();
	}
	*ts = *arg;

	_cgo_sys_thread_start(ts);	/* OS-dependent half */
}
