// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "libcgo.h"

#include <stdlib.h>

/* Stub for calling malloc from Go */
static void
x_cgo_malloc(void *p)
{
	struct a {
		long long n;
		void *ret;
	} *a = p;

	a->ret = malloc(a->n);
}

void (*_cgo_malloc)(void*) = x_cgo_malloc;

/* Stub for calling from Go */
static void
x_cgo_free(void *p)
{
	struct a {
		void *arg;
	} *a = p;

	free(a->arg);
}

void (*_cgo_free)(void*) = x_cgo_free;

/* Stub for creating a new thread */
static void
xlibcgo_thread_start(ThreadStart *arg)
{
	ThreadStart *ts;

	/* Make our own copy that can persist after we return. */
	ts = malloc(sizeof *ts);
	if(ts == nil) {
		fprintf(stderr, "libcgo: out of memory in thread_start\n");
		abort();
	}
	*ts = *arg;

	libcgo_sys_thread_start(ts);	/* OS-dependent half */
}

void (*libcgo_thread_start)(ThreadStart*) = xlibcgo_thread_start;

/* Stub for calling setenv */
static void
xlibcgo_setenv(char **arg)
{
	setenv(arg[0], arg[1], 1);
}

void (*libcgo_setenv)(char**) = xlibcgo_setenv;
