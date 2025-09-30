// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

#include <string.h>
#include "libcgo.h"
#include "libcgo_unix.h"

void (*x_cgo_inittls)(void **tlsg, void **tlsbase);
static void (*setg_gcc)(void*);

void
x_cgo_init(G *g, void (*setg)(void*), void **tlsbase)
{
	setg_gcc = setg;
	_cgo_set_stacklo(g, NULL);
}

extern void crosscall_ppc64(void (*fn)(void), void *g);

void*
threadentry(void *v)
{
	ThreadStart ts;

	ts = *(ThreadStart*)v;
	_cgo_tsan_acquire();
	free(v);
	_cgo_tsan_release();

	// Save g for this thread in C TLS
	setg_gcc((void*)ts.g);

	crosscall_ppc64(ts.fn, (void*)ts.g);
	return nil;
}
