// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

extern void crosscall_s390x(void (*fn)(void), void *g);

void*
threadentry(void *v)
{
	ThreadStart ts;

	ts = *(ThreadStart*)v;
	free(v);

	// Save g for this thread in C TLS
	setg_gcc((void*)ts.g);

	crosscall_s390x(ts.fn, (void*)ts.g);
	return nil;
}
