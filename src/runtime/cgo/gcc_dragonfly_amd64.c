// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <string.h>
#include "libcgo.h"
#include "libcgo_unix.h"

static void (*setg_gcc)(void*);

void
x_cgo_init(G *g, void (*setg)(void*))
{
	setg_gcc = setg;
	_cgo_set_stacklo(g, NULL);
}

extern void crosscall1(void (*fn)(void), void (*setg_gcc)(void*), void *g);
void*
threadentry(void *v)
{
	ThreadStart ts;

	ts = *(ThreadStart*)v;
	free(v);

	crosscall1(ts.fn, setg_gcc, (void*)ts.g);
	return nil;
}
