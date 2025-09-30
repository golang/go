// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (386 || arm || loong64 || mips || mipsle || mips64 || mips64le || riscv64)

#include <string.h>
#include "libcgo.h"
#include "libcgo_unix.h"

void (*x_cgo_inittls)(void **tlsg, void **tlsbase) __attribute__((common));
static void (*setg_gcc)(void*);

void
x_cgo_init(G *g, void (*setg)(void*), void **tlsg, void **tlsbase)
{
	setg_gcc = setg;

	_cgo_set_stacklo(g, NULL);

	if (x_cgo_inittls) {
		x_cgo_inittls(tlsg, tlsbase);
	}
}

extern void crosscall1(void (*fn)(void), void (*setg_gcc)(void*), void *g);
void*
threadentry(void *v)
{
	ThreadStart ts;

	ts = *(ThreadStart*)v;
	free(v);

	crosscall1(ts.fn, setg_gcc, ts.g);
	return nil;
}
