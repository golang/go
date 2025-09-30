// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <string.h>
#include <ucontext.h>
#include "libcgo.h"
#include "libcgo_unix.h"

static void (*setg_gcc)(void*);

void
x_cgo_init(G *g, void (*setg)(void*))
{
	ucontext_t ctx;

	setg_gcc = setg;
	if (getcontext(&ctx) != 0)
		perror("runtime/cgo: getcontext failed");
	g->stacklo = (uintptr_t)ctx.uc_stack.ss_sp;

	// Solaris processes report a tiny stack when run with "ulimit -s unlimited".
	// Correct that as best we can: assume it's at least 1 MB.
	// See golang.org/issue/12210.
	if(ctx.uc_stack.ss_size < 1024*1024)
		g->stacklo -= 1024*1024 - ctx.uc_stack.ss_size;

	// Sanity check the results now, rather than getting a
	// morestack on g0 crash.
	if (g->stacklo >= g->stackhi) {
		fatalf("bad stack bounds: lo=%p hi=%p", (void*)(g->stacklo), (void*)(g->stackhi));
	}
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
