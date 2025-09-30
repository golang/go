// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <errno.h>
#include <string.h>
#include "libcgo.h"
#include "libcgo_unix.h"

static void (*setg_gcc)(void*);

void
x_cgo_init(G *g, void (*setg)(void*))
{
	uintptr *pbounds;

	// Deal with memory sanitizer/clang interaction.
	// See gcc_linux_amd64.c for details.
	setg_gcc = setg;
	pbounds = (uintptr*)malloc(2 * sizeof(uintptr));
	if (pbounds == NULL) {
		fatalf("malloc failed: %s", strerror(errno));
	}
	_cgo_set_stacklo(g, pbounds);
	free(pbounds);
}

extern void crosscall1(void (*fn)(void), void (*setg_gcc)(void*), void *g);
void*
threadentry(void *v)
{
	ThreadStart ts;

	ts = *(ThreadStart*)v;
	_cgo_tsan_acquire();
	free(v);
	_cgo_tsan_release();

	crosscall1(ts.fn, setg_gcc, (void*)ts.g);
	return nil;
}
