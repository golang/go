// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <errno.h>
#include <string.h> // strerror
#include <stdlib.h>
#include "libcgo.h"
#include "libcgo_unix.h"

static void (*setg_gcc)(void*);

// This will be set in gcc_android.c for android-specific customization.
void (*x_cgo_inittls)(void **tlsg, void **tlsbase) __attribute__((common));

void
x_cgo_init(G *g, void (*setg)(void*), void **tlsg, void **tlsbase)
{
	uintptr *pbounds;

	/* The memory sanitizer distributed with versions of clang
	   before 3.8 has a bug: if you call mmap before malloc, mmap
	   may return an address that is later overwritten by the msan
	   library.  Avoid this problem by forcing a call to malloc
	   here, before we ever call malloc.

	   This is only required for the memory sanitizer, so it's
	   unfortunate that we always run it.  It should be possible
	   to remove this when we no longer care about versions of
	   clang before 3.8.  The test for this is
	   misc/cgo/testsanitizers.

	   GCC works hard to eliminate a seemingly unnecessary call to
	   malloc, so we actually use the memory we allocate.  */

	setg_gcc = setg;
	pbounds = (uintptr*)malloc(2 * sizeof(uintptr));
	if (pbounds == NULL) {
		fatalf("malloc failed: %s", strerror(errno));
	}
	_cgo_set_stacklo(g, pbounds);
	free(pbounds);

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
	_cgo_tsan_acquire();
	free(v);
	_cgo_tsan_release();

	crosscall1(ts.fn, setg_gcc, (void*)ts.g);
	return nil;
}
