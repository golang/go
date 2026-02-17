// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix && !solaris

#include <signal.h>
#include <string.h>
#include <errno.h>
#include "libcgo.h"
#include "libcgo_unix.h"

// Platform-specific hooks.
void (*x_cgo_inittls)(void **tlsg, void **tlsbase) __attribute__((weak));
void (*x_cgo_init_platform)(void) __attribute__((weak));
void (*x_cgo_threadentry_platform)(void) __attribute__((weak));

static void (*setg_gcc)(void*);

// _cgo_set_stacklo sets g->stacklo based on the stack size.
// This is common code called from x_cgo_init, which is itself
// called by rt0_go in the runtime package.
static void
_cgo_set_stacklo(G *g)
{
	uintptr bounds[2];

	x_cgo_getstackbound(bounds);

	g->stacklo = bounds[0];

	// Sanity check the results now, rather than getting a
	// morestack on g0 crash.
	if (g->stacklo >= g->stackhi) {
		fprintf(stderr, "runtime/cgo: bad stack bounds: lo=%p hi=%p\n", (void*)(g->stacklo), (void*)(g->stackhi));
		abort();
	}
}

static void
clang_init()
{
#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))
	/* The memory sanitizer distributed with versions of clang
	   before 3.8 has a bug: if you call mmap before malloc, mmap
	   may return an address that is later overwritten by the msan
	   library. Avoid this problem by forcing a call to malloc
	   here, before we ever call malloc.

	   This is only required for the memory sanitizer, so it's
	   unfortunate that we always run it. It should be possible
	   to remove this when we no longer care about versions of
	   clang before 3.8. The test for this is
	   cmd/cgo/internal/testsanitizers .  */
	uintptr *p;
	p = (uintptr*)malloc(sizeof(uintptr));
	if (p == NULL) {
		fatalf("malloc failed: %s", strerror(errno));
	}
	/* GCC works hard to eliminate a seemingly unnecessary call to
	   malloc, so we actually touch the memory we allocate.  */
	((volatile char *)p)[0] = 0;
	free(p);
#endif
}

void
x_cgo_init(G *g, void (*setg)(void*), void **tlsg, void **tlsbase)
{
	clang_init();
	setg_gcc = setg;
	_cgo_set_stacklo(g);

	if (x_cgo_inittls) {
		x_cgo_inittls(tlsg, tlsbase);
	}
	if (x_cgo_init_platform) {
		x_cgo_init_platform();
	}
}

// TODO: change crosscall_ppc64 and crosscall_s390x so that it matches crosscall1
// signature and behavior.
#if defined(__powerpc64__)
extern void crosscall_ppc64(void (*fn)(void), void *g);
#elif defined(__s390x__)
extern void crosscall_s390x(void (*fn)(void), void *g);
#else
extern void crosscall1(void (*fn)(void), void (*setg_gcc)(void*), void *g);
#endif

void*
threadentry(void *v)
{
	ThreadStart ts;

	ts = *(ThreadStart*)v;
	_cgo_tsan_acquire();
	free(v);
	_cgo_tsan_release();

	if (x_cgo_threadentry_platform != NULL) {
		x_cgo_threadentry_platform();
	}

#if defined(__powerpc64__)
	// Save g for this thread in C TLS
	setg_gcc((void*)ts.g);
	crosscall_ppc64(ts.fn, (void*)ts.g);
#elif defined(__s390x__)
	// Save g for this thread in C TLS
	setg_gcc((void*)ts.g);
	crosscall_s390x(ts.fn, (void*)ts.g);
#else
	crosscall1(ts.fn, setg_gcc, ts.g);
#endif
	return NULL;
}
