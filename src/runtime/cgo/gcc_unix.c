// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

#include "libcgo.h"
#include "libcgo_unix.h"

// Platform-specific hooks.
void (*x_cgo_inittls)(void **tlsg, void **tlsbase) __attribute__((weak));
void (*x_cgo_init_platform)(void) __attribute__((weak));
void (*x_cgo_threadentry_platform)(void) __attribute__((weak));

#ifdef __linux__
// Weak reference to glibc-specific function. Resolves to the actual
// function on glibc, NULL on musl/bionic/other libc implementations.
// Used by the Go runtime to determine whether .init_array functions
// receive argc/argv (glibc extension) or garbage (ELF ABI standard).
extern const char *gnu_get_libc_version(void) __attribute__((weak));

// Set to 1 during x_cgo_init if running on glibc.
// Accessed by the Go runtime as _cgo_isglibc.
// Use uint8 to match the Go-side byte declaration.
uint8_t x_cgo_isglibc;
#endif

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

void
x_cgo_init(G *g, void (*setg)(void*), void **tlsg, void **tlsbase)
{
	setg_gcc = setg;
	_cgo_set_stacklo(g);

#ifdef __linux__
	x_cgo_isglibc = (gnu_get_libc_version != nil) ? 1 : 0;
#endif

	if (x_cgo_inittls) {
		x_cgo_inittls(tlsg, tlsbase);
	}
	if (x_cgo_init_platform) {
		x_cgo_init_platform();
	}
}

void (* _cgo_init)(G*, void (*)(void*), void **, void **) = x_cgo_init;

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

	crosscall1(ts.fn, setg_gcc, ts.g);
	return NULL;
}
