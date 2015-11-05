// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <string.h> /* for strerror */
#include <pthread.h>
#include <signal.h>
#include "libcgo.h"

static void* threadentry(void*);
static pthread_key_t k1;

#define magic1 (0x23581321U)

static void
inittls(void)
{
	uint32 x;
	pthread_key_t tofree[128], k;
	int i, ntofree;

	/*
	 * Same logic, code as gcc_android_amd64.c:/inittls.
	 * Note that this is a temporary hack that should be fixed soon.
	 *
	 * TODO: fix this.
	 *
	 * The linker and runtime hard-code this constant offset
	 * from %gs where we expect to find g. Disgusting.
	 *
	 * Known to src/cmd/link/internal/ld/sym.go:/0xf8
	 * and to src/runtime/sys_linux_386.s:/0xf8 or /GOOS_android.
	 * TODO(hyangah): check 0xb0 works with API23+
	 *
	 * As disgusting as on the darwin/386, darwin/amd64.
	 */
	ntofree = 0;
	for(;;) {
		if(pthread_key_create(&k, nil) < 0) {
			fprintf(stderr, "runtime/cgo: pthread_key_create failed\n");
			abort();
		}
		pthread_setspecific(k, (void*)magic1);
		asm volatile("movl %%gs:0xf8, %0" : "=r"(x));
		pthread_setspecific(k, 0);
		if (x == magic1) {
			k1 = k;
			break;
		}
		if(ntofree >= nelem(tofree)) {
			fprintf(stderr, "runtime/cgo: could not obtain pthread_keys\n");
			fprintf(stderr, "\ttried");
			for(i=0; i<ntofree; i++)
				fprintf(stderr, " %#x", (unsigned)tofree[i]);
			fprintf(stderr, "\n");
			abort();
		}
		tofree[ntofree++] = k;
	}
	// TODO: output to stderr is not useful for apps.
	// Can we fall back to Android's log library?

	/*
	 * We got the key we wanted.  Free the others.
	 */
	for(i=0; i<ntofree; i++) {
		pthread_key_delete(tofree[i]);
	}
}


static void*
threadentry(void *v)
{
	ThreadStart ts;

	ts = *(ThreadStart*)v;
	free(v);

	pthread_setspecific(k1, (void*)ts.g);

	crosscall_386(ts.fn);
	return nil;
}

void (*x_cgo_inittls)(void) = inittls;
void* (*x_cgo_threadentry)(void*) = threadentry;
