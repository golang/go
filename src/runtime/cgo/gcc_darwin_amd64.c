// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <string.h> /* for strerror */
#include <pthread.h>
#include <signal.h>
#include "libcgo.h"

static void* threadentry(void*);
static pthread_key_t k1;

#define magic1 (0x23581321345589ULL)

static void
inittls(void)
{
	uint64 x;
	pthread_key_t tofree[128], k;
	int i, ntofree;

	/*
	 * Same logic, code as darwin_386.c:/inittls, except that words
	 * are 8 bytes long now, and the thread-local storage starts
	 * at 0x60 on Leopard / Snow Leopard. So the offset is
	 * 0x60+8*0x108 = 0x8a0.
	 *
	 * The linker and runtime hard-code this constant offset
	 * from %gs where we expect to find g.
	 * Known to ../../../liblink/sym.c:/8a0
	 * and to ../sys_darwin_amd64.s:/8a0
	 *
	 * As disgusting as on the 386; same justification.
	 */
	ntofree = 0;
	for(;;) {
		if(pthread_key_create(&k, nil) < 0) {
			fprintf(stderr, "runtime/cgo: pthread_key_create failed\n");
			abort();
		}
		pthread_setspecific(k, (void*)magic1);
		asm volatile("movq %%gs:0x8a0, %0" : "=r"(x));
		pthread_setspecific(k, 0);
		if(x == magic1) {
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

	/*
	 * We got the key we wanted.  Free the others.
	 */
	for(i=0; i<ntofree; i++)
		pthread_key_delete(tofree[i]);
}

void
x_cgo_init(G *g)
{
	pthread_attr_t attr;
	size_t size;

	pthread_attr_init(&attr);
	pthread_attr_getstacksize(&attr, &size);
	g->stacklo = (uintptr)&attr - size + 4096;
	pthread_attr_destroy(&attr);

	inittls();
}


void
_cgo_sys_thread_start(ThreadStart *ts)
{
	pthread_attr_t attr;
	sigset_t ign, oset;
	pthread_t p;
	size_t size;
	int err;

	sigfillset(&ign);
	pthread_sigmask(SIG_SETMASK, &ign, &oset);

	pthread_attr_init(&attr);
	pthread_attr_getstacksize(&attr, &size);
	// Leave stacklo=0 and set stackhi=size; mstack will do the rest.
	ts->g->stackhi = size;
	err = pthread_create(&p, &attr, threadentry, ts);

	pthread_sigmask(SIG_SETMASK, &oset, nil);

	if (err != 0) {
		fprintf(stderr, "runtime/cgo: pthread_create failed: %s\n", strerror(err));
		abort();
	}
}

static void*
threadentry(void *v)
{
	ThreadStart ts;

	ts = *(ThreadStart*)v;
	free(v);

	pthread_setspecific(k1, (void*)ts.g);

	crosscall_amd64(ts.fn);
	return nil;
}
