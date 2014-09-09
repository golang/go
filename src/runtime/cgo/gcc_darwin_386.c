// Copyright 2009 The Go Authors.  All rights reserved.
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
	 * Allocate thread-local storage slot for g.
	 * The key numbers start at 0x100, and we expect to be
	 * one of the early calls to pthread_key_create, so we
	 * should be able to get a pretty low number.
	 *
	 * In Darwin/386 pthreads, %gs points at the thread
	 * structure, and each key is an index into the thread-local
	 * storage array that begins at offset 0x48 within in that structure.
	 * It may happen that we are not quite the first function to try
	 * to allocate thread-local storage keys, so instead of depending
	 * on getting 0x100, we try for 0x108, allocating keys until
	 * we get the one we want and then freeing the ones we didn't want.
	 *
	 * Thus the final offset to use in %gs references is
	 * 0x48+4*0x108 = 0x468.
	 *
	 * The linker and runtime hard-code this constant offset
	 * from %gs where we expect to find g.
	 * Known to ../../../liblink/sym.c:/468
	 * and to ../sys_darwin_386.s:/468
	 *
	 * This is truly disgusting and a bit fragile, but taking care
	 * of it here protects the rest of the system from damage.
	 * The alternative would be to use a global variable that
	 * held the offset and refer to that variable each time we
	 * need a %gs variable (g).  That approach would
	 * require an extra instruction and memory reference in
	 * every stack growth prolog and would also require
	 * rewriting the code that 8c generates for extern registers.
	 *
	 * Things get more disgusting on OS X 10.7 Lion.
	 * The 0x48 base mentioned above is the offset of the tsd
	 * array within the per-thread structure on Leopard and Snow Leopard.
	 * On Lion, the base moved a little, so while the math above
	 * still applies, the base is different.  Thus, we cannot
	 * look for specific key values if we want to build binaries
	 * that run on both systems.  Instead, forget about the
	 * specific key values and just allocate and initialize per-thread
	 * storage until we find a key that writes to the memory location
	 * we want.  Then keep that key.
	 */
	ntofree = 0;
	for(;;) {
		if(pthread_key_create(&k, nil) < 0) {
			fprintf(stderr, "runtime/cgo: pthread_key_create failed\n");
			abort();
		}
		pthread_setspecific(k, (void*)magic1);
		asm volatile("movl %%gs:0x468, %0" : "=r"(x));
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

	crosscall_386(ts.fn);
	return nil;
}
