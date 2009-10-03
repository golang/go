// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <pthread.h>
#include "libcgo.h"

static void* threadentry(void*);
static pthread_key_t k1, k2;

/* gccism: arrange for inittls to be called at dynamic load time */
static void inittls(void) __attribute__((constructor));

static void
inittls(void)
{
	uint32 x, y;
	pthread_key_t tofree[16], k;
	int i, ntofree;
	int havek1, havek2;

	/*
	 * Allocate thread-local storage slots for m, g.
	 * The key numbers start at 0x100, and we expect to be
	 * one of the early calls to pthread_key_create, so we
	 * should be able to get pretty low numbers.
	 *
	 * In Darwin/386 pthreads, %gs points at the thread
	 * structure, and each key is an index into the thread-local
	 * storage array that begins at offset 0x48 within in that structure.
	 * It may happen that we are not quite the first function to try
	 * to allocate thread-local storage keys, so instead of depending
	 * on getting 0x100 and 0x101, we try for 0x108 and 0x109,
	 * allocating keys until we get the ones we want and then freeing
	 * the ones we didn't want.
	 *
	 * Thus the final offsets to use in %gs references are
	 * 0x48+4*0x108 = 0x468 and 0x48+4*0x109 = 0x46c.
	 *
	 * The linker and runtime hard-code these constant offsets
	 * from %gs where we expect to find m and g.  The code
	 * below verifies that the constants are correct once it has
	 * obtained the keys.  Known to ../cmd/8l/obj.c:/468
	 * and to ../pkg/runtime/darwin/386/sys.s:/468
	 *
	 * This is truly disgusting and a bit fragile, but taking care
	 * of it here protects the rest of the system from damage.
	 * The alternative would be to use a global variable that
	 * held the offset and refer to that variable each time we
	 * need a %gs variable (m or g).  That approach would
	 * require an extra instruction and memory reference in
	 * every stack growth prolog and would also require
	 * rewriting the code that 8c generates for extern registers.
	 */
	havek1 = 0;
	havek2 = 0;
	ntofree = 0;
	while(!havek1 || !havek2) {
		if(pthread_key_create(&k, nil) < 0) {
			fprintf(stderr, "libcgo: pthread_key_create failed\n");
			abort();
		}
		if(k == 0x108) {
			havek1 = 1;
			k1 = k;
			continue;
		}
		if(k == 0x109) {
			havek2 = 1;
			k2 = k;
			continue;
		}
		if(ntofree >= nelem(tofree)) {
			fprintf(stderr, "libcgo: could not obtain pthread_keys\n");
			fprintf(stderr, "\twanted 0x108 and 0x109\n");
			fprintf(stderr, "\tgot");
			for(i=0; i<ntofree; i++)
				fprintf(stderr, " %#x", tofree[i]);
			fprintf(stderr, "\n");
			abort();
		}
		tofree[ntofree++] = k;
	}

	for(i=0; i<ntofree; i++)
		pthread_key_delete(tofree[i]);

	/*
	 * We got the keys we wanted.  Make sure that we observe
	 * updates to k1 at 0x468, to verify that the TLS array
	 * offset from %gs hasn't changed.
	 */
	pthread_setspecific(k1, (void*)0x12345678);
	asm volatile("movl %%gs:0x468, %0" : "=r"(x));

	pthread_setspecific(k1, (void*)0x87654321);
	asm volatile("movl %%gs:0x468, %0" : "=r"(y));

	if(x != 0x12345678 || y != 0x87654321) {
		printf("libcgo: thread-local storage %#x not at %%gs:0x468 - x=%#x y=%#x\n", k1, x, y);
		abort();
	}
}

void
initcgo(void)
{
}

void
libcgo_sys_thread_start(ThreadStart *ts)
{
	pthread_attr_t attr;
	pthread_t p;
	size_t size;

	pthread_attr_init(&attr);
	pthread_attr_getstacksize(&attr, &size);
	ts->g->stackguard = size;
	pthread_create(&p, &attr, threadentry, ts);
}

static void*
threadentry(void *v)
{
	ThreadStart ts;

	ts = *(ThreadStart*)v;
	free(v);

	ts.g->stackbase = (uintptr)&ts;

	/*
	 * libcgo_sys_thread_start set stackguard to stack size;
	 * change to actual guard pointer.
	 */
	ts.g->stackguard = (uintptr)&ts - ts.g->stackguard + 4096;

	pthread_setspecific(k1, (void*)ts.g);
	pthread_setspecific(k2, (void*)ts.m);

	crosscall_386(ts.fn);
	return nil;
}
