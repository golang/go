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
	uint64 x, y;
	pthread_key_t tofree[16], k;
	int i, ntofree;
	int havek1, havek2;

	/*
	 * Same logic, code as darwin_386.c:/inittls, except that words
	 * are 8 bytes long now, and the thread-local storage starts at 0x60.
	 * So the offsets are
	 * 0x60+8*0x108 = 0x8a0 and 0x60+8*0x109 = 0x8a8.
	 *
	 * The linker and runtime hard-code these constant offsets
	 * from %gs where we expect to find m and g.  The code
	 * below verifies that the constants are correct once it has
	 * obtained the keys.  Known to ../cmd/6l/obj.c:/8a0
	 * and to ../pkg/runtime/darwin/amd64/sys.s:/8a0
	 *
	 * As disgusting as on the 386; same justification.
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
	 * updates to k1 at 0x8a0, to verify that the TLS array
	 * offset from %gs hasn't changed.
	 */
	pthread_setspecific(k1, (void*)0x123456789abcdef0ULL);
	asm volatile("movq %%gs:0x8a0, %0" : "=r"(x));

	pthread_setspecific(k2, (void*)0x0fedcba987654321);
	asm volatile("movq %%gs:0x8a8, %0" : "=r"(y));

	if(x != 0x123456789abcdef0ULL || y != 0x0fedcba987654321) {
		printf("libcgo: thread-local storage %#x not at %%gs:0x8a0 - x=%#llx y=%#llx\n", k1, x, y);
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

	crosscall_amd64(ts.fn);
	return nil;
}
