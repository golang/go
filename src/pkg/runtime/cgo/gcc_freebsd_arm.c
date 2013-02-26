// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <pthread.h>
#include <string.h>
#include "libcgo.h"

static void *threadentry(void*);

// We have to resort to TLS variable to save g(R10) and
// m(R9). One reason is that external code might trigger
// SIGSEGV, and our runtime.sigtramp don't even know we
// are in external code, and will continue to use R10/R9,
// this might as well result in another SIGSEGV.
// Note: all three functions will clobber R0, and the last
// two can be called from 5c ABI code.
void __aeabi_read_tp(void) __attribute__((naked));
void cgo_tls_set_gm(void) __attribute__((naked));
void cgo_tls_get_gm(void) __attribute__((naked));

void
__aeabi_read_tp(void)
{
	// read @ 0xffff1000
	__asm__ __volatile__ (
		"ldr r0, =0xffff1000\n\t"
		"ldr r0, [r0]\n\t"
		"mov pc, lr\n\t"
	);
}

// g (R10) at 8(TP), m (R9) at 12(TP)
void
cgo_tls_get_gm(void)
{
	__asm__ __volatile__ (
		"push {lr}\n\t"
		"bl __aeabi_read_tp\n\t"
		"ldr r10, [r0, #8]\n\t"
		"ldr r9, [r0, #12]\n\t"
		"pop {pc}\n\t"
	);
}

void
cgo_tls_set_gm(void)
{
	__asm__ __volatile__ (
		"push {lr}\n\t"
		"bl __aeabi_read_tp\n\t"
		"str r10, [r0, #8]\n\t"
		"str r9, [r0, #12]\n\t"
		"pop {pc}\n\t"
	);
}

// both cgo_tls_{get,set}_gm can be called from runtime
void (*cgo_load_gm)(void) = cgo_tls_get_gm;
void (*cgo_save_gm)(void) = cgo_tls_set_gm;

static void
xinitcgo(G *g)
{
	pthread_attr_t attr;
	size_t size;
	cgo_tls_set_gm(); // save g and m for the initial thread

	pthread_attr_init(&attr);
	pthread_attr_getstacksize(&attr, &size);
	g->stackguard = (uintptr)&attr - size + 4096;
	pthread_attr_destroy(&attr);
}

void (*initcgo)(G*) = xinitcgo;

void
libcgo_sys_thread_start(ThreadStart *ts)
{
	pthread_attr_t attr;
	pthread_t p;
	size_t size;
	int err;

	// Not sure why the memset is necessary here,
	// but without it, we get a bogus stack size
	// out of pthread_attr_getstacksize.  C'est la Linux.
	memset(&attr, 0, sizeof attr);
	pthread_attr_init(&attr);
	size = 0;
	pthread_attr_getstacksize(&attr, &size);
	ts->g->stackguard = size;
	err = pthread_create(&p, &attr, threadentry, ts);
	if (err != 0) {
		fprintf(stderr, "runtime/cgo: pthread_create failed: %s\n", strerror(err));
		abort();
	}
}

extern void crosscall_arm2(void (*fn)(void), void *g, void *m);
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
	ts.g->stackguard = (uintptr)&ts - ts.g->stackguard + 4096 * 2;

	crosscall_arm2(ts.fn, (void *)ts.g, (void *)ts.m);
	return nil;
}
