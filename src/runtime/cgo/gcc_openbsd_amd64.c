// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <sys/types.h>
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include "libcgo.h"

static void* threadentry(void*);
static void (*setg_gcc)(void*);

// TCB_SIZE is sizeof(struct thread_control_block),
// as defined in /usr/src/lib/librthread/tcb.h
#define TCB_SIZE (4 * sizeof(void *))
#define TLS_SIZE (2 * sizeof(void *))

void *__get_tcb(void);
void __set_tcb(void *);

static int (*sys_pthread_create)(pthread_t *thread, const pthread_attr_t *attr,
	void *(*start_routine)(void *), void *arg);

struct thread_args {
	void *(*func)(void *);
	void *arg;
};

static void
tcb_fixup(int mainthread)
{
	void *newtcb, *oldtcb;

	// The OpenBSD ld.so(1) does not currently support PT_TLS. As a result,
	// we need to allocate our own TLS space while preserving the existing
	// TCB that has been setup via librthread.

	newtcb = malloc(TCB_SIZE + TLS_SIZE);
	if(newtcb == NULL)
		abort();

	// The signal trampoline expects the TLS slots to be zeroed.
	bzero(newtcb, TLS_SIZE);

	oldtcb = __get_tcb();
	bcopy(oldtcb, newtcb + TLS_SIZE, TCB_SIZE);
	__set_tcb(newtcb + TLS_SIZE);

	// NOTE(jsing, minux): we can't free oldtcb without causing double-free
	// problem. so newtcb will be memory leaks. Get rid of this when OpenBSD
	// has proper support for PT_TLS.
}

static void *
thread_start_wrapper(void *arg)
{
	struct thread_args args = *(struct thread_args *)arg;

	free(arg);
	tcb_fixup(0);

	return args.func(args.arg);
}

static void init_pthread_wrapper(void) {
	void *handle;

	// Locate symbol for the system pthread_create function.
	handle = dlopen("libpthread.so", RTLD_LAZY);
	if(handle == NULL) {
		fprintf(stderr, "runtime/cgo: dlopen failed to load libpthread: %s\n", dlerror());
		abort();
	}
	sys_pthread_create = dlsym(handle, "pthread_create");
	if(sys_pthread_create == NULL) {
		fprintf(stderr, "runtime/cgo: dlsym failed to find pthread_create: %s\n", dlerror());
		abort();
	}
	dlclose(handle);
}

static pthread_once_t init_pthread_wrapper_once = PTHREAD_ONCE_INIT;

int
pthread_create(pthread_t *thread, const pthread_attr_t *attr,
	void *(*start_routine)(void *), void *arg)
{
	struct thread_args *p;

	// we must initialize our wrapper in pthread_create, because it is valid to call
	// pthread_create in a static constructor, and in fact, our test for issue 9456
	// does just that.
	if(pthread_once(&init_pthread_wrapper_once, init_pthread_wrapper) != 0) {
		fprintf(stderr, "runtime/cgo: failed to initialize pthread_create wrapper\n");
		abort();
	}

	p = malloc(sizeof(*p));
	if(p == NULL) {
		errno = ENOMEM;
		return -1;
	}
	p->func = start_routine;
	p->arg = arg;

	return sys_pthread_create(thread, attr, thread_start_wrapper, p);
}

void
x_cgo_init(G *g, void (*setg)(void*))
{
	pthread_attr_t attr;
	size_t size;

	setg_gcc = setg;
	pthread_attr_init(&attr);
	pthread_attr_getstacksize(&attr, &size);
	g->stacklo = (uintptr)&attr - size + 4096;
	pthread_attr_destroy(&attr);

	if(pthread_once(&init_pthread_wrapper_once, init_pthread_wrapper) != 0) {
		fprintf(stderr, "runtime/cgo: failed to initialize pthread_create wrapper\n");
		abort();
	}

	tcb_fixup(1);
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
	err = sys_pthread_create(&p, &attr, threadentry, ts);

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

	tcb_fixup(0);

	ts = *(ThreadStart*)v;
	free(v);

	/*
	 * Set specific keys.
	 */
	setg_gcc((void*)ts.g);

	crosscall_amd64(ts.fn);
	return nil;
}
