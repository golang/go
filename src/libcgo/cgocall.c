// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define _GNU_SOURCE
#include <stdio.h>
#include <errno.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define nil ((void*)0)

/*
 * gcc implementation of src/pkg/runtime/linux/thread.c
 */
typedef struct Lock Lock;
typedef struct Note Note;
typedef uint32_t uint32;

struct Lock
{
	uint32 key;
	uint32 sema;	// ignored
};

struct Note
{
	Lock lock;
	uint32 pad;
};

static struct timespec longtime =
{
	1<<30,	// 34 years
	0
};

static int
cas(uint32 *val, uint32 old, uint32 new)
{
	int ret;

	__asm__ __volatile__(
		"lock; cmpxchgl %2, 0(%3)\n"
		"setz %%al\n"
	:	"=a" (ret)
	:	"a" (old),
		"r" (new),
		"r" (val)
	:	"memory", "cc"
	);

	return ret & 1;
}

static void
futexsleep(uint32 *addr, uint32 val)
{
	int ret;

	ret = syscall(SYS_futex, (int*)addr, FUTEX_WAIT, val, &longtime, nil, 0);
	if(ret >= 0 || errno == EAGAIN || errno == EINTR)
		return;
	fprintf(stderr, "futexsleep: %s\n", strerror(errno));
	*(int*)0 = 0;
}

static void
futexwakeup(uint32 *addr)
{
	int ret;

	ret = syscall(SYS_futex, (int*)addr, FUTEX_WAKE, 1, nil, nil, 0);
	if(ret >= 0)
		return;
	fprintf(stderr, "futexwakeup: %s\n", strerror(errno));
	*(int*)0 = 0;
}

static void
futexlock(Lock *l)
{
	uint32 v;

again:
	v = l->key;
	if((v&1) == 0){
		if(cas(&l->key, v, v|1)){
			// Lock wasn't held; we grabbed it.
			return;
		}
		goto again;
	}

	if(!cas(&l->key, v, v+2))
		goto again;

	futexsleep(&l->key, v+2);
	for(;;){
		v = l->key;
		if((int)v < 2) {
			fprintf(stderr, "futexsleep: invalid key %d\n", (int)v);
			*(int*)0 = 0;
		}
		if(cas(&l->key, v, v-2))
			break;
	}
	goto again;
}

static void
futexunlock(Lock *l)
{
	uint32 v;

again:
	v = l->key;
	if((v&1) == 0)
		*(int*)0 = 0;
	if(!cas(&l->key, v, v&~1))
		goto again;

	// If there were waiters, wake one.
	if(v & ~1)
		futexwakeup(&l->key);
}

static void
lock(Lock *l)
{
	futexlock(l);
}

static void
unlock(Lock *l)
{
	futexunlock(l);
}

void
noteclear(Note *n)
{
	n->lock.key = 0;
	futexlock(&n->lock);
}

static void
notewakeup(Note *n)
{
	futexunlock(&n->lock);
}

static void
notesleep(Note *n)
{
	futexlock(&n->lock);
	futexunlock(&n->lock);
}

/*
 * runtime Cgo server.
 * gcc half of src/pkg/runtime/cgocall.c
 */

typedef struct CgoWork CgoWork;
typedef struct CgoServer CgoServer;
typedef struct Cgo Cgo;

struct Cgo
{
	Lock lock;
	CgoServer *idle;
	CgoWork *whead;
	CgoWork *wtail;
};

struct CgoServer
{
	CgoServer *next;
	Note note;
	CgoWork *work;
};

struct CgoWork
{
	CgoWork *next;
	Note note;
	void (*fn)(void*);
	void *arg;
};

Cgo cgo;

static void newserver(void);

void
initcgo(void)
{
	newserver();
}

static void* go_pthread(void*);

/*
 * allocate servers to handle any work that has piled up
 * and one more server to sit idle and wait for new work.
 */
static void
newserver(void)
{
	CgoServer *f;
	CgoWork *w, *next;
	pthread_t p;

	lock(&cgo.lock);
	// kick off new servers with work to do
	for(w=cgo.whead; w; w=next) {
		next = w;
		w->next = nil;
		f = malloc(sizeof *f);
		memset(f, 0, sizeof *f);
		f->work = w;
		noteclear(&f->note);
		notewakeup(&f->note);
		if(pthread_create(&p, nil, go_pthread, f) < 0) {
			fprintf(stderr, "pthread_create: %s\n", strerror(errno));
			*(int*)0 = 0;
		}
	}
	cgo.whead = nil;
	cgo.wtail = nil;

	// kick off one more server to sit idle
	if(cgo.idle == nil) {
		f = malloc(sizeof *f);
		memset(f, 0, sizeof *f);
		f->next = cgo.idle;
		noteclear(&f->note);
		cgo.idle = f;
		if(pthread_create(&p, nil, go_pthread, f) < 0) {
			fprintf(stderr, "pthread_create: %s\n", strerror(errno));
			*(int*)0 = 0;
		}
	}
	unlock(&cgo.lock);
}

static void*
go_pthread(void *v)
{
	CgoServer *f;
	CgoWork *w;

	// newserver queued us; wait for work
	f = v;
	goto wait;

	for(;;) {
		// kick off new server to handle requests while we work
		newserver();

		// do work
		w = f->work;
		w->fn(w->arg);
		notewakeup(&w->note);
		f->work = nil;

		// take some work if available
		lock(&cgo.lock);
		if((w = cgo.whead) != nil) {
			cgo.whead = w->next;
			if(cgo.whead == nil)
				cgo.wtail = nil;
			unlock(&cgo.lock);
			f->work = w;
			continue;
		}

		// otherwise queue
		f->work = nil;
		noteclear(&f->note);
		f->next = cgo.idle;
		cgo.idle = f;
		unlock(&cgo.lock);

wait:
		// wait for work
		notesleep(&f->note);
	}
}

// Helper.

void
_cgo_malloc(void *p)
{
	struct a {
		long long n;
		void *ret;
	} *a = p;

	a->ret = malloc(a->n);
}
