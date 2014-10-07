// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build gc

#include "_cgo_export.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Test calling panic from C.  This is what SWIG does.  */

extern void crosscall2(void (*fn)(void *, int), void *, int);
extern void _cgo_panic(void *, int);
extern void _cgo_allocate(void *, int);

void
callPanic(void)
{
	struct { const char *p; } a;
	a.p = "panic from C";
	crosscall2(_cgo_panic, &a, sizeof a);
	*(int*)1 = 1;
}

/* Test calling cgo_allocate from C. This is what SWIG does. */

typedef struct List List;
struct List
{
	List *next;
	int x;
};

void
callCgoAllocate(void)
{
	int i;
	struct { size_t n; void *ret; } a;
	List *l, *head, **tail;

	// Make sure this doesn't crash.
	// And make sure it returns non-nil.
	a.n = 0;
	a.ret = 0;
	crosscall2(_cgo_allocate, &a, sizeof a);
	if(a.ret == 0) {
		fprintf(stderr, "callCgoAllocate: alloc 0 returned nil\n");
		exit(2);
	}
	
	head = 0;
	tail = &head;
	for(i=0; i<100; i++) {
		a.n = sizeof *l;
		crosscall2(_cgo_allocate, &a, sizeof a);
		l = a.ret;
		l->x = i;
		l->next = 0;
		*tail = l;
		tail = &l->next;
	}
	
	gc();
	
	l = head;
	for(i=0; i<100; i++) {
		if(l->x != i) {
			fprintf(stderr, "callCgoAllocate: lost memory\n");
			exit(2);
		}
		l = l->next;
	}
	if(l != 0) {
		fprintf(stderr, "callCgoAllocate: lost memory\n");
		exit(2);
	}
}

