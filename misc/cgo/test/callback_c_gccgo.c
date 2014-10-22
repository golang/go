// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build gccgo

#include "_cgo_export.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Test calling panic from C.  This is what SWIG does.  */

extern void _cgo_panic(const char *);
extern void *_cgo_allocate(size_t);

void
callPanic(void)
{
	_cgo_panic("panic from C");
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
	List *l, *head, **tail;
	
	// Make sure this doesn't crash.
	// And make sure it returns non-nil.
	if(_cgo_allocate(0) == 0) {
		fprintf(stderr, "callCgoAllocate: alloc 0 returned nil\n");
		exit(2);
	}

	head = 0;
	tail = &head;
	for(i=0; i<100; i++) {
		l = _cgo_allocate(sizeof *l);
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

