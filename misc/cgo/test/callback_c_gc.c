// Copyright 2013 The Go Authors. All rights reserved.
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
