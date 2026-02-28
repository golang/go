// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gccgo

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
