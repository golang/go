// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "malloc.h"

void*
stackalloc(uint32 n)
{
	void *v;

	v = alloc(n);
//printf("stackalloc %d = %p\n", n, v);
	return v;
}

void
stackfree(void *v)
{
//printf("stackfree %p\n", v);
	free(v);
}
