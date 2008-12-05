// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "malloc.h"

void*
stackalloc(uint32 n)
{
	void *v;
	int32 *ref;

	v = alloc(n);
//printf("stackalloc %d = %p\n", n, v);
	ref = nil;
	findobj(v, nil, nil, &ref);
	*ref = RefStack;
	return v;
}

void
stackfree(void *v)
{
//printf("stackfree %p\n", v);
	free(v);
}

void*
mal(uint32 n)
{
	return alloc(n);
}

void
sys·mal(uint32 n, uint8 *ret)
{
	ret = alloc(n);
	FLUSH(&ret);
}
