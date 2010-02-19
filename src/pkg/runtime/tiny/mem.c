// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "malloc.h"

// Assume there's an arbitrary amount of memory starting at "end".
// Sizing PC memory is beyond the scope of this demo.

void*
SysAlloc(uintptr ask)
{
	static byte *p;
	extern byte end[];
	byte *q;
	
	if(p == nil) {
		p = end;
		p += 7 & -(uintptr)p;
	}
	ask += 7 & -ask;

	q = p;
	p += ask;
	·memclr(q, ask);
	return q;
}

void
SysFree(void *v, uintptr n)
{
	USED(v, n);
}

void
SysUnused(void *v, uintptr n)
{
	USED(v, n);
}

