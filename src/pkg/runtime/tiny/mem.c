// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "malloc.h"

// Assume there's an arbitrary amount of memory starting at "end".
// Sizing PC memory is beyond the scope of this demo.

static byte *allocp;

void*
SysAlloc(uintptr ask)
{
	extern byte end[];
	byte *q;
	
	if(allocp == nil) {
		allocp = end;
		allocp += 7 & -(uintptr)allocp;
	}
	ask += 7 & -ask;

	q = allocp;
	allocp += ask;
	·memclr(q, ask);
	return q;
}

void
SysFree(void *v, uintptr n)
{
	// Push pointer back if this is a free
	// of the most recent SysAlloc.
	n += 7 & -n;
	if(allocp == v+n)
		allocp -= n;
}

void
SysUnused(void *v, uintptr n)
{
	USED(v, n);
}

