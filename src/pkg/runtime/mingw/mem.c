// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "os.h"
#include "defs.h"
#include "malloc.h"

void*
SysAlloc(uintptr n)
{
	return stdcall(VirtualAlloc, nil, n, 0x3000, 0x40);
}

void
SysUnused(void *v, uintptr n)
{
	USED(v);
	USED(n);
}

void
SysFree(void *v, uintptr n)
{
	USED(v);
	USED(n);
}

