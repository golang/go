// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "os.h"
#include "defs.h"
#include "malloc.h"

enum {
	MEM_COMMIT = 0x1000,
	MEM_RESERVE = 0x2000,
	MEM_RELEASE = 0x8000,
	
	PAGE_EXECUTE_READWRITE = 0x40,
};

void*
SysAlloc(uintptr n)
{
	return stdcall(VirtualAlloc, 4, nil, n, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
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
	return stdcall(VirtualFree, 3, v, n, MEM_RELEASE);
}
