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

static void
abort(int8 *name)
{
	uintptr errno;

	errno = (uintptr)runtime·stdcall(runtime·GetLastError, 0);
	runtime·printf("%s failed with errno=%d\n", name, errno);
	runtime·throw(name);
}

void*
runtime·SysAlloc(uintptr n)
{
	void *v;

	v = runtime·stdcall(runtime·VirtualAlloc, 4, nil, n, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
	if(v == 0)
		abort("VirtualAlloc");
	return v;
}

void
runtime·SysUnused(void *v, uintptr n)
{
	USED(v);
	USED(n);
}

void
runtime·SysFree(void *v, uintptr n)
{
	uintptr r;

	r = (uintptr)runtime·stdcall(runtime·VirtualFree, 3, v, 0, MEM_RELEASE);
	if(r == 0)
		abort("VirtualFree");
}

void
runtime·SysMemInit(void)
{
}
