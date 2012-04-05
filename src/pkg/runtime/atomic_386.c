// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

#pragma textflag 7
uint32
runtime·atomicload(uint32 volatile* addr)
{
	return *addr;
}

#pragma textflag 7
void*
runtime·atomicloadp(void* volatile* addr)
{
	return *addr;
}

#pragma textflag 7
uint64
runtime·xadd64(uint64 volatile* addr, int64 v)
{
	uint64 old;

	old = *addr;
	while(!runtime·cas64(addr, &old, old+v)) {
		// nothing
	}
	return old+v;
}
