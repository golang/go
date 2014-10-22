// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "textflag.h"

#pragma textflag NOSPLIT
uint32
runtime·atomicload(uint32 volatile* addr)
{
	return *addr;
}

#pragma textflag NOSPLIT
void*
runtime·atomicloadp(void* volatile* addr)
{
	return *addr;
}

#pragma textflag NOSPLIT
uint64
runtime·xadd64(uint64 volatile* addr, int64 v)
{
	uint64 old;

	do
		old = *addr;
	while(!runtime·cas64(addr, old, old+v));

	return old+v;
}

#pragma textflag NOSPLIT
uint64
runtime·xchg64(uint64 volatile* addr, uint64 v)
{
	uint64 old;

	do
		old = *addr;
	while(!runtime·cas64(addr, old, v));

	return old;
}
