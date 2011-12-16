// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

// Atomic add and return new value.
#pragma textflag 7
uint32
runtime·xadd(uint32 volatile *val, int32 delta)
{
	uint32 oval, nval;

	for(;;){
		oval = *val;
		nval = oval + delta;
		if(runtime·cas(val, oval, nval))
			return nval;
	}
}

#pragma textflag 7
uint32
runtime·xchg(uint32 volatile* addr, uint32 v)
{
	uint32 old;

	for(;;) {
		old = *addr;
		if(runtime·cas(addr, old, v))
			return old;
	}
}

#pragma textflag 7
void
runtime·procyield(uint32 cnt)
{
	uint32 volatile i;

	for(i = 0; i < cnt; i++) {
	}
}

#pragma textflag 7
uint32
runtime·atomicload(uint32 volatile* addr)
{
	return runtime·xadd(addr, 0);
}

#pragma textflag 7
void*
runtime·atomicloadp(void* volatile* addr)
{
	return (void*)runtime·xadd((uint32 volatile*)addr, 0);
}

#pragma textflag 7
void
runtime·atomicstorep(void* volatile* addr, void* v)
{
	void *old;

	for(;;) {
		old = *addr;
		if(runtime·casp(addr, old, v))
			return;
	}
}

#pragma textflag 7
void
runtime·atomicstore(uint32 volatile* addr, uint32 v)
{
	uint32 old;
	
	for(;;) {
		old = *addr;
		if(runtime·cas(addr, old, v))
			return;
	}
}