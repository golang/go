// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"

static struct {
	Lock l;
	byte pad[CacheLineSize-sizeof(Lock)];
} locktab[57];

#define LOCK(addr) (&locktab[((uintptr)(addr)>>3)%nelem(locktab)].l)

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

#pragma textflag 7
bool
runtime·cas64(uint64 volatile *addr, uint64 *old, uint64 new)
{
	bool res;
	
	runtime·lock(LOCK(addr));
	if(*addr == *old) {
		*addr = new;
		res = true;
	} else {
		*old = *addr;
		res = false;
	}
	runtime·unlock(LOCK(addr));
	return res;
}

#pragma textflag 7
uint64
runtime·xadd64(uint64 volatile *addr, int64 delta)
{
	uint64 res;
	
	runtime·lock(LOCK(addr));
	res = *addr + delta;
	*addr = res;
	runtime·unlock(LOCK(addr));
	return res;
}

#pragma textflag 7
uint64
runtime·xchg64(uint64 volatile *addr, uint64 v)
{
	uint64 res;

	runtime·lock(LOCK(addr));
	res = *addr;
	*addr = v;
	runtime·unlock(LOCK(addr));
	return res;
}

#pragma textflag 7
uint64
runtime·atomicload64(uint64 volatile *addr)
{
	uint64 res;
	
	runtime·lock(LOCK(addr));
	res = *addr;
	runtime·unlock(LOCK(addr));
	return res;
}

#pragma textflag 7
void
runtime·atomicstore64(uint64 volatile *addr, uint64 v)
{
	runtime·lock(LOCK(addr));
	*addr = v;
	runtime·unlock(LOCK(addr));
}
