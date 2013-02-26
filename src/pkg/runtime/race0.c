// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Stub implementation of the race detector API.
// +build !race

#include "runtime.h"

uintptr
runtime·raceinit(void)
{
	return 0;
}

void
runtime·racefini(void)
{
}


void
runtime·racemapshadow(void *addr, uintptr size)
{
	USED(addr);
	USED(size);
}

void
runtime·racewritepc(void *addr, void *callpc, void *pc)
{
	USED(addr);
	USED(callpc);
	USED(pc);
}

void
runtime·racereadpc(void *addr, void *callpc, void *pc)
{
	USED(addr);
	USED(callpc);
	USED(pc);
}

void
runtime·racewriterangepc(void *addr, uintptr sz, uintptr step, void *callpc, void *pc)
{
	USED(addr);
	USED(sz);
	USED(step);
	USED(callpc);
	USED(pc);
}

void
runtime·racereadrangepc(void *addr, uintptr sz, uintptr step, void *callpc, void *pc)
{
	USED(addr);
	USED(sz);
	USED(step);
	USED(callpc);
	USED(pc);
}

void
runtime·raceacquire(void *addr)
{
	USED(addr);
}

void
runtime·raceacquireg(G *gp, void *addr)
{
	USED(gp);
	USED(addr);
}

void
runtime·racerelease(void *addr)
{
	USED(addr);
}

void
runtime·racereleaseg(G *gp, void *addr)
{
	USED(gp);
	USED(addr);
}

void
runtime·racereleasemerge(void *addr)
{
	USED(addr);
}

void
runtime·racereleasemergeg(G *gp, void *addr)
{
	USED(gp);
	USED(addr);
}

void
runtime·racefingo(void)
{
}

void
runtime·racemalloc(void *p, uintptr sz, void *pc)
{
	USED(p);
	USED(sz);
	USED(pc);
}

void
runtime·racefree(void *p)
{
	USED(p);
}

uintptr
runtime·racegostart(void *pc)
{
	USED(pc);
	return 0;
}

void
runtime·racegoend(void)
{
}
