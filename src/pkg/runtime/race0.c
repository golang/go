// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Stub implementation of the race detector API.
// +build !race

#include "runtime.h"

void
runtime·raceinit(void)
{
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
runtime·racewritepc(void *addr, void *pc)
{
	USED(addr);
	USED(pc);
}

void
runtime·racereadpc(void *addr, void *pc)
{
	USED(addr);
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

void
runtime·racegostart(int32 goid, void *pc)
{
	USED(goid);
	USED(pc);
}

void
runtime·racegoend(int32 goid)
{
	USED(goid);
}
