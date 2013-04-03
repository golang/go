// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Implementation of the race detector API.
// +build race

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "race.h"

void runtime∕race·Initialize(uintptr *racectx);
void runtime∕race·MapShadow(void *addr, uintptr size);
void runtime∕race·Finalize(void);
void runtime∕race·FinalizerGoroutine(uintptr racectx);
void runtime∕race·Read(uintptr racectx, void *addr, void *pc);
void runtime∕race·Write(uintptr racectx, void *addr, void *pc);
void runtime∕race·ReadRange(uintptr racectx, void *addr, uintptr sz, uintptr step, void *pc);
void runtime∕race·WriteRange(uintptr racectx, void *addr, uintptr sz, uintptr step, void *pc);
void runtime∕race·FuncEnter(uintptr racectx, void *pc);
void runtime∕race·FuncExit(uintptr racectx);
void runtime∕race·Malloc(uintptr racectx, void *p, uintptr sz, void *pc);
void runtime∕race·Free(void *p);
void runtime∕race·GoStart(uintptr racectx, uintptr *chracectx, void *pc);
void runtime∕race·GoEnd(uintptr racectx);
void runtime∕race·Acquire(uintptr racectx, void *addr);
void runtime∕race·Release(uintptr racectx, void *addr);
void runtime∕race·ReleaseMerge(uintptr racectx, void *addr);

extern byte noptrdata[];
extern byte enoptrbss[];

static bool onstack(uintptr argp);

uintptr
runtime·raceinit(void)
{
	uintptr racectx, start, size;

	m->racecall = true;
	runtime∕race·Initialize(&racectx);
	// Round data segment to page boundaries, because it's used in mmap().
	start = (uintptr)noptrdata & ~(PageSize-1);
	size = ROUND((uintptr)enoptrbss - start, PageSize);
	runtime∕race·MapShadow((void*)start, size);
	m->racecall = false;
	return racectx;
}

void
runtime·racefini(void)
{
	m->racecall = true;
	runtime∕race·Finalize();
	m->racecall = false;
}

void
runtime·racemapshadow(void *addr, uintptr size)
{
	m->racecall = true;
	runtime∕race·MapShadow(addr, size);
	m->racecall = false;
}

// Called from instrumented code.
// If we split stack, getcallerpc() can return runtime·lessstack().
#pragma textflag 7
void
runtime·racewrite(uintptr addr)
{
	if(!onstack(addr)) {
		m->racecall = true;
		runtime∕race·Write(g->racectx, (void*)addr, runtime·getcallerpc(&addr));
		m->racecall = false;
	}
}

// Called from instrumented code.
// If we split stack, getcallerpc() can return runtime·lessstack().
#pragma textflag 7
void
runtime·raceread(uintptr addr)
{
	if(!onstack(addr)) {
		m->racecall = true;
		runtime∕race·Read(g->racectx, (void*)addr, runtime·getcallerpc(&addr));
		m->racecall = false;
	}
}

// Called from runtime·racefuncenter (assembly).
#pragma textflag 7
void
runtime·racefuncenter1(uintptr pc)
{
	// If the caller PC is lessstack, use slower runtime·callers
	// to walk across the stack split to find the real caller.
	if(pc == (uintptr)runtime·lessstack)
		runtime·callers(2, &pc, 1);

	m->racecall = true;
	runtime∕race·FuncEnter(g->racectx, (void*)pc);
	m->racecall = false;
}

// Called from instrumented code.
#pragma textflag 7
void
runtime·racefuncexit(void)
{
	m->racecall = true;
	runtime∕race·FuncExit(g->racectx);
	m->racecall = false;
}

void
runtime·racemalloc(void *p, uintptr sz, void *pc)
{
	// use m->curg because runtime·stackalloc() is called from g0
	if(m->curg == nil)
		return;
	m->racecall = true;
	runtime∕race·Malloc(m->curg->racectx, p, sz, pc);
	m->racecall = false;
}

void
runtime·racefree(void *p)
{
	m->racecall = true;
	runtime∕race·Free(p);
	m->racecall = false;
}

uintptr
runtime·racegostart(void *pc)
{
	uintptr racectx;

	m->racecall = true;
	runtime∕race·GoStart(g->racectx, &racectx, pc);
	m->racecall = false;
	return racectx;
}

void
runtime·racegoend(void)
{
	m->racecall = true;
	runtime∕race·GoEnd(g->racectx);
	m->racecall = false;
}

static void
memoryaccess(void *addr, uintptr callpc, uintptr pc, bool write)
{
	uintptr racectx;

	if(!onstack((uintptr)addr)) {
		m->racecall = true;
		racectx = g->racectx;
		if(callpc) {
			if(callpc == (uintptr)runtime·lessstack)
				runtime·callers(3, &callpc, 1);
			runtime∕race·FuncEnter(racectx, (void*)callpc);
		}
		if(write)
			runtime∕race·Write(racectx, addr, (void*)pc);
		else
			runtime∕race·Read(racectx, addr, (void*)pc);
		if(callpc)
			runtime∕race·FuncExit(racectx);
		m->racecall = false;
	}
}

void
runtime·racewritepc(void *addr, void *callpc, void *pc)
{
	memoryaccess(addr, (uintptr)callpc, (uintptr)pc, true);
}

void
runtime·racereadpc(void *addr, void *callpc, void *pc)
{
	memoryaccess(addr, (uintptr)callpc, (uintptr)pc, false);
}

static void
rangeaccess(void *addr, uintptr size, uintptr step, uintptr callpc, uintptr pc, bool write)
{
	uintptr racectx;

	if(!onstack((uintptr)addr)) {
		m->racecall = true;
		racectx = g->racectx;
		if(callpc) {
			if(callpc == (uintptr)runtime·lessstack)
				runtime·callers(3, &callpc, 1);
			runtime∕race·FuncEnter(racectx, (void*)callpc);
		}
		if(write)
			runtime∕race·WriteRange(racectx, addr, size, step, (void*)pc);
		else
			runtime∕race·ReadRange(racectx, addr, size, step, (void*)pc);
		if(callpc)
			runtime∕race·FuncExit(racectx);
		m->racecall = false;
	}
}

void
runtime·racewriterangepc(void *addr, uintptr sz, uintptr step, void *callpc, void *pc)
{
	rangeaccess(addr, sz, step, (uintptr)callpc, (uintptr)pc, true);
}

void
runtime·racereadrangepc(void *addr, uintptr sz, uintptr step, void *callpc, void *pc)
{
	rangeaccess(addr, sz, step, (uintptr)callpc, (uintptr)pc, false);
}

void
runtime·raceacquire(void *addr)
{
	runtime·raceacquireg(g, addr);
}

void
runtime·raceacquireg(G *gp, void *addr)
{
	if(g->raceignore)
		return;
	m->racecall = true;
	runtime∕race·Acquire(gp->racectx, addr);
	m->racecall = false;
}

void
runtime·racerelease(void *addr)
{
	runtime·racereleaseg(g, addr);
}

void
runtime·racereleaseg(G *gp, void *addr)
{
	if(g->raceignore)
		return;
	m->racecall = true;
	runtime∕race·Release(gp->racectx, addr);
	m->racecall = false;
}

void
runtime·racereleasemerge(void *addr)
{
	runtime·racereleasemergeg(g, addr);
}

void
runtime·racereleasemergeg(G *gp, void *addr)
{
	if(g->raceignore)
		return;
	m->racecall = true;
	runtime∕race·ReleaseMerge(gp->racectx, addr);
	m->racecall = false;
}

void
runtime·racefingo(void)
{
	m->racecall = true;
	runtime∕race·FinalizerGoroutine(g->racectx);
	m->racecall = false;
}

// func RaceAcquire(addr unsafe.Pointer)
void
runtime·RaceAcquire(void *addr)
{
	runtime·raceacquire(addr);
}

// func RaceRelease(addr unsafe.Pointer)
void
runtime·RaceRelease(void *addr)
{
	runtime·racerelease(addr);
}

// func RaceReleaseMerge(addr unsafe.Pointer)
void
runtime·RaceReleaseMerge(void *addr)
{
	runtime·racereleasemerge(addr);
}

// func RaceSemacquire(s *uint32)
void runtime·RaceSemacquire(uint32 *s)
{
	runtime·semacquire(s);
}

// func RaceSemrelease(s *uint32)
void runtime·RaceSemrelease(uint32 *s)
{
	runtime·semrelease(s);
}

// func RaceRead(addr unsafe.Pointer)
#pragma textflag 7
void
runtime·RaceRead(void *addr)
{
	memoryaccess(addr, 0, (uintptr)runtime·getcallerpc(&addr), false);
}

// func RaceWrite(addr unsafe.Pointer)
#pragma textflag 7
void
runtime·RaceWrite(void *addr)
{
	memoryaccess(addr, 0, (uintptr)runtime·getcallerpc(&addr), true);
}

// func RaceDisable()
void runtime·RaceDisable(void)
{
	g->raceignore++;
}

// func RaceEnable()
void runtime·RaceEnable(void)
{
	g->raceignore--;
}

static bool
onstack(uintptr argp)
{
	// noptrdata, data, bss, noptrbss
	// the layout is in ../../cmd/ld/data.c
	if((byte*)argp >= noptrdata && (byte*)argp < enoptrbss)
		return false;
	if((byte*)argp >= runtime·mheap->arena_start && (byte*)argp < runtime·mheap->arena_used)
		return false;
	return true;
}
