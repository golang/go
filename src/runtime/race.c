// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Implementation of the race detector API.
// +build race

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "race.h"
#include "type.h"
#include "typekind.h"
#include "textflag.h"

// Race runtime functions called via runtime·racecall.
void __tsan_init(void);
void __tsan_fini(void);
void __tsan_map_shadow(void);
void __tsan_finalizer_goroutine(void);
void __tsan_go_start(void);
void __tsan_go_end(void);
void __tsan_malloc(void);
void __tsan_acquire(void);
void __tsan_release(void);
void __tsan_release_merge(void);
void __tsan_go_ignore_sync_begin(void);
void __tsan_go_ignore_sync_end(void);

// Mimic what cmd/cgo would do.
#pragma cgo_import_static __tsan_init
#pragma cgo_import_static __tsan_fini
#pragma cgo_import_static __tsan_map_shadow
#pragma cgo_import_static __tsan_finalizer_goroutine
#pragma cgo_import_static __tsan_go_start
#pragma cgo_import_static __tsan_go_end
#pragma cgo_import_static __tsan_malloc
#pragma cgo_import_static __tsan_acquire
#pragma cgo_import_static __tsan_release
#pragma cgo_import_static __tsan_release_merge
#pragma cgo_import_static __tsan_go_ignore_sync_begin
#pragma cgo_import_static __tsan_go_ignore_sync_end

// These are called from race_amd64.s.
#pragma cgo_import_static __tsan_read
#pragma cgo_import_static __tsan_read_pc
#pragma cgo_import_static __tsan_read_range
#pragma cgo_import_static __tsan_write
#pragma cgo_import_static __tsan_write_pc
#pragma cgo_import_static __tsan_write_range
#pragma cgo_import_static __tsan_func_enter
#pragma cgo_import_static __tsan_func_exit

#pragma cgo_import_static __tsan_go_atomic32_load
#pragma cgo_import_static __tsan_go_atomic64_load
#pragma cgo_import_static __tsan_go_atomic32_store
#pragma cgo_import_static __tsan_go_atomic64_store
#pragma cgo_import_static __tsan_go_atomic32_exchange
#pragma cgo_import_static __tsan_go_atomic64_exchange
#pragma cgo_import_static __tsan_go_atomic32_fetch_add
#pragma cgo_import_static __tsan_go_atomic64_fetch_add
#pragma cgo_import_static __tsan_go_atomic32_compare_exchange
#pragma cgo_import_static __tsan_go_atomic64_compare_exchange

extern byte runtime·noptrdata[];
extern byte runtime·enoptrdata[];
extern byte runtime·data[];
extern byte runtime·edata[];
extern byte runtime·bss[];
extern byte runtime·ebss[];
extern byte runtime·noptrbss[];
extern byte runtime·enoptrbss[];

// start/end of global data (data+bss).
uintptr runtime·racedatastart;
uintptr runtime·racedataend;
// start/end of heap for race_amd64.s
uintptr runtime·racearenastart;
uintptr runtime·racearenaend;

void runtime·racefuncenter(void *callpc);
void runtime·racefuncexit(void);
void runtime·racereadrangepc1(void *addr, uintptr sz, void *pc);
void runtime·racewriterangepc1(void *addr, uintptr sz, void *pc);
void runtime·racesymbolizethunk(void*);

// racecall allows calling an arbitrary function f from C race runtime
// with up to 4 uintptr arguments.
void runtime·racecall(void(*f)(void), ...);

// checks if the address has shadow (i.e. heap or data/bss)
#pragma textflag NOSPLIT
static bool
isvalidaddr(uintptr addr)
{
	if(addr >= runtime·racearenastart && addr < runtime·racearenaend)
		return true;
	if(addr >= runtime·racedatastart && addr < runtime·racedataend)
		return true;
	return false;
}

#pragma textflag NOSPLIT
uintptr
runtime·raceinit(void)
{
	uintptr racectx, start, end, size;

	// cgo is required to initialize libc, which is used by race runtime
	if(!runtime·iscgo)
		runtime·throw("raceinit: race build must use cgo");
	runtime·racecall(__tsan_init, &racectx, runtime·racesymbolizethunk);
	// Round data segment to page boundaries, because it's used in mmap().
	// The relevant sections are noptrdata, data, bss, noptrbss.
	// In external linking mode, there may be other non-Go data mixed in,
	// and the sections may even occur out of order.
	// Work out a conservative range of addresses.
	start = ~(uintptr)0;
	end = 0;
	if(start > (uintptr)runtime·noptrdata)
		start = (uintptr)runtime·noptrdata;
	if(start > (uintptr)runtime·data)
		start = (uintptr)runtime·data;
	if(start > (uintptr)runtime·noptrbss)
		start = (uintptr)runtime·noptrbss;
	if(start > (uintptr)runtime·bss)
		start = (uintptr)runtime·bss;
	if(end < (uintptr)runtime·enoptrdata)
		end = (uintptr)runtime·enoptrdata;
	if(end < (uintptr)runtime·edata)
		end = (uintptr)runtime·edata;
	if(end < (uintptr)runtime·enoptrbss)
		end = (uintptr)runtime·enoptrbss;
	if(end < (uintptr)runtime·ebss)
		end = (uintptr)runtime·ebss;
	start = start & ~(PageSize-1);
	size = ROUND(end - start, PageSize);
	runtime·racecall(__tsan_map_shadow, start, size);
	runtime·racedatastart = start;
	runtime·racedataend = start + size;
	return racectx;
}

#pragma textflag NOSPLIT
void
runtime·racefini(void)
{
	runtime·racecall(__tsan_fini);
}

#pragma textflag NOSPLIT
void
runtime·racemapshadow(void *addr, uintptr size)
{
	if(runtime·racearenastart == 0)
		runtime·racearenastart = (uintptr)addr;
	if(runtime·racearenaend < (uintptr)addr+size)
		runtime·racearenaend = (uintptr)addr+size;
	runtime·racecall(__tsan_map_shadow, addr, size);
}

#pragma textflag NOSPLIT
void
runtime·racemalloc(void *p, uintptr sz)
{
	runtime·racecall(__tsan_malloc, p, sz);
}

#pragma textflag NOSPLIT
uintptr
runtime·racegostart(void *pc)
{
	uintptr racectx;
	G *spawng;

	if(g->m->curg != nil)
		spawng = g->m->curg;
	else
		spawng = g;

	runtime·racecall(__tsan_go_start, spawng->racectx, &racectx, pc);
	return racectx;
}

#pragma textflag NOSPLIT
void
runtime·racegoend(void)
{
	runtime·racecall(__tsan_go_end, g->racectx);
}

#pragma textflag NOSPLIT
void
runtime·racewriterangepc(void *addr, uintptr sz, void *callpc, void *pc)
{
	if(g != g->m->curg) {
		// The call is coming from manual instrumentation of Go code running on g0/gsignal.
		// Not interesting.
		return;
	}
	if(callpc != nil)
		runtime·racefuncenter(callpc);
	runtime·racewriterangepc1(addr, sz, pc);
	if(callpc != nil)
		runtime·racefuncexit();
}

#pragma textflag NOSPLIT
void
runtime·racereadrangepc(void *addr, uintptr sz, void *callpc, void *pc)
{
	if(g != g->m->curg) {
		// The call is coming from manual instrumentation of Go code running on g0/gsignal.
		// Not interesting.
		return;
	}
	if(callpc != nil)
		runtime·racefuncenter(callpc);
	runtime·racereadrangepc1(addr, sz, pc);
	if(callpc != nil)
		runtime·racefuncexit();
}

#pragma textflag NOSPLIT
void
runtime·racewriteobjectpc(void *addr, Type *t, void *callpc, void *pc)
{
	uint8 kind;

	kind = t->kind & KindMask;
	if(kind == KindArray || kind == KindStruct)
		runtime·racewriterangepc(addr, t->size, callpc, pc);
	else
		runtime·racewritepc(addr, callpc, pc);
}

#pragma textflag NOSPLIT
void
runtime·racereadobjectpc(void *addr, Type *t, void *callpc, void *pc)
{
	uint8 kind;

	kind = t->kind & KindMask;
	if(kind == KindArray || kind == KindStruct)
		runtime·racereadrangepc(addr, t->size, callpc, pc);
	else
		runtime·racereadpc(addr, callpc, pc);
}

#pragma textflag NOSPLIT
void
runtime·raceacquire(void *addr)
{
	runtime·raceacquireg(g, addr);
}

#pragma textflag NOSPLIT
void
runtime·raceacquireg(G *gp, void *addr)
{
	if(g->raceignore || !isvalidaddr((uintptr)addr))
		return;
	runtime·racecall(__tsan_acquire, gp->racectx, addr);
}

#pragma textflag NOSPLIT
void
runtime·racerelease(void *addr)
{
	if(g->raceignore || !isvalidaddr((uintptr)addr))
		return;
	runtime·racereleaseg(g, addr);
}

#pragma textflag NOSPLIT
void
runtime·racereleaseg(G *gp, void *addr)
{
	if(g->raceignore || !isvalidaddr((uintptr)addr))
		return;
	runtime·racecall(__tsan_release, gp->racectx, addr);
}

#pragma textflag NOSPLIT
void
runtime·racereleasemerge(void *addr)
{
	runtime·racereleasemergeg(g, addr);
}

#pragma textflag NOSPLIT
void
runtime·racereleasemergeg(G *gp, void *addr)
{
	if(g->raceignore || !isvalidaddr((uintptr)addr))
		return;
	runtime·racecall(__tsan_release_merge, gp->racectx, addr);
}

#pragma textflag NOSPLIT
void
runtime·racefingo(void)
{
	runtime·racecall(__tsan_finalizer_goroutine, g->racectx);
}

// func RaceAcquire(addr unsafe.Pointer)
#pragma textflag NOSPLIT
void
runtime·RaceAcquire(void *addr)
{
	runtime·raceacquire(addr);
}

// func RaceRelease(addr unsafe.Pointer)
#pragma textflag NOSPLIT
void
runtime·RaceRelease(void *addr)
{
	runtime·racerelease(addr);
}

// func RaceReleaseMerge(addr unsafe.Pointer)
#pragma textflag NOSPLIT
void
runtime·RaceReleaseMerge(void *addr)
{
	runtime·racereleasemerge(addr);
}

// func RaceDisable()
#pragma textflag NOSPLIT
void
runtime·RaceDisable(void)
{
	if(g->raceignore++ == 0)
		runtime·racecall(__tsan_go_ignore_sync_begin, g->racectx);
}

// func RaceEnable()
#pragma textflag NOSPLIT
void
runtime·RaceEnable(void)
{
	if(--g->raceignore == 0)
		runtime·racecall(__tsan_go_ignore_sync_end, g->racectx);
}
