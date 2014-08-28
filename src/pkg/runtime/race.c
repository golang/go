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
#include "../../cmd/ld/textflag.h"

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

// These are called from race_amd64.s.
#pragma cgo_import_static __tsan_read
#pragma cgo_import_static __tsan_read_pc
#pragma cgo_import_static __tsan_read_range
#pragma cgo_import_static __tsan_write
#pragma cgo_import_static __tsan_write_pc
#pragma cgo_import_static __tsan_write_range
#pragma cgo_import_static __tsan_func_enter
#pragma cgo_import_static __tsan_func_exit

extern byte runtime·noptrdata[];
extern byte runtime·enoptrbss[];
  
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
static bool
isvalidaddr(uintptr addr)
{
	if(addr >= runtime·racearenastart && addr < runtime·racearenaend)
		return true;
	if(addr >= (uintptr)runtime·noptrdata && addr < (uintptr)runtime·enoptrbss)
		return true;
	return false;
}

uintptr
runtime·raceinit(void)
{
	uintptr racectx, start, size;

	// cgo is required to initialize libc, which is used by race runtime
	if(!runtime·iscgo)
		runtime·throw("raceinit: race build must use cgo");
	runtime·racecall(__tsan_init, &racectx, runtime·racesymbolizethunk);
	// Round data segment to page boundaries, because it's used in mmap().
	start = (uintptr)runtime·noptrdata & ~(PageSize-1);
	size = ROUND((uintptr)runtime·enoptrbss - start, PageSize);
	runtime·racecall(__tsan_map_shadow, start, size);
	return racectx;
}

void
runtime·racefini(void)
{
	runtime·racecall(__tsan_fini);
}

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

void
runtime·racegoend(void)
{
	runtime·racecall(__tsan_go_end, g->racectx);
}

void
runtime·racewriterangepc(void *addr, uintptr sz, void *callpc, void *pc)
{
	if(callpc != nil)
		runtime·racefuncenter(callpc);
	runtime·racewriterangepc1(addr, sz, pc);
	if(callpc != nil)
		runtime·racefuncexit();
}

void
runtime·racereadrangepc(void *addr, uintptr sz, void *callpc, void *pc)
{
	if(callpc != nil)
		runtime·racefuncenter(callpc);
	runtime·racereadrangepc1(addr, sz, pc);
	if(callpc != nil)
		runtime·racefuncexit();
}

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

void
runtime·raceacquire(void *addr)
{
	runtime·raceacquireg(g, addr);
}

void
runtime·raceacquireg(G *gp, void *addr)
{
	if(g->raceignore || !isvalidaddr((uintptr)addr))
		return;
	runtime·racecall(__tsan_acquire, gp->racectx, addr);
}

void
runtime·racerelease(void *addr)
{
	if(g->raceignore || !isvalidaddr((uintptr)addr))
		return;
	runtime·racereleaseg(g, addr);
}

void
runtime·racereleaseg(G *gp, void *addr)
{
	if(g->raceignore || !isvalidaddr((uintptr)addr))
		return;
	runtime·racecall(__tsan_release, gp->racectx, addr);
}

void
runtime·racereleasemerge(void *addr)
{
	runtime·racereleasemergeg(g, addr);
}

void
runtime·racereleasemergeg(G *gp, void *addr)
{
	if(g->raceignore || !isvalidaddr((uintptr)addr))
		return;
	runtime·racecall(__tsan_release_merge, gp->racectx, addr);
}

void
runtime·racefingo(void)
{
	runtime·racecall(__tsan_finalizer_goroutine, g->racectx);
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
void
runtime·RaceSemacquire(uint32 *s)
{
	runtime·semacquire(s, false);
}

// func RaceSemrelease(s *uint32)
void
runtime·RaceSemrelease(uint32 *s)
{
	runtime·semrelease(s);
}

// func RaceDisable()
void
runtime·RaceDisable(void)
{
	g->raceignore++;
}

// func RaceEnable()
void
runtime·RaceEnable(void)
{
	g->raceignore--;
}

typedef struct SymbolizeContext SymbolizeContext;
struct SymbolizeContext
{
	uintptr	pc;
	int8*	func;
	int8*	file;
	uintptr	line;
	uintptr	off;
	uintptr	res;
};

// Callback from C into Go, runs on g0.
void
runtime·racesymbolize(SymbolizeContext *ctx)
{
	Func *f;
	String file;

	f = runtime·findfunc(ctx->pc);
	if(f == nil) {
		ctx->func = "??";
		ctx->file = "-";
		ctx->line = 0;
		ctx->off = ctx->pc;
		ctx->res = 1;
		return;
	}
	ctx->func = runtime·funcname(f);
	ctx->line = runtime·funcline(f, ctx->pc, &file);
	ctx->file = (int8*)file.str;  // assume zero-terminated
	ctx->off = ctx->pc - f->entry;
	ctx->res = 1;
}
