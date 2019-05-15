// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race

package runtime

import (
	"unsafe"
)

// Public race detection API, present iff build with -race.

func RaceRead(addr unsafe.Pointer)
func RaceWrite(addr unsafe.Pointer)
func RaceReadRange(addr unsafe.Pointer, len int)
func RaceWriteRange(addr unsafe.Pointer, len int)

func RaceErrors() int {
	var n uint64
	racecall(&__tsan_report_count, uintptr(unsafe.Pointer(&n)), 0, 0, 0)
	return int(n)
}

//go:nosplit

// RaceAcquire/RaceRelease/RaceReleaseMerge establish happens-before relations
// between goroutines. These inform the race detector about actual synchronization
// that it can't see for some reason (e.g. synchronization within RaceDisable/RaceEnable
// sections of code).
// RaceAcquire establishes a happens-before relation with the preceding
// RaceReleaseMerge on addr up to and including the last RaceRelease on addr.
// In terms of the C memory model (C11 §5.1.2.4, §7.17.3),
// RaceAcquire is equivalent to atomic_load(memory_order_acquire).
func RaceAcquire(addr unsafe.Pointer) {
	raceacquire(addr)
}

//go:nosplit

// RaceRelease performs a release operation on addr that
// can synchronize with a later RaceAcquire on addr.
//
// In terms of the C memory model, RaceRelease is equivalent to
// atomic_store(memory_order_release).
func RaceRelease(addr unsafe.Pointer) {
	racerelease(addr)
}

//go:nosplit

// RaceReleaseMerge is like RaceRelease, but also establishes a happens-before
// relation with the preceding RaceRelease or RaceReleaseMerge on addr.
//
// In terms of the C memory model, RaceReleaseMerge is equivalent to
// atomic_exchange(memory_order_release).
func RaceReleaseMerge(addr unsafe.Pointer) {
	racereleasemerge(addr)
}

//go:nosplit

// RaceDisable disables handling of race synchronization events in the current goroutine.
// Handling is re-enabled with RaceEnable. RaceDisable/RaceEnable can be nested.
// Non-synchronization events (memory accesses, function entry/exit) still affect
// the race detector.
func RaceDisable() {
	_g_ := getg()
	if _g_.raceignore == 0 {
		racecall(&__tsan_go_ignore_sync_begin, _g_.racectx, 0, 0, 0)
	}
	_g_.raceignore++
}

//go:nosplit

// RaceEnable re-enables handling of race events in the current goroutine.
func RaceEnable() {
	_g_ := getg()
	_g_.raceignore--
	if _g_.raceignore == 0 {
		racecall(&__tsan_go_ignore_sync_end, _g_.racectx, 0, 0, 0)
	}
}

// Private interface for the runtime.

const raceenabled = true

// For all functions accepting callerpc and pc,
// callerpc is a return PC of the function that calls this function,
// pc is start PC of the function that calls this function.
func raceReadObjectPC(t *_type, addr unsafe.Pointer, callerpc, pc uintptr) {
	kind := t.kind & kindMask
	if kind == kindArray || kind == kindStruct {
		// for composite objects we have to read every address
		// because a write might happen to any subobject.
		racereadrangepc(addr, t.size, callerpc, pc)
	} else {
		// for non-composite objects we can read just the start
		// address, as any write must write the first byte.
		racereadpc(addr, callerpc, pc)
	}
}

func raceWriteObjectPC(t *_type, addr unsafe.Pointer, callerpc, pc uintptr) {
	kind := t.kind & kindMask
	if kind == kindArray || kind == kindStruct {
		// for composite objects we have to write every address
		// because a write might happen to any subobject.
		racewriterangepc(addr, t.size, callerpc, pc)
	} else {
		// for non-composite objects we can write just the start
		// address, as any write must write the first byte.
		racewritepc(addr, callerpc, pc)
	}
}

//go:noescape
func racereadpc(addr unsafe.Pointer, callpc, pc uintptr)

//go:noescape
func racewritepc(addr unsafe.Pointer, callpc, pc uintptr)

type symbolizeCodeContext struct {
	pc   uintptr
	fn   *byte
	file *byte
	line uintptr
	off  uintptr
	res  uintptr
}

var qq = [...]byte{'?', '?', 0}
var dash = [...]byte{'-', 0}

const (
	raceGetProcCmd = iota
	raceSymbolizeCodeCmd
	raceSymbolizeDataCmd
)

// Callback from C into Go, runs on g0.
func racecallback(cmd uintptr, ctx unsafe.Pointer) {
	switch cmd {
	case raceGetProcCmd:
		throw("should have been handled by racecallbackthunk")
	case raceSymbolizeCodeCmd:
		raceSymbolizeCode((*symbolizeCodeContext)(ctx))
	case raceSymbolizeDataCmd:
		raceSymbolizeData((*symbolizeDataContext)(ctx))
	default:
		throw("unknown command")
	}
}

func raceSymbolizeCode(ctx *symbolizeCodeContext) {
	f := findfunc(ctx.pc)._Func()
	if f != nil {
		file, line := f.FileLine(ctx.pc)
		if line != 0 {
			ctx.fn = cfuncname(f.funcInfo())
			ctx.line = uintptr(line)
			ctx.file = &bytes(file)[0] // assume NUL-terminated
			ctx.off = ctx.pc - f.Entry()
			ctx.res = 1
			return
		}
	}
	ctx.fn = &qq[0]
	ctx.file = &dash[0]
	ctx.line = 0
	ctx.off = ctx.pc
	ctx.res = 1
}

type symbolizeDataContext struct {
	addr  uintptr
	heap  uintptr
	start uintptr
	size  uintptr
	name  *byte
	file  *byte
	line  uintptr
	res   uintptr
}

func raceSymbolizeData(ctx *symbolizeDataContext) {
	if base, span, _ := findObject(ctx.addr, 0, 0); base != 0 {
		ctx.heap = 1
		ctx.start = base
		ctx.size = span.elemsize
		ctx.res = 1
	}
}

// Race runtime functions called via runtime·racecall.
//go:linkname __tsan_init __tsan_init
var __tsan_init byte

//go:linkname __tsan_fini __tsan_fini
var __tsan_fini byte

//go:linkname __tsan_proc_create __tsan_proc_create
var __tsan_proc_create byte

//go:linkname __tsan_proc_destroy __tsan_proc_destroy
var __tsan_proc_destroy byte

//go:linkname __tsan_map_shadow __tsan_map_shadow
var __tsan_map_shadow byte

//go:linkname __tsan_finalizer_goroutine __tsan_finalizer_goroutine
var __tsan_finalizer_goroutine byte

//go:linkname __tsan_go_start __tsan_go_start
var __tsan_go_start byte

//go:linkname __tsan_go_end __tsan_go_end
var __tsan_go_end byte

//go:linkname __tsan_malloc __tsan_malloc
var __tsan_malloc byte

//go:linkname __tsan_free __tsan_free
var __tsan_free byte

//go:linkname __tsan_acquire __tsan_acquire
var __tsan_acquire byte

//go:linkname __tsan_release __tsan_release
var __tsan_release byte

//go:linkname __tsan_release_merge __tsan_release_merge
var __tsan_release_merge byte

//go:linkname __tsan_go_ignore_sync_begin __tsan_go_ignore_sync_begin
var __tsan_go_ignore_sync_begin byte

//go:linkname __tsan_go_ignore_sync_end __tsan_go_ignore_sync_end
var __tsan_go_ignore_sync_end byte

//go:linkname __tsan_report_count __tsan_report_count
var __tsan_report_count byte

// Mimic what cmd/cgo would do.
//go:cgo_import_static __tsan_init
//go:cgo_import_static __tsan_fini
//go:cgo_import_static __tsan_proc_create
//go:cgo_import_static __tsan_proc_destroy
//go:cgo_import_static __tsan_map_shadow
//go:cgo_import_static __tsan_finalizer_goroutine
//go:cgo_import_static __tsan_go_start
//go:cgo_import_static __tsan_go_end
//go:cgo_import_static __tsan_malloc
//go:cgo_import_static __tsan_free
//go:cgo_import_static __tsan_acquire
//go:cgo_import_static __tsan_release
//go:cgo_import_static __tsan_release_merge
//go:cgo_import_static __tsan_go_ignore_sync_begin
//go:cgo_import_static __tsan_go_ignore_sync_end
//go:cgo_import_static __tsan_report_count

// These are called from race_amd64.s.
//go:cgo_import_static __tsan_read
//go:cgo_import_static __tsan_read_pc
//go:cgo_import_static __tsan_read_range
//go:cgo_import_static __tsan_write
//go:cgo_import_static __tsan_write_pc
//go:cgo_import_static __tsan_write_range
//go:cgo_import_static __tsan_func_enter
//go:cgo_import_static __tsan_func_exit

//go:cgo_import_static __tsan_go_atomic32_load
//go:cgo_import_static __tsan_go_atomic64_load
//go:cgo_import_static __tsan_go_atomic32_store
//go:cgo_import_static __tsan_go_atomic64_store
//go:cgo_import_static __tsan_go_atomic32_exchange
//go:cgo_import_static __tsan_go_atomic64_exchange
//go:cgo_import_static __tsan_go_atomic32_fetch_add
//go:cgo_import_static __tsan_go_atomic64_fetch_add
//go:cgo_import_static __tsan_go_atomic32_compare_exchange
//go:cgo_import_static __tsan_go_atomic64_compare_exchange

// start/end of global data (data+bss).
var racedatastart uintptr
var racedataend uintptr

// start/end of heap for race_amd64.s
var racearenastart uintptr
var racearenaend uintptr

func racefuncenter(callpc uintptr)
func racefuncenterfp(fp uintptr)
func racefuncexit()
func raceread(addr uintptr)
func racewrite(addr uintptr)
func racereadrange(addr, size uintptr)
func racewriterange(addr, size uintptr)
func racereadrangepc1(addr, size, pc uintptr)
func racewriterangepc1(addr, size, pc uintptr)
func racecallbackthunk(uintptr)

// racecall allows calling an arbitrary function f from C race runtime
// with up to 4 uintptr arguments.
func racecall(fn *byte, arg0, arg1, arg2, arg3 uintptr)

// checks if the address has shadow (i.e. heap or data/bss)
//go:nosplit
func isvalidaddr(addr unsafe.Pointer) bool {
	return racearenastart <= uintptr(addr) && uintptr(addr) < racearenaend ||
		racedatastart <= uintptr(addr) && uintptr(addr) < racedataend
}

//go:nosplit
func raceinit() (gctx, pctx uintptr) {
	// cgo is required to initialize libc, which is used by race runtime
	if !iscgo {
		throw("raceinit: race build must use cgo")
	}

	racecall(&__tsan_init, uintptr(unsafe.Pointer(&gctx)), uintptr(unsafe.Pointer(&pctx)), funcPC(racecallbackthunk), 0)

	// Round data segment to page boundaries, because it's used in mmap().
	start := ^uintptr(0)
	end := uintptr(0)
	if start > firstmoduledata.noptrdata {
		start = firstmoduledata.noptrdata
	}
	if start > firstmoduledata.data {
		start = firstmoduledata.data
	}
	if start > firstmoduledata.noptrbss {
		start = firstmoduledata.noptrbss
	}
	if start > firstmoduledata.bss {
		start = firstmoduledata.bss
	}
	if end < firstmoduledata.enoptrdata {
		end = firstmoduledata.enoptrdata
	}
	if end < firstmoduledata.edata {
		end = firstmoduledata.edata
	}
	if end < firstmoduledata.enoptrbss {
		end = firstmoduledata.enoptrbss
	}
	if end < firstmoduledata.ebss {
		end = firstmoduledata.ebss
	}
	size := round(end-start, _PageSize)
	racecall(&__tsan_map_shadow, start, size, 0, 0)
	racedatastart = start
	racedataend = start + size

	return
}

var raceFiniLock mutex

//go:nosplit
func racefini() {
	// racefini() can only be called once to avoid races.
	// This eventually (via __tsan_fini) calls C.exit which has
	// undefined behavior if called more than once. If the lock is
	// already held it's assumed that the first caller exits the program
	// so other calls can hang forever without an issue.
	lock(&raceFiniLock)
	racecall(&__tsan_fini, 0, 0, 0, 0)
}

//go:nosplit
func raceproccreate() uintptr {
	var ctx uintptr
	racecall(&__tsan_proc_create, uintptr(unsafe.Pointer(&ctx)), 0, 0, 0)
	return ctx
}

//go:nosplit
func raceprocdestroy(ctx uintptr) {
	racecall(&__tsan_proc_destroy, ctx, 0, 0, 0)
}

//go:nosplit
func racemapshadow(addr unsafe.Pointer, size uintptr) {
	if racearenastart == 0 {
		racearenastart = uintptr(addr)
	}
	if racearenaend < uintptr(addr)+size {
		racearenaend = uintptr(addr) + size
	}
	racecall(&__tsan_map_shadow, uintptr(addr), size, 0, 0)
}

//go:nosplit
func racemalloc(p unsafe.Pointer, sz uintptr) {
	racecall(&__tsan_malloc, 0, 0, uintptr(p), sz)
}

//go:nosplit
func racefree(p unsafe.Pointer, sz uintptr) {
	racecall(&__tsan_free, uintptr(p), sz, 0, 0)
}

//go:nosplit
func racegostart(pc uintptr) uintptr {
	_g_ := getg()
	var spawng *g
	if _g_.m.curg != nil {
		spawng = _g_.m.curg
	} else {
		spawng = _g_
	}

	var racectx uintptr
	racecall(&__tsan_go_start, spawng.racectx, uintptr(unsafe.Pointer(&racectx)), pc, 0)
	return racectx
}

//go:nosplit
func racegoend() {
	racecall(&__tsan_go_end, getg().racectx, 0, 0, 0)
}

//go:nosplit
func racewriterangepc(addr unsafe.Pointer, sz, callpc, pc uintptr) {
	_g_ := getg()
	if _g_ != _g_.m.curg {
		// The call is coming from manual instrumentation of Go code running on g0/gsignal.
		// Not interesting.
		return
	}
	if callpc != 0 {
		racefuncenter(callpc)
	}
	racewriterangepc1(uintptr(addr), sz, pc)
	if callpc != 0 {
		racefuncexit()
	}
}

//go:nosplit
func racereadrangepc(addr unsafe.Pointer, sz, callpc, pc uintptr) {
	_g_ := getg()
	if _g_ != _g_.m.curg {
		// The call is coming from manual instrumentation of Go code running on g0/gsignal.
		// Not interesting.
		return
	}
	if callpc != 0 {
		racefuncenter(callpc)
	}
	racereadrangepc1(uintptr(addr), sz, pc)
	if callpc != 0 {
		racefuncexit()
	}
}

//go:nosplit
func raceacquire(addr unsafe.Pointer) {
	raceacquireg(getg(), addr)
}

//go:nosplit
func raceacquireg(gp *g, addr unsafe.Pointer) {
	if getg().raceignore != 0 || !isvalidaddr(addr) {
		return
	}
	racecall(&__tsan_acquire, gp.racectx, uintptr(addr), 0, 0)
}

//go:nosplit
func racerelease(addr unsafe.Pointer) {
	racereleaseg(getg(), addr)
}

//go:nosplit
func racereleaseg(gp *g, addr unsafe.Pointer) {
	if getg().raceignore != 0 || !isvalidaddr(addr) {
		return
	}
	racecall(&__tsan_release, gp.racectx, uintptr(addr), 0, 0)
}

//go:nosplit
func racereleasemerge(addr unsafe.Pointer) {
	racereleasemergeg(getg(), addr)
}

//go:nosplit
func racereleasemergeg(gp *g, addr unsafe.Pointer) {
	if getg().raceignore != 0 || !isvalidaddr(addr) {
		return
	}
	racecall(&__tsan_release_merge, gp.racectx, uintptr(addr), 0, 0)
}

//go:nosplit
func racefingo() {
	racecall(&__tsan_finalizer_goroutine, getg().racectx, 0, 0, 0)
}

// The declarations below generate ABI wrappers for functions
// implemented in assembly in this package but declared in another
// package.

//go:linkname abigen_sync_atomic_LoadInt32 sync/atomic.LoadInt32
func abigen_sync_atomic_LoadInt32(addr *int32) (val int32)

//go:linkname abigen_sync_atomic_LoadInt64 sync/atomic.LoadInt64
func abigen_sync_atomic_LoadInt64(addr *int64) (val int64)

//go:linkname abigen_sync_atomic_LoadUint32 sync/atomic.LoadUint32
func abigen_sync_atomic_LoadUint32(addr *uint32) (val uint32)

//go:linkname abigen_sync_atomic_LoadUint64 sync/atomic.LoadUint64
func abigen_sync_atomic_LoadUint64(addr *uint64) (val uint64)

//go:linkname abigen_sync_atomic_LoadUintptr sync/atomic.LoadUintptr
func abigen_sync_atomic_LoadUintptr(addr *uintptr) (val uintptr)

//go:linkname abigen_sync_atomic_LoadPointer sync/atomic.LoadPointer
func abigen_sync_atomic_LoadPointer(addr *unsafe.Pointer) (val unsafe.Pointer)

//go:linkname abigen_sync_atomic_StoreInt32 sync/atomic.StoreInt32
func abigen_sync_atomic_StoreInt32(addr *int32, val int32)

//go:linkname abigen_sync_atomic_StoreInt64 sync/atomic.StoreInt64
func abigen_sync_atomic_StoreInt64(addr *int64, val int64)

//go:linkname abigen_sync_atomic_StoreUint32 sync/atomic.StoreUint32
func abigen_sync_atomic_StoreUint32(addr *uint32, val uint32)

//go:linkname abigen_sync_atomic_StoreUint64 sync/atomic.StoreUint64
func abigen_sync_atomic_StoreUint64(addr *uint64, val uint64)

//go:linkname abigen_sync_atomic_SwapInt32 sync/atomic.SwapInt32
func abigen_sync_atomic_SwapInt32(addr *int32, new int32) (old int32)

//go:linkname abigen_sync_atomic_SwapInt64 sync/atomic.SwapInt64
func abigen_sync_atomic_SwapInt64(addr *int64, new int64) (old int64)

//go:linkname abigen_sync_atomic_SwapUint32 sync/atomic.SwapUint32
func abigen_sync_atomic_SwapUint32(addr *uint32, new uint32) (old uint32)

//go:linkname abigen_sync_atomic_SwapUint64 sync/atomic.SwapUint64
func abigen_sync_atomic_SwapUint64(addr *uint64, new uint64) (old uint64)

//go:linkname abigen_sync_atomic_AddInt32 sync/atomic.AddInt32
func abigen_sync_atomic_AddInt32(addr *int32, delta int32) (new int32)

//go:linkname abigen_sync_atomic_AddUint32 sync/atomic.AddUint32
func abigen_sync_atomic_AddUint32(addr *uint32, delta uint32) (new uint32)

//go:linkname abigen_sync_atomic_AddInt64 sync/atomic.AddInt64
func abigen_sync_atomic_AddInt64(addr *int64, delta int64) (new int64)

//go:linkname abigen_sync_atomic_AddUint64 sync/atomic.AddUint64
func abigen_sync_atomic_AddUint64(addr *uint64, delta uint64) (new uint64)

//go:linkname abigen_sync_atomic_AddUintptr sync/atomic.AddUintptr
func abigen_sync_atomic_AddUintptr(addr *uintptr, delta uintptr) (new uintptr)

//go:linkname abigen_sync_atomic_CompareAndSwapInt32 sync/atomic.CompareAndSwapInt32
func abigen_sync_atomic_CompareAndSwapInt32(addr *int32, old, new int32) (swapped bool)

//go:linkname abigen_sync_atomic_CompareAndSwapInt64 sync/atomic.CompareAndSwapInt64
func abigen_sync_atomic_CompareAndSwapInt64(addr *int64, old, new int64) (swapped bool)

//go:linkname abigen_sync_atomic_CompareAndSwapUint32 sync/atomic.CompareAndSwapUint32
func abigen_sync_atomic_CompareAndSwapUint32(addr *uint32, old, new uint32) (swapped bool)

//go:linkname abigen_sync_atomic_CompareAndSwapUint64 sync/atomic.CompareAndSwapUint64
func abigen_sync_atomic_CompareAndSwapUint64(addr *uint64, old, new uint64) (swapped bool)
