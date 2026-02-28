// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build race

package runtime

import (
	"internal/abi"
	"unsafe"
)

// Public race detection API, present iff build with -race.

func RaceRead(addr unsafe.Pointer)

//go:linkname race_Read internal/race.Read
//go:nosplit
func race_Read(addr unsafe.Pointer) {
	RaceRead(addr)
}

func RaceWrite(addr unsafe.Pointer)

//go:linkname race_Write internal/race.Write
//go:nosplit
func race_Write(addr unsafe.Pointer) {
	RaceWrite(addr)
}

func RaceReadRange(addr unsafe.Pointer, len int)

//go:linkname race_ReadRange internal/race.ReadRange
//go:nosplit
func race_ReadRange(addr unsafe.Pointer, len int) {
	RaceReadRange(addr, len)
}

func RaceWriteRange(addr unsafe.Pointer, len int)

//go:linkname race_WriteRange internal/race.WriteRange
//go:nosplit
func race_WriteRange(addr unsafe.Pointer, len int) {
	RaceWriteRange(addr, len)
}

func RaceErrors() int {
	var n uint64
	racecall(&__tsan_report_count, uintptr(unsafe.Pointer(&n)), 0, 0, 0)
	return int(n)
}

//go:linkname race_Errors internal/race.Errors
//go:nosplit
func race_Errors() int {
	return RaceErrors()
}

// RaceAcquire/RaceRelease/RaceReleaseMerge establish happens-before relations
// between goroutines. These inform the race detector about actual synchronization
// that it can't see for some reason (e.g. synchronization within RaceDisable/RaceEnable
// sections of code).
// RaceAcquire establishes a happens-before relation with the preceding
// RaceReleaseMerge on addr up to and including the last RaceRelease on addr.
// In terms of the C memory model (C11 §5.1.2.4, §7.17.3),
// RaceAcquire is equivalent to atomic_load(memory_order_acquire).
//
//go:nosplit
func RaceAcquire(addr unsafe.Pointer) {
	raceacquire(addr)
}

//go:linkname race_Acquire internal/race.Acquire
//go:nosplit
func race_Acquire(addr unsafe.Pointer) {
	RaceAcquire(addr)
}

// RaceRelease performs a release operation on addr that
// can synchronize with a later RaceAcquire on addr.
//
// In terms of the C memory model, RaceRelease is equivalent to
// atomic_store(memory_order_release).
//
//go:nosplit
func RaceRelease(addr unsafe.Pointer) {
	racerelease(addr)
}

//go:linkname race_Release internal/race.Release
//go:nosplit
func race_Release(addr unsafe.Pointer) {
	RaceRelease(addr)
}

// RaceReleaseMerge is like RaceRelease, but also establishes a happens-before
// relation with the preceding RaceRelease or RaceReleaseMerge on addr.
//
// In terms of the C memory model, RaceReleaseMerge is equivalent to
// atomic_exchange(memory_order_release).
//
//go:nosplit
func RaceReleaseMerge(addr unsafe.Pointer) {
	racereleasemerge(addr)
}

//go:linkname race_ReleaseMerge internal/race.ReleaseMerge
//go:nosplit
func race_ReleaseMerge(addr unsafe.Pointer) {
	RaceReleaseMerge(addr)
}

// RaceDisable disables handling of race synchronization events in the current goroutine.
// Handling is re-enabled with RaceEnable. RaceDisable/RaceEnable can be nested.
// Non-synchronization events (memory accesses, function entry/exit) still affect
// the race detector.
//
//go:nosplit
func RaceDisable() {
	gp := getg()
	if gp.raceignore == 0 {
		racecall(&__tsan_go_ignore_sync_begin, gp.racectx, 0, 0, 0)
	}
	gp.raceignore++
}

//go:linkname race_Disable internal/race.Disable
//go:nosplit
func race_Disable() {
	RaceDisable()
}

// RaceEnable re-enables handling of race events in the current goroutine.
//
//go:nosplit
func RaceEnable() {
	gp := getg()
	gp.raceignore--
	if gp.raceignore == 0 {
		racecall(&__tsan_go_ignore_sync_end, gp.racectx, 0, 0, 0)
	}
}

//go:linkname race_Enable internal/race.Enable
//go:nosplit
func race_Enable() {
	RaceEnable()
}

// Private interface for the runtime.

const raceenabled = true

// For all functions accepting callerpc and pc,
// callerpc is a return PC of the function that calls this function,
// pc is start PC of the function that calls this function.
func raceReadObjectPC(t *_type, addr unsafe.Pointer, callerpc, pc uintptr) {
	kind := t.Kind()
	if kind == abi.Array || kind == abi.Struct {
		// for composite objects we have to read every address
		// because a write might happen to any subobject.
		racereadrangepc(addr, t.Size_, callerpc, pc)
	} else {
		// for non-composite objects we can read just the start
		// address, as any write must write the first byte.
		racereadpc(addr, callerpc, pc)
	}
}

//go:linkname race_ReadObjectPC internal/race.ReadObjectPC
func race_ReadObjectPC(t *abi.Type, addr unsafe.Pointer, callerpc, pc uintptr) {
	raceReadObjectPC(t, addr, callerpc, pc)
}

func raceWriteObjectPC(t *_type, addr unsafe.Pointer, callerpc, pc uintptr) {
	kind := t.Kind()
	if kind == abi.Array || kind == abi.Struct {
		// for composite objects we have to write every address
		// because a write might happen to any subobject.
		racewriterangepc(addr, t.Size_, callerpc, pc)
	} else {
		// for non-composite objects we can write just the start
		// address, as any write must write the first byte.
		racewritepc(addr, callerpc, pc)
	}
}

//go:linkname race_WriteObjectPC internal/race.WriteObjectPC
func race_WriteObjectPC(t *abi.Type, addr unsafe.Pointer, callerpc, pc uintptr) {
	raceWriteObjectPC(t, addr, callerpc, pc)
}

//go:noescape
func racereadpc(addr unsafe.Pointer, callpc, pc uintptr)

//go:noescape
func racewritepc(addr unsafe.Pointer, callpc, pc uintptr)

//go:linkname race_ReadPC internal/race.ReadPC
func race_ReadPC(addr unsafe.Pointer, callerpc, pc uintptr) {
	racereadpc(addr, callerpc, pc)
}

//go:linkname race_WritePC internal/race.WritePC
func race_WritePC(addr unsafe.Pointer, callerpc, pc uintptr) {
	racewritepc(addr, callerpc, pc)
}

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

// raceSymbolizeCode reads ctx.pc and populates the rest of *ctx with
// information about the code at that pc.
//
// The race detector has already subtracted 1 from pcs, so they point to the last
// byte of call instructions (including calls to runtime.racewrite and friends).
//
// If the incoming pc is part of an inlined function, *ctx is populated
// with information about the inlined function, and on return ctx.pc is set
// to a pc in the logically containing function. (The race detector should call this
// function again with that pc.)
//
// If the incoming pc is not part of an inlined function, the return pc is unchanged.
func raceSymbolizeCode(ctx *symbolizeCodeContext) {
	pc := ctx.pc
	fi := findfunc(pc)
	if fi.valid() {
		u, uf := newInlineUnwinder(fi, pc)
		for ; uf.valid(); uf = u.next(uf) {
			sf := u.srcFunc(uf)
			if sf.funcID == abi.FuncIDWrapper && u.isInlined(uf) {
				// Ignore wrappers, unless we're at the outermost frame of u.
				// A non-inlined wrapper frame always means we have a physical
				// frame consisting entirely of wrappers, in which case we'll
				// take an outermost wrapper over nothing.
				continue
			}

			name := sf.name()
			file, line := u.fileLine(uf)
			if line == 0 {
				// Failure to symbolize
				continue
			}
			ctx.fn = &bytes(name)[0] // assume NUL-terminated
			ctx.line = uintptr(line)
			ctx.file = &bytes(file)[0] // assume NUL-terminated
			ctx.off = pc - fi.entry()
			ctx.res = 1
			if u.isInlined(uf) {
				// Set ctx.pc to the "caller" so the race detector calls this again
				// to further unwind.
				uf = u.next(uf)
				ctx.pc = uf.pc
			}
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
		// TODO: Does this need to handle malloc headers?
		ctx.heap = 1
		ctx.start = base
		ctx.size = span.elemsize
		ctx.res = 1
	}
}

// Race runtime functions called via runtime·racecall.
//
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

//go:linkname __tsan_release_acquire __tsan_release_acquire
var __tsan_release_acquire byte

//go:linkname __tsan_release_merge __tsan_release_merge
var __tsan_release_merge byte

//go:linkname __tsan_go_ignore_sync_begin __tsan_go_ignore_sync_begin
var __tsan_go_ignore_sync_begin byte

//go:linkname __tsan_go_ignore_sync_end __tsan_go_ignore_sync_end
var __tsan_go_ignore_sync_end byte

//go:linkname __tsan_report_count __tsan_report_count
var __tsan_report_count byte

// Mimic what cmd/cgo would do.
//
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
//go:cgo_import_static __tsan_release_acquire
//go:cgo_import_static __tsan_release_merge
//go:cgo_import_static __tsan_go_ignore_sync_begin
//go:cgo_import_static __tsan_go_ignore_sync_end
//go:cgo_import_static __tsan_report_count

// These are called from race_amd64.s.
//
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
//go:cgo_import_static __tsan_go_atomic32_fetch_and
//go:cgo_import_static __tsan_go_atomic64_fetch_and
//go:cgo_import_static __tsan_go_atomic32_fetch_or
//go:cgo_import_static __tsan_go_atomic64_fetch_or
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

// racecall allows calling an arbitrary function fn from C race runtime
// with up to 4 uintptr arguments.
func racecall(fn *byte, arg0, arg1, arg2, arg3 uintptr)

// checks if the address has shadow (i.e. heap or data/bss).
//
//go:nosplit
func isvalidaddr(addr unsafe.Pointer) bool {
	return racearenastart <= uintptr(addr) && uintptr(addr) < racearenaend ||
		racedatastart <= uintptr(addr) && uintptr(addr) < racedataend
}

//go:nosplit
func raceinit() (gctx, pctx uintptr) {
	lockInit(&raceFiniLock, lockRankRaceFini)

	// On most machines, cgo is required to initialize libc, which is used by race runtime.
	if !iscgo && GOOS != "darwin" {
		throw("raceinit: race build must use cgo")
	}

	racecall(&__tsan_init, uintptr(unsafe.Pointer(&gctx)), uintptr(unsafe.Pointer(&pctx)), abi.FuncPCABI0(racecallbackthunk), 0)

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
	// Use exact bounds for boundary check in racecalladdr. See issue 73483.
	racedatastart = start
	racedataend = end
	// Round data segment to page boundaries for race detector (TODO: still needed?)
	start = alignDown(start, _PageSize)
	end = alignUp(end, _PageSize)
	racecall(&__tsan_map_shadow, start, end-start, 0, 0)

	return
}

//go:nosplit
func racefini() {
	// racefini() can only be called once to avoid races.
	// This eventually (via __tsan_fini) calls C.exit which has
	// undefined behavior if called more than once. If the lock is
	// already held it's assumed that the first caller exits the program
	// so other calls can hang forever without an issue.
	lock(&raceFiniLock)

	// __tsan_fini will run C atexit functions and C++ destructors,
	// which can theoretically call back into Go.
	// Tell the scheduler we entering external code.
	entersyscall()

	// We're entering external code that may call ExitProcess on
	// Windows.
	osPreemptExtEnter(getg().m)

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
	gp := getg()
	var spawng *g
	if gp.m.curg != nil {
		spawng = gp.m.curg
	} else {
		spawng = gp
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
func racectxstart(pc, spawnctx uintptr) uintptr {
	var racectx uintptr
	racecall(&__tsan_go_start, spawnctx, uintptr(unsafe.Pointer(&racectx)), pc, 0)
	return racectx
}

//go:nosplit
func racectxend(racectx uintptr) {
	racecall(&__tsan_go_end, racectx, 0, 0, 0)
}

//go:nosplit
func racewriterangepc(addr unsafe.Pointer, sz, callpc, pc uintptr) {
	gp := getg()
	if gp != gp.m.curg {
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
	gp := getg()
	if gp != gp.m.curg {
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
func raceacquirectx(racectx uintptr, addr unsafe.Pointer) {
	if !isvalidaddr(addr) {
		return
	}
	racecall(&__tsan_acquire, racectx, uintptr(addr), 0, 0)
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
func racereleaseacquire(addr unsafe.Pointer) {
	racereleaseacquireg(getg(), addr)
}

//go:nosplit
func racereleaseacquireg(gp *g, addr unsafe.Pointer) {
	if getg().raceignore != 0 || !isvalidaddr(addr) {
		return
	}
	racecall(&__tsan_release_acquire, gp.racectx, uintptr(addr), 0, 0)
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

//go:linkname abigen_sync_atomic_AndInt32 sync/atomic.AndInt32
func abigen_sync_atomic_AndInt32(addr *int32, mask int32) (old int32)

//go:linkname abigen_sync_atomic_AndUint32 sync/atomic.AndUint32
func abigen_sync_atomic_AndUint32(addr *uint32, mask uint32) (old uint32)

//go:linkname abigen_sync_atomic_AndInt64 sync/atomic.AndInt64
func abigen_sync_atomic_AndInt64(addr *int64, mask int64) (old int64)

//go:linkname abigen_sync_atomic_AndUint64 sync/atomic.AndUint64
func abigen_sync_atomic_AndUint64(addr *uint64, mask uint64) (old uint64)

//go:linkname abigen_sync_atomic_AndUintptr sync/atomic.AndUintptr
func abigen_sync_atomic_AndUintptr(addr *uintptr, mask uintptr) (old uintptr)

//go:linkname abigen_sync_atomic_OrInt32 sync/atomic.OrInt32
func abigen_sync_atomic_OrInt32(addr *int32, mask int32) (old int32)

//go:linkname abigen_sync_atomic_OrUint32 sync/atomic.OrUint32
func abigen_sync_atomic_OrUint32(addr *uint32, mask uint32) (old uint32)

//go:linkname abigen_sync_atomic_OrInt64 sync/atomic.OrInt64
func abigen_sync_atomic_OrInt64(addr *int64, mask int64) (old int64)

//go:linkname abigen_sync_atomic_OrUint64 sync/atomic.OrUint64
func abigen_sync_atomic_OrUint64(addr *uint64, mask uint64) (old uint64)

//go:linkname abigen_sync_atomic_OrUintptr sync/atomic.OrUintptr
func abigen_sync_atomic_OrUintptr(addr *uintptr, mask uintptr) (old uintptr)

//go:linkname abigen_sync_atomic_CompareAndSwapInt32 sync/atomic.CompareAndSwapInt32
func abigen_sync_atomic_CompareAndSwapInt32(addr *int32, old, new int32) (swapped bool)

//go:linkname abigen_sync_atomic_CompareAndSwapInt64 sync/atomic.CompareAndSwapInt64
func abigen_sync_atomic_CompareAndSwapInt64(addr *int64, old, new int64) (swapped bool)

//go:linkname abigen_sync_atomic_CompareAndSwapUint32 sync/atomic.CompareAndSwapUint32
func abigen_sync_atomic_CompareAndSwapUint32(addr *uint32, old, new uint32) (swapped bool)

//go:linkname abigen_sync_atomic_CompareAndSwapUint64 sync/atomic.CompareAndSwapUint64
func abigen_sync_atomic_CompareAndSwapUint64(addr *uint64, old, new uint64) (swapped bool)
