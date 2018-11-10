// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race

// Public race detection API, present iff build with -race.

package runtime

import (
	"unsafe"
)

func RaceRead(addr unsafe.Pointer)
func RaceWrite(addr unsafe.Pointer)
func RaceReadRange(addr unsafe.Pointer, len int)
func RaceWriteRange(addr unsafe.Pointer, len int)

func RaceErrors() int {
	var n uint64
	racecall(&__tsan_report_count, uintptr(unsafe.Pointer(&n)), 0, 0, 0)
	return int(n)
}

// private interface for the runtime
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
	f := FuncForPC(ctx.pc)
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
	if _, x, n := findObject(unsafe.Pointer(ctx.addr)); x != nil {
		ctx.heap = 1
		ctx.start = uintptr(x)
		ctx.size = n
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

func racefuncenter(uintptr)
func racefuncexit()
func racereadrangepc1(uintptr, uintptr, uintptr)
func racewriterangepc1(uintptr, uintptr, uintptr)
func racecallbackthunk(uintptr)

// racecall allows calling an arbitrary function f from C race runtime
// with up to 4 uintptr arguments.
func racecall(*byte, uintptr, uintptr, uintptr, uintptr)

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

//go:nosplit

func RaceAcquire(addr unsafe.Pointer) {
	raceacquire(addr)
}

//go:nosplit

func RaceRelease(addr unsafe.Pointer) {
	racerelease(addr)
}

//go:nosplit

func RaceReleaseMerge(addr unsafe.Pointer) {
	racereleasemerge(addr)
}

//go:nosplit

// RaceDisable disables handling of race events in the current goroutine.
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
