// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Implementation of the race detector API.
// +build race

package runtime

import "unsafe"

// Race runtime functions called via runtime·racecall.
//go:linkname __tsan_init __tsan_init
var __tsan_init byte

//go:linkname __tsan_fini __tsan_fini
var __tsan_fini byte

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

// Mimic what cmd/cgo would do.
//go:cgo_import_static __tsan_init
//go:cgo_import_static __tsan_fini
//go:cgo_import_static __tsan_map_shadow
//go:cgo_import_static __tsan_finalizer_goroutine
//go:cgo_import_static __tsan_go_start
//go:cgo_import_static __tsan_go_end
//go:cgo_import_static __tsan_malloc
//go:cgo_import_static __tsan_acquire
//go:cgo_import_static __tsan_release
//go:cgo_import_static __tsan_release_merge
//go:cgo_import_static __tsan_go_ignore_sync_begin
//go:cgo_import_static __tsan_go_ignore_sync_end

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
func racesymbolizethunk(uintptr)

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
func raceinit() uintptr {
	// cgo is required to initialize libc, which is used by race runtime
	if !iscgo {
		throw("raceinit: race build must use cgo")
	}

	var racectx uintptr
	racecall(&__tsan_init, uintptr(unsafe.Pointer(&racectx)), funcPC(racesymbolizethunk), 0, 0)

	// Round data segment to page boundaries, because it's used in mmap().
	start := ^uintptr(0)
	end := uintptr(0)
	if start > uintptr(unsafe.Pointer(&noptrdata)) {
		start = uintptr(unsafe.Pointer(&noptrdata))
	}
	if start > uintptr(unsafe.Pointer(&data)) {
		start = uintptr(unsafe.Pointer(&data))
	}
	if start > uintptr(unsafe.Pointer(&noptrbss)) {
		start = uintptr(unsafe.Pointer(&noptrbss))
	}
	if start > uintptr(unsafe.Pointer(&bss)) {
		start = uintptr(unsafe.Pointer(&bss))
	}
	if end < uintptr(unsafe.Pointer(&enoptrdata)) {
		end = uintptr(unsafe.Pointer(&enoptrdata))
	}
	if end < uintptr(unsafe.Pointer(&edata)) {
		end = uintptr(unsafe.Pointer(&edata))
	}
	if end < uintptr(unsafe.Pointer(&enoptrbss)) {
		end = uintptr(unsafe.Pointer(&enoptrbss))
	}
	if end < uintptr(unsafe.Pointer(&ebss)) {
		end = uintptr(unsafe.Pointer(&ebss))
	}
	size := round(end-start, _PageSize)
	racecall(&__tsan_map_shadow, start, size, 0, 0)
	racedatastart = start
	racedataend = start + size

	return racectx
}

//go:nosplit
func racefini() {
	racecall(&__tsan_fini, 0, 0, 0, 0)
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
	racecall(&__tsan_malloc, uintptr(p), sz, 0, 0)
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
func racewriteobjectpc(addr unsafe.Pointer, t *_type, callpc, pc uintptr) {
	kind := t.kind & _KindMask
	if kind == _KindArray || kind == _KindStruct {
		racewriterangepc(addr, t.size, callpc, pc)
	} else {
		racewritepc(addr, callpc, pc)
	}
}

//go:nosplit
func racereadobjectpc(addr unsafe.Pointer, t *_type, callpc, pc uintptr) {
	kind := t.kind & _KindMask
	if kind == _KindArray || kind == _KindStruct {
		racereadrangepc(addr, t.size, callpc, pc)
	} else {
		racereadpc(addr, callpc, pc)
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
	_g_ := getg()
	if _g_.raceignore != 0 || !isvalidaddr(addr) {
		return
	}
	racereleaseg(_g_, addr)
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

// RaceEnable re-enables handling of race events in the current goroutine.
func RaceDisable() {
	_g_ := getg()
	if _g_.raceignore == 0 {
		racecall(&__tsan_go_ignore_sync_begin, _g_.racectx, 0, 0, 0)
	}
	_g_.raceignore++
}

//go:nosplit

// RaceDisable disables handling of race events in the current goroutine.
func RaceEnable() {
	_g_ := getg()
	_g_.raceignore--
	if _g_.raceignore == 0 {
		racecall(&__tsan_go_ignore_sync_end, _g_.racectx, 0, 0, 0)
	}
}
