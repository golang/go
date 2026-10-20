// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasm

package runtime

import "unsafe"

// The constants of the split of the wasm linear memory under -gc=wasm
// (SPEC Sec 19, T-301.2) into a low permanent region and a high scratch
// region.
//
//	| data and bss | low permanent region | guard page | scratch region ... |
//
// The low region has a fixed extent: logosWasmHeapInit takes
// logosWasmLowBytes and a guard page from the break at startup and the
// bounds never change afterwards. Permanent runtime memory, the requests of
// sysAllocOS and of sysReserveOS without an address hint (metadata,
// persistent chunks, GC bit arenas, page summaries), is served from the low
// region, so it never sits between heap arenas. Everything else (heap
// arenas and the alignment gaps before them) comes from the scratch break
// bloc. memory.grow runs only from sbrk, which only moves the scratch
// break, so growth extends the high region alone, and sysFreeOS lowers the
// scratch break and never the low region: the guard page, which belongs to
// neither region and is never freed, keeps the break above the low region
// even while scratch is empty.
//
// Each region has its own free list. Memory freed inside the low region
// joins logosLowFreelist and serves later permanent requests only; it never
// reaches memFreelist, so a heap arena reservation, which takes any block of
// memFreelist that is large enough, can never be served from low memory. A
// permanent request goes to the low free list first, then to the low bump
// pointer, and last, with the low region full, to memAlloc like upstream;
// the bytes of that last kind are counted in spilled. The low region does
// not grow.
//
// All access holds memlock, as in mem_sbrk.go. logosWasmLowBytes is the
// extent of the low region and logosWasmGuardBytes the guard page.
const (
	logosWasmLowBytes   = 4 << 20
	logosWasmGuardBytes = 64 << 10
)

type logosWasmRegions struct {
	base, end, bloc uintptr
	scratch0        uintptr
	spilled         uintptr
}

var logosWasmLow logosWasmRegions

// logosLowFreelist is the free list of the low region, sorted in ascending
// order like memFreelist.
var logosLowFreelist memHdrPtr

// logosWasmHeapInit reserves the low permanent region at startup. It runs
// from osinit, after the break is set, before anything allocates. A binary
// that is not linked with -gc=wasm, or a memory.grow that fails, leaves the
// heap unsplit.
func logosWasmHeapInit() {
	if !logosWasmMode() {
		return
	}
	p := uintptr(sbrk(logosWasmLowBytes + logosWasmGuardBytes))
	if p == 0 {
		return
	}
	end := p + logosWasmLowBytes
	logosWasmLow = logosWasmRegions{base: p, end: end, bloc: p, scratch0: end + logosWasmGuardBytes}
}

// logosWasmInLow reports whether v lies in the low permanent region.
func logosWasmInLow(v unsafe.Pointer) bool {
	return uintptr(v) >= logosWasmLow.base && uintptr(v) < logosWasmLow.end
}

// logosSwapFreelists exchanges the active free list with the low one, so
// the upstream list routines operate on the other region.
func logosSwapFreelists() {
	memFreelist, logosLowFreelist = logosLowFreelist, memFreelist
}

// logosMemAllocPerm allocates n bytes of permanent memory: from the low
// free list, then the low bump pointer, and with the region full from
// memAlloc like upstream.
//
// memlock must be held, and the caller must be on the system stack.
//
//go:systemstack
func logosMemAllocPerm(n uintptr) unsafe.Pointer {
	l := &logosWasmLow
	if l.end == 0 {
		return memAlloc(n)
	}
	logosSwapFreelists()
	p := memAllocNoGrow(n)
	logosSwapFreelists()
	if p != nil {
		return p
	}
	n = memRound(n)
	if l.end-l.bloc >= n {
		p := l.bloc
		l.bloc += n
		return unsafe.Pointer(p)
	}
	p = memAlloc(n)
	if p != nil {
		l.spilled += n
	}
	return p
}

// logosMemFree makes [v, v+n) available for reallocation by the list of
// its region.
//
// memlock must be held, and the caller must be on the system stack.
//
//go:systemstack
func logosMemFree(v unsafe.Pointer, n uintptr) {
	if !logosWasmInLow(v) {
		memFree(v, n)
		return
	}
	logosSwapFreelists()
	memFree(v, n)
	logosSwapFreelists()
}
