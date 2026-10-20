// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasm

package runtime

import "unsafe"

// Test seam for M19-S3: the low permanent region and the high scratch
// region of the wasm linear memory, driven through the same entry points
// the runtime uses (sysAllocOS, sysReserveOS at the break, sysFreeOS).

const (
	LogosWasmLowBytes   = logosWasmLowBytes
	LogosWasmGuardBytes = logosWasmGuardBytes
)

// LogosWasmScratchStart returns the first address of the scratch region.
func LogosWasmScratchStart() uintptr {
	return logosWasmLow.scratch0
}

// LogosWasmFreeOverlaps reports whether a block of the scratch free list
// overlaps [lo, hi).
func LogosWasmFreeOverlaps(lo, hi uintptr) bool {
	overlaps := false
	systemstack(func() {
		lock(&memlock)
		for p := memFreelist.ptr(); p != nil; p = p.next.ptr() {
			start := uintptr(unsafe.Pointer(p))
			if start < hi && start+p.size > lo {
				overlaps = true
			}
		}
		unlock(&memlock)
	})
	return overlaps
}

// LogosWasmStep records one scratch growth step: the address sysReserveOS
// returned, the low region after the step, and the scratch break.
type LogosWasmStep struct {
	Addr, Base, End, Used, Break uintptr
}

// LogosWasmLowRegion reports the fixed bounds of the low permanent
// region, the bytes handed out from it, and the bytes of permanent
// requests that spilled into scratch. base is zero when the heap is not
// split.
func LogosWasmLowRegion() (base, end, used, spilled uintptr) {
	l := &logosWasmLow
	return l.base, l.end, l.bloc - l.base, l.spilled
}

// LogosWasmBreak returns the scratch break.
func LogosWasmBreak() uintptr {
	return bloc
}

// LogosWasmMemoryPages returns the size of the linear memory in pages.
func LogosWasmMemoryPages() uintptr {
	return uintptr(currentMemory())
}

// LogosWasmScratchSteps grows the scratch break steps times by n bytes
// through sysReserveOS at the break, the way heap arena growth extends
// it, and records each step in out. It allocates nothing, so no
// collection can move the break between steps. It returns the number of
// steps recorded.
func LogosWasmScratchSteps(steps int, n uintptr, out []LogosWasmStep) int {
	done := 0
	for done < steps && done < len(out) {
		p := uintptr(sysReserveOS(add(nil, bloc), n, ""))
		if p == 0 {
			break
		}
		l := &logosWasmLow
		out[done] = LogosWasmStep{Addr: p, Base: l.base, End: l.end, Used: l.bloc - l.base, Break: bloc}
		done++
	}
	return done
}

// LogosWasmScratchFree releases [p, p+n) through sysFreeOS.
func LogosWasmScratchFree(p, n uintptr) {
	sysFreeOS(add(nil, p), n)
}

// LogosWasmPermAlloc takes n bytes of permanent memory through
// sysAllocOS and reports whether the block came back zeroed.
func LogosWasmPermAlloc(n uintptr) (p uintptr, zeroed bool) {
	p = uintptr(sysAllocOS(n, ""))
	if p == 0 {
		return 0, false
	}
	for off := uintptr(0); off < n; off += 8 {
		if *(*uintptr)(add(nil, p+off)) != 0 {
			return p, false
		}
	}
	return p, true
}

// LogosWasmPermReserve takes n bytes through sysReserveOS without an
// address hint, the path of the page summary and scavenge index.
func LogosWasmPermReserve(n uintptr) uintptr {
	return uintptr(sysReserveOS(nil, n, ""))
}

// LogosWasmPermFree gives a block from LogosWasmPermAlloc back.
func LogosWasmPermFree(p, n uintptr) {
	sysFreeOS(add(nil, p), n)
}

// LogosWasmInLow reports whether p lies in the low permanent region.
func LogosWasmInLow(p uintptr) bool {
	return logosWasmInLow(add(nil, p))
}

// LogosWasmLowFreeBytes returns the bytes on the free list of the low
// region.
func LogosWasmLowFreeBytes() uintptr {
	var total uintptr
	systemstack(func() {
		lock(&memlock)
		for p := logosLowFreelist.ptr(); p != nil; p = p.next.ptr() {
			total += p.size
		}
		unlock(&memlock)
	})
	return total
}

// LogosWasmReserveAligned reserves size bytes aligned to align the way a
// heap arena reservation does, and returns the block and its size.
func LogosWasmReserveAligned(size, align uintptr) (p, got uintptr) {
	v, n := sysReserveAlignedSbrk(size, align)
	return uintptr(v), n
}
