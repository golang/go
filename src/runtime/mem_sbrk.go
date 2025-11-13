// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build plan9 || wasm

package runtime

import "unsafe"

const isSbrkPlatform = true

const memDebug = false

// Memory management on sbrk systems (including the linear memory
// on Wasm).

// bloc is the runtime's sense of the break, which can go up or
// down. blocMax is the system's break, also the high water mark
// of bloc. The runtime uses memory up to bloc. The memory
// between bloc and blocMax is allocated by the OS but not used
// by the runtime.
//
// When the runtime needs to grow the heap address range, it
// increases bloc. When it needs to grow beyond blocMax, it calls
// the system sbrk to allocate more memory (and therefore
// increase blocMax).
//
// When the runtime frees memory at the end of the address space,
// it decreases bloc, but does not reduces the system break (as
// the OS doesn't support it). When the runtime frees memory in
// the middle of the address space, the memory goes to a free
// list.

var bloc uintptr    // The runtime's sense of break. Can go up or down.
var blocMax uintptr // The break of the OS. Only increase.
var memlock mutex

type memHdr struct {
	next memHdrPtr
	size uintptr
}

var memFreelist memHdrPtr // sorted in ascending order

type memHdrPtr uintptr

func (p memHdrPtr) ptr() *memHdr   { return (*memHdr)(unsafe.Pointer(p)) }
func (p *memHdrPtr) set(x *memHdr) { *p = memHdrPtr(unsafe.Pointer(x)) }

// memAlloc allocates n bytes from the brk reservation, or if it's full,
// the system.
//
// memlock must be held.
//
// memAlloc must be called on the system stack, otherwise a stack growth
// could cause us to call back into it. Since memlock is held, that could
// lead to a self-deadlock.
//
//go:systemstack
func memAlloc(n uintptr) unsafe.Pointer {
	if p := memAllocNoGrow(n); p != nil {
		return p
	}
	return sbrk(n)
}

// memAllocNoGrow attempts to allocate n bytes from the existing brk.
//
// memlock must be held.
//
// memAlloc must be called on the system stack, otherwise a stack growth
// could cause us to call back into it. Since memlock is held, that could
// lead to a self-deadlock.
//
//go:systemstack
func memAllocNoGrow(n uintptr) unsafe.Pointer {
	n = memRound(n)
	var prevp *memHdr
	for p := memFreelist.ptr(); p != nil; p = p.next.ptr() {
		if p.size >= n {
			if p.size == n {
				if prevp != nil {
					prevp.next = p.next
				} else {
					memFreelist = p.next
				}
			} else {
				p.size -= n
				p = (*memHdr)(add(unsafe.Pointer(p), p.size))
			}
			*p = memHdr{}
			return unsafe.Pointer(p)
		}
		prevp = p
	}
	return nil
}

// memFree makes [ap, ap+n) available for reallocation by memAlloc.
//
// memlock must be held.
//
// memAlloc must be called on the system stack, otherwise a stack growth
// could cause us to call back into it. Since memlock is held, that could
// lead to a self-deadlock.
//
//go:systemstack
func memFree(ap unsafe.Pointer, n uintptr) {
	n = memRound(n)
	memclrNoHeapPointers(ap, n)
	bp := (*memHdr)(ap)
	bp.size = n
	bpn := uintptr(ap)
	if memFreelist == 0 {
		bp.next = 0
		memFreelist.set(bp)
		return
	}
	p := memFreelist.ptr()
	if bpn < uintptr(unsafe.Pointer(p)) {
		memFreelist.set(bp)
		if bpn+bp.size == uintptr(unsafe.Pointer(p)) {
			bp.size += p.size
			bp.next = p.next
			*p = memHdr{}
		} else {
			bp.next.set(p)
		}
		return
	}
	for ; p.next != 0; p = p.next.ptr() {
		if bpn > uintptr(unsafe.Pointer(p)) && bpn < uintptr(unsafe.Pointer(p.next)) {
			break
		}
	}
	if bpn+bp.size == uintptr(unsafe.Pointer(p.next)) {
		bp.size += p.next.ptr().size
		bp.next = p.next.ptr().next
		*p.next.ptr() = memHdr{}
	} else {
		bp.next = p.next
	}
	if uintptr(unsafe.Pointer(p))+p.size == bpn {
		p.size += bp.size
		p.next = bp.next
		*bp = memHdr{}
	} else {
		p.next.set(bp)
	}
}

// memCheck checks invariants around free list management.
//
// memlock must be held.
//
// memAlloc must be called on the system stack, otherwise a stack growth
// could cause us to call back into it. Since memlock is held, that could
// lead to a self-deadlock.
//
//go:systemstack
func memCheck() {
	if !memDebug {
		return
	}
	for p := memFreelist.ptr(); p != nil && p.next != 0; p = p.next.ptr() {
		if uintptr(unsafe.Pointer(p)) == uintptr(unsafe.Pointer(p.next)) {
			print("runtime: ", unsafe.Pointer(p), " == ", unsafe.Pointer(p.next), "\n")
			throw("mem: infinite loop")
		}
		if uintptr(unsafe.Pointer(p)) > uintptr(unsafe.Pointer(p.next)) {
			print("runtime: ", unsafe.Pointer(p), " > ", unsafe.Pointer(p.next), "\n")
			throw("mem: unordered list")
		}
		if uintptr(unsafe.Pointer(p))+p.size > uintptr(unsafe.Pointer(p.next)) {
			print("runtime: ", unsafe.Pointer(p), "+", p.size, " > ", unsafe.Pointer(p.next), "\n")
			throw("mem: overlapping blocks")
		}
		for b := add(unsafe.Pointer(p), unsafe.Sizeof(memHdr{})); uintptr(b) < uintptr(unsafe.Pointer(p))+p.size; b = add(b, 1) {
			if *(*byte)(b) != 0 {
				print("runtime: value at addr ", b, " with offset ", uintptr(b)-uintptr(unsafe.Pointer(p)), " in block ", p, " of size ", p.size, " is not zero\n")
				throw("mem: uninitialised memory")
			}
		}
	}
}

func memRound(p uintptr) uintptr {
	return alignUp(p, physPageSize)
}

func initBloc() {
	bloc = memRound(firstmoduledata.end)
	blocMax = bloc
}

func sysAllocOS(n uintptr, _ string) unsafe.Pointer {
	var p uintptr
	systemstack(func() {
		lock(&memlock)
		p = uintptr(memAlloc(n))
		memCheck()
		unlock(&memlock)
	})
	return unsafe.Pointer(p)
}

func sysFreeOS(v unsafe.Pointer, n uintptr) {
	systemstack(func() {
		lock(&memlock)
		if uintptr(v)+n == bloc {
			// Address range being freed is at the end of memory,
			// so record a new lower value for end of memory.
			// Can't actually shrink address space because segment is shared.
			memclrNoHeapPointers(v, n)
			bloc -= n
		} else {
			memFree(v, n)
			memCheck()
		}
		unlock(&memlock)
	})
}

func sysUnusedOS(v unsafe.Pointer, n uintptr) {
}

func sysUsedOS(v unsafe.Pointer, n uintptr) {
}

func sysHugePageOS(v unsafe.Pointer, n uintptr) {
}

func sysNoHugePageOS(v unsafe.Pointer, n uintptr) {
}

func sysHugePageCollapseOS(v unsafe.Pointer, n uintptr) {
}

func sysMapOS(v unsafe.Pointer, n uintptr, _ string) {
}

func sysFaultOS(v unsafe.Pointer, n uintptr) {
}

func sysReserveOS(v unsafe.Pointer, n uintptr, _ string) unsafe.Pointer {
	var p uintptr
	systemstack(func() {
		lock(&memlock)
		if uintptr(v) == bloc {
			// Address hint is the current end of memory,
			// so try to extend the address space.
			p = uintptr(sbrk(n))
		}
		if p == 0 && v == nil {
			p = uintptr(memAlloc(n))
			memCheck()
		}
		unlock(&memlock)
	})
	return unsafe.Pointer(p)
}

func sysReserveAlignedSbrk(size, align uintptr) (unsafe.Pointer, uintptr) {
	var p uintptr
	systemstack(func() {
		lock(&memlock)
		if base := memAllocNoGrow(size + align); base != nil {
			// We can satisfy the reservation from the free list.
			// Trim off the unaligned parts.
			start := alignUp(uintptr(base), align)
			if startLen := start - uintptr(base); startLen > 0 {
				memFree(base, startLen)
			}
			end := start + size
			if endLen := (uintptr(base) + size + align) - end; endLen > 0 {
				memFree(unsafe.Pointer(end), endLen)
			}
			memCheck()
			unlock(&memlock)
			p = start
			return
		}

		// Round up bloc to align, then allocate size.
		p = alignUp(bloc, align)
		r := sbrk(p + size - bloc)
		if r == nil {
			p, size = 0, 0
		} else if l := p - uintptr(r); l > 0 {
			// Free the area we skipped over for alignment.
			memFree(r, l)
			memCheck()
		}
		unlock(&memlock)
	})
	return unsafe.Pointer(p), size
}

func needZeroAfterSysUnusedOS() bool {
	return true
}
