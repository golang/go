// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const memDebug = false

var bloc uintptr
var blocMax uintptr
var memlock mutex

type memHdr struct {
	next memHdrPtr
	size uintptr
}

var memFreelist memHdrPtr // sorted in ascending order

type memHdrPtr uintptr

func (p memHdrPtr) ptr() *memHdr   { return (*memHdr)(unsafe.Pointer(p)) }
func (p *memHdrPtr) set(x *memHdr) { *p = memHdrPtr(unsafe.Pointer(x)) }

func memAlloc(n uintptr) unsafe.Pointer {
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
	return sbrk(n)
}

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

func memCheck() {
	if memDebug == false {
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
	return (p + _PAGESIZE - 1) &^ (_PAGESIZE - 1)
}

func initBloc() {
	bloc = memRound(firstmoduledata.end)
	blocMax = bloc
}

func sbrk(n uintptr) unsafe.Pointer {
	// Plan 9 sbrk from /sys/src/libc/9sys/sbrk.c
	bl := bloc
	n = memRound(n)
	if bl+n > blocMax {
		if brk_(unsafe.Pointer(bl+n)) < 0 {
			return nil
		}
		blocMax = bl + n
	}
	bloc += n
	return unsafe.Pointer(bl)
}

func sysAllocOS(n uintptr) unsafe.Pointer {
	lock(&memlock)
	p := memAlloc(n)
	memCheck()
	unlock(&memlock)
	return p
}

func sysFreeOS(v unsafe.Pointer, n uintptr) {
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
}

func sysUnusedOS(v unsafe.Pointer, n uintptr) {
}

func sysUsedOS(v unsafe.Pointer, n uintptr) {
}

func sysHugePageOS(v unsafe.Pointer, n uintptr) {
}

func sysMapOS(v unsafe.Pointer, n uintptr) {
}

func sysFaultOS(v unsafe.Pointer, n uintptr) {
}

func sysReserveOS(v unsafe.Pointer, n uintptr) unsafe.Pointer {
	lock(&memlock)
	var p unsafe.Pointer
	if uintptr(v) == bloc {
		// Address hint is the current end of memory,
		// so try to extend the address space.
		p = sbrk(n)
	}
	if p == nil && v == nil {
		p = memAlloc(n)
		memCheck()
	}
	unlock(&memlock)
	return p
}
