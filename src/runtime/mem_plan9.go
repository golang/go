// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const memDebug = false

var bloc uintptr
var memlock mutex

type memHdr struct {
	next *memHdr
	size uintptr
}

var memFreelist *memHdr // sorted in ascending order

func memAlloc(n uintptr) unsafe.Pointer {
	n = memRound(n)
	var prevp *memHdr
	for p := memFreelist; p != nil; p = p.next {
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
			memclr(unsafe.Pointer(p), unsafe.Sizeof(memHdr{}))
			return unsafe.Pointer(p)
		}
		prevp = p
	}
	return sbrk(n)
}

func memFree(ap unsafe.Pointer, n uintptr) {
	n = memRound(n)
	memclr(ap, n)
	bp := (*memHdr)(ap)
	bp.size = n
	bpn := uintptr(ap)
	if memFreelist == nil {
		bp.next = nil
		memFreelist = bp
		return
	}
	p := memFreelist
	if bpn < uintptr(unsafe.Pointer(p)) {
		memFreelist = bp
		if bpn+bp.size == uintptr(unsafe.Pointer(p)) {
			bp.size += p.size
			bp.next = p.next
			memclr(unsafe.Pointer(p), unsafe.Sizeof(memHdr{}))
		} else {
			bp.next = p
		}
		return
	}
	for ; p.next != nil; p = p.next {
		if bpn > uintptr(unsafe.Pointer(p)) && bpn < uintptr(unsafe.Pointer(p.next)) {
			break
		}
	}
	if bpn+bp.size == uintptr(unsafe.Pointer(p.next)) {
		bp.size += p.next.size
		bp.next = p.next.next
		memclr(unsafe.Pointer(p.next), unsafe.Sizeof(memHdr{}))
	} else {
		bp.next = p.next
	}
	if uintptr(unsafe.Pointer(p))+p.size == bpn {
		p.size += bp.size
		p.next = bp.next
		memclr(unsafe.Pointer(bp), unsafe.Sizeof(memHdr{}))
	} else {
		p.next = bp
	}
}

func memCheck() {
	if memDebug == false {
		return
	}
	for p := memFreelist; p != nil && p.next != nil; p = p.next {
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
}

func sbrk(n uintptr) unsafe.Pointer {
	// Plan 9 sbrk from /sys/src/libc/9sys/sbrk.c
	bl := bloc
	n = memRound(n)
	if brk_(unsafe.Pointer(bl+n)) < 0 {
		return nil
	}
	bloc += n
	return unsafe.Pointer(bl)
}

func sysAlloc(n uintptr, sysStat *uint64) unsafe.Pointer {
	lock(&memlock)
	p := memAlloc(n)
	memCheck()
	unlock(&memlock)
	if p != nil {
		mSysStatInc(sysStat, n)
	}
	return p
}

func sysFree(v unsafe.Pointer, n uintptr, sysStat *uint64) {
	mSysStatDec(sysStat, n)
	lock(&memlock)
	memFree(v, n)
	memCheck()
	unlock(&memlock)
}

func sysUnused(v unsafe.Pointer, n uintptr) {
}

func sysUsed(v unsafe.Pointer, n uintptr) {
}

func sysMap(v unsafe.Pointer, n uintptr, reserved bool, sysStat *uint64) {
	// sysReserve has already allocated all heap memory,
	// but has not adjusted stats.
	mSysStatInc(sysStat, n)
}

func sysFault(v unsafe.Pointer, n uintptr) {
}

func sysReserve(v unsafe.Pointer, n uintptr, reserved *bool) unsafe.Pointer {
	*reserved = true
	lock(&memlock)
	p := memAlloc(n)
	memCheck()
	unlock(&memlock)
	return p
}
