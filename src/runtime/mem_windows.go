// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

const (
	_MEM_COMMIT   = 0x1000
	_MEM_RESERVE  = 0x2000
	_MEM_DECOMMIT = 0x4000
	_MEM_RELEASE  = 0x8000

	_PAGE_READWRITE = 0x0004
	_PAGE_NOACCESS  = 0x0001

	_ERROR_NOT_ENOUGH_MEMORY = 8
	_ERROR_COMMITMENT_LIMIT  = 1455
)

// Don't split the stack as this function may be invoked without a valid G,
// which prevents us from allocating more stack.
//go:nosplit
func sysAlloc(n uintptr, sysStat *uint64) unsafe.Pointer {
	mSysStatInc(sysStat, n)
	return unsafe.Pointer(stdcall4(_VirtualAlloc, 0, n, _MEM_COMMIT|_MEM_RESERVE, _PAGE_READWRITE))
}

func sysUnused(v unsafe.Pointer, n uintptr) {
	r := stdcall3(_VirtualFree, uintptr(v), n, _MEM_DECOMMIT)
	if r != 0 {
		return
	}

	// Decommit failed. Usual reason is that we've merged memory from two different
	// VirtualAlloc calls, and Windows will only let each VirtualFree handle pages from
	// a single VirtualAlloc. It is okay to specify a subset of the pages from a single alloc,
	// just not pages from multiple allocs. This is a rare case, arising only when we're
	// trying to give memory back to the operating system, which happens on a time
	// scale of minutes. It doesn't have to be terribly fast. Instead of extra bookkeeping
	// on all our VirtualAlloc calls, try freeing successively smaller pieces until
	// we manage to free something, and then repeat. This ends up being O(n log n)
	// in the worst case, but that's fast enough.
	for n > 0 {
		small := n
		for small >= 4096 && stdcall3(_VirtualFree, uintptr(v), small, _MEM_DECOMMIT) == 0 {
			small /= 2
			small &^= 4096 - 1
		}
		if small < 4096 {
			print("runtime: VirtualFree of ", small, " bytes failed with errno=", getlasterror(), "\n")
			throw("runtime: failed to decommit pages")
		}
		v = add(v, small)
		n -= small
	}
}

func sysUsed(v unsafe.Pointer, n uintptr) {
	p := stdcall4(_VirtualAlloc, uintptr(v), n, _MEM_COMMIT, _PAGE_READWRITE)
	if p == uintptr(v) {
		return
	}

	// Commit failed. See SysUnused.
	// Hold on to n here so we can give back a better error message
	// for certain cases.
	k := n
	for k > 0 {
		small := k
		for small >= 4096 && stdcall4(_VirtualAlloc, uintptr(v), small, _MEM_COMMIT, _PAGE_READWRITE) == 0 {
			small /= 2
			small &^= 4096 - 1
		}
		if small < 4096 {
			errno := getlasterror()
			switch errno {
			case _ERROR_NOT_ENOUGH_MEMORY, _ERROR_COMMITMENT_LIMIT:
				print("runtime: VirtualAlloc of ", n, " bytes failed with errno=", errno, "\n")
				throw("out of memory")
			default:
				print("runtime: VirtualAlloc of ", small, " bytes failed with errno=", errno, "\n")
				throw("runtime: failed to commit pages")
			}
		}
		v = add(v, small)
		k -= small
	}
}

func sysHugePage(v unsafe.Pointer, n uintptr) {
}

// Don't split the stack as this function may be invoked without a valid G,
// which prevents us from allocating more stack.
//go:nosplit
func sysFree(v unsafe.Pointer, n uintptr, sysStat *uint64) {
	mSysStatDec(sysStat, n)
	r := stdcall3(_VirtualFree, uintptr(v), 0, _MEM_RELEASE)
	if r == 0 {
		print("runtime: VirtualFree of ", n, " bytes failed with errno=", getlasterror(), "\n")
		throw("runtime: failed to release pages")
	}
}

func sysFault(v unsafe.Pointer, n uintptr) {
	// SysUnused makes the memory inaccessible and prevents its reuse
	sysUnused(v, n)
}

func sysReserve(v unsafe.Pointer, n uintptr) unsafe.Pointer {
	// v is just a hint.
	// First try at v.
	// This will fail if any of [v, v+n) is already reserved.
	v = unsafe.Pointer(stdcall4(_VirtualAlloc, uintptr(v), n, _MEM_RESERVE, _PAGE_READWRITE))
	if v != nil {
		return v
	}

	// Next let the kernel choose the address.
	return unsafe.Pointer(stdcall4(_VirtualAlloc, 0, n, _MEM_RESERVE, _PAGE_READWRITE))
}

func sysMap(v unsafe.Pointer, n uintptr, sysStat *uint64) {
	mSysStatInc(sysStat, n)
}
