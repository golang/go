// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// Don't split the stack as this function may be invoked without a valid G,
// which prevents us from allocating more stack.
//go:nosplit
func sysAlloc(n uintptr, sysStat *uint64) unsafe.Pointer {
	v := unsafe.Pointer(mmap(nil, n, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_PRIVATE, -1, 0))
	if uintptr(v) < 4096 {
		return nil
	}
	mSysStatInc(sysStat, n)
	return v
}

func sysUnused(v unsafe.Pointer, n uintptr) {
	// Linux's MADV_DONTNEED is like BSD's MADV_FREE.
	madvise(v, n, _MADV_FREE)
}

func sysUsed(v unsafe.Pointer, n uintptr) {
}

// Don't split the stack as this function may be invoked without a valid G,
// which prevents us from allocating more stack.
//go:nosplit
func sysFree(v unsafe.Pointer, n uintptr, sysStat *uint64) {
	mSysStatDec(sysStat, n)
	munmap(v, n)
}

func sysFault(v unsafe.Pointer, n uintptr) {
	mmap(v, n, _PROT_NONE, _MAP_ANON|_MAP_PRIVATE|_MAP_FIXED, -1, 0)
}

func sysReserve(v unsafe.Pointer, n uintptr, reserved *bool) unsafe.Pointer {
	*reserved = true
	p := unsafe.Pointer(mmap(v, n, _PROT_NONE, _MAP_ANON|_MAP_PRIVATE, -1, 0))
	if uintptr(p) < 4096 {
		return nil
	}
	return p
}

const (
	_ENOMEM = 12
)

func sysMap(v unsafe.Pointer, n uintptr, reserved bool, sysStat *uint64) {
	mSysStatInc(sysStat, n)
	p := unsafe.Pointer(mmap(v, n, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_FIXED|_MAP_PRIVATE, -1, 0))
	if uintptr(p) == _ENOMEM {
		throw("runtime: out of memory")
	}
	if p != v {
		throw("runtime: cannot map pages in arena address space")
	}
}
