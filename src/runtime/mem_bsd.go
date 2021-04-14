// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build dragonfly freebsd netbsd openbsd solaris

package runtime

import (
	"unsafe"
)

// Don't split the stack as this function may be invoked without a valid G,
// which prevents us from allocating more stack.
//go:nosplit
func sysAlloc(n uintptr, sysStat *sysMemStat) unsafe.Pointer {
	v, err := mmap(nil, n, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_PRIVATE, -1, 0)
	if err != 0 {
		return nil
	}
	sysStat.add(int64(n))
	return v
}

func sysUnused(v unsafe.Pointer, n uintptr) {
	madvise(v, n, _MADV_FREE)
}

func sysUsed(v unsafe.Pointer, n uintptr) {
}

func sysHugePage(v unsafe.Pointer, n uintptr) {
}

// Don't split the stack as this function may be invoked without a valid G,
// which prevents us from allocating more stack.
//go:nosplit
func sysFree(v unsafe.Pointer, n uintptr, sysStat *sysMemStat) {
	sysStat.add(-int64(n))
	munmap(v, n)
}

func sysFault(v unsafe.Pointer, n uintptr) {
	mmap(v, n, _PROT_NONE, _MAP_ANON|_MAP_PRIVATE|_MAP_FIXED, -1, 0)
}

// Indicates not to reserve swap space for the mapping.
const _sunosMAP_NORESERVE = 0x40

func sysReserve(v unsafe.Pointer, n uintptr) unsafe.Pointer {
	flags := int32(_MAP_ANON | _MAP_PRIVATE)
	if GOOS == "solaris" || GOOS == "illumos" {
		// Be explicit that we don't want to reserve swap space
		// for PROT_NONE anonymous mappings. This avoids an issue
		// wherein large mappings can cause fork to fail.
		flags |= _sunosMAP_NORESERVE
	}
	p, err := mmap(v, n, _PROT_NONE, flags, -1, 0)
	if err != 0 {
		return nil
	}
	return p
}

const _sunosEAGAIN = 11
const _ENOMEM = 12

func sysMap(v unsafe.Pointer, n uintptr, sysStat *sysMemStat) {
	sysStat.add(int64(n))

	p, err := mmap(v, n, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_FIXED|_MAP_PRIVATE, -1, 0)
	if err == _ENOMEM || ((GOOS == "solaris" || GOOS == "illumos") && err == _sunosEAGAIN) {
		throw("runtime: out of memory")
	}
	if p != v || err != 0 {
		throw("runtime: cannot map pages in arena address space")
	}
}
