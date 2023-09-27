// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || netbsd || openbsd || solaris

package runtime

import (
	"unsafe"
)

// Don't split the stack as this function may be invoked without a valid G,
// which prevents us from allocating more stack.
//
//go:nosplit
func sysAllocOS(n uintptr) unsafe.Pointer {
	v, err := mmap(nil, n, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_PRIVATE, -1, 0)
	if err != 0 {
		return nil
	}
	return v
}

func sysUnusedOS(v unsafe.Pointer, n uintptr) {
	if debug.madvdontneed != 0 {
		madvise(v, n, _MADV_DONTNEED)
	} else {
		madvise(v, n, _MADV_FREE)
	}
}

func sysUsedOS(v unsafe.Pointer, n uintptr) {
}

func sysHugePageOS(v unsafe.Pointer, n uintptr) {
}

func sysNoHugePageOS(v unsafe.Pointer, n uintptr) {
}

func sysHugePageCollapseOS(v unsafe.Pointer, n uintptr) {
}

// Don't split the stack as this function may be invoked without a valid G,
// which prevents us from allocating more stack.
//
//go:nosplit
func sysFreeOS(v unsafe.Pointer, n uintptr) {
	munmap(v, n)
}

func sysFaultOS(v unsafe.Pointer, n uintptr) {
	mmap(v, n, _PROT_NONE, _MAP_ANON|_MAP_PRIVATE|_MAP_FIXED, -1, 0)
}

// Indicates not to reserve swap space for the mapping.
const _sunosMAP_NORESERVE = 0x40

func sysReserveOS(v unsafe.Pointer, n uintptr) unsafe.Pointer {
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

func sysMapOS(v unsafe.Pointer, n uintptr) {
	p, err := mmap(v, n, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_FIXED|_MAP_PRIVATE, -1, 0)
	if err == _ENOMEM || ((GOOS == "solaris" || GOOS == "illumos") && err == _sunosEAGAIN) {
		throw("runtime: out of memory")
	}
	if p != v || err != 0 {
		print("runtime: mmap(", v, ", ", n, ") returned ", p, ", ", err, "\n")
		throw("runtime: cannot map pages in arena address space")
	}
}
