// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

// Don't split the stack as this function may be invoked without a valid G,
// which prevents us from allocating more stack.
//go:nosplit
func sysAlloc(n uintptr, sysStat *uint64) unsafe.Pointer {
	v, err := mmap(nil, n, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_PRIVATE, -1, 0)
	if err != 0 {
		return nil
	}
	mSysStatInc(sysStat, n)
	return v
}

func sysUnused(v unsafe.Pointer, n uintptr) {
	// MADV_FREE_REUSABLE is like MADV_FREE except it also propagates
	// accounting information about the process to task_info.
	madvise(v, n, _MADV_FREE_REUSABLE)
}

func sysUsed(v unsafe.Pointer, n uintptr) {
	// MADV_FREE_REUSE is necessary to keep the kernel's accounting
	// accurate. If called on any memory region that hasn't been
	// MADV_FREE_REUSABLE'd, it's a no-op.
	madvise(v, n, _MADV_FREE_REUSE)
}

func sysHugePage(v unsafe.Pointer, n uintptr) {
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

func sysReserve(v unsafe.Pointer, n uintptr) unsafe.Pointer {
	flags := int32(_MAP_ANON | _MAP_PRIVATE)
	if raceenabled {
		// Currently the race detector expects memory to live within a certain
		// range, and on Darwin 10.10 mmap is prone to ignoring hints, moreso
		// than later versions and other BSDs (#26475). So, even though it's
		// potentially dangerous to MAP_FIXED, we do it in the race detection
		// case because it'll help maintain the race detector's invariants.
		//
		// TODO(mknyszek): Drop this once support for Darwin 10.10 is dropped,
		// and reconsider this when #24133 is addressed.
		flags |= _MAP_FIXED
	}
	p, err := mmap(v, n, _PROT_NONE, flags, -1, 0)
	if err != 0 {
		return nil
	}
	return p
}

const _ENOMEM = 12

func sysMap(v unsafe.Pointer, n uintptr, sysStat *uint64) {
	mSysStatInc(sysStat, n)

	p, err := mmap(v, n, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_FIXED|_MAP_PRIVATE, -1, 0)
	if err == _ENOMEM {
		throw("runtime: out of memory")
	}
	if p != v || err != 0 {
		throw("runtime: cannot map pages in arena address space")
	}
}
