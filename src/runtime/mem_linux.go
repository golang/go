// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
	"unsafe"
)

const (
	_EACCES = 13
	_EINVAL = 22
)

// Don't split the stack as this method may be invoked without a valid G, which
// prevents us from allocating more stack.
//
//go:nosplit
func sysAllocOS(n uintptr) unsafe.Pointer {
	p, err := mmap(nil, n, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_PRIVATE, -1, 0)
	if err != 0 {
		if err == _EACCES {
			print("runtime: mmap: access denied\n")
			exit(2)
		}
		if err == _EAGAIN {
			print("runtime: mmap: too much locked memory (check 'ulimit -l').\n")
			exit(2)
		}
		return nil
	}
	return p
}

var adviseUnused = uint32(_MADV_FREE)

func sysUnusedOS(v unsafe.Pointer, n uintptr) {
	if uintptr(v)&(physPageSize-1) != 0 || n&(physPageSize-1) != 0 {
		// madvise will round this to any physical page
		// *covered* by this range, so an unaligned madvise
		// will release more memory than intended.
		throw("unaligned sysUnused")
	}

	var advise uint32
	if debug.madvdontneed != 0 {
		advise = _MADV_DONTNEED
	} else {
		advise = atomic.Load(&adviseUnused)
	}
	if errno := madvise(v, n, int32(advise)); advise == _MADV_FREE && errno != 0 {
		// MADV_FREE was added in Linux 4.5. Fall back to MADV_DONTNEED if it is
		// not supported.
		atomic.Store(&adviseUnused, _MADV_DONTNEED)
		madvise(v, n, _MADV_DONTNEED)
	}

	if debug.harddecommit > 0 {
		p, err := mmap(v, n, _PROT_NONE, _MAP_ANON|_MAP_FIXED|_MAP_PRIVATE, -1, 0)
		if p != v || err != 0 {
			throw("runtime: cannot disable permissions in address space")
		}
	}
}

func sysUsedOS(v unsafe.Pointer, n uintptr) {
	if debug.harddecommit > 0 {
		p, err := mmap(v, n, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_FIXED|_MAP_PRIVATE, -1, 0)
		if err == _ENOMEM {
			throw("runtime: out of memory")
		}
		if p != v || err != 0 {
			throw("runtime: cannot remap pages in address space")
		}
		return
	}
}

func sysHugePageOS(v unsafe.Pointer, n uintptr) {
	if physHugePageSize != 0 {
		// Round v up to a huge page boundary.
		beg := alignUp(uintptr(v), physHugePageSize)
		// Round v+n down to a huge page boundary.
		end := alignDown(uintptr(v)+n, physHugePageSize)

		if beg < end {
			madvise(unsafe.Pointer(beg), end-beg, _MADV_HUGEPAGE)
		}
	}
}

func sysNoHugePageOS(v unsafe.Pointer, n uintptr) {
	if uintptr(v)&(physPageSize-1) != 0 {
		// The Linux implementation requires that the address
		// addr be page-aligned, and allows length to be zero.
		throw("unaligned sysNoHugePageOS")
	}
	madvise(v, n, _MADV_NOHUGEPAGE)
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

func sysReserveOS(v unsafe.Pointer, n uintptr) unsafe.Pointer {
	p, err := mmap(v, n, _PROT_NONE, _MAP_ANON|_MAP_PRIVATE, -1, 0)
	if err != 0 {
		return nil
	}
	return p
}

func sysMapOS(v unsafe.Pointer, n uintptr) {
	p, err := mmap(v, n, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_FIXED|_MAP_PRIVATE, -1, 0)
	if err == _ENOMEM {
		throw("runtime: out of memory")
	}
	if p != v || err != 0 {
		print("runtime: mmap(", v, ", ", n, ") returned ", p, ", ", err, "\n")
		throw("runtime: cannot map pages in arena address space")
	}
}
