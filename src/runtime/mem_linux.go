// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/atomic"
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
func sysAllocOS(n uintptr, vmaName string) unsafe.Pointer {
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
	setVMAName(p, n, vmaName)
	return p
}

var adviseUnused = uint32(_MADV_FREE)

const madviseUnsupported = 0

func sysUnusedOS(v unsafe.Pointer, n uintptr) {
	if uintptr(v)&(physPageSize-1) != 0 || n&(physPageSize-1) != 0 {
		// madvise will round this to any physical page
		// *covered* by this range, so an unaligned madvise
		// will release more memory than intended.
		throw("unaligned sysUnused")
	}

	advise := atomic.Load(&adviseUnused)
	if debug.madvdontneed != 0 && advise != madviseUnsupported {
		advise = _MADV_DONTNEED
	}
	switch advise {
	case _MADV_FREE:
		if madvise(v, n, _MADV_FREE) == 0 {
			break
		}
		atomic.Store(&adviseUnused, _MADV_DONTNEED)
		fallthrough
	case _MADV_DONTNEED:
		// MADV_FREE was added in Linux 4.5. Fall back on MADV_DONTNEED if it's
		// not supported.
		if madvise(v, n, _MADV_DONTNEED) == 0 {
			break
		}
		atomic.Store(&adviseUnused, madviseUnsupported)
		fallthrough
	case madviseUnsupported:
		// Since Linux 3.18, support for madvise is optional.
		// Fall back on mmap if it's not supported.
		// _MAP_ANON|_MAP_FIXED|_MAP_PRIVATE will unmap all the
		// pages in the old mapping, and remap the memory region.
		p, err := mmap(v, n, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_FIXED|_MAP_PRIVATE, -1, 0)
		if err == 0 && p != nil {
			setVMAName(p, n, "unused")
		}
	}

	if debug.harddecommit > 0 {
		p, err := mmap(v, n, _PROT_NONE, _MAP_ANON|_MAP_FIXED|_MAP_PRIVATE, -1, 0)
		if p != v || err != 0 {
			throw("runtime: cannot disable permissions in address space")
		}
		setVMAName(p, n, "unused")
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
		setVMAName(p, n, "used")
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

func sysHugePageCollapseOS(v unsafe.Pointer, n uintptr) {
	if uintptr(v)&(physPageSize-1) != 0 {
		// The Linux implementation requires that the address
		// addr be page-aligned, and allows length to be zero.
		throw("unaligned sysHugePageCollapseOS")
	}
	if physHugePageSize == 0 {
		return
	}
	// N.B. If you find yourself debugging this code, note that
	// this call can fail with EAGAIN because it's best-effort.
	// Also, when it returns an error, it's only for the last
	// huge page in the region requested.
	//
	// It can also sometimes return EINVAL if the corresponding
	// region hasn't been backed by physical memory. This is
	// difficult to guarantee in general, and it also means
	// there's no way to distinguish whether this syscall is
	// actually available. Oops.
	//
	// Anyway, that's why this call just doesn't bother checking
	// any errors.
	madvise(v, n, _MADV_COLLAPSE)
}

// Don't split the stack as this function may be invoked without a valid G,
// which prevents us from allocating more stack.
//
//go:nosplit
func sysFreeOS(v unsafe.Pointer, n uintptr) {
	munmap(v, n)
}

func sysFaultOS(v unsafe.Pointer, n uintptr) {
	mprotect(v, n, _PROT_NONE)
	madvise(v, n, _MADV_DONTNEED)
}

func sysReserveOS(v unsafe.Pointer, n uintptr, vmaName string) unsafe.Pointer {
	p, err := mmap(v, n, _PROT_NONE, _MAP_ANON|_MAP_PRIVATE, -1, 0)
	if err != 0 {
		return nil
	}
	setVMAName(p, n, vmaName)
	return p
}

func sysMapOS(v unsafe.Pointer, n uintptr, vmaName string) {
	p, err := mmap(v, n, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_FIXED|_MAP_PRIVATE, -1, 0)
	if err == _ENOMEM {
		throw("runtime: out of memory")
	}
	if p != v || err != 0 {
		print("runtime: mmap(", v, ", ", n, ") returned ", p, ", ", err, "\n")
		throw("runtime: cannot map pages in arena address space")
	}
	setVMAName(p, n, vmaName)

	// Disable huge pages if the GODEBUG for it is set.
	//
	// Note that there are a few sysHugePage calls that can override this, but
	// they're all for GC metadata.
	if debug.disablethp != 0 {
		sysNoHugePageOS(v, n)
	}
}
