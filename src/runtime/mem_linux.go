// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const (
	_PAGE_SIZE = _PhysPageSize
	_EACCES    = 13
)

// NOTE: vec must be just 1 byte long here.
// Mincore returns ENOMEM if any of the pages are unmapped,
// but we want to know that all of the pages are unmapped.
// To make these the same, we can only ask about one page
// at a time. See golang.org/issue/7476.
var addrspace_vec [1]byte

func addrspace_free(v unsafe.Pointer, n uintptr) bool {
	var chunk uintptr
	for off := uintptr(0); off < n; off += chunk {
		chunk = _PAGE_SIZE * uintptr(len(addrspace_vec))
		if chunk > (n - off) {
			chunk = n - off
		}
		errval := mincore(unsafe.Pointer(uintptr(v)+off), chunk, &addrspace_vec[0])
		// ENOMEM means unmapped, which is what we want.
		// Anything else we assume means the pages are mapped.
		if errval != -_ENOMEM {
			return false
		}
	}
	return true
}

func mmap_fixed(v unsafe.Pointer, n uintptr, prot, flags, fd int32, offset uint32) unsafe.Pointer {
	p := mmap(v, n, prot, flags, fd, offset)
	// On some systems, mmap ignores v without
	// MAP_FIXED, so retry if the address space is free.
	if p != v && addrspace_free(v, n) {
		if uintptr(p) > 4096 {
			munmap(p, n)
		}
		p = mmap(v, n, prot, flags|_MAP_FIXED, fd, offset)
	}
	return p
}

// Don't split the stack as this method may be invoked without a valid G, which
// prevents us from allocating more stack.
//go:nosplit
func sysAlloc(n uintptr, sysStat *uint64) unsafe.Pointer {
	p := mmap(nil, n, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_PRIVATE, -1, 0)
	if uintptr(p) < 4096 {
		if uintptr(p) == _EACCES {
			print("runtime: mmap: access denied\n")
			exit(2)
		}
		if uintptr(p) == _EAGAIN {
			print("runtime: mmap: too much locked memory (check 'ulimit -l').\n")
			exit(2)
		}
		return nil
	}
	mSysStatInc(sysStat, n)
	return p
}

func sysUnused(v unsafe.Pointer, n uintptr) {
	var s uintptr = hugePageSize // division by constant 0 is a compile-time error :(
	if s != 0 && (uintptr(v)%s != 0 || n%s != 0) {
		// See issue 8832
		// Linux kernel bug: https://bugzilla.kernel.org/show_bug.cgi?id=93111
		// Mark the region as NOHUGEPAGE so the kernel's khugepaged
		// doesn't undo our DONTNEED request.  khugepaged likes to migrate
		// regions which are only partially mapped to huge pages, including
		// regions with some DONTNEED marks.  That needlessly allocates physical
		// memory for our DONTNEED regions.
		madvise(v, n, _MADV_NOHUGEPAGE)
	}
	madvise(v, n, _MADV_DONTNEED)
}

func sysUsed(v unsafe.Pointer, n uintptr) {
	if hugePageSize != 0 {
		// Undo the NOHUGEPAGE marks from sysUnused.  There is no alignment check
		// around this call as spans may have been merged in the interim.
		// Note that this might enable huge pages for regions which were
		// previously disabled.  Unfortunately there is no easy way to detect
		// what the previous state was, and in any case we probably want huge
		// pages to back our heap if the kernel can arrange that.
		madvise(v, n, _MADV_HUGEPAGE)
	}
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
	// On 64-bit, people with ulimit -v set complain if we reserve too
	// much address space.  Instead, assume that the reservation is okay
	// if we can reserve at least 64K and check the assumption in SysMap.
	// Only user-mode Linux (UML) rejects these requests.
	if ptrSize == 8 && uint64(n) > 1<<32 {
		p := mmap_fixed(v, 64<<10, _PROT_NONE, _MAP_ANON|_MAP_PRIVATE, -1, 0)
		if p != v {
			if uintptr(p) >= 4096 {
				munmap(p, 64<<10)
			}
			return nil
		}
		munmap(p, 64<<10)
		*reserved = false
		return v
	}

	p := mmap(v, n, _PROT_NONE, _MAP_ANON|_MAP_PRIVATE, -1, 0)
	if uintptr(p) < 4096 {
		return nil
	}
	*reserved = true
	return p
}

func sysMap(v unsafe.Pointer, n uintptr, reserved bool, sysStat *uint64) {
	mSysStatInc(sysStat, n)

	// On 64-bit, we don't actually have v reserved, so tread carefully.
	if !reserved {
		p := mmap_fixed(v, n, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_PRIVATE, -1, 0)
		if uintptr(p) == _ENOMEM {
			throw("runtime: out of memory")
		}
		if p != v {
			print("runtime: address space conflict: map(", v, ") = ", p, "\n")
			throw("runtime: address space conflict")
		}
		return
	}

	p := mmap(v, n, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_FIXED|_MAP_PRIVATE, -1, 0)
	if uintptr(p) == _ENOMEM {
		throw("runtime: out of memory")
	}
	if p != v {
		throw("runtime: cannot map pages in arena address space")
	}
}
