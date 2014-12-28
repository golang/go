// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build dragonfly freebsd nacl netbsd openbsd solaris

package runtime

import "unsafe"

//go:nosplit
func sysAlloc(n uintptr, stat *uint64) unsafe.Pointer {
	v := unsafe.Pointer(mmap(nil, n, _PROT_READ|_PROT_WRITE, _MAP_ANON|_MAP_PRIVATE, -1, 0))
	if uintptr(v) < 4096 {
		return nil
	}
	xadd64(stat, int64(n))
	return v
}

func sysUnused(v unsafe.Pointer, n uintptr) {
	madvise(v, n, _MADV_FREE)
}

func sysUsed(v unsafe.Pointer, n uintptr) {
}

func sysFree(v unsafe.Pointer, n uintptr, stat *uint64) {
	xadd64(stat, -int64(n))
	munmap(v, n)
}

func sysFault(v unsafe.Pointer, n uintptr) {
	mmap(v, n, _PROT_NONE, _MAP_ANON|_MAP_PRIVATE|_MAP_FIXED, -1, 0)
}

func sysReserve(v unsafe.Pointer, n uintptr, reserved *bool) unsafe.Pointer {
	// On 64-bit, people with ulimit -v set complain if we reserve too
	// much address space.  Instead, assume that the reservation is okay
	// and check the assumption in SysMap.
	if ptrSize == 8 && uint64(n) > 1<<32 || goos_nacl != 0 {
		*reserved = false
		return v
	}

	p := unsafe.Pointer(mmap(v, n, _PROT_NONE, _MAP_ANON|_MAP_PRIVATE, -1, 0))
	if uintptr(p) < 4096 {
		return nil
	}
	*reserved = true
	return p
}

func sysMap(v unsafe.Pointer, n uintptr, reserved bool, stat *uint64) {
	const _ENOMEM = 12

	xadd64(stat, int64(n))

	// On 64-bit, we don't actually have v reserved, so tread carefully.
	if !reserved {
		flags := int32(_MAP_ANON | _MAP_PRIVATE)
		if GOOS == "dragonfly" {
			// TODO(jsing): For some reason DragonFly seems to return
			// memory at a different address than we requested, even when
			// there should be no reason for it to do so. This can be
			// avoided by using MAP_FIXED, but I'm not sure we should need
			// to do this - we do not on other platforms.
			flags |= _MAP_FIXED
		}
		p := mmap(v, n, _PROT_READ|_PROT_WRITE, flags, -1, 0)
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
