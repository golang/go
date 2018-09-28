// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

// Don't split the stack as this method may be invoked without a valid G, which
// prevents us from allocating more stack.
//go:nosplit
func sysAlloc(n uintptr, sysStat *uint64) unsafe.Pointer {
	p, err := mmap(nil, n, _PROT_READ|_PROT_WRITE, _MAP_ANONYMOUS|_MAP_PRIVATE, -1, 0)
	if err != 0 {
		if err == _EACCES {
			print("runtime: mmap: access denied\n")
			exit(2)
		}
		if err == _EAGAIN {
			print("runtime: mmap: too much locked memory (check 'ulimit -l').\n")
			exit(2)
		}
		//println("sysAlloc failed: ", err)
		return nil
	}
	mSysStatInc(sysStat, n)
	return p
}

func sysUnused(v unsafe.Pointer, n uintptr) {
	madvise(v, n, _MADV_DONTNEED)
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
	mmap(v, n, _PROT_NONE, _MAP_ANONYMOUS|_MAP_PRIVATE|_MAP_FIXED, -1, 0)
}

func sysReserve(v unsafe.Pointer, n uintptr) unsafe.Pointer {
	p, err := mmap(v, n, _PROT_NONE, _MAP_ANONYMOUS|_MAP_PRIVATE, -1, 0)
	if err != 0 {
		return nil
	}
	return p
}

func sysMap(v unsafe.Pointer, n uintptr, sysStat *uint64) {
	mSysStatInc(sysStat, n)

	// AIX does not allow mapping a range that is already mapped.
	// So always unmap first even if it is already unmapped.
	munmap(v, n)
	p, err := mmap(v, n, _PROT_READ|_PROT_WRITE, _MAP_ANONYMOUS|_MAP_FIXED|_MAP_PRIVATE, -1, 0)

	if err == _ENOMEM {
		throw("runtime: out of memory")
	}
	if p != v || err != 0 {
		throw("runtime: cannot map pages in arena address space")
	}
}
