// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// The calls to nop are to keep these functions from being inlined.
// If they are inlined we have no guarantee that later rewrites of the
// code by optimizers will preserve the relative order of memory accesses.

//go:nosplit
func atomicload(ptr *uint32) uint32 {
	nop()
	return *ptr
}

//go:nosplit
func atomicloadp(ptr unsafe.Pointer) unsafe.Pointer {
	nop()
	return *(*unsafe.Pointer)(ptr)
}

//go:nosplit
func xadd64(ptr *uint64, delta int64) uint64 {
	for {
		old := *ptr
		if cas64(ptr, old, old+uint64(delta)) {
			return old + uint64(delta)
		}
	}
}

//go:nosplit
func xchg64(ptr *uint64, new uint64) uint64 {
	for {
		old := *ptr
		if cas64(ptr, old, new) {
			return old
		}
	}
}

//go:noescape
func xadd(ptr *uint32, delta int32) uint32

//go:noescape
func xchg(ptr *uint32, new uint32) uint32

// NO go:noescape annotation; see atomic_pointer.go.
func xchgp1(ptr unsafe.Pointer, new unsafe.Pointer) unsafe.Pointer

//go:noescape
func xchguintptr(ptr *uintptr, new uintptr) uintptr

//go:noescape
func atomicload64(ptr *uint64) uint64

//go:noescape
func atomicor8(ptr *uint8, val uint8)

//go:noescape
func cas64(ptr *uint64, old, new uint64) bool

//go:noescape
func atomicstore(ptr *uint32, val uint32)

//go:noescape
func atomicstore64(ptr *uint64, val uint64)

// NO go:noescape annotation; see atomic_pointer.go.
func atomicstorep1(ptr unsafe.Pointer, val unsafe.Pointer)
