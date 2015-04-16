// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

//go:noescape
func xadd(ptr *uint32, delta int32) uint32

//go:noescape
func xadd64(ptr *uint64, delta int64) uint64

//go:noescape
//go:linkname xadduintptr runtime.xadd64
func xadduintptr(ptr *uintptr, delta uintptr) uintptr

//go:noescape
func xchg(ptr *uint32, new uint32) uint32

//go:noescape
func xchg64(ptr *uint64, new uint64) uint64

// NO go:noescape annotation; see atomic_pointer.go.
func xchgp1(ptr unsafe.Pointer, new unsafe.Pointer) unsafe.Pointer

//go:noescape
func xchguintptr(ptr *uintptr, new uintptr) uintptr

//go:noescape
func atomicload(ptr *uint32) uint32

//go:noescape
func atomicload64(ptr *uint64) uint64

//go:noescape
func atomicloadp(ptr unsafe.Pointer) unsafe.Pointer

//go:nosplit
func atomicor8(addr *uint8, v uint8) {
	// TODO(dfc) implement this in asm.
	// Align down to 4 bytes and use 32-bit CAS.
	uaddr := uintptr(unsafe.Pointer(addr))
	addr32 := (*uint32)(unsafe.Pointer(uaddr &^ 3))
	word := uint32(v) << ((uaddr & 3) * 8) // little endian
	for {
		old := *addr32
		if cas(addr32, old, old|word) {
			return
		}
	}
}

//go:nosplit
func atomicand8(addr *uint8, v uint8) {
	// TODO(dfc) implement this in asm.
	// Align down to 4 bytes and use 32-bit CAS.
	uaddr := uintptr(unsafe.Pointer(addr))
	addr32 := (*uint32)(unsafe.Pointer(uaddr &^ 3))
	word := uint32(v) << ((uaddr & 3) * 8)    // little endian
	mask := uint32(0xFF) << ((uaddr & 3) * 8) // little endian
	word |= ^mask
	for {
		old := *addr32
		if cas(addr32, old, old&word) {
			return
		}
	}
}

//go:noescape
func cas64(ptr *uint64, old, new uint64) bool

//go:noescape
func atomicstore(ptr *uint32, val uint32)

//go:noescape
func atomicstore64(ptr *uint64, val uint64)

// NO go:noescape annotation; see atomic_pointer.go.
func atomicstorep1(ptr unsafe.Pointer, val unsafe.Pointer)
