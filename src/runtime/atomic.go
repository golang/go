// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !arm

package runtime

import "unsafe"

//go:noescape
func xadd(ptr *uint32, delta int32) uint32

//go:noescape
func xadd64(ptr *uint64, delta int64) uint64

//go:noescape
func xchg(ptr *uint32, new uint32) uint32

//go:noescape
func xchg64(ptr *uint64, new uint64) uint64

// Cannot use noescape here: ptr does not but new does escape.
// Instead use noescape(ptr) in wrapper below.
func xchgp1(ptr unsafe.Pointer, new unsafe.Pointer) unsafe.Pointer

//go:nosplit
func xchgp(ptr unsafe.Pointer, new unsafe.Pointer) unsafe.Pointer {
	old := xchgp1(noescape(ptr), new)
	writebarrierptr_nostore((*uintptr)(ptr), uintptr(new))
	return old
}

//go:noescape
func xchguintptr(ptr *uintptr, new uintptr) uintptr

//go:noescape
func atomicload(ptr *uint32) uint32

//go:noescape
func atomicload64(ptr *uint64) uint64

//go:noescape
func atomicloadp(ptr unsafe.Pointer) unsafe.Pointer

//go:noescape
func atomicor8(ptr *uint8, val uint8)

//go:noescape
func cas64(ptr *uint64, old, new uint64) bool

//go:noescape
func atomicstore(ptr *uint32, val uint32)

//go:noescape
func atomicstore64(ptr *uint64, val uint64)

// Cannot use noescape here: ptr does not but val does escape.
// Instead use noescape(ptr) in wrapper below.
func atomicstorep1(ptr unsafe.Pointer, val unsafe.Pointer)

//go:nosplit
func atomicstorep(ptr unsafe.Pointer, val unsafe.Pointer) {
	atomicstorep1(noescape(ptr), val)
	// TODO(rsc): Why does the compiler think writebarrierptr_nostore's dst argument escapes?
	writebarrierptr_nostore((*uintptr)(noescape(ptr)), uintptr(val))
}

// Cannot use noescape here: ptr does not but new does escape.
// Instead use noescape(ptr) in wrapper below.
func casp1(ptr *unsafe.Pointer, old, new unsafe.Pointer) bool

//go:nosplit
func casp(ptr *unsafe.Pointer, old, new unsafe.Pointer) bool {
	ok := casp1((*unsafe.Pointer)(noescape(unsafe.Pointer(ptr))), old, new)
	if !ok {
		return false
	}
	writebarrierptr_nostore((*uintptr)(unsafe.Pointer(ptr)), uintptr(new))
	return true
}
