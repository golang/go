// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atomic

import "unsafe"

//go:nosplit
//go:noinline
func Load(ptr *uint32) uint32 {
	return *ptr
}

//go:nosplit
//go:noinline
func Loadp(ptr unsafe.Pointer) unsafe.Pointer {
	return *(*unsafe.Pointer)(ptr)
}

//go:nosplit
//go:noinline
func Load64(ptr *uint64) uint64 {
	return *ptr
}

//go:noinline
//go:nosplit
func Store(ptr *uint32, val uint32) {
	*ptr = val
}

//go:noinline
//go:nosplit
func Store64(ptr *uint64, val uint64) {
	*ptr = val
}

// NO go:noescape annotation; see atomic_pointer.go.
//go:noinline
//go:nosplit
func StorepNoWB(ptr unsafe.Pointer, val unsafe.Pointer) {
	*(*uintptr)(ptr) = uintptr(val)
}

//go:noescape
func And8(ptr *uint8, val uint8)

//go:noescape
func Or8(ptr *uint8, val uint8)

// NOTE: Do not add atomicxor8 (XOR is not idempotent).

//go:noescape
func Xadd(ptr *uint32, delta int32) uint32

//go:noescape
func Xadd64(ptr *uint64, delta int64) uint64

//go:noescape
func Xadduintptr(ptr *uintptr, delta uintptr) uintptr

//go:noescape
func Xchg(ptr *uint32, new uint32) uint32

//go:noescape
func Xchg64(ptr *uint64, new uint64) uint64

//go:noescape
func Xchguintptr(ptr *uintptr, new uintptr) uintptr

//go:noescape
func Cas64(ptr *uint64, old, new uint64) bool
