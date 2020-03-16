// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(neelance): implement with actual atomic operations as soon as threads are available
// See https://github.com/WebAssembly/design/issues/1073

// Export some functions via linkname to assembly in sync/atomic.
//go:linkname Load
//go:linkname Loadp
//go:linkname Load64
//go:linkname Loaduintptr
//go:linkname Xadd
//go:linkname Xadd64
//go:linkname Xadduintptr
//go:linkname Xchg
//go:linkname Xchg64
//go:linkname Xchguintptr
//go:linkname Cas
//go:linkname Cas64
//go:linkname Casuintptr
//go:linkname Store
//go:linkname Store64
//go:linkname Storeuintptr

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
func LoadAcq(ptr *uint32) uint32 {
	return *ptr
}

//go:nosplit
//go:noinline
func Load8(ptr *uint8) uint8 {
	return *ptr
}

//go:nosplit
//go:noinline
func Load64(ptr *uint64) uint64 {
	return *ptr
}

//go:nosplit
//go:noinline
func Xadd(ptr *uint32, delta int32) uint32 {
	new := *ptr + uint32(delta)
	*ptr = new
	return new
}

//go:nosplit
//go:noinline
func Xadd64(ptr *uint64, delta int64) uint64 {
	new := *ptr + uint64(delta)
	*ptr = new
	return new
}

//go:nosplit
//go:noinline
func Xadduintptr(ptr *uintptr, delta uintptr) uintptr {
	new := *ptr + delta
	*ptr = new
	return new
}

//go:nosplit
//go:noinline
func Xchg(ptr *uint32, new uint32) uint32 {
	old := *ptr
	*ptr = new
	return old
}

//go:nosplit
//go:noinline
func Xchg64(ptr *uint64, new uint64) uint64 {
	old := *ptr
	*ptr = new
	return old
}

//go:nosplit
//go:noinline
func Xchguintptr(ptr *uintptr, new uintptr) uintptr {
	old := *ptr
	*ptr = new
	return old
}

//go:nosplit
//go:noinline
func And8(ptr *uint8, val uint8) {
	*ptr = *ptr & val
}

//go:nosplit
//go:noinline
func Or8(ptr *uint8, val uint8) {
	*ptr = *ptr | val
}

// NOTE: Do not add atomicxor8 (XOR is not idempotent).

//go:nosplit
//go:noinline
func Cas64(ptr *uint64, old, new uint64) bool {
	if *ptr == old {
		*ptr = new
		return true
	}
	return false
}

//go:nosplit
//go:noinline
func Store(ptr *uint32, val uint32) {
	*ptr = val
}

//go:nosplit
//go:noinline
func StoreRel(ptr *uint32, val uint32) {
	*ptr = val
}

//go:nosplit
//go:noinline
func Store8(ptr *uint8, val uint8) {
	*ptr = val
}

//go:nosplit
//go:noinline
func Store64(ptr *uint64, val uint64) {
	*ptr = val
}

//go:notinheap
type noWB struct{}

//go:noinline
//go:nosplit
func StorepNoWB(ptr unsafe.Pointer, val unsafe.Pointer) {
	*(**noWB)(ptr) = (*noWB)(val)
}

//go:nosplit
//go:noinline
func Cas(ptr *uint32, old, new uint32) bool {
	if *ptr == old {
		*ptr = new
		return true
	}
	return false
}

//go:nosplit
//go:noinline
func Casp1(ptr *unsafe.Pointer, old, new unsafe.Pointer) bool {
	if *ptr == old {
		*ptr = new
		return true
	}
	return false
}

//go:nosplit
//go:noinline
func Casuintptr(ptr *uintptr, old, new uintptr) bool {
	if *ptr == old {
		*ptr = new
		return true
	}
	return false
}

//go:nosplit
//go:noinline
func CasRel(ptr *uint32, old, new uint32) bool {
	if *ptr == old {
		*ptr = new
		return true
	}
	return false
}

//go:nosplit
//go:noinline
func Storeuintptr(ptr *uintptr, new uintptr) {
	*ptr = new
}

//go:nosplit
//go:noinline
func Loaduintptr(ptr *uintptr) uintptr {
	return *ptr
}

//go:nosplit
//go:noinline
func Loaduint(ptr *uint) uint {
	return *ptr
}

//go:nosplit
//go:noinline
func Loadint64(ptr *int64) int64 {
	return *ptr
}

//go:nosplit
//go:noinline
func Xaddint64(ptr *int64, delta int64) int64 {
	new := *ptr + delta
	*ptr = new
	return new
}
