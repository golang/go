// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
	"unsafe"
)

// These functions cannot have go:noescape annotations,
// because while ptr does not escape, new does.
// If new is marked as not escaping, the compiler will make incorrect
// escape analysis decisions about the pointer value being stored.
// Instead, these are wrappers around the actual atomics (casp1 and so on)
// that use noescape to convey which arguments do not escape.

// atomicstorep performs *ptr = new atomically and invokes a write barrier.
//
//go:nosplit
func atomicstorep(ptr unsafe.Pointer, new unsafe.Pointer) {
	writebarrierptr_prewrite((*uintptr)(ptr), uintptr(new))
	atomic.StorepNoWB(noescape(ptr), new)
}

//go:nosplit
func casp(ptr *unsafe.Pointer, old, new unsafe.Pointer) bool {
	// The write barrier is only necessary if the CAS succeeds,
	// but since it needs to happen before the write becomes
	// public, we have to do it conservatively all the time.
	writebarrierptr_prewrite((*uintptr)(unsafe.Pointer(ptr)), uintptr(new))
	return atomic.Casp1((*unsafe.Pointer)(noescape(unsafe.Pointer(ptr))), noescape(old), new)
}

// Like above, but implement in terms of sync/atomic's uintptr operations.
// We cannot just call the runtime routines, because the race detector expects
// to be able to intercept the sync/atomic forms but not the runtime forms.

//go:linkname sync_atomic_StoreUintptr sync/atomic.StoreUintptr
func sync_atomic_StoreUintptr(ptr *uintptr, new uintptr)

//go:linkname sync_atomic_StorePointer sync/atomic.StorePointer
//go:nosplit
func sync_atomic_StorePointer(ptr *unsafe.Pointer, new unsafe.Pointer) {
	writebarrierptr_prewrite((*uintptr)(unsafe.Pointer(ptr)), uintptr(new))
	sync_atomic_StoreUintptr((*uintptr)(unsafe.Pointer(ptr)), uintptr(new))
}

//go:linkname sync_atomic_SwapUintptr sync/atomic.SwapUintptr
func sync_atomic_SwapUintptr(ptr *uintptr, new uintptr) uintptr

//go:linkname sync_atomic_SwapPointer sync/atomic.SwapPointer
//go:nosplit
func sync_atomic_SwapPointer(ptr *unsafe.Pointer, new unsafe.Pointer) unsafe.Pointer {
	writebarrierptr_prewrite((*uintptr)(unsafe.Pointer(ptr)), uintptr(new))
	old := unsafe.Pointer(sync_atomic_SwapUintptr((*uintptr)(noescape(unsafe.Pointer(ptr))), uintptr(new)))
	return old
}

//go:linkname sync_atomic_CompareAndSwapUintptr sync/atomic.CompareAndSwapUintptr
func sync_atomic_CompareAndSwapUintptr(ptr *uintptr, old, new uintptr) bool

//go:linkname sync_atomic_CompareAndSwapPointer sync/atomic.CompareAndSwapPointer
//go:nosplit
func sync_atomic_CompareAndSwapPointer(ptr *unsafe.Pointer, old, new unsafe.Pointer) bool {
	writebarrierptr_prewrite((*uintptr)(unsafe.Pointer(ptr)), uintptr(new))
	return sync_atomic_CompareAndSwapUintptr((*uintptr)(noescape(unsafe.Pointer(ptr))), uintptr(old), uintptr(new))
}
