// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goexperiment"
	"internal/runtime/atomic"
	"unsafe"
)

// These functions cannot have go:noescape annotations,
// because while ptr does not escape, new does.
// If new is marked as not escaping, the compiler will make incorrect
// escape analysis decisions about the pointer value being stored.

// atomicwb performs a write barrier before an atomic pointer write.
// The caller should guard the call with "if writeBarrier.enabled".
//
// atomicwb should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/bytedance/gopkg
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname atomicwb
//go:nosplit
func atomicwb(ptr *unsafe.Pointer, new unsafe.Pointer) {
	slot := (*uintptr)(unsafe.Pointer(ptr))
	buf := getg().m.p.ptr().wbBuf.get2()
	buf[0] = *slot
	buf[1] = uintptr(new)
}

// atomicstorep performs *ptr = new atomically and invokes a write barrier.
//
//go:nosplit
func atomicstorep(ptr unsafe.Pointer, new unsafe.Pointer) {
	if writeBarrier.enabled {
		atomicwb((*unsafe.Pointer)(ptr), new)
	}
	if goexperiment.CgoCheck2 {
		cgoCheckPtrWrite((*unsafe.Pointer)(ptr), new)
	}
	atomic.StorepNoWB(noescape(ptr), new)
}

// atomic_storePointer is the implementation of runtime/internal/UnsafePointer.Store
// (like StoreNoWB but with the write barrier).
//
//go:nosplit
//go:linkname atomic_storePointer internal/runtime/atomic.storePointer
func atomic_storePointer(ptr *unsafe.Pointer, new unsafe.Pointer) {
	atomicstorep(unsafe.Pointer(ptr), new)
}

// atomic_casPointer is the implementation of runtime/internal/UnsafePointer.CompareAndSwap
// (like CompareAndSwapNoWB but with the write barrier).
//
//go:nosplit
//go:linkname atomic_casPointer internal/runtime/atomic.casPointer
func atomic_casPointer(ptr *unsafe.Pointer, old, new unsafe.Pointer) bool {
	if writeBarrier.enabled {
		atomicwb(ptr, new)
	}
	if goexperiment.CgoCheck2 {
		cgoCheckPtrWrite(ptr, new)
	}
	return atomic.Casp1(ptr, old, new)
}

// Like above, but implement in terms of sync/atomic's uintptr operations.
// We cannot just call the runtime routines, because the race detector expects
// to be able to intercept the sync/atomic forms but not the runtime forms.

//go:linkname sync_atomic_StoreUintptr sync/atomic.StoreUintptr
func sync_atomic_StoreUintptr(ptr *uintptr, new uintptr)

//go:linkname sync_atomic_StorePointer sync/atomic.StorePointer
//go:nosplit
func sync_atomic_StorePointer(ptr *unsafe.Pointer, new unsafe.Pointer) {
	if writeBarrier.enabled {
		atomicwb(ptr, new)
	}
	if goexperiment.CgoCheck2 {
		cgoCheckPtrWrite(ptr, new)
	}
	sync_atomic_StoreUintptr((*uintptr)(unsafe.Pointer(ptr)), uintptr(new))
}

//go:linkname sync_atomic_SwapUintptr sync/atomic.SwapUintptr
func sync_atomic_SwapUintptr(ptr *uintptr, new uintptr) uintptr

//go:linkname sync_atomic_SwapPointer sync/atomic.SwapPointer
//go:nosplit
func sync_atomic_SwapPointer(ptr *unsafe.Pointer, new unsafe.Pointer) unsafe.Pointer {
	if writeBarrier.enabled {
		atomicwb(ptr, new)
	}
	if goexperiment.CgoCheck2 {
		cgoCheckPtrWrite(ptr, new)
	}
	old := unsafe.Pointer(sync_atomic_SwapUintptr((*uintptr)(noescape(unsafe.Pointer(ptr))), uintptr(new)))
	return old
}

//go:linkname sync_atomic_CompareAndSwapUintptr sync/atomic.CompareAndSwapUintptr
func sync_atomic_CompareAndSwapUintptr(ptr *uintptr, old, new uintptr) bool

//go:linkname sync_atomic_CompareAndSwapPointer sync/atomic.CompareAndSwapPointer
//go:nosplit
func sync_atomic_CompareAndSwapPointer(ptr *unsafe.Pointer, old, new unsafe.Pointer) bool {
	if writeBarrier.enabled {
		atomicwb(ptr, new)
	}
	if goexperiment.CgoCheck2 {
		cgoCheckPtrWrite(ptr, new)
	}
	return sync_atomic_CompareAndSwapUintptr((*uintptr)(noescape(unsafe.Pointer(ptr))), uintptr(old), uintptr(new))
}
