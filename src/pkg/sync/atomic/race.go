// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race

package atomic

import (
	"runtime"
	"unsafe"
)

// We use runtime.RaceRead() inside of atomic operations to catch races
// between atomic and non-atomic operations.  It will also catch races
// between Mutex.Lock() and mutex overwrite (mu = Mutex{}).  Since we use
// only RaceRead() we won't catch races with non-atomic loads.
// Otherwise (if we use RaceWrite()) we will report races
// between atomic operations (false positives).

var mtx uint32 = 1 // same for all

func CompareAndSwapInt32(val *int32, old, new int32) bool {
	return CompareAndSwapUint32((*uint32)(unsafe.Pointer(val)), uint32(old), uint32(new))
}

func CompareAndSwapUint32(val *uint32, old, new uint32) (swapped bool) {
	_ = *val
	swapped = false
	runtime.RaceSemacquire(&mtx)
	runtime.RaceRead(unsafe.Pointer(val))
	runtime.RaceAcquire(unsafe.Pointer(val))
	if *val == old {
		*val = new
		swapped = true
		runtime.RaceReleaseMerge(unsafe.Pointer(val))
	}
	runtime.RaceSemrelease(&mtx)
	return
}

func CompareAndSwapInt64(val *int64, old, new int64) bool {
	return CompareAndSwapUint64((*uint64)(unsafe.Pointer(val)), uint64(old), uint64(new))
}

func CompareAndSwapUint64(val *uint64, old, new uint64) (swapped bool) {
	_ = *val
	swapped = false
	runtime.RaceSemacquire(&mtx)
	runtime.RaceRead(unsafe.Pointer(val))
	runtime.RaceAcquire(unsafe.Pointer(val))
	if *val == old {
		*val = new
		swapped = true
		runtime.RaceReleaseMerge(unsafe.Pointer(val))
	}
	runtime.RaceSemrelease(&mtx)
	return
}

func CompareAndSwapPointer(val *unsafe.Pointer, old, new unsafe.Pointer) (swapped bool) {
	_ = *val
	swapped = false
	runtime.RaceSemacquire(&mtx)
	runtime.RaceRead(unsafe.Pointer(val))
	runtime.RaceAcquire(unsafe.Pointer(val))
	if *val == old {
		*val = new
		swapped = true
		runtime.RaceReleaseMerge(unsafe.Pointer(val))
	}
	runtime.RaceSemrelease(&mtx)
	return
}

func CompareAndSwapUintptr(val *uintptr, old, new uintptr) (swapped bool) {
	_ = *val
	swapped = false
	runtime.RaceSemacquire(&mtx)
	runtime.RaceRead(unsafe.Pointer(val))
	runtime.RaceAcquire(unsafe.Pointer(val))
	if *val == old {
		*val = new
		swapped = true
		runtime.RaceReleaseMerge(unsafe.Pointer(val))
	}
	runtime.RaceSemrelease(&mtx)
	return
}

func AddInt32(val *int32, delta int32) int32 {
	return int32(AddUint32((*uint32)(unsafe.Pointer(val)), uint32(delta)))
}

func AddUint32(val *uint32, delta uint32) (new uint32) {
	_ = *val
	runtime.RaceSemacquire(&mtx)
	runtime.RaceRead(unsafe.Pointer(val))
	runtime.RaceAcquire(unsafe.Pointer(val))
	*val = *val + delta
	new = *val
	runtime.RaceReleaseMerge(unsafe.Pointer(val))
	runtime.RaceSemrelease(&mtx)

	return
}

func AddInt64(val *int64, delta int64) int64 {
	return int64(AddUint64((*uint64)(unsafe.Pointer(val)), uint64(delta)))
}

func AddUint64(val *uint64, delta uint64) (new uint64) {
	_ = *val
	runtime.RaceSemacquire(&mtx)
	runtime.RaceRead(unsafe.Pointer(val))
	runtime.RaceAcquire(unsafe.Pointer(val))
	*val = *val + delta
	new = *val
	runtime.RaceReleaseMerge(unsafe.Pointer(val))
	runtime.RaceSemrelease(&mtx)

	return
}

func AddUintptr(val *uintptr, delta uintptr) (new uintptr) {
	_ = *val
	runtime.RaceSemacquire(&mtx)
	runtime.RaceRead(unsafe.Pointer(val))
	runtime.RaceAcquire(unsafe.Pointer(val))
	*val = *val + delta
	new = *val
	runtime.RaceReleaseMerge(unsafe.Pointer(val))
	runtime.RaceSemrelease(&mtx)

	return
}

func LoadInt32(addr *int32) int32 {
	return int32(LoadUint32((*uint32)(unsafe.Pointer(addr))))
}

func LoadUint32(addr *uint32) (val uint32) {
	_ = *addr
	runtime.RaceSemacquire(&mtx)
	runtime.RaceRead(unsafe.Pointer(addr))
	runtime.RaceAcquire(unsafe.Pointer(addr))
	val = *addr
	runtime.RaceSemrelease(&mtx)
	return
}

func LoadInt64(addr *int64) int64 {
	return int64(LoadUint64((*uint64)(unsafe.Pointer(addr))))
}

func LoadUint64(addr *uint64) (val uint64) {
	_ = *addr
	runtime.RaceSemacquire(&mtx)
	runtime.RaceRead(unsafe.Pointer(addr))
	runtime.RaceAcquire(unsafe.Pointer(addr))
	val = *addr
	runtime.RaceSemrelease(&mtx)
	return
}

func LoadPointer(addr *unsafe.Pointer) (val unsafe.Pointer) {
	_ = *addr
	runtime.RaceSemacquire(&mtx)
	runtime.RaceRead(unsafe.Pointer(addr))
	runtime.RaceAcquire(unsafe.Pointer(addr))
	val = *addr
	runtime.RaceSemrelease(&mtx)
	return
}

func LoadUintptr(addr *uintptr) (val uintptr) {
	_ = *addr
	runtime.RaceSemacquire(&mtx)
	runtime.RaceRead(unsafe.Pointer(addr))
	runtime.RaceAcquire(unsafe.Pointer(addr))
	val = *addr
	runtime.RaceSemrelease(&mtx)
	return
}

func StoreInt32(addr *int32, val int32) {
	StoreUint32((*uint32)(unsafe.Pointer(addr)), uint32(val))
}

func StoreUint32(addr *uint32, val uint32) {
	_ = *addr
	runtime.RaceSemacquire(&mtx)
	runtime.RaceRead(unsafe.Pointer(addr))
	*addr = val
	runtime.RaceRelease(unsafe.Pointer(addr))
	runtime.RaceSemrelease(&mtx)
}

func StoreInt64(addr *int64, val int64) {
	StoreUint64((*uint64)(unsafe.Pointer(addr)), uint64(val))
}

func StoreUint64(addr *uint64, val uint64) {
	_ = *addr
	runtime.RaceSemacquire(&mtx)
	runtime.RaceRead(unsafe.Pointer(addr))
	*addr = val
	runtime.RaceRelease(unsafe.Pointer(addr))
	runtime.RaceSemrelease(&mtx)
}

func StorePointer(addr *unsafe.Pointer, val unsafe.Pointer) {
	_ = *addr
	runtime.RaceSemacquire(&mtx)
	runtime.RaceRead(unsafe.Pointer(addr))
	*addr = val
	runtime.RaceRelease(unsafe.Pointer(addr))
	runtime.RaceSemrelease(&mtx)
}

func StoreUintptr(addr *uintptr, val uintptr) {
	_ = *addr
	runtime.RaceSemacquire(&mtx)
	runtime.RaceRead(unsafe.Pointer(addr))
	*addr = val
	runtime.RaceRelease(unsafe.Pointer(addr))
	runtime.RaceSemrelease(&mtx)
}
