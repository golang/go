// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

var locktab [57]struct {
	l   mutex
	pad [_CacheLineSize - unsafe.Sizeof(mutex{})]byte
}

func addrLock(addr *uint64) *mutex {
	return &locktab[(uintptr(unsafe.Pointer(addr))>>3)%uintptr(len(locktab))].l
}

// Atomic add and return new value.
//go:nosplit
func xadd(val *uint32, delta int32) uint32 {
	for {
		oval := *val
		nval := oval + uint32(delta)
		if cas(val, oval, nval) {
			return nval
		}
	}
}

//go:nosplit
func xchg(addr *uint32, v uint32) uint32 {
	for {
		old := *addr
		if cas(addr, old, v) {
			return old
		}
	}
}

//go:nosplit
func xchgp1(addr unsafe.Pointer, v unsafe.Pointer) unsafe.Pointer {
	for {
		old := *(*unsafe.Pointer)(addr)
		if casp1((*unsafe.Pointer)(addr), old, v) {
			return old
		}
	}
}

//go:nosplit
func xchguintptr(addr *uintptr, v uintptr) uintptr {
	return uintptr(xchg((*uint32)(unsafe.Pointer(addr)), uint32(v)))
}

//go:nosplit
func atomicload(addr *uint32) uint32 {
	return xadd(addr, 0)
}

//go:nosplit
func atomicloadp(addr unsafe.Pointer) unsafe.Pointer {
	return unsafe.Pointer(uintptr(xadd((*uint32)(addr), 0)))
}

//go:nosplit
func atomicstorep1(addr unsafe.Pointer, v unsafe.Pointer) {
	for {
		old := *(*unsafe.Pointer)(addr)
		if casp1((*unsafe.Pointer)(addr), old, v) {
			return
		}
	}
}

//go:nosplit
func atomicstore(addr *uint32, v uint32) {
	for {
		old := *addr
		if cas(addr, old, v) {
			return
		}
	}
}

//go:nosplit
func cas64(addr *uint64, old, new uint64) bool {
	var ok bool
	systemstack(func() {
		lock(addrLock(addr))
		if *addr == old {
			*addr = new
			ok = true
		}
		unlock(addrLock(addr))
	})
	return ok
}

//go:nosplit
func xadd64(addr *uint64, delta int64) uint64 {
	var r uint64
	systemstack(func() {
		lock(addrLock(addr))
		r = *addr + uint64(delta)
		*addr = r
		unlock(addrLock(addr))
	})
	return r
}

//go:nosplit
func xchg64(addr *uint64, v uint64) uint64 {
	var r uint64
	systemstack(func() {
		lock(addrLock(addr))
		r = *addr
		*addr = v
		unlock(addrLock(addr))
	})
	return r
}

//go:nosplit
func atomicload64(addr *uint64) uint64 {
	var r uint64
	systemstack(func() {
		lock(addrLock(addr))
		r = *addr
		unlock(addrLock(addr))
	})
	return r
}

//go:nosplit
func atomicstore64(addr *uint64, v uint64) {
	systemstack(func() {
		lock(addrLock(addr))
		*addr = v
		unlock(addrLock(addr))
	})
}

//go:nosplit
func atomicor8(addr *uint8, v uint8) {
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
