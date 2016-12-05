// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips mipsle

package atomic

import (
	"runtime/internal/sys"
	"unsafe"
)

// TODO implement lock striping
var lock struct {
	state uint32
	pad   [sys.CacheLineSize - 4]byte
}

//go:noescape
func spinLock(state *uint32)

//go:noescape
func spinUnlock(state *uint32)

//go:nosplit
func lockAndCheck(addr *uint64) {
	// ensure 8-byte alignement
	if uintptr(unsafe.Pointer(addr))&7 != 0 {
		addr = nil
	}
	// force dereference before taking lock
	_ = *addr

	spinLock(&lock.state)
}

//go:nosplit
func unlock() {
	spinUnlock(&lock.state)
}

//go:nosplit
func unlockNoFence() {
	lock.state = 0
}

//go:nosplit
func Xadd64(addr *uint64, delta int64) (new uint64) {
	lockAndCheck(addr)

	new = *addr + uint64(delta)
	*addr = new

	unlock()
	return
}

//go:nosplit
func Xchg64(addr *uint64, new uint64) (old uint64) {
	lockAndCheck(addr)

	old = *addr
	*addr = new

	unlock()
	return
}

//go:nosplit
func Cas64(addr *uint64, old, new uint64) (swapped bool) {
	lockAndCheck(addr)

	if (*addr) == old {
		*addr = new
		unlock()
		return true
	}

	unlockNoFence()
	return false
}

//go:nosplit
func Load64(addr *uint64) (val uint64) {
	lockAndCheck(addr)

	val = *addr

	unlock()
	return
}

//go:nosplit
func Store64(addr *uint64, val uint64) {
	lockAndCheck(addr)

	*addr = val

	unlock()
	return
}

//go:noescape
func Xadd(ptr *uint32, delta int32) uint32

//go:noescape
func Xadduintptr(ptr *uintptr, delta uintptr) uintptr

//go:noescape
func Xchg(ptr *uint32, new uint32) uint32

//go:noescape
func Xchguintptr(ptr *uintptr, new uintptr) uintptr

//go:noescape
func Load(ptr *uint32) uint32

//go:noescape
func Loadp(ptr unsafe.Pointer) unsafe.Pointer

//go:noescape
func And8(ptr *uint8, val uint8)

//go:noescape
func Or8(ptr *uint8, val uint8)

//go:noescape
func Store(ptr *uint32, val uint32)

// NO go:noescape annotation; see atomic_pointer.go.
func StorepNoWB(ptr unsafe.Pointer, val unsafe.Pointer)
