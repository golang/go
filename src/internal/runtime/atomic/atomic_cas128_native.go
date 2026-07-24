// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !arm64

package atomic

import (
	"internal/cpu"
	"unsafe"
)

// goCas128 is the Go-side lock-table fallback for Cas128
//
//go:nosplit
func goCas128(ptr *[2]uint64, old1, old2, new1, new2 uint64) bool {
	if uintptr(unsafe.Pointer(ptr))&15 != 0 {
		panicUnaligned128()
	}
	_ = *ptr // fault on nil before taking the lock
	l := pairAddrLock(ptr)
	l.lock()
	ok := false
	if ptr[0] == old1 && ptr[1] == old2 {
		ptr[0] = new1
		ptr[1] = new2
		ok = true
	}
	l.unlock()
	return ok
}

type pairSpinlock struct {
	v uint32
}

//go:nosplit
func (l *pairSpinlock) lock() {
	for !Cas(&l.v, 0, 1) {
	}
}

//go:nosplit
func (l *pairSpinlock) unlock() {
	Store(&l.v, 0)
}

// pairLocktab is a small open-addressed table of spinlocks keyed by the
// 16-byte-aligned address of the pair. Non-power-of-2 size to keep buckets
// from aliasing common allocator strides.
var pairLocktab [57]struct {
	l   pairSpinlock
	pad [cpu.CacheLinePadSize - unsafe.Sizeof(pairSpinlock{})]byte
}

//go:nosplit
func pairAddrLock(addr *[2]uint64) *pairSpinlock {
	return &pairLocktab[(uintptr(unsafe.Pointer(addr))>>4)%uintptr(len(pairLocktab))].l
}
