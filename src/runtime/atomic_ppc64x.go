// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

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

// xchgp cannot have a go:noescape annotation, because
// while ptr does not escape, new does. If new is marked as
// not escaping, the compiler will make incorrect escape analysis
// decisions about the value being xchg'ed.
// Instead, make xchgp a wrapper around the actual atomic.
// When calling the wrapper we mark ptr as noescape explicitly.

//go:nosplit
func xchgp(ptr unsafe.Pointer, new unsafe.Pointer) unsafe.Pointer {
	return xchgp1(noescape(ptr), new)
}

func xchgp1(ptr unsafe.Pointer, new unsafe.Pointer) unsafe.Pointer

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

// atomicstorep cannot have a go:noescape annotation.
// See comment above for xchgp.

//go:nosplit
func atomicstorep(ptr unsafe.Pointer, new unsafe.Pointer) {
	atomicstorep1(noescape(ptr), new)
}

func atomicstorep1(ptr unsafe.Pointer, val unsafe.Pointer)
