// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

package atomic

import "unsafe"

//go:noescape
func Xadd(ptr *uint32, delta int32) uint32

//go:noescape
func Xadd64(ptr *uint64, delta int64) uint64

//go:noescape
func Xadduintptr(ptr *uintptr, delta uintptr) uintptr

//go:noescape
func Xchg8(ptr *uint8, new uint8) uint8

//go:noescape
func Xchg(ptr *uint32, new uint32) uint32

//go:noescape
func Xchg64(ptr *uint64, new uint64) uint64

//go:noescape
func Xchguintptr(ptr *uintptr, new uintptr) uintptr

//go:noescape
func Load(ptr *uint32) uint32

//go:noescape
func Load8(ptr *uint8) uint8

//go:noescape
func Load64(ptr *uint64) uint64

// NO go:noescape annotation; *ptr escapes if result escapes (#31525)
func Loadp(ptr unsafe.Pointer) unsafe.Pointer

//go:noescape
func LoadAcq(ptr *uint32) uint32

//go:noescape
func LoadAcq64(ptr *uint64) uint64

//go:noescape
func LoadAcquintptr(ptr *uintptr) uintptr

//go:noescape
func And8(ptr *uint8, val uint8)

//go:noescape
func Or8(ptr *uint8, val uint8)

// NOTE: Do not add atomicxor8 (XOR is not idempotent).

//go:noescape
func And(ptr *uint32, val uint32)

//go:noescape
func Or(ptr *uint32, val uint32)

//go:noescape
func And32(ptr *uint32, val uint32) uint32

//go:noescape
func Or32(ptr *uint32, val uint32) uint32

//go:noescape
func And64(ptr *uint64, val uint64) uint64

//go:noescape
func Or64(ptr *uint64, val uint64) uint64

//go:noescape
func Anduintptr(ptr *uintptr, val uintptr) uintptr

//go:noescape
func Oruintptr(ptr *uintptr, val uintptr) uintptr

//go:noescape
func Cas64(ptr *uint64, old, new uint64) bool

//go:noescape
func CasRel(ptr *uint32, old, new uint32) bool

//go:noescape
func Store(ptr *uint32, val uint32)

//go:noescape
func Store8(ptr *uint8, val uint8)

//go:noescape
func Store64(ptr *uint64, val uint64)

//go:noescape
func StoreRel(ptr *uint32, val uint32)

//go:noescape
func StoreRel64(ptr *uint64, val uint64)

//go:noescape
func StoreReluintptr(ptr *uintptr, val uintptr)

// NO go:noescape annotation; see atomic_pointer.go.
func StorepNoWB(ptr unsafe.Pointer, val unsafe.Pointer)
