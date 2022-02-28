// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !wasm

package atomic

import "unsafe"

//go:noescape
func Cas(ptr *uint32, old, new uint32) bool

// NO go:noescape annotation; see atomic_pointer.go.
func Casp1(ptr *unsafe.Pointer, old, new unsafe.Pointer) bool

//go:noescape
func Casint32(ptr *int32, old, new int32) bool

//go:noescape
func Casint64(ptr *int64, old, new int64) bool

//go:noescape
func Casuintptr(ptr *uintptr, old, new uintptr) bool

//go:noescape
func Storeint32(ptr *int32, new int32)

//go:noescape
func Storeint64(ptr *int64, new int64)

//go:noescape
func Storeuintptr(ptr *uintptr, new uintptr)

//go:noescape
func Loaduintptr(ptr *uintptr) uintptr

//go:noescape
func Loaduint(ptr *uint) uint

// TODO(matloob): Should these functions have the go:noescape annotation?

//go:noescape
func Loadint32(ptr *int32) int32

//go:noescape
func Loadint64(ptr *int64) int64

//go:noescape
func Xaddint32(ptr *int32, delta int32) int32

//go:noescape
func Xaddint64(ptr *int64, delta int64) int64

//go:noescape
func Xchgint32(ptr *int32, new int32) int32

//go:noescape
func Xchgint64(ptr *int64, new int64) int64
