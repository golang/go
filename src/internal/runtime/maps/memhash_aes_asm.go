// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (amd64 && !goexperiment.simd) || arm64 || 386

package maps

import (
	"unsafe"
)

const memHashUsesVAES = false

// stabs for assembly implementations
//
//go:noescape
func memHashAES(p unsafe.Pointer, h, s uintptr) uintptr

//go:noescape
func memHash32AES(k uint32, h uintptr) uintptr

//go:noescape
func memHash64AES(k uint64, h uintptr) uintptr
