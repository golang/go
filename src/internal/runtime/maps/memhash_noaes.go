// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(amd64 || arm64 || 386)

package maps

import (
	"unsafe"
)

// AES hashing not implemented for these architectures
const memHashAESImplemented = false
const memHashUsesVAES = false

func memHash32AES(k uint32, h uintptr) uintptr {
	panic("memHash32AES not implemented")
}

func memHash64AES(k uint64, h uintptr) uintptr {
	panic("memHash64AES not implemented")
}

func memHashAES(p unsafe.Pointer, h, s uintptr) uintptr {
	panic("memHashAES not implemented")
}

func MemHash(p unsafe.Pointer, h, s uintptr) uintptr {
	return memHashFallback(p, h, s)
}

func MemHash32(k uint32, h uintptr) uintptr {
	return memHash32Fallback(k, h)
}

func MemHash64(k uint64, h uintptr) uintptr {
	return memHash64Fallback(k, h)
}

func StrHash(s string, h uintptr) uintptr {
	return memHashFallback(unsafe.Pointer(unsafe.StringData(s)), h, uintptr(len(s)))
}
