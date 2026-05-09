// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm64 || 386

package maps

import (
	"unsafe"
)

const memHashAESImplemented = true

func MemHash(p unsafe.Pointer, h, s uintptr) uintptr {
	if UseAeshash {
		return memHashAES(p, h, s)
	}
	return memHashFallback(p, h, s)
}

func MemHash32(k uint32, h uintptr) uintptr {
	if UseAeshash {
		return memHash32AES(k, h)
	}
	return memHash32Fallback(k, h)
}

func MemHash64(k uint64, h uintptr) uintptr {
	if UseAeshash {
		return memHash64AES(k, h)
	}
	return memHash64Fallback(k, h)
}

func StrHash(s string, h uintptr) uintptr {
	return MemHash(unsafe.Pointer(unsafe.StringData(s)), h, uintptr(len(s)))
}
