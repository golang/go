// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm || wasm

// Export some functions via linkname to assembly in sync/atomic.
//
//go:linkname And32
//go:linkname Or32
//go:linkname And64
//go:linkname Or64
//go:linkname Anduintptr
//go:linkname Oruintptr

package atomic

import _ "unsafe" // For linkname

//go:nosplit
func And32(ptr *uint32, val uint32) uint32 {
	for {
		old := *ptr
		if Cas(ptr, old, old&val) {
			return old
		}
	}
}

//go:nosplit
func Or32(ptr *uint32, val uint32) uint32 {
	for {
		old := *ptr
		if Cas(ptr, old, old|val) {
			return old
		}
	}
}

//go:nosplit
func And64(ptr *uint64, val uint64) uint64 {
	for {
		old := *ptr
		if Cas64(ptr, old, old&val) {
			return old
		}
	}
}

//go:nosplit
func Or64(ptr *uint64, val uint64) uint64 {
	for {
		old := *ptr
		if Cas64(ptr, old, old|val) {
			return old
		}
	}
}

//go:nosplit
func Anduintptr(ptr *uintptr, val uintptr) uintptr {
	for {
		old := *ptr
		if Casuintptr(ptr, old, old&val) {
			return old
		}
	}
}

//go:nosplit
func Oruintptr(ptr *uintptr, val uintptr) uintptr {
	for {
		old := *ptr
		if Casuintptr(ptr, old, old|val) {
			return old
		}
	}
}
