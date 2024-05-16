// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build msan

package runtime

import (
	"unsafe"
)

// Public memory sanitizer API.

func MSanRead(addr unsafe.Pointer, len int) {
	msanread(addr, uintptr(len))
}

func MSanWrite(addr unsafe.Pointer, len int) {
	msanwrite(addr, uintptr(len))
}

// Private interface for the runtime.
const msanenabled = true

// If we are running on the system stack, the C program may have
// marked part of that stack as uninitialized. We don't instrument
// the runtime, but operations like a slice copy can call msanread
// anyhow for values on the stack. Just ignore msanread when running
// on the system stack. The other msan functions are fine.
//
//go:linkname msanread
//go:nosplit
func msanread(addr unsafe.Pointer, sz uintptr) {
	gp := getg()
	if gp == nil || gp.m == nil || gp == gp.m.g0 || gp == gp.m.gsignal {
		return
	}
	domsanread(addr, sz)
}

//go:noescape
func domsanread(addr unsafe.Pointer, sz uintptr)

//go:linkname msanwrite
//go:noescape
func msanwrite(addr unsafe.Pointer, sz uintptr)

//go:linkname msanmalloc
//go:noescape
func msanmalloc(addr unsafe.Pointer, sz uintptr)

//go:linkname msanfree
//go:noescape
func msanfree(addr unsafe.Pointer, sz uintptr)

//go:linkname msanmove
//go:noescape
func msanmove(dst, src unsafe.Pointer, sz uintptr)

// These are called from msan_GOARCH.s
//
//go:cgo_import_static __msan_read_go
//go:cgo_import_static __msan_write_go
//go:cgo_import_static __msan_malloc_go
//go:cgo_import_static __msan_free_go
//go:cgo_import_static __msan_memmove
