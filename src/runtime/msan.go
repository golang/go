// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build msan

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

//go:noescape
func msanread(addr unsafe.Pointer, sz uintptr)

//go:noescape
func msanwrite(addr unsafe.Pointer, sz uintptr)

//go:noescape
func msanmalloc(addr unsafe.Pointer, sz uintptr)

//go:noescape
func msanfree(addr unsafe.Pointer, sz uintptr)

// These are called from msan_amd64.s
//go:cgo_import_static __msan_read_go
//go:cgo_import_static __msan_write_go
//go:cgo_import_static __msan_malloc_go
//go:cgo_import_static __msan_free_go
