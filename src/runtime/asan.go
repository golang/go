// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build asan

package runtime

import (
	"unsafe"
)

// Public address sanitizer API.

func ASanRead(addr unsafe.Pointer, len int) {
	asanread(addr, uintptr(len))
}

func ASanWrite(addr unsafe.Pointer, len int) {
	asanwrite(addr, uintptr(len))
}

// Private interface for the runtime.
const asanenabled = true

//go:noescape
func asanread(addr unsafe.Pointer, sz uintptr)

//go:noescape
func asanwrite(addr unsafe.Pointer, sz uintptr)

//go:noescape
func asanunpoison(addr unsafe.Pointer, sz uintptr)

//go:noescape
func asanpoison(addr unsafe.Pointer, sz uintptr)

// These are called from asan_GOARCH.s
//go:cgo_import_static __asan_read_go
//go:cgo_import_static __asan_write_go
//go:cgo_import_static __asan_unpoison_go
//go:cgo_import_static __asan_poison_go
