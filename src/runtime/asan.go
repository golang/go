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
	sp := getcallersp()
	pc := getcallerpc()
	doasanread(addr, uintptr(len), sp, pc)
}

func ASanWrite(addr unsafe.Pointer, len int) {
	sp := getcallersp()
	pc := getcallerpc()
	doasanwrite(addr, uintptr(len), sp, pc)
}

// Private interface for the runtime.
const asanenabled = true

// asan{read,write} are nosplit because they may be called between
// fork and exec, when the stack must not grow. See issue #50391.

//go:nosplit
func asanread(addr unsafe.Pointer, sz uintptr) {
	sp := getcallersp()
	pc := getcallerpc()
	doasanread(addr, sz, sp, pc)
}

//go:nosplit
func asanwrite(addr unsafe.Pointer, sz uintptr) {
	sp := getcallersp()
	pc := getcallerpc()
	doasanwrite(addr, sz, sp, pc)
}

//go:noescape
func doasanread(addr unsafe.Pointer, sz, sp, pc uintptr)

//go:noescape
func doasanwrite(addr unsafe.Pointer, sz, sp, pc uintptr)

//go:noescape
func asanunpoison(addr unsafe.Pointer, sz uintptr)

//go:noescape
func asanpoison(addr unsafe.Pointer, sz uintptr)

//go:noescape
func asanregisterglobals(addr unsafe.Pointer, n uintptr)

// These are called from asan_GOARCH.s
//
//go:cgo_import_static __asan_read_go
//go:cgo_import_static __asan_write_go
//go:cgo_import_static __asan_unpoison_go
//go:cgo_import_static __asan_poison_go
//go:cgo_import_static __asan_register_globals_go
