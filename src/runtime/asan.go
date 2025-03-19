// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build asan

package runtime

import (
	"internal/runtime/sys"
	"unsafe"
)

// Public address sanitizer API.
func ASanRead(addr unsafe.Pointer, len int) {
	sp := sys.GetCallerSP()
	pc := sys.GetCallerPC()
	doasanread(addr, uintptr(len), sp, pc)
}

func ASanWrite(addr unsafe.Pointer, len int) {
	sp := sys.GetCallerSP()
	pc := sys.GetCallerPC()
	doasanwrite(addr, uintptr(len), sp, pc)
}

// Private interface for the runtime.
const asanenabled = true

// asan{read,write} are nosplit because they may be called between
// fork and exec, when the stack must not grow. See issue #50391.

//go:linkname asanread
//go:nosplit
func asanread(addr unsafe.Pointer, sz uintptr) {
	sp := sys.GetCallerSP()
	pc := sys.GetCallerPC()
	doasanread(addr, sz, sp, pc)
}

//go:linkname asanwrite
//go:nosplit
func asanwrite(addr unsafe.Pointer, sz uintptr) {
	sp := sys.GetCallerSP()
	pc := sys.GetCallerPC()
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

//go:noescape
func lsanregisterrootregion(addr unsafe.Pointer, n uintptr)

//go:noescape
func lsanunregisterrootregion(addr unsafe.Pointer, n uintptr)

func lsandoleakcheck()

// These are called from asan_GOARCH.s
//
//go:cgo_import_static __asan_read_go
//go:cgo_import_static __asan_write_go
//go:cgo_import_static __asan_unpoison_go
//go:cgo_import_static __asan_poison_go
//go:cgo_import_static __asan_register_globals_go
//go:cgo_import_static __lsan_register_root_region_go
//go:cgo_import_static __lsan_unregister_root_region_go
//go:cgo_import_static __lsan_do_leak_check_go
