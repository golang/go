// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package cgo

import "unsafe"

// _cgo_stub_export is only used to ensure there's at least one symbol
// in the .def file passed to the external linker.
// If there are no exported symbols, the unfortunate behavior of
// the binutils linker is to also strip the relocations table,
// resulting in non-PIE binary. The other option is the
// --export-all-symbols flag, but we don't need to export all symbols
// and this may overflow the export table (#40795).
// See https://sourceware.org/bugzilla/show_bug.cgi?id=19011
//
//go:cgo_export_static _cgo_stub_export
//go:linkname _cgo_stub_export _cgo_stub_export
var _cgo_stub_export uintptr

// No pthreads on Windows, these are always zero.

//go:linkname _cgo_init _cgo_init
var _cgo_init unsafe.Pointer

//go:linkname _cgo_thread_start _cgo_thread_start
var _cgo_thread_start unsafe.Pointer

//go:linkname _cgo_sys_thread_create _cgo_sys_thread_create
var _cgo_sys_thread_create unsafe.Pointer

//go:linkname _cgo_bindm _cgo_bindm
var _cgo_bindm unsafe.Pointer

//go:linkname _cgo_getstackbound _cgo_getstackbound
var _cgo_getstackbound unsafe.Pointer
