// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package cgo

import _ "unsafe" // for go:linkname

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
