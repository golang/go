// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,amd64

package cgo

// Import "unsafe" because we use go:linkname.
import _ "unsafe"

// When using cgo, call the C library for mmap, so that we call into
// any sanitizer interceptors.  This supports using the memory
// sanitizer with Go programs.  The memory sanitizer only applies to
// C/C++ code; this permits that code to see the Go code as normal
// program addresses that have been initialized.

//go:cgo_import_static x_cgo_mmap
//go:linkname x_cgo_mmap x_cgo_mmap
//go:linkname _cgo_mmap _cgo_mmap
var x_cgo_mmap byte
var _cgo_mmap = &x_cgo_mmap
