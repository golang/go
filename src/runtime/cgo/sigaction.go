// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,amd64 freebsd,amd64 linux,arm64

package cgo

// Import "unsafe" because we use go:linkname.
import _ "unsafe"

// When using cgo, call the C library for sigaction, so that we call into
// any sanitizer interceptors. This supports using the memory
// sanitizer with Go programs. The memory sanitizer only applies to
// C/C++ code; this permits that code to see the Go runtime's existing signal
// handlers when registering new signal handlers for the process.

//go:cgo_import_static x_cgo_sigaction
//go:linkname x_cgo_sigaction x_cgo_sigaction
//go:linkname _cgo_sigaction _cgo_sigaction
var x_cgo_sigaction byte
var _cgo_sigaction = &x_cgo_sigaction
