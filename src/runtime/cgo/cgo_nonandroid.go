// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && (darwin || dragonfly || freebsd || netbsd || openbsd || (linux && !ppc64 && !ppc64le && !android))

package cgo

//go:nosplit
//go:norace
func cgoInitTLS(tlsg, tlsbase uintptr) {}

// cgoFatal reports a fatal runtime/cgo error and aborts.
// msg must end with "\n\x00".
//
//go:nosplit
//go:norace
func cgoFatal(msg string) {
	runtime_writeErrStr(msg[:len(msg)-1])
	runtime_abort()
}
