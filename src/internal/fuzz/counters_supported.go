// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (darwin || linux || windows || freebsd) && (amd64 || arm64)

package fuzz

import (
	"unsafe"
)

// coverage returns a []byte containing unique 8-bit counters for each edge of
// the instrumented source code. This coverage data will only be generated if
// `-d=libfuzzer` is set at build time. This can be used to understand the code
// coverage of a test execution.
func coverage() []byte {
	addr := unsafe.Pointer(&_counters)
	size := uintptr(unsafe.Pointer(&_ecounters)) - uintptr(addr)
	return unsafe.Slice((*byte)(addr), int(size))
}
