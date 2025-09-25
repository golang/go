// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"internal/abi"
	"runtime"
	"unsafe"
)

func init() {
	register("SetCgoTracebackNoCgo", SetCgoTracebackNoCgo)
}

func cgoTraceback() {
	panic("unexpectedly reached cgo traceback function")
}

func cgoContext() {
	panic("unexpectedly reached cgo context function")
}

func cgoSymbolizer() {
	panic("unexpectedly reached cgo symbolizer function")
}

// SetCgoTraceback is a no-op in non-cgo binaries.
func SetCgoTracebackNoCgo() {
	traceback := unsafe.Pointer(abi.FuncPCABIInternal(cgoTraceback))
	context := unsafe.Pointer(abi.FuncPCABIInternal(cgoContext))
	symbolizer := unsafe.Pointer(abi.FuncPCABIInternal(cgoSymbolizer))
	runtime.SetCgoTraceback(0, traceback, context, symbolizer)

	// In a cgo binary, runtime.(*Frames).Next calls the cgo symbolizer for
	// any non-Go frames. Pass in a bogus frame to verify that Next does
	// not attempt to call the cgo symbolizer, which would crash in a
	// non-cgo binary like this one.
	frames := runtime.CallersFrames([]uintptr{0x12345678})
	frames.Next()

	fmt.Println("OK")
}
