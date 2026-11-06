// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && unix && !aix

package cgo

import (
	"internal/abi"
	"unsafe"
)

// cgoCallTrampoline, pthreadKeyDestructor, and x_cgo_threadentry are C-callable
// assembly entrypoints.
func cgoCallTrampoline()
func pthreadKeyDestructor()
func x_cgo_threadentry()

//go:nosplit
//go:norace
func cgoCallTrampolinePC() unsafe.Pointer {
	return unsafe.Pointer(abi.FuncPCABI0(cgoCallTrampoline))
}

//go:nosplit
//go:norace
func pthreadKeyDestructorPC() uintptr {
	return abi.FuncPCABI0(pthreadKeyDestructor)
}

//go:nosplit
//go:norace
func threadentryPC() uintptr {
	return abi.FuncPCABI0(x_cgo_threadentry)
}
