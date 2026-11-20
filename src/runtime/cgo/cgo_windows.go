// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && windows

package cgo

import "unsafe"

//go:cgo_import_dynamic _cgo_SwitchToThread SwitchToThread%0 "kernel32.dll"
//go:linkname _cgo_SwitchToThread _cgo_SwitchToThread
var _cgo_SwitchToThread unsafe.Pointer

//go:nosplit
//go:norace
func cgoYieldWaitRuntimeInit() {
	asmcgocallNoG(uintptr(_cgo_SwitchToThread), 0, 0, 0, 0, 0, 0)
}

//go:norace
func cgoNotifyRuntimeInitDoneOS() {}
