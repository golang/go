// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package runtime

import "unsafe"

var _cgo_clearenv unsafe.Pointer // pointer to C function

// Clear the C environment if cgo is loaded.
func clearenv_c() {
	if _cgo_clearenv == nil {
		return
	}
	asmcgocall(_cgo_clearenv, nil)
}

//go:linkname syscall_runtimeClearenv syscall.runtimeClearenv
func syscall_runtimeClearenv(env map[string]int) {
	clearenv_c()
	// Did we just unset GODEBUG?
	if _, ok := env["GODEBUG"]; ok {
		godebugEnv.Store(nil)
		godebugNotify(true)
	}
}
