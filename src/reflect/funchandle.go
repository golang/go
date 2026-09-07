// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import "internal/goarch"

// These functions are implemented by runtime on wasm32. Calls must be
// guarded by goarch.IsWasm32 so other architectures do not reference them.
func wasm32FuncHandleToPC(handle uintptr) uintptr
func wasm32FuncPCToHandle(pc uintptr) uintptr

// funcValuePC returns the entry PC for a function value's code word.
func funcValuePC(fn uintptr) uintptr {
	if goarch.IsWasm32 != 0 {
		return wasm32FuncHandleToPC(fn)
	}
	return fn
}

// funcHandle returns the code word used to call the function at pc.
func funcHandle(pc uintptr) uintptr {
	if goarch.IsWasm32 != 0 {
		return wasm32FuncPCToHandle(pc)
	}
	return pc
}
