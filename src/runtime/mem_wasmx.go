// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasm || wasm32

package runtime

import "unsafe"

func sbrk(n uintptr) unsafe.Pointer {
	bl := bloc
	if n > ^uintptr(0)-(physPageSize-1) {
		return nil
	}
	n = memRound(n)
	if n > ^uintptr(0)-bl {
		return nil
	}
	if bl+n > blocMax {
		grow := (bl + n - blocMax) / physPageSize
		size := growMemory(int32(grow))
		if size < 0 {
			return nil
		}
		resetMemoryDataView()
		blocMax = bl + n
	}
	bloc += n
	return unsafe.Pointer(bl)
}

// Implemented in src/runtime/sys_wasm.s and src/runtime/sys_wasm32.s.
func growMemory(pages int32) int32

// Implemented in src/runtime/sys_wasmx.s.
func currentMemory() int32
