// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm

package runtime

import "unsafe"

func sbrk(n uintptr) unsafe.Pointer {
	grow := divRoundUp(n, physPageSize)
	size := currentMemory()

	if growMemory(int32(grow)) < 0 {
		return nil
	}

	resetMemoryDataView()
	return unsafe.Pointer(uintptr(size) * physPageSize)
}

func currentMemory() int32
func growMemory(pages int32) int32

// resetMemoryDataView signals the JS front-end that WebAssembly's memory.grow instruction has been used.
// This allows the front-end to replace the old DataView object with a new one.
//
//go:wasmimport gojs runtime.resetMemoryDataView
func resetMemoryDataView()
