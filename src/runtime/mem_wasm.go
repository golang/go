// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func sbrk(n uintptr) unsafe.Pointer {
	grow := divRoundUp(n, physPageSize)
	size := growMemory(int32(grow))
	if size < 0 {
		return nil
	}
	resetMemoryDataView()
	return unsafe.Pointer(uintptr(size) * physPageSize)
}

// Implemented in src/runtime/sys_wasm.s
func growMemory(pages int32) int32
