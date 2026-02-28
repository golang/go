// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func sbrk(n uintptr) unsafe.Pointer {
	bl := bloc
	n = memRound(n)
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

// Implemented in src/runtime/sys_wasm.s
func growMemory(pages int32) int32
func currentMemory() int32
