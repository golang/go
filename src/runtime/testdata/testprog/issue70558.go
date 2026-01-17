// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

func init() {
	register("VirtualAllocFailure", VirtualAllocFailure)
}

func VirtualAllocFailure() {
	// Allocate a size that is guaranteed to fail VirtualAlloc on Windows
	// immediately, without trying to expand the pagefile.
	// https://learn.microsoft.com/en-us/windows/win32/memory/memory-limits-for-windows-releases
	var size int64
	if unsafe.Sizeof(int(0)) == 8 {
		// On 64-bit Windows, the user address space is 128 TB.
		size = (1 << 47) + (1 << 46) // 192 TB
	} else {
		// On 32-bit Windows, the user address space is 2 GB.
		size = 1<<31 - 1 // ~2 GB
	}

	b := make([]byte, size)
	println(b[0])
}
