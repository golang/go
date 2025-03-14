// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atomic

import (
	"internal/goarch"
	"unsafe"
)

//go:nosplit
func goXchg8(addr *uint8, v uint8) uint8 {
	// Align down to 4 bytes and use 32-bit CAS.
	addr32 := (*uint32)(unsafe.Pointer(uintptr(unsafe.Pointer(addr)) &^ 3))
	shift := (uintptr(unsafe.Pointer(addr)) & 3)
	if goarch.BigEndian {
		shift = shift ^ 3
	}
	shift = shift * 8
	word := uint32(v) << shift
	mask := uint32(0xFF) << shift

	for {
		old := *addr32 // Read the old 32-bit value
		// Clear the old 8 bits then insert the new value
		if Cas(addr32, old, (old&^mask)|word) {
			// Return the old 8-bit value
			return uint8((old & mask) >> shift)
		}
	}
}
