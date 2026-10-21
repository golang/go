// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly

package runtime

import "unsafe"

// From DragonFly's <sys/sysctl.h>.
const logosHWPhysmem = 5

// logosPhysicalMemory returns hw.physmem in bytes, or 0 when sysctl fails.
// The length sysctl returns says whether the kernel stored 4 or 8 bytes;
// DragonFly is little-endian only, so a 4-byte value sits in the low half of
// mem.
func logosPhysicalMemory() uint64 {
	mib := [2]uint32{_CTL_HW, logosHWPhysmem}
	var mem uint64
	n := unsafe.Sizeof(mem)
	if sysctl(&mib[0], 2, (*byte)(unsafe.Pointer(&mem)), &n, nil, 0) < 0 {
		return 0
	}
	if n == 4 {
		return uint64(uint32(mem))
	}
	return mem
}
