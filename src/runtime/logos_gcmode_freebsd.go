// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build freebsd

package runtime

import "unsafe"

// From FreeBSD's <sys/sysctl.h>.
const logosHWPhysmem = 5

// logosPhysicalMemory returns hw.physmem in bytes, or 0 when sysctl fails.
func logosPhysicalMemory() uint64 {
	mib := [2]uint32{_CTL_HW, logosHWPhysmem}
	var mem uintptr
	n := unsafe.Sizeof(mem)
	if sysctl(&mib[0], 2, (*byte)(unsafe.Pointer(&mem)), &n, nil, 0) < 0 {
		return 0
	}
	return uint64(mem)
}
