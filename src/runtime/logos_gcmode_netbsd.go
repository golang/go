// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netbsd

package runtime

import "unsafe"

// From NetBSD's <sys/sysctl.h>.
const logosHWPhysmem64 = 13

// logosPhysicalMemory returns hw.physmem64 in bytes, or 0 when sysctl fails.
func logosPhysicalMemory() uint64 {
	mib := [2]uint32{_CTL_HW, logosHWPhysmem64}
	var mem uint64
	n := unsafe.Sizeof(mem)
	if sysctl(&mib[0], 2, (*byte)(unsafe.Pointer(&mem)), &n, nil, 0) < 0 {
		return 0
	}
	return mem
}
