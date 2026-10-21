// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build openbsd

package runtime

// From OpenBSD's <sys/sysctl.h>.
const logosHWPhysmem64 = 19

// logosPhysicalMemory returns hw.physmem64 in bytes, or 0 when sysctl fails.
func logosPhysicalMemory() uint64 {
	mem, _ := sysctlUint64([]uint32{_CTL_HW, logosHWPhysmem64})
	return mem
}
