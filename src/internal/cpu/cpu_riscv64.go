// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

const CacheLinePadSize = 64

// RISC-V doesn't have a 'cpuid' equivalent. On Linux we rely on the riscv_hwprobe syscall.

func doinit() {
	options = []option{
		{Name: "fastmisaligned", Feature: &RISCV64.HasFastMisaligned},
		{Name: "v", Feature: &RISCV64.HasV},
		{Name: "zbb", Feature: &RISCV64.HasZbb},
	}
	osInit()
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}
