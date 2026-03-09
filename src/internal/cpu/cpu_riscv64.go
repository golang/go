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
		{Name: "zvbb", Feature: &RISCV64.HasZvbb},
		{Name: "zvbc", Feature: &RISCV64.HasZvbc},
		{Name: "zvkg", Feature: &RISCV64.HasZvkg},
		{Name: "zvkned", Feature: &RISCV64.HasZvkned},
		{Name: "zvknha", Feature: &RISCV64.HasZvknha},
		{Name: "zvknhb", Feature: &RISCV64.HasZvknhb},
		{Name: "zvksed", Feature: &RISCV64.HasZvksed},
		{Name: "zvksh", Feature: &RISCV64.HasZvksh},
		{Name: "zvkt", Feature: &RISCV64.HasZvkt},
	}
	osInit()
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}
