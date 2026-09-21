// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

const CacheLinePadSize = 64

// RISC-V doesn't have a 'cpuid' equivalent. On Linux we rely on the riscv_hwprobe syscall.

// getGORISCV64level is implemented in cpu_riscv64.s. Returns number 20, 22, or 23.
func getGORISCV64level() int32

func doinit() {
	options = []option{
		{Name: "fastmisaligned", Feature: &RISCV64.HasFastMisaligned},
		{Name: "zbc", Feature: &RISCV64.HasZbc},
		{Name: "zvbc", Feature: &RISCV64.HasZvbc},
		{Name: "zvkg", Feature: &RISCV64.HasZvkg},
		{Name: "zvkned", Feature: &RISCV64.HasZvkned},
		{Name: "zvknha", Feature: &RISCV64.HasZvknha},
		{Name: "zvknhb", Feature: &RISCV64.HasZvknhb},
		{Name: "zvksed", Feature: &RISCV64.HasZvksed},
		{Name: "zvksh", Feature: &RISCV64.HasZvksh},
	}
	level := getGORISCV64level()
	if level < 22 {
		// Zbb is mandatory for rva22u64 and higher.
		options = append(options, option{Name: "zbb", Feature: &RISCV64.HasZbb})
	}
	if level < 23 {
		// V, Zvbb and Zvkt are mandatory for rva23u64 and higher.
		options = append(options,
			option{Name: "v", Feature: &RISCV64.HasV},
			option{Name: "zvbb", Feature: &RISCV64.HasZvbb},
			option{Name: "zvkt", Feature: &RISCV64.HasZvkt},
		)
	}

	doDerived = func() {
		// If the vector extension is disabled by GODEBUG, then the VLENB is zero.
		if !RISCV64.HasV {
			RISCV64.VLENB = 0
		}
	}

	osInit()
	if RISCV64.HasV {
		RISCV64.VLENB = readVLENB()
	}
}

func isSet(hwc uint, value uint) bool {
	return hwc&value != 0
}

// Read the vector register length in bytes from the vlenb CSR.
// May only be called when the vector extension is present.
// Implemented in cpu_riscv64.s.
func readVLENB() uint
