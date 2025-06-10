// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build riscv64

package cpu

const cacheLineSize = 64

func initOptions() {
	options = []option{
		{Name: "fastmisaligned", Feature: &RISCV64.HasFastMisaligned},
		{Name: "c", Feature: &RISCV64.HasC},
		{Name: "v", Feature: &RISCV64.HasV},
		{Name: "zba", Feature: &RISCV64.HasZba},
		{Name: "zbb", Feature: &RISCV64.HasZbb},
		{Name: "zbs", Feature: &RISCV64.HasZbs},
		// RISC-V Cryptography Extensions
		{Name: "zvbb", Feature: &RISCV64.HasZvbb},
		{Name: "zvbc", Feature: &RISCV64.HasZvbc},
		{Name: "zvkb", Feature: &RISCV64.HasZvkb},
		{Name: "zvkg", Feature: &RISCV64.HasZvkg},
		{Name: "zvkt", Feature: &RISCV64.HasZvkt},
		{Name: "zvkn", Feature: &RISCV64.HasZvkn},
		{Name: "zvknc", Feature: &RISCV64.HasZvknc},
		{Name: "zvkng", Feature: &RISCV64.HasZvkng},
		{Name: "zvks", Feature: &RISCV64.HasZvks},
		{Name: "zvksc", Feature: &RISCV64.HasZvksc},
		{Name: "zvksg", Feature: &RISCV64.HasZvksg},
	}
}
