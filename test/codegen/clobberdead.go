// asmcheck -gcflags=-clobberdead -gcflags=-d=ssa/mem2reg/off -gcflags=-d=ssa/decompose_addr/off

//go:build amd64 || arm64 || loong64 || riscv64

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

type T [2]*int // contain pointer, not SSA-able (so locals are not registerized)

var p1, p2, p3 T

func F() {
	// 3735936685 is 0xdeaddead. REGTMP is R27 on ARM64, R30 on loong64, X31 on riscv64.
	// clobber x, y at entry. not clobber z (stack object).
	// amd64:`MOVL \$3735936685, command-line-arguments\.x` `MOVL \$3735936685, command-line-arguments\.y` -`MOVL \$3735936685, command-line-arguments\.z`
	// arm64:`MOVW R27, command-line-arguments\.x` `MOVW R27, command-line-arguments\.y` -`MOVW R27, command-line-arguments\.z`
	// loong64:`MOVW R30, command-line-arguments\.x` `MOVW R30, command-line-arguments\.y` -`MOVW R30, command-line-arguments\.z`
	// riscv64:`MOVW X31, command-line-arguments\.x` `MOVW X31, command-line-arguments\.y` -`MOVW X31, command-line-arguments\.z`
	x, y, z := p1, p2, p3
	addrTaken(&z)
	// x is dead at the call (the value of x is loaded before the CALL), y is not
	// amd64:`MOVL \$3735936685, command-line-arguments\.x` -`MOVL \$3735936685, command-line-arguments\.y`
	// arm64:`MOVW R27, command-line-arguments\.x` -`MOVW R27, command-line-arguments\.y`
	// loong64:`MOVW R30, command-line-arguments\.x` -`MOVW R30, command-line-arguments\.y`
	// riscv64:`MOVW X31, command-line-arguments\.x` -`MOVW X31, command-line-arguments\.y`
	use(x)
	// amd64:`MOVL \$3735936685, command-line-arguments\.x` `MOVL \$3735936685, command-line-arguments\.y`
	// arm64:`MOVW R27, command-line-arguments\.x` `MOVW R27, command-line-arguments\.y`
	// loong64:`MOVW R30, command-line-arguments\.x` `MOVW R30, command-line-arguments\.y`
	// riscv64:`MOVW X31, command-line-arguments\.x` `MOVW X31, command-line-arguments\.y`
	use(y)
}

//go:noinline
func use(T) {}

//go:noinline
func addrTaken(*T) {}
