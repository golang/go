// asmcheck -gcflags=-clobberdead

//go:build amd64 || arm64

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

type T [2]*int // contain pointer, not SSA-able (so locals are not registerized)

var p1, p2, p3 T

func F() {
	// 3735936685 is 0xdeaddead. On ARM64 R27 is REGTMP.
	// clobber x, y at entry. not clobber z (stack object).
	// amd64:`MOVL\t\$3735936685, command-line-arguments\.x`, `MOVL\t\$3735936685, command-line-arguments\.y`, -`MOVL\t\$3735936685, command-line-arguments\.z`
	// arm64:`MOVW\tR27, command-line-arguments\.x`, `MOVW\tR27, command-line-arguments\.y`, -`MOVW\tR27, command-line-arguments\.z`
	x, y, z := p1, p2, p3
	addrTaken(&z)
	// x is dead at the call (the value of x is loaded before the CALL), y is not
	// amd64:`MOVL\t\$3735936685, command-line-arguments\.x`, -`MOVL\t\$3735936685, command-line-arguments\.y`
	// arm64:`MOVW\tR27, command-line-arguments\.x`, -`MOVW\tR27, command-line-arguments\.y`
	use(x)
	// amd64:`MOVL\t\$3735936685, command-line-arguments\.x`, `MOVL\t\$3735936685, command-line-arguments\.y`
	// arm64:`MOVW\tR27, command-line-arguments\.x`, `MOVW\tR27, command-line-arguments\.y`
	use(y)
}

//go:noinline
func use(T) {}

//go:noinline
func addrTaken(*T) {}
