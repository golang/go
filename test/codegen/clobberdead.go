// asmcheck -gcflags=-clobberdead

// +build amd64 arm64

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

type T [2]*int // contain pointer, not SSA-able (so locals are not registerized)

var p1, p2, p3 T

func F() {
	// 3735936685 is 0xdeaddead. On ARM64 R27 is REGTMP.
	// clobber x, y at entry. not clobber z (stack object).
	// amd64:`MOVL\t\$3735936685, ""\.x`, `MOVL\t\$3735936685, ""\.y`, -`MOVL\t\$3735936685, ""\.z`
	// arm64:`MOVW\tR27, ""\.x`, `MOVW\tR27, ""\.y`, -`MOVW\tR27, ""\.z`
	x, y, z := p1, p2, p3
	addrTaken(&z)
	// x is dead at the call (the value of x is loaded before the CALL), y is not
	// amd64:`MOVL\t\$3735936685, ""\.x`, -`MOVL\t\$3735936685, ""\.y`
	// arm64:`MOVW\tR27, ""\.x`, -`MOVW\tR27, ""\.y`
	use(x)
	// amd64:`MOVL\t\$3735936685, ""\.x`, `MOVL\t\$3735936685, ""\.y`
	// arm64:`MOVW\tR27, ""\.x`, `MOVW\tR27, ""\.y`
	use(y)
}

//go:noinline
func use(T) {}

//go:noinline
func addrTaken(*T) {}
