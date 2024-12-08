// asmcheck -gcflags=-clobberdeadreg

//go:build amd64

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

type S struct {
	a, b, c, d, e, f int
}

func F(a, b, c int, d S) {
	// -2401018187971961171 is 0xdeaddeaddeaddead
	// amd64:`MOVQ\t\$-2401018187971961171, AX`, `MOVQ\t\$-2401018187971961171, BX`, `MOVQ\t\$-2401018187971961171, CX`
	// amd64:`MOVQ\t\$-2401018187971961171, DX`, `MOVQ\t\$-2401018187971961171, SI`, `MOVQ\t\$-2401018187971961171, DI`
	// amd64:`MOVQ\t\$-2401018187971961171, R8`, `MOVQ\t\$-2401018187971961171, R9`, `MOVQ\t\$-2401018187971961171, R10`
	// amd64:`MOVQ\t\$-2401018187971961171, R11`, `MOVQ\t\$-2401018187971961171, R12`, `MOVQ\t\$-2401018187971961171, R13`
	// amd64:-`MOVQ\t\$-2401018187971961171, BP` // frame pointer is not clobbered
	StackArgsCall([10]int{a, b, c})
	// amd64:`MOVQ\t\$-2401018187971961171, R12`, `MOVQ\t\$-2401018187971961171, R13`, `MOVQ\t\$-2401018187971961171, DX`
	// amd64:-`MOVQ\t\$-2401018187971961171, AX`, -`MOVQ\t\$-2401018187971961171, R11` // register args are not clobbered
	RegArgsCall(a, b, c, d)
}

//go:noinline
func StackArgsCall([10]int) {}

//go:noinline
//go:registerparams
func RegArgsCall(int, int, int, S) {}
