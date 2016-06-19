// Derived from Inferno utils/6c/reg.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/reg.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package amd64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj/x86"
)

const (
	NREGVAR = 32
)

var regname = []string{
	".AX",
	".CX",
	".DX",
	".BX",
	".SP",
	".BP",
	".SI",
	".DI",
	".R8",
	".R9",
	".R10",
	".R11",
	".R12",
	".R13",
	".R14",
	".R15",
	".X0",
	".X1",
	".X2",
	".X3",
	".X4",
	".X5",
	".X6",
	".X7",
	".X8",
	".X9",
	".X10",
	".X11",
	".X12",
	".X13",
	".X14",
	".X15",
}

func regnames(n *int) []string {
	*n = NREGVAR
	return regname
}

func excludedregs() uint64 {
	return RtoB(x86.REG_SP)
}

func doregbits(r int) uint64 {
	b := uint64(0)
	if r >= x86.REG_AX && r <= x86.REG_R15 {
		b |= RtoB(r)
	} else if r >= x86.REG_AL && r <= x86.REG_R15B {
		b |= RtoB(r - x86.REG_AL + x86.REG_AX)
	} else if r >= x86.REG_AH && r <= x86.REG_BH {
		b |= RtoB(r - x86.REG_AH + x86.REG_AX)
	} else if r >= x86.REG_X0 && r <= x86.REG_X0+15 {
		b |= FtoB(r)
	}
	return b
}

// For ProgInfo.
const (
	AX  = 1 << (x86.REG_AX - x86.REG_AX)
	BX  = 1 << (x86.REG_BX - x86.REG_AX)
	CX  = 1 << (x86.REG_CX - x86.REG_AX)
	DX  = 1 << (x86.REG_DX - x86.REG_AX)
	DI  = 1 << (x86.REG_DI - x86.REG_AX)
	SI  = 1 << (x86.REG_SI - x86.REG_AX)
	R15 = 1 << (x86.REG_R15 - x86.REG_AX)
	X0  = 1 << 16
)

func RtoB(r int) uint64 {
	if r < x86.REG_AX || r > x86.REG_R15 {
		return 0
	}
	return 1 << uint(r-x86.REG_AX)
}

func BtoR(b uint64) int {
	b &= 0xffff
	if gc.Nacl {
		b &^= (1<<(x86.REG_BP-x86.REG_AX) | 1<<(x86.REG_R15-x86.REG_AX))
	} else if gc.Ctxt.Framepointer_enabled {
		// BP is part of the calling convention if framepointer_enabled.
		b &^= (1 << (x86.REG_BP - x86.REG_AX))
	}
	if b == 0 {
		return 0
	}
	return gc.Bitno(b) + x86.REG_AX
}

/*
 *	bit	reg
 *	16	X0
 *	...
 *	31	X15
 */
func FtoB(f int) uint64 {
	if f < x86.REG_X0 || f > x86.REG_X15 {
		return 0
	}
	return 1 << uint(f-x86.REG_X0+16)
}

func BtoF(b uint64) int {
	b &= 0xFFFF0000
	if b == 0 {
		return 0
	}
	return gc.Bitno(b) - 16 + x86.REG_X0
}
