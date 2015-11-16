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
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
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

package mips64

import "cmd/internal/obj/mips"
import "cmd/compile/internal/gc"

const (
	NREGVAR = 64 /* 32 general + 32 floating */
)

var regname = []string{
	".R0",
	".R1",
	".R2",
	".R3",
	".R4",
	".R5",
	".R6",
	".R7",
	".R8",
	".R9",
	".R10",
	".R11",
	".R12",
	".R13",
	".R14",
	".R15",
	".R16",
	".R17",
	".R18",
	".R19",
	".R20",
	".R21",
	".R22",
	".R23",
	".R24",
	".R25",
	".R26",
	".R27",
	".R28",
	".R29",
	".R30",
	".R31",
	".F0",
	".F1",
	".F2",
	".F3",
	".F4",
	".F5",
	".F6",
	".F7",
	".F8",
	".F9",
	".F10",
	".F11",
	".F12",
	".F13",
	".F14",
	".F15",
	".F16",
	".F17",
	".F18",
	".F19",
	".F20",
	".F21",
	".F22",
	".F23",
	".F24",
	".F25",
	".F26",
	".F27",
	".F28",
	".F29",
	".F30",
	".F31",
}

func regnames(n *int) []string {
	*n = NREGVAR
	return regname
}

func excludedregs() uint64 {
	// Exclude registers with fixed functions
	regbits := uint64(1<<0 | RtoB(mips.REGSP) | RtoB(mips.REGG) | RtoB(mips.REGTMP) | RtoB(mips.REGLINK) | RtoB(mips.REG_R26) | RtoB(mips.REG_R27))

	// Also exclude floating point registers with fixed constants
	regbits |= RtoB(mips.FREGZERO) | RtoB(mips.FREGHALF) | RtoB(mips.FREGONE) | RtoB(mips.FREGTWO)

	return regbits
}

func doregbits(r int) uint64 {
	return 0
}

/*
 * track register variables including external registers:
 *	bit	reg
 *	0	R0
 *	1	R1
 *	...	...
 *	31	R31
 *	32+0	F0
 *	32+1	F1
 *	...	...
 *	32+31	F31
 */
func RtoB(r int) uint64 {
	if r > mips.REG_R0 && r <= mips.REG_R31 {
		return 1 << uint(r-mips.REG_R0)
	}
	if r >= mips.REG_F0 && r <= mips.REG_F31 {
		return 1 << uint(32+r-mips.REG_F0)
	}
	return 0
}

func BtoR(b uint64) int {
	b &= 0xffffffff
	if b == 0 {
		return 0
	}
	return gc.Bitno(b) + mips.REG_R0
}

func BtoF(b uint64) int {
	b >>= 32
	if b == 0 {
		return 0
	}
	return gc.Bitno(b) + mips.REG_F0
}
