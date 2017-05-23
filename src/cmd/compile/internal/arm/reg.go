// Inferno utils/5c/reg.c
// http://code.google.com/p/inferno-os/source/browse/utils/5c/reg.c
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

package arm

import "cmd/internal/obj/arm"
import "cmd/compile/internal/gc"

const (
	NREGVAR = 32
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
}

func regnames(n *int) []string {
	*n = NREGVAR
	return regname
}

func excludedregs() uint64 {
	return RtoB(arm.REGSP) | RtoB(arm.REGLINK) | RtoB(arm.REGPC)
}

func doregbits(r int) uint64 {
	return 0
}

/*
 *	bit	reg
 *	0	R0
 *	1	R1
 *	...	...
 *	10	R10
 *	12  R12
 *
 *	bit	reg
 *	18	F2
 *	19	F3
 *	...	...
 *	31	F15
 */
func RtoB(r int) uint64 {
	if arm.REG_R0 <= r && r <= arm.REG_R15 {
		if r >= arm.REGTMP-2 && r != arm.REG_R12 { // excluded R9 and R10 for m and g, but not R12
			return 0
		}
		return 1 << uint(r-arm.REG_R0)
	}

	if arm.REG_F0 <= r && r <= arm.REG_F15 {
		if r < arm.REG_F2 || r > arm.REG_F0+arm.NFREG-1 {
			return 0
		}
		return 1 << uint((r-arm.REG_F0)+16)
	}

	return 0
}

func BtoR(b uint64) int {
	// TODO Allow R0 and R1, but be careful with a 0 return
	// TODO Allow R9. Only R10 is reserved now (just g, not m).
	b &= 0x11fc // excluded R9 and R10 for m and g, but not R12
	if b == 0 {
		return 0
	}
	return gc.Bitno(b) + arm.REG_R0
}

func BtoF(b uint64) int {
	b &= 0xfffc0000
	if b == 0 {
		return 0
	}
	return gc.Bitno(b) - 16 + arm.REG_F0
}
