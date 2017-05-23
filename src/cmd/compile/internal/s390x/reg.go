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

package s390x

import "cmd/internal/obj/s390x"
import "cmd/compile/internal/gc"

const (
	NREGVAR = 32 /* 16 general + 16 floating */
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
	// Exclude registers with fixed functions
	return RtoB(s390x.REG_R0) |
		RtoB(s390x.REGSP) |
		RtoB(s390x.REGG) |
		RtoB(s390x.REGTMP) |
		RtoB(s390x.REGTMP2) |
		RtoB(s390x.REG_LR)
}

func doregbits(r int) uint64 {
	return 0
}

/*
 * track register variables including external registers:
 *	bit	reg
 *	0	R0
 *	...	...
 *	15	R15
 *	16+0	F0
 *	16+1	F1
 *	...	...
 *	16+15	F15
 */
func RtoB(r int) uint64 {
	if r >= s390x.REG_R0 && r <= s390x.REG_R15 {
		return 1 << uint(r-s390x.REG_R0)
	}
	if r >= s390x.REG_F0 && r <= s390x.REG_F15 {
		return 1 << uint(16+r-s390x.REG_F0)
	}
	return 0
}

func BtoR(b uint64) int {
	b &= 0xffff
	if b == 0 {
		return 0
	}
	return gc.Bitno(b) + s390x.REG_R0
}

func BtoF(b uint64) int {
	b >>= 16
	b &= 0xffff
	if b == 0 {
		return 0
	}
	return gc.Bitno(b) + s390x.REG_F0
}
