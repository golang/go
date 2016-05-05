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

package x86

import "cmd/internal/obj/x86"
import "cmd/compile/internal/gc"

const (
	NREGVAR = 16 /* 8 integer + 8 floating */
)

var regname = []string{
	".ax",
	".cx",
	".dx",
	".bx",
	".sp",
	".bp",
	".si",
	".di",
	".x0",
	".x1",
	".x2",
	".x3",
	".x4",
	".x5",
	".x6",
	".x7",
}

func regnames(n *int) []string {
	*n = NREGVAR
	return regname
}

func excludedregs() uint64 {
	if gc.Ctxt.Flag_shared {
		return RtoB(x86.REG_SP) | RtoB(x86.REG_CX)
	} else {
		return RtoB(x86.REG_SP)
	}
}

func doregbits(r int) uint64 {
	b := uint64(0)
	if r >= x86.REG_AX && r <= x86.REG_DI {
		b |= RtoB(r)
	} else if r >= x86.REG_AL && r <= x86.REG_BL {
		b |= RtoB(r - x86.REG_AL + x86.REG_AX)
	} else if r >= x86.REG_AH && r <= x86.REG_BH {
		b |= RtoB(r - x86.REG_AH + x86.REG_AX)
	} else if r >= x86.REG_X0 && r <= x86.REG_X0+7 {
		b |= FtoB(r)
	}
	return b
}

func RtoB(r int) uint64 {
	if r < x86.REG_AX || r > x86.REG_DI {
		return 0
	}
	return 1 << uint(r-x86.REG_AX)
}

func BtoR(b uint64) int {
	b &= 0xff
	if b == 0 {
		return 0
	}
	return gc.Bitno(b) + x86.REG_AX
}

func FtoB(f int) uint64 {
	if f < x86.REG_X0 || f > x86.REG_X7 {
		return 0
	}
	return 1 << uint(f-x86.REG_X0+8)
}

func BtoF(b uint64) int {
	b &= 0xFF00
	if b == 0 {
		return 0
	}
	return gc.Bitno(b) - 8 + x86.REG_X0
}
