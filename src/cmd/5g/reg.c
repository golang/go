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


#include <u.h>
#include <libc.h>
#include "gg.h"
#include "../gc/popt.h"

enum {
	NREGVAR = 32,
};

static char* regname[] = {
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
};

char**
regnames(int *n)
{
	*n = NREGVAR;
	return regname;
}

uint64
excludedregs(void)
{
	return RtoB(REGSP)|RtoB(REGLINK)|RtoB(REGPC);
}

uint64
doregbits(int r)
{
	USED(r);
	return 0;
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
uint64
RtoB(int r)
{
	if(REG_R0 <= r && r <= REG_R15) {
		if(r >= REGTMP-2 && r != REG_R12)	// excluded R9 and R10 for m and g, but not R12
			return 0;
		return 1ULL << (r - REG_R0);
	}
	
	if(REG_F0 <= r && r <= REG_F15) {
		if(r < REG_F2 || r > REG_F0+NFREG-1)
			return 0;
		return 1ULL << ((r - REG_F0) + 16);
	}
	
	return 0;
}

int
BtoR(uint64 b)
{
	// TODO Allow R0 and R1, but be careful with a 0 return
	// TODO Allow R9. Only R10 is reserved now (just g, not m).
	b &= 0x11fcL;	// excluded R9 and R10 for m and g, but not R12
	if(b == 0)
		return 0;
	return bitno(b) + REG_R0;
}

int
BtoF(uint64 b)
{
	b &= 0xfffc0000L;
	if(b == 0)
		return 0;
	return bitno(b) - 16 + REG_F0;
}
