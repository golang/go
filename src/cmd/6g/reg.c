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

#include <u.h>
#include <libc.h>
#include "gg.h"
#include "../gc/popt.h"

enum {
	NREGVAR = 32,
};

static char* regname[] = {
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
	return RtoB(REG_SP);
}

uint64
doregbits(int r)
{
	uint64 b;

	b = 0;
	if(r >= REG_AX && r <= REG_R15)
		b |= RtoB(r);
	else
	if(r >= REG_AL && r <= REG_R15B)
		b |= RtoB(r-REG_AL+REG_AX);
	else
	if(r >= REG_AH && r <= REG_BH)
		b |= RtoB(r-REG_AH+REG_AX);
	else
	if(r >= REG_X0 && r <= REG_X0+15)
		b |= FtoB(r);
	return b;
}

uint64
RtoB(int r)
{

	if(r < REG_AX || r > REG_R15)
		return 0;
	return 1ULL << (r-REG_AX);
}

int
BtoR(uint64 b)
{
	b &= 0xffffULL;
	if(nacl)
		b &= ~((1<<(REG_BP-REG_AX)) | (1<<(REG_R15-REG_AX)));
	else if(framepointer_enabled)
		// BP is part of the calling convention if framepointer_enabled.
		b &= ~(1<<(REG_BP-REG_AX));
	if(b == 0)
		return 0;
	return bitno(b) + REG_AX;
}

/*
 *	bit	reg
 *	16	X0
 *	...
 *	31	X15
 */
uint64
FtoB(int f)
{
	if(f < REG_X0 || f > REG_X15)
		return 0;
	return 1ULL << (f - REG_X0 + 16);
}

int
BtoF(uint64 b)
{

	b &= 0xFFFF0000L;
	if(b == 0)
		return 0;
	return bitno(b) - 16 + REG_X0;
}
