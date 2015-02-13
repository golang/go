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
	NREGVAR = 16,	/* 8 integer + 8 floating */
};

static char* regname[] = {
	".ax", ".cx", ".dx", ".bx", ".sp", ".bp", ".si", ".di",
	".x0", ".x1", ".x2", ".x3", ".x4", ".x5", ".x6", ".x7",
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
	if(r >= REG_AX && r <= REG_DI)
		b |= RtoB(r);
	else
	if(r >= REG_AL && r <= REG_BL)
		b |= RtoB(r-REG_AL+REG_AX);
	else
	if(r >= REG_AH && r <= REG_BH)
		b |= RtoB(r-REG_AH+REG_AX);
	else
	if(r >= REG_X0 && r <= REG_X0+7)
		b |= FtoB(r);
	return b;
}

uint64
RtoB(int r)
{

	if(r < REG_AX || r > REG_DI)
		return 0;
	return 1ULL << (r-REG_AX);
}

int
BtoR(uint64 b)
{

	b &= 0xffL;
	if(b == 0)
		return 0;
	return bitno(b) + REG_AX;
}

uint64
FtoB(int f)
{
	if(f < REG_X0 || f > REG_X7)
		return 0;
	return 1ULL << (f - REG_X0 + 8);
}

int
BtoF(uint64 b)
{
	b &= 0xFF00L;
	if(b == 0)
		return 0;
	return bitno(b) - 8 + REG_X0;
}
