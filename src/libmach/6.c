// Inferno libmach/6.c
// http://code.google.com/p/inferno-os/source/browse/utils/libmach/6.c
//
// 	Copyright © 1994-1999 Lucent Technologies Inc.
// 	Power PC support Copyright © 1995-2004 C H Forsyth (forsyth@terzarima.net).
// 	Portions Copyright © 1997-1999 Vita Nuova Limited.
// 	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).
// 	Revisions Copyright © 2000-2004 Lucent Technologies Inc. and others.
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

/*
 * amd64 definition
 */
#include <u.h>
#include <libc.h>
#include <bio.h>
#include "ureg_amd64.h"
#include <mach.h>

#define	REGOFF(x)	offsetof(struct Ureg, x)

#define	REGSIZE		sizeof(struct Ureg)
#define FP_CTLS(x)	(REGSIZE+2*(x))
#define FP_CTL(x)	(REGSIZE+4*(x))
#define FP_REG(x)	(FP_CTL(8)+16*(x))
#define XM_REG(x)	(FP_CTL(8)+8*16+16*(x))

#define	FPREGSIZE	512	/* TO DO? currently only 0x1A0 used */

Reglist amd64reglist[] = {
	{"AX",		REGOFF(ax),	RINT, 'Y'},
	{"BX",		REGOFF(bx),	RINT, 'Y'},
	{"CX",		REGOFF(cx),	RINT, 'Y'},
	{"DX",		REGOFF(dx),	RINT, 'Y'},
	{"SI",		REGOFF(si),	RINT, 'Y'},
	{"DI",		REGOFF(di),	RINT, 'Y'},
	{"BP",		REGOFF(bp),	RINT, 'Y'},
	{"R8",		REGOFF(r8),	RINT, 'Y'},
	{"R9",		REGOFF(r9),	RINT, 'Y'},
	{"R10",		REGOFF(r10),	RINT, 'Y'},
	{"R11",		REGOFF(r11),	RINT, 'Y'},
	{"R12",		REGOFF(r12),	RINT, 'Y'},
	{"R13",		REGOFF(r13),	RINT, 'Y'},
	{"R14",		REGOFF(r14),	RINT, 'Y'},
	{"R15",		REGOFF(r15),	RINT, 'Y'},
	{"DS",		REGOFF(ds),	RINT, 'x'},
	{"ES",		REGOFF(es),	RINT, 'x'},
	{"FS",		REGOFF(fs),	RINT, 'x'},
	{"GS",		REGOFF(gs),	RINT, 'x'},
	{"TYPE",	REGOFF(type), 	RINT, 'Y'},
	{"TRAP",	REGOFF(type), 	RINT, 'Y'},	/* alias for acid */
	{"ERROR",	REGOFF(error),	RINT, 'Y'},
	{"IP",		REGOFF(ip),	RINT, 'Y'},
	{"PC",		REGOFF(ip),	RINT, 'Y'},	/* alias for acid */
	{"CS",		REGOFF(cs),	RINT, 'Y'},
	{"FLAGS",	REGOFF(flags),	RINT, 'Y'},
	{"SP",		REGOFF(sp),	RINT, 'Y'},
	{"SS",		REGOFF(ss),	RINT, 'Y'},

	{"FCW",		FP_CTLS(0),	RFLT, 'x'},
	{"FSW",		FP_CTLS(1),	RFLT, 'x'},
	{"FTW",		FP_CTLS(2),	RFLT, 'b'},
	{"FOP",		FP_CTLS(3),	RFLT, 'x'},
	{"RIP",		FP_CTL(2),	RFLT, 'Y'},
	{"RDP",		FP_CTL(4),	RFLT, 'Y'},
	{"MXCSR",	FP_CTL(6),	RFLT, 'X'},
	{"MXCSRMASK",	FP_CTL(7),	RFLT, 'X'},
	{"M0",		FP_REG(0),	RFLT, 'F'},	/* assumes double */
	{"M1",		FP_REG(1),	RFLT, 'F'},
	{"M2",		FP_REG(2),	RFLT, 'F'},
	{"M3",		FP_REG(3),	RFLT, 'F'},
	{"M4",		FP_REG(4),	RFLT, 'F'},
	{"M5",		FP_REG(5),	RFLT, 'F'},
	{"M6",		FP_REG(6),	RFLT, 'F'},
	{"M7",		FP_REG(7),	RFLT, 'F'},
	{"X0",		XM_REG(0),	RFLT, 'F'},	/* assumes double */
	{"X1",		XM_REG(1),	RFLT, 'F'},
	{"X2",		XM_REG(2),	RFLT, 'F'},
	{"X3",		XM_REG(3),	RFLT, 'F'},
	{"X4",		XM_REG(4),	RFLT, 'F'},
	{"X5",		XM_REG(5),	RFLT, 'F'},
	{"X6",		XM_REG(6),	RFLT, 'F'},
	{"X7",		XM_REG(7),	RFLT, 'F'},
	{"X8",		XM_REG(8),	RFLT, 'F'},
	{"X9",		XM_REG(9),	RFLT, 'F'},
	{"X10",		XM_REG(10),	RFLT, 'F'},
	{"X11",		XM_REG(11),	RFLT, 'F'},
	{"X12",		XM_REG(12),	RFLT, 'F'},
	{"X13",		XM_REG(13),	RFLT, 'F'},
	{"X14",		XM_REG(14),	RFLT, 'F'},
	{"X15",		XM_REG(15),	RFLT, 'F'},
	{"X16",		XM_REG(16),	RFLT, 'F'},
/*
	{"F0",		FP_REG(7),	RFLT, '3'},
	{"F1",		FP_REG(6),	RFLT, '3'},
	{"F2",		FP_REG(5),	RFLT, '3'},
	{"F3",		FP_REG(4),	RFLT, '3'},
	{"F4",		FP_REG(3),	RFLT, '3'},
	{"F5",		FP_REG(2),	RFLT, '3'},
	{"F6",		FP_REG(1),	RFLT, '3'},
	{"F7",		FP_REG(0),	RFLT, '3'},
*/
	{  0 }
};

Mach mamd64=
{
	"amd64",
	MAMD64,			/* machine type */
	amd64reglist,		/* register list */
	REGSIZE,		/* size of registers in bytes */
	FPREGSIZE,		/* size of fp registers in bytes */
	"PC",			/* name of PC */
	"SP",			/* name of SP */
	0,			/* link register */
	"setSB",		/* static base register name (bogus anyways) */
	0,			/* static base register value */
	0x1000,			/* page size */
	0xFFFFFFFF80110000ULL,	/* kernel base */
	0xFFFF800000000000ULL,	/* kernel text mask */
	0x00007FFFFFFFF000ULL,	/* user stack top */
	1,			/* quantization of pc */
	8,			/* szaddr */
	4,			/* szreg */
	4,			/* szfloat */
	8,			/* szdouble */
};
