// Inferno libmach/8.c
// http://code.google.com/p/inferno-os/source/browse/utils/libmach/8.c
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
 * 386 definition
 */
#include <u.h>
#include <libc.h>
#include <bio.h>
#include <ureg_x86.h>
#include <mach.h>

#define	REGOFF(x)	(uintptr)(&((struct Ureg *) 0)->x)

#define PC		REGOFF(pc)
#define SP		REGOFF(sp)
#define	AX		REGOFF(ax)

#define	REGSIZE		sizeof(struct Ureg)
#define FP_CTL(x)	(REGSIZE+4*(x))
#define FP_REG(x)	(FP_CTL(7)+10*(x))
#define	FPREGSIZE	(7*4+8*10)

Reglist i386reglist[] = {
	{"DI",		REGOFF(di),	RINT, 'X'},
	{"SI",		REGOFF(si),	RINT, 'X'},
	{"BP",		REGOFF(bp),	RINT, 'X'},
	{"BX",		REGOFF(bx),	RINT, 'X'},
	{"DX",		REGOFF(dx),	RINT, 'X'},
	{"CX",		REGOFF(cx),	RINT, 'X'},
	{"AX",		REGOFF(ax),	RINT, 'X'},
	{"GS",		REGOFF(gs),	RINT, 'X'},
	{"FS",		REGOFF(fs),	RINT, 'X'},
	{"ES",		REGOFF(es),	RINT, 'X'},
	{"DS",		REGOFF(ds),	RINT, 'X'},
	{"TRAP",	REGOFF(trap), 	RINT, 'X'},
	{"ECODE",	REGOFF(ecode),	RINT, 'X'},
	{"PC",		PC,		RINT, 'X'},
	{"CS",		REGOFF(cs),	RINT, 'X'},
	{"EFLAGS",	REGOFF(flags),	RINT, 'X'},
	{"SP",		SP,		RINT, 'X'},
	{"SS",		REGOFF(ss),	RINT, 'X'},

	{"E0",		FP_CTL(0),	RFLT, 'X'},
	{"E1",		FP_CTL(1),	RFLT, 'X'},
	{"E2",		FP_CTL(2),	RFLT, 'X'},
	{"E3",		FP_CTL(3),	RFLT, 'X'},
	{"E4",		FP_CTL(4),	RFLT, 'X'},
	{"E5",		FP_CTL(5),	RFLT, 'X'},
	{"E6",		FP_CTL(6),	RFLT, 'X'},
	{"F0",		FP_REG(0),	RFLT, '3'},
	{"F1",		FP_REG(1),	RFLT, '3'},
	{"F2",		FP_REG(2),	RFLT, '3'},
	{"F3",		FP_REG(3),	RFLT, '3'},
	{"F4",		FP_REG(4),	RFLT, '3'},
	{"F5",		FP_REG(5),	RFLT, '3'},
	{"F6",		FP_REG(6),	RFLT, '3'},
	{"F7",		FP_REG(7),	RFLT, '3'},
	{  0 }
};

Mach mi386 =
{
	"386",
	MI386,		/* machine type */
	i386reglist,	/* register list */
	REGSIZE,	/* size of registers in bytes */
	FPREGSIZE,	/* size of fp registers in bytes */
	"PC",		/* name of PC */
	"SP",		/* name of SP */
	0,		/* link register */
	"setSB",	/* static base register name (bogus anyways) */
	0,		/* static base register value */
	0x1000,		/* page size */
	0x80100000ULL,	/* kernel base */
	0xF0000000ULL,	/* kernel text mask */
	0xFFFFFFFFULL,	/* user stack top */
	1,		/* quantization of pc */
	4,		/* szaddr */
	4,		/* szreg */
	4,		/* szfloat */
	8,		/* szdouble */
};
