// Inferno libmach/5.c
// http://code.google.com/p/inferno-os/source/browse/utils/libmach/5.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.
//	Power PC support Copyright © 1995-2004 C H Forsyth (forsyth@terzarima.net).
//	Portions Copyright © 1997-1999 Vita Nuova Limited.
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).
//	Revisions Copyright © 2000-2004 Lucent Technologies Inc. and others.
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
 * arm definition
 */
#include <u.h>
#include <libc.h>
#include <bio.h>
#include "ureg_arm.h"
#include <mach.h>

#define	REGOFF(x)	(uintptr) (&((struct Ureg *) 0)->x)

#define SP		REGOFF(r13)
#define PC		REGOFF(pc)

#define	REGSIZE		sizeof(struct Ureg)

Reglist armreglist[] =
{
	{"LINK",	REGOFF(link),		RINT|RRDONLY, 'X'},
	{"TYPE",	REGOFF(type),		RINT|RRDONLY, 'X'},
	{"PSR",		REGOFF(psr),		RINT|RRDONLY, 'X'},
	{"PC",		PC,			RINT, 'X'},
	{"SP",		SP,			RINT, 'X'},
	{"R15",		PC,			RINT, 'X'},
	{"R14",		REGOFF(r14),		RINT, 'X'},
	{"R13",		REGOFF(r13),		RINT, 'X'},
	{"R12",		REGOFF(r12),		RINT, 'X'},
	{"R11",		REGOFF(r11),		RINT, 'X'},
	{"R10",		REGOFF(r10),		RINT, 'X'},
	{"R9",		REGOFF(r9),		RINT, 'X'},
	{"R8",		REGOFF(r8),		RINT, 'X'},
	{"R7",		REGOFF(r7),		RINT, 'X'},
	{"R6",		REGOFF(r6),		RINT, 'X'},
	{"R5",		REGOFF(r5),		RINT, 'X'},
	{"R4",		REGOFF(r4),		RINT, 'X'},
	{"R3",		REGOFF(r3),		RINT, 'X'},
	{"R2",		REGOFF(r2),		RINT, 'X'},
	{"R1",		REGOFF(r1),		RINT, 'X'},
	{"R0",		REGOFF(r0),		RINT, 'X'},
	{  0 }
};

	/* the machine description */
Mach marm =
{
	"arm",
	MARM,		/* machine type */
	armreglist,	/* register set */
	REGSIZE,	/* register set size */
	0,		/* fp register set size */
	"PC",		/* name of PC */
	"SP",		/* name of SP */
	"R15",		/* name of link register */
	"setR12",	/* static base register name */
	0,		/* static base register value */
	0x1000,		/* page size */
	0xC0000000,	/* kernel base */
	0,		/* kernel text mask */
	4,		/* quantization of pc */
	4,		/* szaddr */
	4,		/* szreg */
	4,		/* szfloat */
	8,		/* szdouble */
};
