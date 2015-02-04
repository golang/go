// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include "gg.h"

int	thechar	= '5';
char*	thestring	= "arm";
LinkArch*	thelinkarch = &linkarm;

void
linkarchinit(void)
{
}

vlong MAXWIDTH = (1LL<<32) - 1;

/*
 * go declares several platform-specific type aliases:
 * int, uint, float, and uintptr
 */
Typedef	typedefs[] =
{
	{"int",		TINT,		TINT32},
	{"uint",		TUINT,		TUINT32},
	{"uintptr",	TUINTPTR,	TUINT32},
	{0}
};

void
betypeinit(void)
{
	widthptr = 4;
	widthint = 4;
	widthreg = 4;

	listinit5();
}

void
main(int argc, char **argv)
{
	arch.thechar = thechar;
	arch.thestring = thestring;
	arch.thelinkarch = thelinkarch;
	arch.typedefs = typedefs;
	arch.REGSP = REGSP;
	arch.REGCTXT = REGCTXT;
	arch.MAXWIDTH = MAXWIDTH;
	arch.anyregalloc = anyregalloc;
	arch.betypeinit = betypeinit;
	arch.bgen = bgen;
	arch.cgen = cgen;
	arch.cgen_call = cgen_call;
	arch.cgen_callinter = cgen_callinter;
	arch.cgen_ret = cgen_ret;
	arch.clearfat = clearfat;
	arch.dumpit = dumpit;
	arch.excise = excise;
	arch.expandchecks = expandchecks;
	arch.gclean = gclean;
	arch.ginit = ginit;
	arch.gins = gins;
	arch.ginscall = ginscall;
	arch.igen = igen;
	arch.linkarchinit = linkarchinit;
	arch.peep = peep;
	arch.proginfo = proginfo;
	arch.regalloc = regalloc;
	arch.regfree = regfree;
	arch.regtyp = regtyp;
	arch.sameaddr = sameaddr;
	arch.smallindir = smallindir;
	arch.stackaddr = stackaddr;
	arch.excludedregs = excludedregs;
	arch.RtoB = RtoB;
	arch.FtoB = RtoB;
	arch.BtoR = BtoR;
	arch.BtoF = BtoF;
	arch.optoas = optoas;
	arch.doregbits = doregbits;
	arch.regnames = regnames;
	
	gcmain(argc, argv);
}
