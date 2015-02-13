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
	thearch.thechar = thechar;
	thearch.thestring = thestring;
	thearch.thelinkarch = thelinkarch;
	thearch.typedefs = typedefs;
	thearch.REGSP = REGSP;
	thearch.REGCTXT = REGCTXT;
	thearch.MAXWIDTH = MAXWIDTH;
	thearch.anyregalloc = anyregalloc;
	thearch.betypeinit = betypeinit;
	thearch.bgen = bgen;
	thearch.cgen = cgen;
	thearch.cgen_call = cgen_call;
	thearch.cgen_callinter = cgen_callinter;
	thearch.cgen_ret = cgen_ret;
	thearch.clearfat = clearfat;
	thearch.defframe = defframe;
	thearch.excise = excise;
	thearch.expandchecks = expandchecks;
	thearch.gclean = gclean;
	thearch.ginit = ginit;
	thearch.gins = gins;
	thearch.ginscall = ginscall;
	thearch.igen = igen;
	thearch.linkarchinit = linkarchinit;
	thearch.peep = peep;
	thearch.proginfo = proginfo;
	thearch.regalloc = regalloc;
	thearch.regfree = regfree;
	thearch.regtyp = regtyp;
	thearch.sameaddr = sameaddr;
	thearch.smallindir = smallindir;
	thearch.stackaddr = stackaddr;
	thearch.excludedregs = excludedregs;
	thearch.RtoB = RtoB;
	thearch.FtoB = RtoB;
	thearch.BtoR = BtoR;
	thearch.BtoF = BtoF;
	thearch.optoas = optoas;
	thearch.doregbits = doregbits;
	thearch.regnames = regnames;
	
	gcmain(argc, argv);
}
