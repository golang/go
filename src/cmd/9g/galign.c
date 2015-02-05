// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include "gg.h"

int	thechar	= '9';
char*	thestring = "ppc64";
LinkArch*	thelinkarch;

void
linkarchinit(void)
{
	thestring = getgoarch();
	thearch.thestring = thestring;
	if(strcmp(thestring, "ppc64le") == 0)
		thelinkarch = &linkppc64le;
	else
		thelinkarch = &linkppc64;
	thearch.thelinkarch = thelinkarch;
}

vlong MAXWIDTH = 1LL<<50;

/*
 * go declares several platform-specific type aliases:
 * int, uint, float, and uintptr
 */
Typedef	typedefs[] =
{
	{"int",		TINT,		TINT64},
	{"uint",		TUINT,		TUINT64},
	{"uintptr",	TUINTPTR,	TUINT64},
	{0}
};

void
betypeinit(void)
{
	widthptr = 8;
	widthint = 8;
	widthreg = 8;

	listinit9();
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
