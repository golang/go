// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include "gg.h"

int	thechar	= '6';
char*	thestring	= "amd64";
LinkArch*	thelinkarch = &linkamd64;

void
linkarchinit(void)
{
	if(strcmp(getgoarch(), "amd64p32") == 0) {
		thelinkarch = &linkamd64p32;
		thearch.thelinkarch = thelinkarch;
		thestring = "amd64p32";
		thearch.thestring = "amd64p32";
	}
}

vlong MAXWIDTH = 1LL<<50;

int	addptr = AADDQ;
int	movptr = AMOVQ;
int	leaptr = ALEAQ;
int	cmpptr = ACMPQ;

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
	if(strcmp(getgoarch(), "amd64p32") == 0) {
		widthptr = 4;
		widthint = 4;
		addptr = AADDL;
		movptr = AMOVL;
		leaptr = ALEAL;
		cmpptr = ACMPL;
		typedefs[0].sameas = TINT32;
		typedefs[1].sameas = TUINT32;
		typedefs[2].sameas = TUINT32;
		
	}

	listinit6();
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
	thearch.FtoB = FtoB;
	thearch.BtoR = BtoR;
	thearch.BtoF = BtoF;
	thearch.optoas = optoas;
	thearch.doregbits = doregbits;
	thearch.regnames = regnames;
	
	gcmain(argc, argv);
}
