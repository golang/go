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
		arch.thelinkarch = thelinkarch;
		thestring = "amd64p32";
		arch.thestring = "amd64p32";
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
	arch.proginfo = proginfo;
	arch.regalloc = regalloc;
	arch.regfree = regfree;
	arch.regopt = regopt;
	arch.regtyp = regtyp;
	arch.sameaddr = sameaddr;
	arch.smallindir = smallindir;
	arch.stackaddr = stackaddr;
	
	gcmain(argc, argv);
}
