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
	arch.thestring = thestring;
	if(strcmp(thestring, "ppc64le") == 0)
		thelinkarch = &linkppc64le;
	else
		thelinkarch = &linkppc64;
	arch.thelinkarch = thelinkarch;
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
	arch.thechar = thechar;
	arch.thestring = thestring;
	arch.thelinkarch = thelinkarch;
	arch.typedefs = typedefs;
	arch.MAXWIDTH = MAXWIDTH;
	arch.afunclit = afunclit;
	arch.anyregalloc = anyregalloc;
	arch.betypeinit = betypeinit;
	arch.bgen = bgen;
	arch.cgen = cgen;
	arch.cgen_asop = cgen_asop;
	arch.cgen_call = cgen_call;
	arch.cgen_callinter = cgen_callinter;
	arch.cgen_ret = cgen_ret;
	arch.clearfat = clearfat;
	arch.clearp = clearp;
	arch.defframe = defframe;
	arch.dgostringptr = dgostringptr;
	arch.dgostrlitptr = dgostrlitptr;
	arch.dsname = dsname;
	arch.dsymptr = dsymptr;
	arch.dumpdata = dumpdata;
	arch.dumpit = dumpit;
	arch.excise = excise;
	arch.expandchecks = expandchecks;
	arch.fixautoused = fixautoused;
	arch.gclean = gclean;
	arch.gdata = gdata;
	arch.gdatacomplex = gdatacomplex;
	arch.gdatastring = gdatastring;
	arch.ggloblnod = ggloblnod;
	arch.ggloblsym = ggloblsym;
	arch.ginit = ginit;
	arch.gins = gins;
	arch.ginscall = ginscall;
	arch.gjmp = gjmp;
	arch.gtrack = gtrack;
	arch.gused = gused;
	arch.igen = igen;
	arch.isfat = isfat;
	arch.linkarchinit = linkarchinit;
	arch.markautoused = markautoused;
	arch.naddr = naddr;
	arch.newplist = newplist;
	arch.nodarg = nodarg;
	arch.patch = patch;
	arch.proginfo = proginfo;
	arch.regalloc = regalloc;
	arch.regfree = regfree;
	arch.regopt = regopt;
	arch.regtyp = regtyp;
	arch.sameaddr = sameaddr;
	arch.smallindir = smallindir;
	arch.stackaddr = stackaddr;
	arch.unpatch = unpatch;
	
	gcmain(argc, argv);
}
