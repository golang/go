// Inferno utils/5l/obj.c
// http://code.google.com/p/inferno-os/source/browse/utils/5l/obj.c
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

// Reading object files.

#include	"l.h"
#include	"../ld/lib.h"
#include	"../ld/elf.h"
#include	"../ld/dwarf.h"
#include	<ar.h>

void
main(int argc, char **argv)
{
	linkarchinit();
	ldmain(argc, argv);
}

void
linkarchinit(void)
{
	thestring = getgoarch();
	if(strcmp(thestring, "ppc64le") == 0)
		thelinkarch = &linkppc64le;
	else
		thelinkarch = &linkppc64;

	thearch.thechar = thechar;
	thearch.ptrsize = thelinkarch->ptrsize;
	thearch.intsize = thelinkarch->ptrsize;
	thearch.regsize = thelinkarch->regsize;
	thearch.funcalign = FuncAlign;
	thearch.maxalign = MaxAlign;
	thearch.minlc = MINLC;
	thearch.dwarfregsp = DWARFREGSP;

	thearch.adddynlib = adddynlib;
	thearch.adddynrel = adddynrel;
	thearch.adddynsym = adddynsym;
	thearch.archinit = archinit;
	thearch.archreloc = archreloc;
	thearch.archrelocvariant = archrelocvariant;
	thearch.asmb = asmb;
	thearch.elfreloc1 = elfreloc1;
	thearch.elfsetupplt = elfsetupplt;
	thearch.gentext = gentext;
	thearch.listinit = listinit;
	thearch.machoreloc1 = machoreloc1;
	if(thelinkarch == &linkppc64le) {
		thearch.lput = lputl;
		thearch.wput = wputl;
		thearch.vput = vputl;
	} else {
		thearch.lput = lputb;
		thearch.wput = wputb;
		thearch.vput = vputb;
	}

	// TODO(austin): ABI v1 uses /usr/lib/ld.so.1
	thearch.linuxdynld = "/lib64/ld64.so.1";
	thearch.freebsddynld = "XXX";
	thearch.openbsddynld = "XXX";
	thearch.netbsddynld = "XXX";
	thearch.dragonflydynld = "XXX";
	thearch.solarisdynld = "XXX";
}

void
archinit(void)
{
	// getgoextlinkenabled is based on GO_EXTLINK_ENABLED when
	// Go was built; see ../../make.bash.
	if(linkmode == LinkAuto && strcmp(getgoextlinkenabled(), "0") == 0)
		linkmode = LinkInternal;

	switch(HEADTYPE) {
	default:
		if(linkmode == LinkAuto)
			linkmode = LinkInternal;
		if(linkmode == LinkExternal && strcmp(getgoextlinkenabled(), "1") != 0)
			sysfatal("cannot use -linkmode=external with -H %s", headstr(HEADTYPE));
		break;
	}

	switch(HEADTYPE) {
	default:
		diag("unknown -H option");
		errorexit();
	case Hplan9:	/* plan 9 */
		HEADR = 32L;
		if(INITTEXT == -1)
			INITTEXT = 4128;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4096;
		break;
	case Hlinux:	/* ppc64 elf */
		if(strcmp(thestring, "ppc64") == 0)
			debug['d'] = 1;	// TODO(austin): ELF ABI v1 not supported yet
		elfinit();
		HEADR = ELFRESERVE;
		if(INITTEXT == -1)
			INITTEXT = 0x10000 + HEADR;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 0x10000;
		break;
	case Hnacl:
		elfinit();
		HEADR = 0x10000;
		funcalign = 16;
		if(INITTEXT == -1)
			INITTEXT = 0x20000;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 0x10000;
		break;
	}
	if(INITDAT != 0 && INITRND != 0)
		print("warning: -D0x%ux is ignored because of -R0x%ux\n",
			INITDAT, INITRND);
}
