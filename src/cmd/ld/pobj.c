// Inferno utils/6l/obj.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/obj.c
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

#define	EXTERN
#include	"l.h"
#include	"../ld/lib.h"
#include	"../ld/elf.h"
#include	"../ld/macho.h"
#include	"../ld/dwarf.h"
#include	"../ld/pe.h"
#include	<ar.h>

char	*noname		= "<none>";
char*	paramspace	= "FP";

void
main(int argc, char *argv[])
{
	int i;

	linkarchinit();
	ctxt = linknew(thelinkarch);
	ctxt->thechar = thechar;
	ctxt->thestring = thestring;
	ctxt->diag = diag;
	ctxt->bso = &bso;

	Binit(&bso, 1, OWRITE);
	listinit();
	memset(debug, 0, sizeof(debug));
	nerrors = 0;
	outfile = nil;
	HEADTYPE = -1;
	INITTEXT = -1;
	INITDAT = -1;
	INITRND = -1;
	INITENTRY = 0;
	linkmode = LinkAuto;
	
	// For testing behavior of go command when tools crash.
	// Undocumented, not in standard flag parser to avoid
	// exposing in usage message.
	for(i=1; i<argc; i++)
		if(strcmp(argv[i], "-crash_for_testing") == 0)
			*(volatile int*)0 = 0;
	
	if(thechar == '5' && ctxt->goarm == 5)
		debug['F'] = 1;

	flagcount("1", "use alternate profiling code", &debug['1']);
	if(thechar == '6')
		flagcount("8", "assume 64-bit addresses", &debug['8']);
	flagfn1("B", "info: define ELF NT_GNU_BUILD_ID note", addbuildinfo);
	flagcount("C", "check Go calls to C code", &debug['C']);
	flagint64("D", "addr: data address", &INITDAT);
	flagstr("E", "sym: entry symbol", &INITENTRY);
	if(thechar == '5')
		flagcount("G", "debug pseudo-ops", &debug['G']);
	flagfn1("I", "interp: set ELF interp", setinterp);
	flagfn1("L", "dir: add dir to library path", Lflag);
	flagfn1("H", "head: header type", setheadtype);
	flagcount("K", "add stack underflow checks", &debug['K']);
	if(thechar == '5')
		flagcount("M", "disable software div/mod", &debug['M']);
	flagcount("O", "print pc-line tables", &debug['O']);
	flagcount("Q", "debug byte-register code gen", &debug['Q']);
	if(thechar == '5')
		flagcount("P", "debug code generation", &debug['P']);
	flagint32("R", "rnd: address rounding", &INITRND);
	flagcount("S", "check type signatures", &debug['S']);
	flagint64("T", "addr: text address", &INITTEXT);
	flagfn0("V", "print version and exit", doversion);
	flagcount("W", "disassemble input", &debug['W']);
	flagfn2("X", "name value: define string data", addstrdata);
	flagcount("Z", "clear stack frame on entry", &debug['Z']);
	flagcount("a", "disassemble output", &debug['a']);
	flagcount("c", "dump call graph", &debug['c']);
	flagcount("d", "disable dynamic executable", &debug['d']);
	flagstr("extld", "ld: linker to run in external mode", &extld);
	flagstr("extldflags", "ldflags: flags for external linker", &extldflags);
	flagcount("f", "ignore version mismatch", &debug['f']);
	flagcount("g", "disable go package data checks", &debug['g']);
	flagstr("installsuffix", "suffix: pkg directory suffix", &flag_installsuffix);
	flagstr("k", "sym: set field tracking symbol", &tracksym);
	flagfn1("linkmode", "mode: set link mode (internal, external, auto)", setlinkmode);
	flagcount("n", "dump symbol table", &debug['n']);
	flagstr("o", "outfile: set output file", &outfile);
	flagstr("r", "dir1:dir2:...: set ELF dynamic linker search path", &rpath);
	flagcount("race", "enable race detector", &flag_race);
	flagcount("s", "disable symbol table", &debug['s']);
	if(thechar == '5' || thechar == '6')
		flagcount("shared", "generate shared object (implies -linkmode external)", &flag_shared);
	flagstr("tmpdir", "dir: leave temporary files in this directory", &tmpdir);
	flagcount("u", "reject unsafe packages", &debug['u']);
	flagcount("v", "print link trace", &debug['v']);
	flagcount("w", "disable DWARF generation", &debug['w']);
	
	flagparse(&argc, &argv, usage);
	ctxt->bso = &bso;
	ctxt->debugdivmod = debug['M'];
	ctxt->debugfloat = debug['F'];
	ctxt->debughist = debug['O'];
	ctxt->debugpcln = debug['O'];
	ctxt->debugread = debug['W'];
	ctxt->debugstack = debug['K'];
	ctxt->debugvlog = debug['v'];

	if(argc != 1)
		usage();

	if(outfile == nil) {
		if(HEADTYPE == Hwindows)
			outfile = smprint("%c.out.exe", thechar);
		else
			outfile = smprint("%c.out", thechar);
	}
	libinit(); // creates outfile

	if(HEADTYPE == -1)
		HEADTYPE = headtype(goos);
	ctxt->headtype = HEADTYPE;
	if(headstring == nil)
		headstring = headstr(HEADTYPE);

	archinit();
	ctxt->debugfloat = debug['F'];

	if(debug['v'])
		Bprint(&bso, "HEADER = -H%d -T0x%llux -D0x%llux -R0x%ux\n",
			HEADTYPE, INITTEXT, INITDAT, INITRND);
	Bflush(&bso);

	cbp = buf.cbuf;
	cbc = sizeof(buf.cbuf);

	addlibpath(ctxt, "command line", "command line", argv[0], "main");
	loadlib();
	
	if(thechar == '5') {
		// mark some functions that are only referenced after linker code editing
		if(debug['F'])
			mark(linkrlookup(ctxt, "_sfloat", 0));
		mark(linklookup(ctxt, "runtime.read_tls_fallback", 0));
	}

	checkgo();
	deadcode();
	callgraph();
	paramspace = "SP";	/* (FP) now (SP) on output */

	doelf();
	if(HEADTYPE == Hdarwin)
		domacho();
	dostkcheck();
	if(HEADTYPE == Hwindows)
		dope();
	addexport();
	textaddress();
	pclntab();
	symtab();
	dodata();
	address();
	doweak();
	reloc();
	asmb();
	undef();
	hostlink();
	if(debug['v']) {
		Bprint(&bso, "%5.2f cpu time\n", cputime());
		Bprint(&bso, "%d symbols\n", ctxt->nsymbol);
		Bprint(&bso, "%d sizeof adr\n", sizeof(Addr));
		Bprint(&bso, "%d sizeof prog\n", sizeof(Prog));
		Bprint(&bso, "%lld liveness data\n", liveness);
	}
	Bflush(&bso);

	errorexit();
}
