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

package ld

import (
	"cmd/internal/obj"
	"flag"
	"fmt"
	"os"
	"strings"
)

// Reading object files.

func Ldmain() {
	Ctxt = linknew(Thelinkarch)
	Ctxt.Thechar = int32(Thearch.Thechar)
	Ctxt.Thestring = Thestring
	Ctxt.Diag = Diag
	Ctxt.Bso = &Bso

	Bso = *Binitw(os.Stdout)
	Debug = [128]int{}
	nerrors = 0
	outfile = ""
	HEADTYPE = -1
	INITTEXT = -1
	INITDAT = -1
	INITRND = -1
	INITENTRY = ""
	Linkmode = LinkAuto

	// For testing behavior of go command when tools crash.
	// Undocumented, not in standard flag parser to avoid
	// exposing in usage message.
	for _, arg := range os.Args {
		if arg == "-crash_for_testing" {
			*(*int)(nil) = 0
		}
	}

	if Thearch.Thechar == '5' && Ctxt.Goarm == 5 {
		Debug['F'] = 1
	}

	obj.Flagcount("1", "use alternate profiling code", &Debug['1'])
	if Thearch.Thechar == '6' {
		obj.Flagcount("8", "assume 64-bit addresses", &Debug['8'])
	}
	obj.Flagfn1("B", "info: define ELF NT_GNU_BUILD_ID note", addbuildinfo)
	obj.Flagcount("C", "check Go calls to C code", &Debug['C'])
	obj.Flagint64("D", "addr: data address", &INITDAT)
	obj.Flagstr("E", "sym: entry symbol", &INITENTRY)
	if Thearch.Thechar == '5' {
		obj.Flagcount("G", "debug pseudo-ops", &Debug['G'])
	}
	obj.Flagfn1("I", "interp: set ELF interp", setinterp)
	obj.Flagfn1("L", "dir: add dir to library path", Lflag)
	obj.Flagfn1("H", "head: header type", setheadtype)
	obj.Flagcount("K", "add stack underflow checks", &Debug['K'])
	if Thearch.Thechar == '5' {
		obj.Flagcount("M", "disable software div/mod", &Debug['M'])
	}
	obj.Flagcount("O", "print pc-line tables", &Debug['O'])
	obj.Flagcount("Q", "debug byte-register code gen", &Debug['Q'])
	if Thearch.Thechar == '5' {
		obj.Flagcount("P", "debug code generation", &Debug['P'])
	}
	obj.Flagint32("R", "rnd: address rounding", &INITRND)
	obj.Flagcount("nil", "check type signatures", &Debug['S'])
	obj.Flagint64("T", "addr: text address", &INITTEXT)
	obj.Flagfn0("V", "print version and exit", doversion)
	obj.Flagcount("W", "disassemble input", &Debug['W'])
	obj.Flagfn1("X", "name value: define string data", addstrdata1)
	obj.Flagcount("Z", "clear stack frame on entry", &Debug['Z'])
	obj.Flagcount("a", "disassemble output", &Debug['a'])
	flag.Var(&Buildmode, "buildmode", "build mode to use")
	obj.Flagcount("c", "dump call graph", &Debug['c'])
	obj.Flagcount("d", "disable dynamic executable", &Debug['d'])
	obj.Flagstr("extld", "ld: linker to run in external mode", &extld)
	obj.Flagstr("extldflags", "ldflags: flags for external linker", &extldflags)
	obj.Flagcount("f", "ignore version mismatch", &Debug['f'])
	obj.Flagcount("g", "disable go package data checks", &Debug['g'])
	obj.Flagstr("installsuffix", "suffix: pkg directory suffix", &flag_installsuffix)
	obj.Flagstr("k", "sym: set field tracking symbol", &tracksym)
	obj.Flagfn1("linkmode", "mode: set link mode (internal, external, auto)", setlinkmode)
	flag.BoolVar(&Linkshared, "linkshared", false, "link against installed Go shared libraries")
	obj.Flagcount("n", "dump symbol table", &Debug['n'])
	obj.Flagstr("o", "outfile: set output file", &outfile)
	flag.Var(&rpath, "r", "dir1:dir2:...: set ELF dynamic linker search path")
	obj.Flagcount("race", "enable race detector", &flag_race)
	obj.Flagcount("s", "disable symbol table", &Debug['s'])
	var flagShared int
	if Thearch.Thechar == '5' || Thearch.Thechar == '6' {
		obj.Flagcount("shared", "generate shared object (implies -linkmode external)", &flagShared)
	}
	obj.Flagstr("tmpdir", "dir: leave temporary files in this directory", &tmpdir)
	obj.Flagcount("u", "reject unsafe packages", &Debug['u'])
	obj.Flagcount("v", "print link trace", &Debug['v'])
	obj.Flagcount("w", "disable DWARF generation", &Debug['w'])

	// Clumsy hack to preserve old behavior of -X taking two arguments.
	for i := 0; i < len(os.Args); i++ {
		arg := os.Args[i]
		if (arg == "--X" || arg == "-X") && i+2 < len(os.Args) {
			os.Args[i+2] = "-X=VALUE:" + os.Args[i+2]
			i += 2
		} else if (strings.HasPrefix(arg, "--X=") || strings.HasPrefix(arg, "-X=")) && i+1 < len(os.Args) {
			os.Args[i+1] = "-X=VALUE:" + os.Args[i+1]
			i++
		}
	}
	obj.Flagstr("cpuprofile", "file: write cpu profile to file", &cpuprofile)
	obj.Flagstr("memprofile", "file: write memory profile to file", &memprofile)
	obj.Flagparse(usage)
	startProfile()
	Ctxt.Bso = &Bso
	Ctxt.Debugvlog = int32(Debug['v'])
	if flagShared != 0 {
		if Buildmode == BuildmodeExe {
			Buildmode = BuildmodeCShared
		} else if Buildmode != BuildmodeCShared {
			Diag("-shared and -buildmode=%s are incompatible\n", Buildmode.String())
			Errorexit()
		}
	}

	if Buildmode != BuildmodeShared && flag.NArg() != 1 {
		usage()
	}

	if outfile == "" {
		if HEADTYPE == Hwindows {
			outfile = fmt.Sprintf("%c.out.exe", Thearch.Thechar)
		} else {
			outfile = fmt.Sprintf("%c.out", Thearch.Thechar)
		}
	}

	libinit() // creates outfile

	if HEADTYPE == -1 {
		HEADTYPE = int32(headtype(goos))
	}
	Ctxt.Headtype = int(HEADTYPE)
	if headstring == "" {
		headstring = Headstr(int(HEADTYPE))
	}

	Thearch.Archinit()

	if Linkshared && !Iself {
		Diag("-linkshared can only be used on elf systems")
		Errorexit()
	}

	if Debug['v'] != 0 {
		fmt.Fprintf(&Bso, "HEADER = -H%d -T0x%x -D0x%x -R0x%x\n", HEADTYPE, uint64(INITTEXT), uint64(INITDAT), uint32(INITRND))
	}
	Bflush(&Bso)

	if Buildmode == BuildmodeShared {
		for i := 0; i < flag.NArg(); i++ {
			arg := flag.Arg(i)
			parts := strings.SplitN(arg, "=", 2)
			var pkgpath, file string
			if len(parts) == 1 {
				pkgpath, file = "main", arg
			} else {
				pkgpath, file = parts[0], parts[1]
			}
			addlibpath(Ctxt, "command line", "command line", file, pkgpath, "")
		}
	} else {
		addlibpath(Ctxt, "command line", "command line", flag.Arg(0), "main", "")
	}
	loadlib()

	if Thearch.Thechar == '5' {
		// mark some functions that are only referenced after linker code editing
		if Debug['F'] != 0 {
			mark(Linkrlookup(Ctxt, "_sfloat", 0))
		}
		mark(Linklookup(Ctxt, "runtime.read_tls_fallback", 0))
	}

	checkgo()
	deadcode()
	callgraph()

	doelf()
	if HEADTYPE == Hdarwin {
		domacho()
	}
	dostkcheck()
	if HEADTYPE == Hwindows {
		dope()
	}
	addexport()
	Thearch.Gentext() // trampolines, call stubs, etc.
	textaddress()
	pclntab()
	findfunctab()
	symtab()
	dodata()
	address()
	doweak()
	reloc()
	Thearch.Asmb()
	undef()
	hostlink()
	archive()
	if Debug['v'] != 0 {
		fmt.Fprintf(&Bso, "%5.2f cpu time\n", obj.Cputime())
		fmt.Fprintf(&Bso, "%d symbols\n", Ctxt.Nsymbol)
		fmt.Fprintf(&Bso, "%d liveness data\n", liveness)
	}

	Bflush(&Bso)

	Errorexit()
}
