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
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
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
	"bufio"
	"cmd/internal/obj"
	"cmd/internal/sys"
	"flag"
	"fmt"
	"os"
	"strings"
)

var (
	pkglistfornote []byte
	buildid        string
)

func Ldmain() {
	ctxt := linknew(SysArch)
	ctxt.Bso = bufio.NewWriter(os.Stdout)

	Debug = [128]bool{}
	nerrors = 0
	HEADTYPE = -1
	Linkmode = LinkAuto

	// For testing behavior of go command when tools crash silently.
	// Undocumented, not in standard flag parser to avoid
	// exposing in usage message.
	for _, arg := range os.Args {
		if arg == "-crash_for_testing" {
			os.Exit(2)
		}
	}

	if SysArch.Family == sys.AMD64 && obj.Getgoos() == "plan9" {
		flag.BoolVar(&Debug['8'], "8", false, "use 64-bit addresses in symbol table")
	}
	obj.Flagfn1("B", "add an ELF NT_GNU_BUILD_ID `note` when using ELF", addbuildinfo)
	flag.BoolVar(&Debug['C'], "C", false, "check Go calls to C code")
	flag.Int64Var(&INITDAT, "D", -1, "set data segment `address`")
	flag.StringVar(&INITENTRY, "E", "", "set `entry` symbol name")
	obj.Flagfn1("I", "use `linker` as ELF dynamic linker", setinterp)
	obj.Flagfn1("L", "add specified `directory` to library path", func(a string) { Lflag(ctxt, a) })
	obj.Flagfn1("H", "set header `type`", setheadtype)
	flag.IntVar(&INITRND, "R", -1, "set address rounding `quantum`")
	flag.Int64Var(&INITTEXT, "T", -1, "set text segment `address`")
	obj.Flagfn0("V", "print version and exit", doversion)
	obj.Flagfn1("X", "add string value `definition` of the form importpath.name=value", func(s string) { addstrdata1(ctxt, s) })
	flag.BoolVar(&Debug['a'], "a", false, "disassemble output")
	flag.StringVar(&buildid, "buildid", "", "record `id` as Go toolchain build id")
	flag.Var(&Buildmode, "buildmode", "set build `mode`")
	flag.BoolVar(&Debug['c'], "c", false, "dump call graph")
	flag.BoolVar(&Debug['d'], "d", false, "disable dynamic executable")
	flag.BoolVar(&flag_dumpdep, "dumpdep", false, "dump symbol dependency graph")
	flag.StringVar(&extar, "extar", "", "archive program for buildmode=c-archive")
	flag.StringVar(&extld, "extld", "", "use `linker` when linking in external mode")
	flag.StringVar(&extldflags, "extldflags", "", "pass `flags` to external linker")
	flag.BoolVar(&Debug['f'], "f", false, "ignore version mismatch")
	flag.BoolVar(&Debug['g'], "g", false, "disable go package data checks")
	flag.BoolVar(&Debug['h'], "h", false, "halt on error")
	flag.StringVar(&flag_installsuffix, "installsuffix", "", "set package directory `suffix`")
	flag.StringVar(&tracksym, "k", "", "set field tracking `symbol`")
	flag.StringVar(&libgccfile, "libgcc", "", "compiler support lib for internal linking; use \"none\" to disable")
	obj.Flagfn1("linkmode", "set link `mode` (internal, external, auto)", setlinkmode)
	flag.BoolVar(&Linkshared, "linkshared", false, "link against installed Go shared libraries")
	flag.BoolVar(&flag_msan, "msan", false, "enable MSan interface")
	flag.BoolVar(&Debug['n'], "n", false, "dump symbol table")
	flag.StringVar(&outfile, "o", "", "write output to `file`")
	flag.Var(&rpath, "r", "set the ELF dynamic linker search `path` to dir1:dir2:...")
	flag.BoolVar(&flag_race, "race", false, "enable race detector")
	flag.BoolVar(&Debug['s'], "s", false, "disable symbol table")
	var flagShared bool
	if SysArch.InFamily(sys.ARM, sys.AMD64) {
		flag.BoolVar(&flagShared, "shared", false, "generate shared object (implies -linkmode external)")
	}
	flag.StringVar(&tmpdir, "tmpdir", "", "use `directory` for temporary files")
	flag.BoolVar(&Debug['u'], "u", false, "reject unsafe packages")
	obj.Flagcount("v", "print link trace", &ctxt.Debugvlog)
	flag.BoolVar(&Debug['w'], "w", false, "disable DWARF generation")

	flag.StringVar(&cpuprofile, "cpuprofile", "", "write cpu profile to `file`")
	flag.StringVar(&memprofile, "memprofile", "", "write memory profile to `file`")
	flag.Int64Var(&memprofilerate, "memprofilerate", 0, "set runtime.MemProfileRate to `rate`")

	obj.Flagparse(usage)

	startProfile()
	ctxt.Bso = ctxt.Bso
	if flagShared {
		if Buildmode == BuildmodeUnset {
			Buildmode = BuildmodeCShared
		} else if Buildmode != BuildmodeCShared {
			Exitf("-shared and -buildmode=%s are incompatible", Buildmode.String())
		}
	}
	if Buildmode == BuildmodeUnset {
		Buildmode = BuildmodeExe
	}

	if Buildmode != BuildmodeShared && flag.NArg() != 1 {
		usage()
	}

	if outfile == "" {
		outfile = "a.out"
		if HEADTYPE == obj.Hwindows {
			outfile += ".exe"
		}
	}

	libinit(ctxt) // creates outfile

	if HEADTYPE == -1 {
		HEADTYPE = int32(headtype(goos))
	}
	ctxt.Headtype = int(HEADTYPE)
	if headstring == "" {
		headstring = Headstr(int(HEADTYPE))
	}

	Thearch.Archinit(ctxt)

	if Linkshared && !Iself {
		Exitf("-linkshared can only be used on elf systems")
	}

	if ctxt.Debugvlog != 0 {
		fmt.Fprintf(ctxt.Bso, "HEADER = -H%d -T0x%x -D0x%x -R0x%x\n", HEADTYPE, uint64(INITTEXT), uint64(INITDAT), uint32(INITRND))
	}
	ctxt.Bso.Flush()

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
			pkglistfornote = append(pkglistfornote, pkgpath...)
			pkglistfornote = append(pkglistfornote, '\n')
			addlibpath(ctxt, "command line", "command line", file, pkgpath, "")
		}
	} else {
		addlibpath(ctxt, "command line", "command line", flag.Arg(0), "main", "")
	}
	ctxt.loadlib()

	ctxt.checkstrdata()
	deadcode(ctxt)
	fieldtrack(ctxt)
	ctxt.callgraph()

	ctxt.doelf()
	if HEADTYPE == obj.Hdarwin {
		ctxt.domacho()
	}
	ctxt.dostkcheck()
	if HEADTYPE == obj.Hwindows {
		ctxt.dope()
	}
	ctxt.addexport()
	Thearch.Gentext(ctxt) // trampolines, call stubs, etc.
	ctxt.textbuildid()
	ctxt.textaddress()
	ctxt.pclntab()
	ctxt.findfunctab()
	ctxt.symtab()
	ctxt.dodata()
	ctxt.address()
	ctxt.reloc()
	Thearch.Asmb(ctxt)
	ctxt.undef()
	ctxt.hostlink()
	ctxt.archive()
	if ctxt.Debugvlog != 0 {
		fmt.Fprintf(ctxt.Bso, "%5.2f cpu time\n", obj.Cputime())
		fmt.Fprintf(ctxt.Bso, "%d symbols\n", len(ctxt.Allsym))
		fmt.Fprintf(ctxt.Bso, "%d liveness data\n", liveness)
	}

	ctxt.Bso.Flush()

	errorexit()
}
