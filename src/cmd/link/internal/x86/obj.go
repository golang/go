// Inferno utils/8l/obj.c
// http://code.google.com/p/inferno-os/source/browse/utils/8l/obj.c
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

package x86

import (
	"cmd/internal/obj"
	"cmd/link/internal/ld"
	"fmt"
	"log"
)

// Reading object files.

func Main() {
	linkarchinit()
	ld.Ldmain()
}

func linkarchinit() {
	ld.Thestring = "386"
	ld.Thelinkarch = &ld.Link386

	ld.Thearch.Thechar = thechar
	ld.Thearch.Ptrsize = ld.Thelinkarch.Ptrsize
	ld.Thearch.Intsize = ld.Thelinkarch.Ptrsize
	ld.Thearch.Regsize = ld.Thelinkarch.Regsize
	ld.Thearch.Funcalign = FuncAlign
	ld.Thearch.Maxalign = MaxAlign
	ld.Thearch.Minlc = MINLC
	ld.Thearch.Dwarfregsp = DWARFREGSP
	ld.Thearch.Dwarfreglr = DWARFREGLR

	ld.Thearch.Adddynrel = adddynrel
	ld.Thearch.Archinit = archinit
	ld.Thearch.Archreloc = archreloc
	ld.Thearch.Archrelocvariant = archrelocvariant
	ld.Thearch.Asmb = asmb
	ld.Thearch.Elfreloc1 = elfreloc1
	ld.Thearch.Elfsetupplt = elfsetupplt
	ld.Thearch.Gentext = gentext
	ld.Thearch.Machoreloc1 = machoreloc1
	ld.Thearch.PEreloc1 = pereloc1
	ld.Thearch.Lput = ld.Lputl
	ld.Thearch.Wput = ld.Wputl
	ld.Thearch.Vput = ld.Vputl

	ld.Thearch.Linuxdynld = "/lib/ld-linux.so.2"
	ld.Thearch.Freebsddynld = "/usr/libexec/ld-elf.so.1"
	ld.Thearch.Openbsddynld = "/usr/libexec/ld.so"
	ld.Thearch.Netbsddynld = "/usr/libexec/ld.elf_so"
	ld.Thearch.Solarisdynld = "/lib/ld.so.1"
}

func archinit() {
	// getgoextlinkenabled is based on GO_EXTLINK_ENABLED when
	// Go was built; see ../../make.bash.
	if ld.Linkmode == ld.LinkAuto && obj.Getgoextlinkenabled() == "0" {
		ld.Linkmode = ld.LinkInternal
	}

	if ld.Buildmode == ld.BuildmodeCShared || ld.Buildmode == ld.BuildmodePIE || ld.DynlinkingGo() {
		ld.Linkmode = ld.LinkExternal
		got := ld.Linklookup(ld.Ctxt, "_GLOBAL_OFFSET_TABLE_", 0)
		got.Type = obj.SDYNIMPORT
		got.Reachable = true
	}

	switch ld.HEADTYPE {
	default:
		if ld.Linkmode == ld.LinkAuto {
			ld.Linkmode = ld.LinkInternal
		}
		if ld.Linkmode == ld.LinkExternal && obj.Getgoextlinkenabled() != "1" {
			log.Fatalf("cannot use -linkmode=external with -H %s", ld.Headstr(int(ld.HEADTYPE)))
		}

	case obj.Hdarwin,
		obj.Hfreebsd,
		obj.Hlinux,
		obj.Hnetbsd,
		obj.Hopenbsd,
		obj.Hwindows:
		break
	}

	switch ld.HEADTYPE {
	default:
		ld.Exitf("unknown -H option: %v", ld.HEADTYPE)

	case obj.Hplan9: /* plan 9 */
		ld.HEADR = 32

		if ld.INITTEXT == -1 {
			ld.INITTEXT = 4096 + 32
		}
		if ld.INITDAT == -1 {
			ld.INITDAT = 0
		}
		if ld.INITRND == -1 {
			ld.INITRND = 4096
		}

	case obj.Hdarwin: /* apple MACH */
		ld.Machoinit()

		ld.HEADR = ld.INITIAL_MACHO_HEADR
		if ld.INITTEXT == -1 {
			ld.INITTEXT = 4096 + int64(ld.HEADR)
		}
		if ld.INITDAT == -1 {
			ld.INITDAT = 0
		}
		if ld.INITRND == -1 {
			ld.INITRND = 4096
		}

	case obj.Hlinux, /* elf32 executable */
		obj.Hfreebsd,
		obj.Hnetbsd,
		obj.Hopenbsd:
		ld.Elfinit()

		ld.HEADR = ld.ELFRESERVE
		if ld.INITTEXT == -1 {
			ld.INITTEXT = 0x08048000 + int64(ld.HEADR)
		}
		if ld.INITDAT == -1 {
			ld.INITDAT = 0
		}
		if ld.INITRND == -1 {
			ld.INITRND = 4096
		}

	case obj.Hnacl:
		ld.Elfinit()
		ld.HEADR = 0x10000
		ld.Funcalign = 32
		if ld.INITTEXT == -1 {
			ld.INITTEXT = 0x20000
		}
		if ld.INITDAT == -1 {
			ld.INITDAT = 0
		}
		if ld.INITRND == -1 {
			ld.INITRND = 0x10000
		}

	case obj.Hwindows: /* PE executable */
		ld.Peinit()

		ld.HEADR = ld.PEFILEHEADR
		if ld.INITTEXT == -1 {
			ld.INITTEXT = ld.PEBASE + int64(ld.PESECTHEADR)
		}
		if ld.INITDAT == -1 {
			ld.INITDAT = 0
		}
		if ld.INITRND == -1 {
			ld.INITRND = ld.PESECTALIGN
		}
	}

	if ld.INITDAT != 0 && ld.INITRND != 0 {
		fmt.Printf("warning: -D0x%x is ignored because of -R0x%x\n", uint64(ld.INITDAT), uint32(ld.INITRND))
	}
}
