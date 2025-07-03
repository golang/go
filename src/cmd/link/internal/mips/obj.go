// Inferno utils/5l/obj.c
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/5l/obj.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2016 The Go Authors. All rights reserved.
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

package mips

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/ld"
	"cmd/link/internal/loader"
	"internal/buildcfg"
)

func Init() (*sys.Arch, ld.Arch) {
	arch := sys.ArchMIPS
	musl := "/lib/ld-musl-mips.so.1"
	if buildcfg.GOARCH == "mipsle" {
		arch = sys.ArchMIPSLE
		musl = "/lib/ld-musl-mipsel.so.1"
	}

	theArch := ld.Arch{
		Funcalign:  FuncAlign,
		Maxalign:   MaxAlign,
		Minalign:   MinAlign,
		Dwarfregsp: DWARFREGSP,
		Dwarfreglr: DWARFREGLR,

		Adddynrel:        adddynrel,
		Archinit:         archinit,
		Archreloc:        archreloc,
		Archrelocvariant: archrelocvariant,
		Extreloc:         extreloc,
		Gentext:          gentext,
		Machoreloc1:      machoreloc1,

		ELF: ld.ELFArch{
			Linuxdynld:     "/lib/ld.so.1",
			LinuxdynldMusl: musl,

			Freebsddynld:   "XXX",
			Openbsddynld:   "XXX",
			Netbsddynld:    "XXX",
			Dragonflydynld: "XXX",
			Solarisdynld:   "XXX",

			Reloc1:    elfreloc1,
			RelocSize: 8,
			SetupPLT:  elfsetupplt,

			// Historically GNU ld creates a read-only
			// .dynamic section.
			DynamicReadOnly: true,
		},
	}

	return arch, theArch
}

func archinit(ctxt *ld.Link) {
	switch ctxt.HeadType {
	default:
		ld.Exitf("unknown -H option: %v", ctxt.HeadType)
	case objabi.Hlinux: /* mips elf */
		ld.Elfinit(ctxt)
		ld.HEADR = ld.ELFRESERVE
		if *ld.FlagRound == -1 {
			*ld.FlagRound = 0x10000
		}
		if *ld.FlagTextAddr == -1 {
			*ld.FlagTextAddr = ld.Rnd(0x10000, *ld.FlagRound) + int64(ld.HEADR)
		}
	}
}

func adddynrel(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s loader.Sym, r loader.Reloc, rIdx int) bool {
	ld.Exitf("adddynrel currently unimplemented for MIPS")
	return false
}
