// Inferno utils/6l/obj.c
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/6l/obj.c
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

package amd64

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/ld"
)

func Init() (*sys.Arch, ld.Arch) {
	arch := sys.ArchAMD64

	theArch := ld.Arch{
		Funcalign:  funcAlign,
		Maxalign:   maxAlign,
		Minalign:   minAlign,
		Dwarfregsp: dwarfRegSP,
		Dwarfreglr: dwarfRegLR,
		// 0xCC is INT $3 - breakpoint instruction
		CodePad: []byte{0xCC},

		Plan9Magic:  uint32(4*26*26 + 7),
		Plan9_64Bit: true,

		Adddynrel:        adddynrel,
		Archinit:         archinit,
		Archreloc:        archreloc,
		Archrelocvariant: archrelocvariant,
		Gentext:          gentext,
		Machoreloc1:      machoreloc1,
		MachorelocSize:   8,
		PEreloc1:         pereloc1,
		TLSIEtoLE:        tlsIEtoLE,

		ELF: ld.ELFArch{
			Linuxdynld:     "/lib64/ld-linux-x86-64.so.2",
			LinuxdynldMusl: "/lib/ld-musl-x86_64.so.1",
			Freebsddynld:   "/libexec/ld-elf.so.1",
			Openbsddynld:   "/usr/libexec/ld.so",
			Netbsddynld:    "/libexec/ld.elf_so",
			Dragonflydynld: "/usr/libexec/ld-elf.so.2",
			Solarisdynld:   "/lib/amd64/ld.so.1",

			Reloc1:    elfreloc1,
			RelocSize: 24,
			SetupPLT:  elfsetupplt,
		},
	}

	return arch, theArch
}

func archinit(ctxt *ld.Link) {
	switch ctxt.HeadType {
	default:
		ld.Exitf("unknown -H option: %v", ctxt.HeadType)

	case objabi.Hplan9: /* plan 9 */
		ld.HEADR = 32 + 8
		if *ld.FlagRound == -1 {
			*ld.FlagRound = 0x200000
		}
		if *ld.FlagTextAddr == -1 {
			*ld.FlagTextAddr = ld.Rnd(0x200000, *ld.FlagRound) + int64(ld.HEADR)
		}

	case objabi.Hdarwin: /* apple MACH */
		ld.HEADR = ld.INITIAL_MACHO_HEADR
		if *ld.FlagRound == -1 {
			*ld.FlagRound = 4096
		}
		if *ld.FlagTextAddr == -1 {
			*ld.FlagTextAddr = ld.Rnd(0x1000000, *ld.FlagRound) + int64(ld.HEADR)
		}

	case objabi.Hlinux, /* elf64 executable */
		objabi.Hfreebsd,   /* freebsd */
		objabi.Hnetbsd,    /* netbsd */
		objabi.Hopenbsd,   /* openbsd */
		objabi.Hdragonfly, /* dragonfly */
		objabi.Hsolaris:   /* solaris */
		ld.Elfinit(ctxt)

		ld.HEADR = ld.ELFRESERVE
		if *ld.FlagRound == -1 {
			*ld.FlagRound = 4096
		}
		if *ld.FlagTextAddr == -1 {
			*ld.FlagTextAddr = ld.Rnd(1<<22, *ld.FlagRound) + int64(ld.HEADR)
		}

	case objabi.Hwindows: /* PE executable */
		// ld.HEADR, ld.FlagTextAddr, ld.FlagRound are set in ld.Peinit
		return
	}
}
