// Inferno utils/8l/obj.c
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/8l/obj.c
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

package x86

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/ld"
)

func Init() (*sys.Arch, ld.Arch) {
	arch := sys.Arch386

	theArch := ld.Arch{
		Funcalign:  funcAlign,
		Maxalign:   maxAlign,
		Minalign:   minAlign,
		Dwarfregsp: dwarfRegSP,
		Dwarfreglr: dwarfRegLR,
		// 0xCC is INT $3 - breakpoint instruction
		CodePad: []byte{0xCC},

		Plan9Magic: uint32(4*11*11 + 7),

		Adddynrel:                 adddynrel,
		Archinit:                  archinit,
		Archreloc:                 archreloc,
		Archrelocvariant:          archrelocvariant,
		Gentext:                   gentext,
		Machoreloc1:               machoreloc1,
		PEreloc1:                  pereloc1,
		ReturnAddressAtTopOfFrame: true,

		ELF: ld.ELFArch{
			Linuxdynld:     "/lib/ld-linux.so.2",
			LinuxdynldMusl: "/lib/ld-musl-i386.so.1",
			Freebsddynld:   "/usr/libexec/ld-elf.so.1",
			Openbsddynld:   "/usr/libexec/ld.so",
			Netbsddynld:    "/usr/libexec/ld.elf_so",
			Solarisdynld:   "/lib/ld.so.1",

			Reloc1:    elfreloc1,
			RelocSize: 8,
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
		ld.HEADR = 32
		if *ld.FlagRound == -1 {
			*ld.FlagRound = 4096
		}
		if *ld.FlagTextAddr == -1 {
			*ld.FlagTextAddr = ld.Rnd(4096, *ld.FlagRound) + int64(ld.HEADR)
		}

	case objabi.Hlinux, /* elf32 executable */
		objabi.Hfreebsd,
		objabi.Hnetbsd,
		objabi.Hopenbsd:
		ld.Elfinit(ctxt)

		ld.HEADR = ld.ELFRESERVE
		if *ld.FlagRound == -1 {
			*ld.FlagRound = 4096
		}
		if *ld.FlagTextAddr == -1 {
			*ld.FlagTextAddr = ld.Rnd(0x08048000, *ld.FlagRound) + int64(ld.HEADR)
		}

	case objabi.Hwindows: /* PE executable */
		// ld.HEADR, ld.FlagTextAddr, ld.FlagRound are set in ld.Peinit
		return
	}
}
