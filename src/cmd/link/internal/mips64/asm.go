// Inferno utils/5l/asm.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/5l/asm.c
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

package mips64

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/ld"
	"cmd/link/internal/sym"
	"debug/elf"
	"fmt"
	"log"
)

func gentext(ctxt *ld.Link) {}

func adddynrel(ctxt *ld.Link, s *sym.Symbol, r *sym.Reloc) bool {
	log.Fatalf("adddynrel not implemented")
	return false
}

func elfreloc1(ctxt *ld.Link, r *sym.Reloc, sectoff int64) bool {
	// mips64 ELF relocation (endian neutral)
	//		offset	uint64
	//		sym		uint32
	//		ssym	uint8
	//		type3	uint8
	//		type2	uint8
	//		type	uint8
	//		addend	int64

	ctxt.Out.Write64(uint64(sectoff))

	elfsym := r.Xsym.ElfsymForReloc()
	ctxt.Out.Write32(uint32(elfsym))
	ctxt.Out.Write8(0)
	ctxt.Out.Write8(0)
	ctxt.Out.Write8(0)
	switch r.Type {
	default:
		return false
	case objabi.R_ADDR:
		switch r.Siz {
		case 4:
			ctxt.Out.Write8(uint8(elf.R_MIPS_32))
		case 8:
			ctxt.Out.Write8(uint8(elf.R_MIPS_64))
		default:
			return false
		}
	case objabi.R_ADDRMIPS:
		ctxt.Out.Write8(uint8(elf.R_MIPS_LO16))
	case objabi.R_ADDRMIPSU:
		ctxt.Out.Write8(uint8(elf.R_MIPS_HI16))
	case objabi.R_ADDRMIPSTLS:
		ctxt.Out.Write8(uint8(elf.R_MIPS_TLS_TPREL_LO16))
	case objabi.R_CALLMIPS,
		objabi.R_JMPMIPS:
		ctxt.Out.Write8(uint8(elf.R_MIPS_26))
	}
	ctxt.Out.Write64(uint64(r.Xadd))

	return true
}

func elfsetupplt(ctxt *ld.Link) {
	return
}

func machoreloc1(arch *sys.Arch, out *ld.OutBuf, s *sym.Symbol, r *sym.Reloc, sectoff int64) bool {
	return false
}

func archreloc(ctxt *ld.Link, r *sym.Reloc, s *sym.Symbol, val int64) (int64, bool) {
	if ctxt.LinkMode == ld.LinkExternal {
		switch r.Type {
		default:
			return val, false
		case objabi.R_ADDRMIPS,
			objabi.R_ADDRMIPSU:
			r.Done = false

			// set up addend for eventual relocation via outer symbol.
			rs := r.Sym
			r.Xadd = r.Add
			for rs.Outer != nil {
				r.Xadd += ld.Symaddr(rs) - ld.Symaddr(rs.Outer)
				rs = rs.Outer
			}

			if rs.Type != sym.SHOSTOBJ && rs.Type != sym.SDYNIMPORT && rs.Sect == nil {
				ld.Errorf(s, "missing section for %s", rs.Name)
			}
			r.Xsym = rs

			return val, true
		case objabi.R_ADDRMIPSTLS,
			objabi.R_CALLMIPS,
			objabi.R_JMPMIPS:
			r.Done = false
			r.Xsym = r.Sym
			r.Xadd = r.Add
			return val, true
		}
	}

	switch r.Type {
	case objabi.R_CONST:
		return r.Add, true
	case objabi.R_GOTOFF:
		return ld.Symaddr(r.Sym) + r.Add - ld.Symaddr(ctxt.Syms.Lookup(".got", 0)), true
	case objabi.R_ADDRMIPS,
		objabi.R_ADDRMIPSU:
		t := ld.Symaddr(r.Sym) + r.Add
		o1 := ctxt.Arch.ByteOrder.Uint32(s.P[r.Off:])
		if r.Type == objabi.R_ADDRMIPS {
			return int64(o1&0xffff0000 | uint32(t)&0xffff), true
		}
		return int64(o1&0xffff0000 | uint32((t+1<<15)>>16)&0xffff), true
	case objabi.R_ADDRMIPSTLS:
		// thread pointer is at 0x7000 offset from the start of TLS data area
		t := ld.Symaddr(r.Sym) + r.Add - 0x7000
		if t < -32768 || t >= 32678 {
			ld.Errorf(s, "TLS offset out of range %d", t)
		}
		o1 := ctxt.Arch.ByteOrder.Uint32(s.P[r.Off:])
		return int64(o1&0xffff0000 | uint32(t)&0xffff), true
	case objabi.R_CALLMIPS,
		objabi.R_JMPMIPS:
		// Low 26 bits = (S + A) >> 2
		t := ld.Symaddr(r.Sym) + r.Add
		o1 := ctxt.Arch.ByteOrder.Uint32(s.P[r.Off:])
		return int64(o1&0xfc000000 | uint32(t>>2)&^0xfc000000), true
	}

	return val, false
}

func archrelocvariant(ctxt *ld.Link, r *sym.Reloc, s *sym.Symbol, t int64) int64 {
	return -1
}

func asmb(ctxt *ld.Link) {
	if ctxt.IsELF {
		ld.Asmbelfsetup()
	}

	sect := ld.Segtext.Sections[0]
	ctxt.Out.SeekSet(int64(sect.Vaddr - ld.Segtext.Vaddr + ld.Segtext.Fileoff))
	ld.Codeblk(ctxt, int64(sect.Vaddr), int64(sect.Length))
	for _, sect = range ld.Segtext.Sections[1:] {
		ctxt.Out.SeekSet(int64(sect.Vaddr - ld.Segtext.Vaddr + ld.Segtext.Fileoff))
		ld.Datblk(ctxt, int64(sect.Vaddr), int64(sect.Length))
	}

	if ld.Segrodata.Filelen > 0 {
		ctxt.Out.SeekSet(int64(ld.Segrodata.Fileoff))
		ld.Datblk(ctxt, int64(ld.Segrodata.Vaddr), int64(ld.Segrodata.Filelen))
	}
	if ld.Segrelrodata.Filelen > 0 {
		ctxt.Out.SeekSet(int64(ld.Segrelrodata.Fileoff))
		ld.Datblk(ctxt, int64(ld.Segrelrodata.Vaddr), int64(ld.Segrelrodata.Filelen))
	}

	ctxt.Out.SeekSet(int64(ld.Segdata.Fileoff))
	ld.Datblk(ctxt, int64(ld.Segdata.Vaddr), int64(ld.Segdata.Filelen))

	ctxt.Out.SeekSet(int64(ld.Segdwarf.Fileoff))
	ld.Dwarfblk(ctxt, int64(ld.Segdwarf.Vaddr), int64(ld.Segdwarf.Filelen))
}

func asmb2(ctxt *ld.Link) {
	/* output symbol table */
	ld.Symsize = 0

	ld.Lcsize = 0
	symo := uint32(0)
	if !*ld.FlagS {
		// TODO: rationalize
		switch ctxt.HeadType {
		default:
			if ctxt.IsELF {
				symo = uint32(ld.Segdwarf.Fileoff + ld.Segdwarf.Filelen)
				symo = uint32(ld.Rnd(int64(symo), int64(*ld.FlagRound)))
			}

		case objabi.Hplan9:
			symo = uint32(ld.Segdata.Fileoff + ld.Segdata.Filelen)
		}

		ctxt.Out.SeekSet(int64(symo))
		switch ctxt.HeadType {
		default:
			if ctxt.IsELF {
				ld.Asmelfsym(ctxt)
				ctxt.Out.Flush()
				ctxt.Out.Write(ld.Elfstrdat)

				if ctxt.LinkMode == ld.LinkExternal {
					ld.Elfemitreloc(ctxt)
				}
			}

		case objabi.Hplan9:
			ld.Asmplan9sym(ctxt)
			ctxt.Out.Flush()

			sym := ctxt.Syms.Lookup("pclntab", 0)
			if sym != nil {
				ld.Lcsize = int32(len(sym.P))
				ctxt.Out.Write(sym.P)
				ctxt.Out.Flush()
			}
		}
	}

	ctxt.Out.SeekSet(0)
	switch ctxt.HeadType {
	default:
	case objabi.Hplan9: /* plan 9 */
		magic := uint32(4*18*18 + 7)
		if ctxt.Arch == sys.ArchMIPS64LE {
			magic = uint32(4*26*26 + 7)
		}
		ctxt.Out.Write32(magic)                      /* magic */
		ctxt.Out.Write32(uint32(ld.Segtext.Filelen)) /* sizes */
		ctxt.Out.Write32(uint32(ld.Segdata.Filelen))
		ctxt.Out.Write32(uint32(ld.Segdata.Length - ld.Segdata.Filelen))
		ctxt.Out.Write32(uint32(ld.Symsize))          /* nsyms */
		ctxt.Out.Write32(uint32(ld.Entryvalue(ctxt))) /* va of entry */
		ctxt.Out.Write32(0)
		ctxt.Out.Write32(uint32(ld.Lcsize))

	case objabi.Hlinux,
		objabi.Hfreebsd,
		objabi.Hnetbsd,
		objabi.Hopenbsd:
		ld.Asmbelf(ctxt, int64(symo))
	}

	ctxt.Out.Flush()
	if *ld.FlagC {
		fmt.Printf("textsize=%d\n", ld.Segtext.Filelen)
		fmt.Printf("datsize=%d\n", ld.Segdata.Filelen)
		fmt.Printf("bsssize=%d\n", ld.Segdata.Length-ld.Segdata.Filelen)
		fmt.Printf("symsize=%d\n", ld.Symsize)
		fmt.Printf("lcsize=%d\n", ld.Lcsize)
		fmt.Printf("total=%d\n", ld.Segtext.Filelen+ld.Segdata.Length+uint64(ld.Symsize)+uint64(ld.Lcsize))
	}
}
