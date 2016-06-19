// Inferno utils/5l/asm.c
// http://code.google.com/p/inferno-os/source/browse/utils/5l/asm.c
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

package arm

import (
	"cmd/internal/obj"
	"cmd/link/internal/ld"
	"fmt"
	"log"
)

// This assembler:
//
//         .align 2
// local.dso_init:
//         ldr r0, .Lmoduledata
// .Lloadfrom:
//         ldr r0, [r0]
//         b runtime.addmoduledata@plt
// .align 2
// .Lmoduledata:
//         .word local.moduledata(GOT_PREL) + (. - (.Lloadfrom + 4))
// assembles to:
//
// 00000000 <local.dso_init>:
//    0:        e59f0004        ldr     r0, [pc, #4]    ; c <local.dso_init+0xc>
//    4:        e5900000        ldr     r0, [r0]
//    8:        eafffffe        b       0 <runtime.addmoduledata>
//                      8: R_ARM_JUMP24 runtime.addmoduledata
//    c:        00000004        .word   0x00000004
//                      c: R_ARM_GOT_PREL       local.moduledata

func gentext() {
	if !ld.DynlinkingGo() {
		return
	}
	addmoduledata := ld.Linklookup(ld.Ctxt, "runtime.addmoduledata", 0)
	if addmoduledata.Type == obj.STEXT {
		// we're linking a module containing the runtime -> no need for
		// an init function
		return
	}
	addmoduledata.Attr |= ld.AttrReachable
	initfunc := ld.Linklookup(ld.Ctxt, "go.link.addmoduledata", 0)
	initfunc.Type = obj.STEXT
	initfunc.Attr |= ld.AttrLocal
	initfunc.Attr |= ld.AttrReachable
	o := func(op uint32) {
		ld.Adduint32(ld.Ctxt, initfunc, op)
	}
	o(0xe59f0004)
	o(0xe08f0000)

	o(0xeafffffe)
	rel := ld.Addrel(initfunc)
	rel.Off = 8
	rel.Siz = 4
	rel.Sym = ld.Linklookup(ld.Ctxt, "runtime.addmoduledata", 0)
	rel.Type = obj.R_CALLARM
	rel.Add = 0xeafffffe // vomit

	o(0x00000000)
	rel = ld.Addrel(initfunc)
	rel.Off = 12
	rel.Siz = 4
	rel.Sym = ld.Ctxt.Moduledata
	rel.Type = obj.R_PCREL
	rel.Add = 4

	ld.Ctxt.Textp = append(ld.Ctxt.Textp, initfunc)
	initarray_entry := ld.Linklookup(ld.Ctxt, "go.link.addmoduledatainit", 0)
	initarray_entry.Attr |= ld.AttrReachable
	initarray_entry.Attr |= ld.AttrLocal
	initarray_entry.Type = obj.SINITARR
	ld.Addaddr(ld.Ctxt, initarray_entry, initfunc)
}

// Preserve highest 8 bits of a, and do addition to lower 24-bit
// of a and b; used to adjust ARM branch instruction's target
func braddoff(a int32, b int32) int32 {
	return int32((uint32(a))&0xff000000 | 0x00ffffff&uint32(a+b))
}

func adddynrel(s *ld.LSym, r *ld.Reloc) {
	targ := r.Sym
	ld.Ctxt.Cursym = s

	switch r.Type {
	default:
		if r.Type >= 256 {
			ld.Diag("unexpected relocation type %d", r.Type)
			return
		}

		// Handle relocations found in ELF object files.
	case 256 + ld.R_ARM_PLT32:
		r.Type = obj.R_CALLARM

		if targ.Type == obj.SDYNIMPORT {
			addpltsym(ld.Ctxt, targ)
			r.Sym = ld.Linklookup(ld.Ctxt, ".plt", 0)
			r.Add = int64(braddoff(int32(r.Add), targ.Plt/4))
		}

		return

	case 256 + ld.R_ARM_THM_PC22: // R_ARM_THM_CALL
		ld.Exitf("R_ARM_THM_CALL, are you using -marm?")
		return

	case 256 + ld.R_ARM_GOT32: // R_ARM_GOT_BREL
		if targ.Type != obj.SDYNIMPORT {
			addgotsyminternal(ld.Ctxt, targ)
		} else {
			addgotsym(ld.Ctxt, targ)
		}

		r.Type = obj.R_CONST // write r->add during relocsym
		r.Sym = nil
		r.Add += int64(targ.Got)
		return

	case 256 + ld.R_ARM_GOT_PREL: // GOT(nil) + A - nil
		if targ.Type != obj.SDYNIMPORT {
			addgotsyminternal(ld.Ctxt, targ)
		} else {
			addgotsym(ld.Ctxt, targ)
		}

		r.Type = obj.R_PCREL
		r.Sym = ld.Linklookup(ld.Ctxt, ".got", 0)
		r.Add += int64(targ.Got) + 4
		return

	case 256 + ld.R_ARM_GOTOFF: // R_ARM_GOTOFF32
		r.Type = obj.R_GOTOFF

		return

	case 256 + ld.R_ARM_GOTPC: // R_ARM_BASE_PREL
		r.Type = obj.R_PCREL

		r.Sym = ld.Linklookup(ld.Ctxt, ".got", 0)
		r.Add += 4
		return

	case 256 + ld.R_ARM_CALL:
		r.Type = obj.R_CALLARM
		if targ.Type == obj.SDYNIMPORT {
			addpltsym(ld.Ctxt, targ)
			r.Sym = ld.Linklookup(ld.Ctxt, ".plt", 0)
			r.Add = int64(braddoff(int32(r.Add), targ.Plt/4))
		}

		return

	case 256 + ld.R_ARM_REL32: // R_ARM_REL32
		r.Type = obj.R_PCREL

		r.Add += 4
		return

	case 256 + ld.R_ARM_ABS32:
		if targ.Type == obj.SDYNIMPORT {
			ld.Diag("unexpected R_ARM_ABS32 relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = obj.R_ADDR
		return

		// we can just ignore this, because we are targeting ARM V5+ anyway
	case 256 + ld.R_ARM_V4BX:
		if r.Sym != nil {
			// R_ARM_V4BX is ABS relocation, so this symbol is a dummy symbol, ignore it
			r.Sym.Type = 0
		}

		r.Sym = nil
		return

	case 256 + ld.R_ARM_PC24,
		256 + ld.R_ARM_JUMP24:
		r.Type = obj.R_CALLARM
		if targ.Type == obj.SDYNIMPORT {
			addpltsym(ld.Ctxt, targ)
			r.Sym = ld.Linklookup(ld.Ctxt, ".plt", 0)
			r.Add = int64(braddoff(int32(r.Add), targ.Plt/4))
		}

		return
	}

	// Handle references to ELF symbols from our own object files.
	if targ.Type != obj.SDYNIMPORT {
		return
	}

	switch r.Type {
	case obj.R_CALLARM:
		addpltsym(ld.Ctxt, targ)
		r.Sym = ld.Linklookup(ld.Ctxt, ".plt", 0)
		r.Add = int64(targ.Plt)
		return

	case obj.R_ADDR:
		if s.Type != obj.SDATA {
			break
		}
		if ld.Iself {
			ld.Adddynsym(ld.Ctxt, targ)
			rel := ld.Linklookup(ld.Ctxt, ".rel", 0)
			ld.Addaddrplus(ld.Ctxt, rel, s, int64(r.Off))
			ld.Adduint32(ld.Ctxt, rel, ld.ELF32_R_INFO(uint32(targ.Dynid), ld.R_ARM_GLOB_DAT)) // we need a nil + A dynamic reloc
			r.Type = obj.R_CONST                                                               // write r->add during relocsym
			r.Sym = nil
			return
		}
	}

	ld.Ctxt.Cursym = s
	ld.Diag("unsupported relocation for dynamic symbol %s (type=%d stype=%d)", targ.Name, r.Type, targ.Type)
}

func elfreloc1(r *ld.Reloc, sectoff int64) int {
	ld.Thearch.Lput(uint32(sectoff))

	elfsym := r.Xsym.ElfsymForReloc()
	switch r.Type {
	default:
		return -1

	case obj.R_ADDR:
		if r.Siz == 4 {
			ld.Thearch.Lput(ld.R_ARM_ABS32 | uint32(elfsym)<<8)
		} else {
			return -1
		}

	case obj.R_PCREL:
		if r.Siz == 4 {
			ld.Thearch.Lput(ld.R_ARM_REL32 | uint32(elfsym)<<8)
		} else {
			return -1
		}

	case obj.R_CALLARM:
		if r.Siz == 4 {
			if r.Add&0xff000000 == 0xeb000000 { // BL
				ld.Thearch.Lput(ld.R_ARM_CALL | uint32(elfsym)<<8)
			} else {
				ld.Thearch.Lput(ld.R_ARM_JUMP24 | uint32(elfsym)<<8)
			}
		} else {
			return -1
		}

	case obj.R_TLS_LE:
		ld.Thearch.Lput(ld.R_ARM_TLS_LE32 | uint32(elfsym)<<8)

	case obj.R_TLS_IE:
		ld.Thearch.Lput(ld.R_ARM_TLS_IE32 | uint32(elfsym)<<8)

	case obj.R_GOTPCREL:
		if r.Siz == 4 {
			ld.Thearch.Lput(ld.R_ARM_GOT_PREL | uint32(elfsym)<<8)
		} else {
			return -1
		}
	}

	return 0
}

func elfsetupplt() {
	plt := ld.Linklookup(ld.Ctxt, ".plt", 0)
	got := ld.Linklookup(ld.Ctxt, ".got.plt", 0)
	if plt.Size == 0 {
		// str lr, [sp, #-4]!
		ld.Adduint32(ld.Ctxt, plt, 0xe52de004)

		// ldr lr, [pc, #4]
		ld.Adduint32(ld.Ctxt, plt, 0xe59fe004)

		// add lr, pc, lr
		ld.Adduint32(ld.Ctxt, plt, 0xe08fe00e)

		// ldr pc, [lr, #8]!
		ld.Adduint32(ld.Ctxt, plt, 0xe5bef008)

		// .word &GLOBAL_OFFSET_TABLE[0] - .
		ld.Addpcrelplus(ld.Ctxt, plt, got, 4)

		// the first .plt entry requires 3 .plt.got entries
		ld.Adduint32(ld.Ctxt, got, 0)

		ld.Adduint32(ld.Ctxt, got, 0)
		ld.Adduint32(ld.Ctxt, got, 0)
	}
}

func machoreloc1(r *ld.Reloc, sectoff int64) int {
	var v uint32

	rs := r.Xsym

	if r.Type == obj.R_PCREL {
		if rs.Type == obj.SHOSTOBJ {
			ld.Diag("pc-relative relocation of external symbol is not supported")
			return -1
		}
		if r.Siz != 4 {
			return -1
		}

		// emit a pair of "scattered" relocations that
		// resolve to the difference of section addresses of
		// the symbol and the instruction
		// this value is added to the field being relocated
		o1 := uint32(sectoff)
		o1 |= 1 << 31 // scattered bit
		o1 |= ld.MACHO_ARM_RELOC_SECTDIFF << 24
		o1 |= 2 << 28 // size = 4

		o2 := uint32(0)
		o2 |= 1 << 31 // scattered bit
		o2 |= ld.MACHO_ARM_RELOC_PAIR << 24
		o2 |= 2 << 28 // size = 4

		ld.Thearch.Lput(o1)
		ld.Thearch.Lput(uint32(ld.Symaddr(rs)))
		ld.Thearch.Lput(o2)
		ld.Thearch.Lput(uint32(ld.Ctxt.Cursym.Value + int64(r.Off)))
		return 0
	}

	if rs.Type == obj.SHOSTOBJ || r.Type == obj.R_CALLARM {
		if rs.Dynid < 0 {
			ld.Diag("reloc %d to non-macho symbol %s type=%d", r.Type, rs.Name, rs.Type)
			return -1
		}

		v = uint32(rs.Dynid)
		v |= 1 << 27 // external relocation
	} else {
		v = uint32(rs.Sect.Extnum)
		if v == 0 {
			ld.Diag("reloc %d to symbol %s in non-macho section %s type=%d", r.Type, rs.Name, rs.Sect.Name, rs.Type)
			return -1
		}
	}

	switch r.Type {
	default:
		return -1

	case obj.R_ADDR:
		v |= ld.MACHO_GENERIC_RELOC_VANILLA << 28

	case obj.R_CALLARM:
		v |= 1 << 24 // pc-relative bit
		v |= ld.MACHO_ARM_RELOC_BR24 << 28
	}

	switch r.Siz {
	default:
		return -1

	case 1:
		v |= 0 << 25

	case 2:
		v |= 1 << 25

	case 4:
		v |= 2 << 25

	case 8:
		v |= 3 << 25
	}

	ld.Thearch.Lput(uint32(sectoff))
	ld.Thearch.Lput(v)
	return 0
}

func archreloc(r *ld.Reloc, s *ld.LSym, val *int64) int {
	if ld.Linkmode == ld.LinkExternal {
		switch r.Type {
		case obj.R_CALLARM:
			r.Done = 0

			// set up addend for eventual relocation via outer symbol.
			rs := r.Sym

			r.Xadd = r.Add
			if r.Xadd&0x800000 != 0 {
				r.Xadd |= ^0xffffff
			}
			r.Xadd *= 4
			for rs.Outer != nil {
				r.Xadd += ld.Symaddr(rs) - ld.Symaddr(rs.Outer)
				rs = rs.Outer
			}

			if rs.Type != obj.SHOSTOBJ && rs.Type != obj.SDYNIMPORT && rs.Sect == nil {
				ld.Diag("missing section for %s", rs.Name)
			}
			r.Xsym = rs

			// ld64 for arm seems to want the symbol table to contain offset
			// into the section rather than pseudo virtual address that contains
			// the section load address.
			// we need to compensate that by removing the instruction's address
			// from addend.
			if ld.HEADTYPE == obj.Hdarwin {
				r.Xadd -= ld.Symaddr(s) + int64(r.Off)
			}

			*val = int64(braddoff(int32(0xff000000&uint32(r.Add)), int32(0xffffff&uint32(r.Xadd/4))))
			return 0
		}

		return -1
	}

	switch r.Type {
	case obj.R_CONST:
		*val = r.Add
		return 0

	case obj.R_GOTOFF:
		*val = ld.Symaddr(r.Sym) + r.Add - ld.Symaddr(ld.Linklookup(ld.Ctxt, ".got", 0))
		return 0

		// The following three arch specific relocations are only for generation of
	// Linux/ARM ELF's PLT entry (3 assembler instruction)
	case obj.R_PLT0: // add ip, pc, #0xXX00000
		if ld.Symaddr(ld.Linklookup(ld.Ctxt, ".got.plt", 0)) < ld.Symaddr(ld.Linklookup(ld.Ctxt, ".plt", 0)) {
			ld.Diag(".got.plt should be placed after .plt section.")
		}
		*val = 0xe28fc600 + (0xff & (int64(uint32(ld.Symaddr(r.Sym)-(ld.Symaddr(ld.Linklookup(ld.Ctxt, ".plt", 0))+int64(r.Off))+r.Add)) >> 20))
		return 0

	case obj.R_PLT1: // add ip, ip, #0xYY000
		*val = 0xe28cca00 + (0xff & (int64(uint32(ld.Symaddr(r.Sym)-(ld.Symaddr(ld.Linklookup(ld.Ctxt, ".plt", 0))+int64(r.Off))+r.Add+4)) >> 12))

		return 0

	case obj.R_PLT2: // ldr pc, [ip, #0xZZZ]!
		*val = 0xe5bcf000 + (0xfff & int64(uint32(ld.Symaddr(r.Sym)-(ld.Symaddr(ld.Linklookup(ld.Ctxt, ".plt", 0))+int64(r.Off))+r.Add+8)))

		return 0

	case obj.R_CALLARM: // bl XXXXXX or b YYYYYY
		*val = int64(braddoff(int32(0xff000000&uint32(r.Add)), int32(0xffffff&uint32((ld.Symaddr(r.Sym)+int64((uint32(r.Add))*4)-(s.Value+int64(r.Off)))/4))))

		return 0
	}

	return -1
}

func archrelocvariant(r *ld.Reloc, s *ld.LSym, t int64) int64 {
	log.Fatalf("unexpected relocation variant")
	return t
}

func addpltreloc(ctxt *ld.Link, plt *ld.LSym, got *ld.LSym, sym *ld.LSym, typ int) *ld.Reloc {
	r := ld.Addrel(plt)
	r.Sym = got
	r.Off = int32(plt.Size)
	r.Siz = 4
	r.Type = int32(typ)
	r.Add = int64(sym.Got) - 8

	plt.Attr |= ld.AttrReachable
	plt.Size += 4
	ld.Symgrow(ctxt, plt, plt.Size)

	return r
}

func addpltsym(ctxt *ld.Link, s *ld.LSym) {
	if s.Plt >= 0 {
		return
	}

	ld.Adddynsym(ctxt, s)

	if ld.Iself {
		plt := ld.Linklookup(ctxt, ".plt", 0)
		got := ld.Linklookup(ctxt, ".got.plt", 0)
		rel := ld.Linklookup(ctxt, ".rel.plt", 0)
		if plt.Size == 0 {
			elfsetupplt()
		}

		// .got entry
		s.Got = int32(got.Size)

		// In theory, all GOT should point to the first PLT entry,
		// Linux/ARM's dynamic linker will do that for us, but FreeBSD/ARM's
		// dynamic linker won't, so we'd better do it ourselves.
		ld.Addaddrplus(ctxt, got, plt, 0)

		// .plt entry, this depends on the .got entry
		s.Plt = int32(plt.Size)

		addpltreloc(ctxt, plt, got, s, obj.R_PLT0) // add lr, pc, #0xXX00000
		addpltreloc(ctxt, plt, got, s, obj.R_PLT1) // add lr, lr, #0xYY000
		addpltreloc(ctxt, plt, got, s, obj.R_PLT2) // ldr pc, [lr, #0xZZZ]!

		// rel
		ld.Addaddrplus(ctxt, rel, got, int64(s.Got))

		ld.Adduint32(ctxt, rel, ld.ELF32_R_INFO(uint32(s.Dynid), ld.R_ARM_JUMP_SLOT))
	} else {
		ld.Diag("addpltsym: unsupported binary format")
	}
}

func addgotsyminternal(ctxt *ld.Link, s *ld.LSym) {
	if s.Got >= 0 {
		return
	}

	got := ld.Linklookup(ctxt, ".got", 0)
	s.Got = int32(got.Size)

	ld.Addaddrplus(ctxt, got, s, 0)

	if ld.Iself {
	} else {
		ld.Diag("addgotsyminternal: unsupported binary format")
	}
}

func addgotsym(ctxt *ld.Link, s *ld.LSym) {
	if s.Got >= 0 {
		return
	}

	ld.Adddynsym(ctxt, s)
	got := ld.Linklookup(ctxt, ".got", 0)
	s.Got = int32(got.Size)
	ld.Adduint32(ctxt, got, 0)

	if ld.Iself {
		rel := ld.Linklookup(ctxt, ".rel", 0)
		ld.Addaddrplus(ctxt, rel, got, int64(s.Got))
		ld.Adduint32(ctxt, rel, ld.ELF32_R_INFO(uint32(s.Dynid), ld.R_ARM_GLOB_DAT))
	} else {
		ld.Diag("addgotsym: unsupported binary format")
	}
}

func asmb() {
	if ld.Debug['v'] != 0 {
		fmt.Fprintf(ld.Bso, "%5.2f asmb\n", obj.Cputime())
	}
	ld.Bso.Flush()

	if ld.Iself {
		ld.Asmbelfsetup()
	}

	sect := ld.Segtext.Sect
	ld.Cseek(int64(sect.Vaddr - ld.Segtext.Vaddr + ld.Segtext.Fileoff))
	ld.Codeblk(int64(sect.Vaddr), int64(sect.Length))
	for sect = sect.Next; sect != nil; sect = sect.Next {
		ld.Cseek(int64(sect.Vaddr - ld.Segtext.Vaddr + ld.Segtext.Fileoff))
		ld.Datblk(int64(sect.Vaddr), int64(sect.Length))
	}

	if ld.Segrodata.Filelen > 0 {
		if ld.Debug['v'] != 0 {
			fmt.Fprintf(ld.Bso, "%5.2f rodatblk\n", obj.Cputime())
		}
		ld.Bso.Flush()

		ld.Cseek(int64(ld.Segrodata.Fileoff))
		ld.Datblk(int64(ld.Segrodata.Vaddr), int64(ld.Segrodata.Filelen))
	}

	if ld.Debug['v'] != 0 {
		fmt.Fprintf(ld.Bso, "%5.2f datblk\n", obj.Cputime())
	}
	ld.Bso.Flush()

	ld.Cseek(int64(ld.Segdata.Fileoff))
	ld.Datblk(int64(ld.Segdata.Vaddr), int64(ld.Segdata.Filelen))

	ld.Cseek(int64(ld.Segdwarf.Fileoff))
	ld.Dwarfblk(int64(ld.Segdwarf.Vaddr), int64(ld.Segdwarf.Filelen))

	machlink := uint32(0)
	if ld.HEADTYPE == obj.Hdarwin {
		machlink = uint32(ld.Domacholink())
	}

	/* output symbol table */
	ld.Symsize = 0

	ld.Lcsize = 0
	symo := uint32(0)
	if ld.Debug['s'] == 0 {
		// TODO: rationalize
		if ld.Debug['v'] != 0 {
			fmt.Fprintf(ld.Bso, "%5.2f sym\n", obj.Cputime())
		}
		ld.Bso.Flush()
		switch ld.HEADTYPE {
		default:
			if ld.Iself {
				symo = uint32(ld.Segdwarf.Fileoff + ld.Segdwarf.Filelen)
				symo = uint32(ld.Rnd(int64(symo), int64(ld.INITRND)))
			}

		case obj.Hplan9:
			symo = uint32(ld.Segdata.Fileoff + ld.Segdata.Filelen)

		case obj.Hdarwin:
			symo = uint32(ld.Segdwarf.Fileoff + uint64(ld.Rnd(int64(ld.Segdwarf.Filelen), int64(ld.INITRND))) + uint64(machlink))
		}

		ld.Cseek(int64(symo))
		switch ld.HEADTYPE {
		default:
			if ld.Iself {
				if ld.Debug['v'] != 0 {
					fmt.Fprintf(ld.Bso, "%5.2f elfsym\n", obj.Cputime())
				}
				ld.Asmelfsym()
				ld.Cflush()
				ld.Cwrite(ld.Elfstrdat)

				if ld.Linkmode == ld.LinkExternal {
					ld.Elfemitreloc()
				}
			}

		case obj.Hplan9:
			ld.Asmplan9sym()
			ld.Cflush()

			sym := ld.Linklookup(ld.Ctxt, "pclntab", 0)
			if sym != nil {
				ld.Lcsize = int32(len(sym.P))
				for i := 0; int32(i) < ld.Lcsize; i++ {
					ld.Cput(sym.P[i])
				}

				ld.Cflush()
			}

		case obj.Hdarwin:
			if ld.Linkmode == ld.LinkExternal {
				ld.Machoemitreloc()
			}
		}
	}

	ld.Ctxt.Cursym = nil
	if ld.Debug['v'] != 0 {
		fmt.Fprintf(ld.Bso, "%5.2f header\n", obj.Cputime())
	}
	ld.Bso.Flush()
	ld.Cseek(0)
	switch ld.HEADTYPE {
	default:
	case obj.Hplan9: /* plan 9 */
		ld.Lputb(0x647)                      /* magic */
		ld.Lputb(uint32(ld.Segtext.Filelen)) /* sizes */
		ld.Lputb(uint32(ld.Segdata.Filelen))
		ld.Lputb(uint32(ld.Segdata.Length - ld.Segdata.Filelen))
		ld.Lputb(uint32(ld.Symsize))      /* nsyms */
		ld.Lputb(uint32(ld.Entryvalue())) /* va of entry */
		ld.Lputb(0)
		ld.Lputb(uint32(ld.Lcsize))

	case obj.Hlinux,
		obj.Hfreebsd,
		obj.Hnetbsd,
		obj.Hopenbsd,
		obj.Hnacl:
		ld.Asmbelf(int64(symo))

	case obj.Hdarwin:
		ld.Asmbmacho()
	}

	ld.Cflush()
	if ld.Debug['c'] != 0 {
		fmt.Printf("textsize=%d\n", ld.Segtext.Filelen)
		fmt.Printf("datsize=%d\n", ld.Segdata.Filelen)
		fmt.Printf("bsssize=%d\n", ld.Segdata.Length-ld.Segdata.Filelen)
		fmt.Printf("symsize=%d\n", ld.Symsize)
		fmt.Printf("lcsize=%d\n", ld.Lcsize)
		fmt.Printf("total=%d\n", ld.Segtext.Filelen+ld.Segdata.Length+uint64(ld.Symsize)+uint64(ld.Lcsize))
	}
}
