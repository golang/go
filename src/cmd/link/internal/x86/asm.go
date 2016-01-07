// Inferno utils/8l/asm.c
// http://code.google.com/p/inferno-os/source/browse/utils/8l/asm.c
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

// Append 4 bytes to s and create a R_CALL relocation targeting t to fill them in.
func addcall(ctxt *ld.Link, s *ld.LSym, t *ld.LSym) {
	s.Reachable = true
	i := s.Size
	s.Size += 4
	ld.Symgrow(ctxt, s, s.Size)
	r := ld.Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Type = obj.R_CALL
	r.Siz = 4
}

func gentext() {
	if !ld.DynlinkingGo() && ld.Buildmode != ld.BuildmodePIE && ld.Buildmode != ld.BuildmodeCShared {
		return
	}

	thunkfunc := ld.Linklookup(ld.Ctxt, "__x86.get_pc_thunk.cx", 0)
	thunkfunc.Type = obj.STEXT
	thunkfunc.Local = true
	thunkfunc.Reachable = true
	o := func(op ...uint8) {
		for _, op1 := range op {
			ld.Adduint8(ld.Ctxt, thunkfunc, op1)
		}
	}
	// 8b 0c 24	mov    (%esp),%ecx
	o(0x8b, 0x0c, 0x24)
	// c3		ret
	o(0xc3)

	if ld.Ctxt.Etextp != nil {
		ld.Ctxt.Etextp.Next = thunkfunc
	} else {
		ld.Ctxt.Textp = thunkfunc
	}
	ld.Ctxt.Etextp = thunkfunc

	addmoduledata := ld.Linklookup(ld.Ctxt, "runtime.addmoduledata", 0)
	if addmoduledata.Type == obj.STEXT {
		// we're linking a module containing the runtime -> no need for
		// an init function
		return
	}

	addmoduledata.Reachable = true

	initfunc := ld.Linklookup(ld.Ctxt, "go.link.addmoduledata", 0)
	initfunc.Type = obj.STEXT
	initfunc.Local = true
	initfunc.Reachable = true
	o = func(op ...uint8) {
		for _, op1 := range op {
			ld.Adduint8(ld.Ctxt, initfunc, op1)
		}
	}

	// go.link.addmoduledata:
	//      53                      push %ebx
	//      e8 00 00 00 00          call __x86.get_pc_thunk.cx + R_CALL __x86.get_pc_thunk.cx
	//      8d 81 00 00 00 00       lea 0x0(%ecx), %eax + R_PCREL ld.Ctxt.Moduledata
	//      8d 99 00 00 00 00       lea 0x0(%ecx), %ebx + R_GOTPC _GLOBAL_OFFSET_TABLE_
	//      e8 00 00 00 00          call runtime.addmoduledata@plt + R_CALL runtime.addmoduledata
	//      5b                      pop %ebx
	//      c3                      ret

	o(0x53)

	o(0xe8)
	addcall(ld.Ctxt, initfunc, ld.Linklookup(ld.Ctxt, "__x86.get_pc_thunk.cx", 0))

	o(0x8d, 0x81)
	ld.Addpcrelplus(ld.Ctxt, initfunc, ld.Ctxt.Moduledata, 6)

	o(0x8d, 0x99)
	i := initfunc.Size
	initfunc.Size += 4
	ld.Symgrow(ld.Ctxt, initfunc, initfunc.Size)
	r := ld.Addrel(initfunc)
	r.Sym = ld.Linklookup(ld.Ctxt, "_GLOBAL_OFFSET_TABLE_", 0)
	r.Off = int32(i)
	r.Type = obj.R_PCREL
	r.Add = 12
	r.Siz = 4

	o(0xe8)
	addcall(ld.Ctxt, initfunc, addmoduledata)

	o(0x5b)

	o(0xc3)

	ld.Ctxt.Etextp.Next = initfunc
	ld.Ctxt.Etextp = initfunc
	initarray_entry := ld.Linklookup(ld.Ctxt, "go.link.addmoduledatainit", 0)
	initarray_entry.Reachable = true
	initarray_entry.Local = true
	initarray_entry.Type = obj.SINITARR
	ld.Addaddr(ld.Ctxt, initarray_entry, initfunc)
}

func adddynrela(rela *ld.LSym, s *ld.LSym, r *ld.Reloc) {
	log.Fatalf("adddynrela not implemented")
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
	case 256 + ld.R_386_PC32:
		if targ.Type == obj.SDYNIMPORT {
			ld.Diag("unexpected R_386_PC32 relocation for dynamic symbol %s", targ.Name)
		}
		if targ.Type == 0 || targ.Type == obj.SXREF {
			ld.Diag("unknown symbol %s in pcrel", targ.Name)
		}
		r.Type = obj.R_PCREL
		r.Add += 4
		return

	case 256 + ld.R_386_PLT32:
		r.Type = obj.R_PCREL
		r.Add += 4
		if targ.Type == obj.SDYNIMPORT {
			addpltsym(ld.Ctxt, targ)
			r.Sym = ld.Linklookup(ld.Ctxt, ".plt", 0)
			r.Add += int64(targ.Plt)
		}

		return

	case 256 + ld.R_386_GOT32, 256 + ld.R_386_GOT32X:
		if targ.Type != obj.SDYNIMPORT {
			// have symbol
			if r.Off >= 2 && s.P[r.Off-2] == 0x8b {
				// turn MOVL of GOT entry into LEAL of symbol address, relative to GOT.
				s.P[r.Off-2] = 0x8d

				r.Type = obj.R_GOTOFF
				return
			}

			if r.Off >= 2 && s.P[r.Off-2] == 0xff && s.P[r.Off-1] == 0xb3 {
				// turn PUSHL of GOT entry into PUSHL of symbol itself.
				// use unnecessary SS prefix to keep instruction same length.
				s.P[r.Off-2] = 0x36

				s.P[r.Off-1] = 0x68
				r.Type = obj.R_ADDR
				return
			}

			ld.Diag("unexpected GOT reloc for non-dynamic symbol %s", targ.Name)
			return
		}

		addgotsym(ld.Ctxt, targ)
		r.Type = obj.R_CONST // write r->add during relocsym
		r.Sym = nil
		r.Add += int64(targ.Got)
		return

	case 256 + ld.R_386_GOTOFF:
		r.Type = obj.R_GOTOFF
		return

	case 256 + ld.R_386_GOTPC:
		r.Type = obj.R_PCREL
		r.Sym = ld.Linklookup(ld.Ctxt, ".got", 0)
		r.Add += 4
		return

	case 256 + ld.R_386_32:
		if targ.Type == obj.SDYNIMPORT {
			ld.Diag("unexpected R_386_32 relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = obj.R_ADDR
		return

	case 512 + ld.MACHO_GENERIC_RELOC_VANILLA*2 + 0:
		r.Type = obj.R_ADDR
		if targ.Type == obj.SDYNIMPORT {
			ld.Diag("unexpected reloc for dynamic symbol %s", targ.Name)
		}
		return

	case 512 + ld.MACHO_GENERIC_RELOC_VANILLA*2 + 1:
		if targ.Type == obj.SDYNIMPORT {
			addpltsym(ld.Ctxt, targ)
			r.Sym = ld.Linklookup(ld.Ctxt, ".plt", 0)
			r.Add = int64(targ.Plt)
			r.Type = obj.R_PCREL
			return
		}

		r.Type = obj.R_PCREL
		return

	case 512 + ld.MACHO_FAKE_GOTPCREL:
		if targ.Type != obj.SDYNIMPORT {
			// have symbol
			// turn MOVL of GOT entry into LEAL of symbol itself
			if r.Off < 2 || s.P[r.Off-2] != 0x8b {
				ld.Diag("unexpected GOT reloc for non-dynamic symbol %s", targ.Name)
				return
			}

			s.P[r.Off-2] = 0x8d
			r.Type = obj.R_PCREL
			return
		}

		addgotsym(ld.Ctxt, targ)
		r.Sym = ld.Linklookup(ld.Ctxt, ".got", 0)
		r.Add += int64(targ.Got)
		r.Type = obj.R_PCREL
		return
	}

	// Handle references to ELF symbols from our own object files.
	if targ.Type != obj.SDYNIMPORT {
		return
	}

	switch r.Type {
	case obj.R_CALL,
		obj.R_PCREL:
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
			ld.Adduint32(ld.Ctxt, rel, ld.ELF32_R_INFO(uint32(targ.Dynid), ld.R_386_32))
			r.Type = obj.R_CONST // write r->add during relocsym
			r.Sym = nil
			return
		}

		if ld.HEADTYPE == obj.Hdarwin && s.Size == PtrSize && r.Off == 0 {
			// Mach-O relocations are a royal pain to lay out.
			// They use a compact stateful bytecode representation
			// that is too much bother to deal with.
			// Instead, interpret the C declaration
			//	void *_Cvar_stderr = &stderr;
			// as making _Cvar_stderr the name of a GOT entry
			// for stderr.  This is separate from the usual GOT entry,
			// just in case the C code assigns to the variable,
			// and of course it only works for single pointers,
			// but we only need to support cgo and that's all it needs.
			ld.Adddynsym(ld.Ctxt, targ)

			got := ld.Linklookup(ld.Ctxt, ".got", 0)
			s.Type = got.Type | obj.SSUB
			s.Outer = got
			s.Sub = got.Sub
			got.Sub = s
			s.Value = got.Size
			ld.Adduint32(ld.Ctxt, got, 0)
			ld.Adduint32(ld.Ctxt, ld.Linklookup(ld.Ctxt, ".linkedit.got", 0), uint32(targ.Dynid))
			r.Type = 256 // ignore during relocsym
			return
		}

		if ld.HEADTYPE == obj.Hwindows && s.Size == PtrSize {
			// nothing to do, the relocation will be laid out in pereloc1
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
			ld.Thearch.Lput(ld.R_386_32 | uint32(elfsym)<<8)
		} else {
			return -1
		}

	case obj.R_GOTPCREL:
		if r.Siz == 4 {
			ld.Thearch.Lput(ld.R_386_GOTPC)
			if r.Xsym.Name != "_GLOBAL_OFFSET_TABLE_" {
				ld.Thearch.Lput(uint32(sectoff))
				ld.Thearch.Lput(ld.R_386_GOT32 | uint32(elfsym)<<8)
			}
		} else {
			return -1
		}

	case obj.R_CALL:
		if r.Siz == 4 {
			if r.Xsym.Type == obj.SDYNIMPORT {
				ld.Thearch.Lput(ld.R_386_PLT32 | uint32(elfsym)<<8)
			} else {
				ld.Thearch.Lput(ld.R_386_PC32 | uint32(elfsym)<<8)
			}
		} else {
			return -1
		}

	case obj.R_PCREL:
		if r.Siz == 4 {
			ld.Thearch.Lput(ld.R_386_PC32 | uint32(elfsym)<<8)
		} else {
			return -1
		}

	case obj.R_TLS_LE:
		if r.Siz == 4 {
			ld.Thearch.Lput(ld.R_386_TLS_LE | uint32(elfsym)<<8)
		} else {
			return -1
		}

	case obj.R_TLS_IE:
		if r.Siz == 4 {
			ld.Thearch.Lput(ld.R_386_GOTPC)
			ld.Thearch.Lput(uint32(sectoff))
			ld.Thearch.Lput(ld.R_386_TLS_GOTIE | uint32(elfsym)<<8)
		} else {
			return -1
		}
	}

	return 0
}

func machoreloc1(r *ld.Reloc, sectoff int64) int {
	var v uint32

	rs := r.Xsym

	if rs.Type == obj.SHOSTOBJ {
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

	case obj.R_CALL,
		obj.R_PCREL:
		v |= 1 << 24 // pc-relative bit
		v |= ld.MACHO_GENERIC_RELOC_VANILLA << 28
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

func pereloc1(r *ld.Reloc, sectoff int64) bool {
	var v uint32

	rs := r.Xsym

	if rs.Dynid < 0 {
		ld.Diag("reloc %d to non-coff symbol %s type=%d", r.Type, rs.Name, rs.Type)
		return false
	}

	ld.Thearch.Lput(uint32(sectoff))
	ld.Thearch.Lput(uint32(rs.Dynid))

	switch r.Type {
	default:
		return false

	case obj.R_ADDR:
		v = ld.IMAGE_REL_I386_DIR32

	case obj.R_CALL,
		obj.R_PCREL:
		v = ld.IMAGE_REL_I386_REL32
	}

	ld.Thearch.Wput(uint16(v))

	return true
}

func archreloc(r *ld.Reloc, s *ld.LSym, val *int64) int {
	if ld.Linkmode == ld.LinkExternal {
		return -1
	}
	switch r.Type {
	case obj.R_CONST:
		*val = r.Add
		return 0

	case obj.R_GOTOFF:
		*val = ld.Symaddr(r.Sym) + r.Add - ld.Symaddr(ld.Linklookup(ld.Ctxt, ".got", 0))
		return 0
	}

	return -1
}

func archrelocvariant(r *ld.Reloc, s *ld.LSym, t int64) int64 {
	log.Fatalf("unexpected relocation variant")
	return t
}

func elfsetupplt() {
	plt := ld.Linklookup(ld.Ctxt, ".plt", 0)
	got := ld.Linklookup(ld.Ctxt, ".got.plt", 0)
	if plt.Size == 0 {
		// pushl got+4
		ld.Adduint8(ld.Ctxt, plt, 0xff)

		ld.Adduint8(ld.Ctxt, plt, 0x35)
		ld.Addaddrplus(ld.Ctxt, plt, got, 4)

		// jmp *got+8
		ld.Adduint8(ld.Ctxt, plt, 0xff)

		ld.Adduint8(ld.Ctxt, plt, 0x25)
		ld.Addaddrplus(ld.Ctxt, plt, got, 8)

		// zero pad
		ld.Adduint32(ld.Ctxt, plt, 0)

		// assume got->size == 0 too
		ld.Addaddrplus(ld.Ctxt, got, ld.Linklookup(ld.Ctxt, ".dynamic", 0), 0)

		ld.Adduint32(ld.Ctxt, got, 0)
		ld.Adduint32(ld.Ctxt, got, 0)
	}
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

		// jmpq *got+size
		ld.Adduint8(ctxt, plt, 0xff)

		ld.Adduint8(ctxt, plt, 0x25)
		ld.Addaddrplus(ctxt, plt, got, got.Size)

		// add to got: pointer to current pos in plt
		ld.Addaddrplus(ctxt, got, plt, plt.Size)

		// pushl $x
		ld.Adduint8(ctxt, plt, 0x68)

		ld.Adduint32(ctxt, plt, uint32(rel.Size))

		// jmp .plt
		ld.Adduint8(ctxt, plt, 0xe9)

		ld.Adduint32(ctxt, plt, uint32(-(plt.Size + 4)))

		// rel
		ld.Addaddrplus(ctxt, rel, got, got.Size-4)

		ld.Adduint32(ctxt, rel, ld.ELF32_R_INFO(uint32(s.Dynid), ld.R_386_JMP_SLOT))

		s.Plt = int32(plt.Size - 16)
	} else if ld.HEADTYPE == obj.Hdarwin {
		// Same laziness as in 6l.

		plt := ld.Linklookup(ctxt, ".plt", 0)

		addgotsym(ctxt, s)

		ld.Adduint32(ctxt, ld.Linklookup(ctxt, ".linkedit.plt", 0), uint32(s.Dynid))

		// jmpq *got+size(IP)
		s.Plt = int32(plt.Size)

		ld.Adduint8(ctxt, plt, 0xff)
		ld.Adduint8(ctxt, plt, 0x25)
		ld.Addaddrplus(ctxt, plt, ld.Linklookup(ctxt, ".got", 0), int64(s.Got))
	} else {
		ld.Diag("addpltsym: unsupported binary format")
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
		ld.Adduint32(ctxt, rel, ld.ELF32_R_INFO(uint32(s.Dynid), ld.R_386_GLOB_DAT))
	} else if ld.HEADTYPE == obj.Hdarwin {
		ld.Adduint32(ctxt, ld.Linklookup(ctxt, ".linkedit.got", 0), uint32(s.Dynid))
	} else {
		ld.Diag("addgotsym: unsupported binary format")
	}
}

func asmb() {
	if ld.Debug['v'] != 0 {
		fmt.Fprintf(&ld.Bso, "%5.2f asmb\n", obj.Cputime())
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
			fmt.Fprintf(&ld.Bso, "%5.2f rodatblk\n", obj.Cputime())
		}
		ld.Bso.Flush()

		ld.Cseek(int64(ld.Segrodata.Fileoff))
		ld.Datblk(int64(ld.Segrodata.Vaddr), int64(ld.Segrodata.Filelen))
	}

	if ld.Debug['v'] != 0 {
		fmt.Fprintf(&ld.Bso, "%5.2f datblk\n", obj.Cputime())
	}
	ld.Bso.Flush()

	ld.Cseek(int64(ld.Segdata.Fileoff))
	ld.Datblk(int64(ld.Segdata.Vaddr), int64(ld.Segdata.Filelen))

	machlink := uint32(0)
	if ld.HEADTYPE == obj.Hdarwin {
		if ld.Debug['v'] != 0 {
			fmt.Fprintf(&ld.Bso, "%5.2f dwarf\n", obj.Cputime())
		}

		dwarfoff := uint32(ld.Rnd(int64(uint64(ld.HEADR)+ld.Segtext.Length), int64(ld.INITRND)) + ld.Rnd(int64(ld.Segdata.Filelen), int64(ld.INITRND)))
		ld.Cseek(int64(dwarfoff))

		ld.Segdwarf.Fileoff = uint64(ld.Cpos())
		ld.Dwarfemitdebugsections()
		ld.Segdwarf.Filelen = uint64(ld.Cpos()) - ld.Segdwarf.Fileoff

		machlink = uint32(ld.Domacholink())
	}

	ld.Symsize = 0
	ld.Spsize = 0
	ld.Lcsize = 0
	symo := uint32(0)
	if ld.Debug['s'] == 0 {
		// TODO: rationalize
		if ld.Debug['v'] != 0 {
			fmt.Fprintf(&ld.Bso, "%5.2f sym\n", obj.Cputime())
		}
		ld.Bso.Flush()
		switch ld.HEADTYPE {
		default:
			if ld.Iself {
				symo = uint32(ld.Segdata.Fileoff + ld.Segdata.Filelen)
				symo = uint32(ld.Rnd(int64(symo), int64(ld.INITRND)))
			}

		case obj.Hplan9:
			symo = uint32(ld.Segdata.Fileoff + ld.Segdata.Filelen)

		case obj.Hdarwin:
			symo = uint32(ld.Segdwarf.Fileoff + uint64(ld.Rnd(int64(ld.Segdwarf.Filelen), int64(ld.INITRND))) + uint64(machlink))

		case obj.Hwindows:
			symo = uint32(ld.Segdata.Fileoff + ld.Segdata.Filelen)
			symo = uint32(ld.Rnd(int64(symo), ld.PEFILEALIGN))
		}

		ld.Cseek(int64(symo))
		switch ld.HEADTYPE {
		default:
			if ld.Iself {
				if ld.Debug['v'] != 0 {
					fmt.Fprintf(&ld.Bso, "%5.2f elfsym\n", obj.Cputime())
				}
				ld.Asmelfsym()
				ld.Cflush()
				ld.Cwrite(ld.Elfstrdat)

				if ld.Debug['v'] != 0 {
					fmt.Fprintf(&ld.Bso, "%5.2f dwarf\n", obj.Cputime())
				}
				ld.Dwarfemitdebugsections()

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
					ld.Cput(uint8(sym.P[i]))
				}

				ld.Cflush()
			}

		case obj.Hwindows:
			if ld.Debug['v'] != 0 {
				fmt.Fprintf(&ld.Bso, "%5.2f dwarf\n", obj.Cputime())
			}
			ld.Dwarfemitdebugsections()

		case obj.Hdarwin:
			if ld.Linkmode == ld.LinkExternal {
				ld.Machoemitreloc()
			}
		}
	}

	if ld.Debug['v'] != 0 {
		fmt.Fprintf(&ld.Bso, "%5.2f headr\n", obj.Cputime())
	}
	ld.Bso.Flush()
	ld.Cseek(0)
	switch ld.HEADTYPE {
	default:
	case obj.Hplan9: /* plan9 */
		magic := int32(4*11*11 + 7)

		ld.Lputb(uint32(magic))              /* magic */
		ld.Lputb(uint32(ld.Segtext.Filelen)) /* sizes */
		ld.Lputb(uint32(ld.Segdata.Filelen))
		ld.Lputb(uint32(ld.Segdata.Length - ld.Segdata.Filelen))
		ld.Lputb(uint32(ld.Symsize))      /* nsyms */
		ld.Lputb(uint32(ld.Entryvalue())) /* va of entry */
		ld.Lputb(uint32(ld.Spsize))       /* sp offsets */
		ld.Lputb(uint32(ld.Lcsize))       /* line offsets */

	case obj.Hdarwin:
		ld.Asmbmacho()

	case obj.Hlinux,
		obj.Hfreebsd,
		obj.Hnetbsd,
		obj.Hopenbsd,
		obj.Hnacl:
		ld.Asmbelf(int64(symo))

	case obj.Hwindows:
		ld.Asmbpe()
	}

	ld.Cflush()
}
