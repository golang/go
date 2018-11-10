// Inferno utils/8l/asm.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/8l/asm.c
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
	"cmd/link/internal/ld"
	"log"
)

// Append 4 bytes to s and create a R_CALL relocation targeting t to fill them in.
func addcall(ctxt *ld.Link, s *ld.Symbol, t *ld.Symbol) {
	s.Attr |= ld.AttrReachable
	i := s.Size
	s.Size += 4
	ld.Symgrow(s, s.Size)
	r := ld.Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Type = objabi.R_CALL
	r.Siz = 4
}

func gentext(ctxt *ld.Link) {
	if ctxt.DynlinkingGo() {
		// We need get_pc_thunk.
	} else {
		switch ld.Buildmode {
		case ld.BuildmodeCArchive:
			if !ld.Iself {
				return
			}
		case ld.BuildmodePIE, ld.BuildmodeCShared, ld.BuildmodePlugin:
			// We need get_pc_thunk.
		default:
			return
		}
	}

	// Generate little thunks that load the PC of the next instruction into a register.
	thunks := make([]*ld.Symbol, 0, 7+len(ctxt.Textp))
	for _, r := range [...]struct {
		name string
		num  uint8
	}{
		{"ax", 0},
		{"cx", 1},
		{"dx", 2},
		{"bx", 3},
		// sp
		{"bp", 5},
		{"si", 6},
		{"di", 7},
	} {
		thunkfunc := ctxt.Syms.Lookup("__x86.get_pc_thunk."+r.name, 0)
		thunkfunc.Type = ld.STEXT
		thunkfunc.Attr |= ld.AttrLocal
		thunkfunc.Attr |= ld.AttrReachable //TODO: remove?
		o := func(op ...uint8) {
			for _, op1 := range op {
				ld.Adduint8(ctxt, thunkfunc, op1)
			}
		}
		// 8b 04 24	mov    (%esp),%eax
		// Destination register is in bits 3-5 of the middle byte, so add that in.
		o(0x8b, 0x04+r.num<<3, 0x24)
		// c3		ret
		o(0xc3)

		thunks = append(thunks, thunkfunc)
	}
	ctxt.Textp = append(thunks, ctxt.Textp...) // keep Textp in dependency order

	addmoduledata := ctxt.Syms.Lookup("runtime.addmoduledata", 0)
	if addmoduledata.Type == ld.STEXT && ld.Buildmode != ld.BuildmodePlugin {
		// we're linking a module containing the runtime -> no need for
		// an init function
		return
	}

	addmoduledata.Attr |= ld.AttrReachable

	initfunc := ctxt.Syms.Lookup("go.link.addmoduledata", 0)
	initfunc.Type = ld.STEXT
	initfunc.Attr |= ld.AttrLocal
	initfunc.Attr |= ld.AttrReachable
	o := func(op ...uint8) {
		for _, op1 := range op {
			ld.Adduint8(ctxt, initfunc, op1)
		}
	}

	// go.link.addmoduledata:
	//      53                      push %ebx
	//      e8 00 00 00 00          call __x86.get_pc_thunk.cx + R_CALL __x86.get_pc_thunk.cx
	//      8d 81 00 00 00 00       lea 0x0(%ecx), %eax + R_PCREL ctxt.Moduledata
	//      8d 99 00 00 00 00       lea 0x0(%ecx), %ebx + R_GOTPC _GLOBAL_OFFSET_TABLE_
	//      e8 00 00 00 00          call runtime.addmoduledata@plt + R_CALL runtime.addmoduledata
	//      5b                      pop %ebx
	//      c3                      ret

	o(0x53)

	o(0xe8)
	addcall(ctxt, initfunc, ctxt.Syms.Lookup("__x86.get_pc_thunk.cx", 0))

	o(0x8d, 0x81)
	ld.Addpcrelplus(ctxt, initfunc, ctxt.Moduledata, 6)

	o(0x8d, 0x99)
	i := initfunc.Size
	initfunc.Size += 4
	ld.Symgrow(initfunc, initfunc.Size)
	r := ld.Addrel(initfunc)
	r.Sym = ctxt.Syms.Lookup("_GLOBAL_OFFSET_TABLE_", 0)
	r.Off = int32(i)
	r.Type = objabi.R_PCREL
	r.Add = 12
	r.Siz = 4

	o(0xe8)
	addcall(ctxt, initfunc, addmoduledata)

	o(0x5b)

	o(0xc3)

	if ld.Buildmode == ld.BuildmodePlugin {
		ctxt.Textp = append(ctxt.Textp, addmoduledata)
	}
	ctxt.Textp = append(ctxt.Textp, initfunc)
	initarray_entry := ctxt.Syms.Lookup("go.link.addmoduledatainit", 0)
	initarray_entry.Attr |= ld.AttrReachable
	initarray_entry.Attr |= ld.AttrLocal
	initarray_entry.Type = ld.SINITARR
	ld.Addaddr(ctxt, initarray_entry, initfunc)
}

func adddynrel(ctxt *ld.Link, s *ld.Symbol, r *ld.Reloc) bool {
	targ := r.Sym

	switch r.Type {
	default:
		if r.Type >= 256 {
			ld.Errorf(s, "unexpected relocation type %d", r.Type)
			return false
		}

		// Handle relocations found in ELF object files.
	case 256 + ld.R_386_PC32:
		if targ.Type == ld.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_386_PC32 relocation for dynamic symbol %s", targ.Name)
		}
		if targ.Type == 0 || targ.Type == ld.SXREF {
			ld.Errorf(s, "unknown symbol %s in pcrel", targ.Name)
		}
		r.Type = objabi.R_PCREL
		r.Add += 4
		return true

	case 256 + ld.R_386_PLT32:
		r.Type = objabi.R_PCREL
		r.Add += 4
		if targ.Type == ld.SDYNIMPORT {
			addpltsym(ctxt, targ)
			r.Sym = ctxt.Syms.Lookup(".plt", 0)
			r.Add += int64(targ.Plt)
		}

		return true

	case 256 + ld.R_386_GOT32, 256 + ld.R_386_GOT32X:
		if targ.Type != ld.SDYNIMPORT {
			// have symbol
			if r.Off >= 2 && s.P[r.Off-2] == 0x8b {
				// turn MOVL of GOT entry into LEAL of symbol address, relative to GOT.
				s.P[r.Off-2] = 0x8d

				r.Type = objabi.R_GOTOFF
				return true
			}

			if r.Off >= 2 && s.P[r.Off-2] == 0xff && s.P[r.Off-1] == 0xb3 {
				// turn PUSHL of GOT entry into PUSHL of symbol itself.
				// use unnecessary SS prefix to keep instruction same length.
				s.P[r.Off-2] = 0x36

				s.P[r.Off-1] = 0x68
				r.Type = objabi.R_ADDR
				return true
			}

			ld.Errorf(s, "unexpected GOT reloc for non-dynamic symbol %s", targ.Name)
			return false
		}

		addgotsym(ctxt, targ)
		r.Type = objabi.R_CONST // write r->add during relocsym
		r.Sym = nil
		r.Add += int64(targ.Got)
		return true

	case 256 + ld.R_386_GOTOFF:
		r.Type = objabi.R_GOTOFF
		return true

	case 256 + ld.R_386_GOTPC:
		r.Type = objabi.R_PCREL
		r.Sym = ctxt.Syms.Lookup(".got", 0)
		r.Add += 4
		return true

	case 256 + ld.R_386_32:
		if targ.Type == ld.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_386_32 relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = objabi.R_ADDR
		return true

	case 512 + ld.MACHO_GENERIC_RELOC_VANILLA*2 + 0:
		r.Type = objabi.R_ADDR
		if targ.Type == ld.SDYNIMPORT {
			ld.Errorf(s, "unexpected reloc for dynamic symbol %s", targ.Name)
		}
		return true

	case 512 + ld.MACHO_GENERIC_RELOC_VANILLA*2 + 1:
		if targ.Type == ld.SDYNIMPORT {
			addpltsym(ctxt, targ)
			r.Sym = ctxt.Syms.Lookup(".plt", 0)
			r.Add = int64(targ.Plt)
			r.Type = objabi.R_PCREL
			return true
		}

		r.Type = objabi.R_PCREL
		return true

	case 512 + ld.MACHO_FAKE_GOTPCREL:
		if targ.Type != ld.SDYNIMPORT {
			// have symbol
			// turn MOVL of GOT entry into LEAL of symbol itself
			if r.Off < 2 || s.P[r.Off-2] != 0x8b {
				ld.Errorf(s, "unexpected GOT reloc for non-dynamic symbol %s", targ.Name)
				return false
			}

			s.P[r.Off-2] = 0x8d
			r.Type = objabi.R_PCREL
			return true
		}

		addgotsym(ctxt, targ)
		r.Sym = ctxt.Syms.Lookup(".got", 0)
		r.Add += int64(targ.Got)
		r.Type = objabi.R_PCREL
		return true
	}

	// Handle references to ELF symbols from our own object files.
	if targ.Type != ld.SDYNIMPORT {
		return true
	}
	switch r.Type {
	case objabi.R_CALL,
		objabi.R_PCREL:
		addpltsym(ctxt, targ)
		r.Sym = ctxt.Syms.Lookup(".plt", 0)
		r.Add = int64(targ.Plt)
		return true

	case objabi.R_ADDR:
		if s.Type != ld.SDATA {
			break
		}
		if ld.Iself {
			ld.Adddynsym(ctxt, targ)
			rel := ctxt.Syms.Lookup(".rel", 0)
			ld.Addaddrplus(ctxt, rel, s, int64(r.Off))
			ld.Adduint32(ctxt, rel, ld.ELF32_R_INFO(uint32(targ.Dynid), ld.R_386_32))
			r.Type = objabi.R_CONST // write r->add during relocsym
			r.Sym = nil
			return true
		}

		if ld.Headtype == objabi.Hdarwin && s.Size == int64(ld.SysArch.PtrSize) && r.Off == 0 {
			// Mach-O relocations are a royal pain to lay out.
			// They use a compact stateful bytecode representation
			// that is too much bother to deal with.
			// Instead, interpret the C declaration
			//	void *_Cvar_stderr = &stderr;
			// as making _Cvar_stderr the name of a GOT entry
			// for stderr. This is separate from the usual GOT entry,
			// just in case the C code assigns to the variable,
			// and of course it only works for single pointers,
			// but we only need to support cgo and that's all it needs.
			ld.Adddynsym(ctxt, targ)

			got := ctxt.Syms.Lookup(".got", 0)
			s.Type = got.Type | ld.SSUB
			s.Outer = got
			s.Sub = got.Sub
			got.Sub = s
			s.Value = got.Size
			ld.Adduint32(ctxt, got, 0)
			ld.Adduint32(ctxt, ctxt.Syms.Lookup(".linkedit.got", 0), uint32(targ.Dynid))
			r.Type = 256 // ignore during relocsym
			return true
		}

		if ld.Headtype == objabi.Hwindows && s.Size == int64(ld.SysArch.PtrSize) {
			// nothing to do, the relocation will be laid out in pereloc1
			return true
		}
	}

	return false
}

func elfreloc1(ctxt *ld.Link, r *ld.Reloc, sectoff int64) int {
	ld.Thearch.Lput(uint32(sectoff))

	elfsym := r.Xsym.ElfsymForReloc()
	switch r.Type {
	default:
		return -1

	case objabi.R_ADDR:
		if r.Siz == 4 {
			ld.Thearch.Lput(ld.R_386_32 | uint32(elfsym)<<8)
		} else {
			return -1
		}

	case objabi.R_GOTPCREL:
		if r.Siz == 4 {
			ld.Thearch.Lput(ld.R_386_GOTPC)
			if r.Xsym.Name != "_GLOBAL_OFFSET_TABLE_" {
				ld.Thearch.Lput(uint32(sectoff))
				ld.Thearch.Lput(ld.R_386_GOT32 | uint32(elfsym)<<8)
			}
		} else {
			return -1
		}

	case objabi.R_CALL:
		if r.Siz == 4 {
			if r.Xsym.Type == ld.SDYNIMPORT {
				ld.Thearch.Lput(ld.R_386_PLT32 | uint32(elfsym)<<8)
			} else {
				ld.Thearch.Lput(ld.R_386_PC32 | uint32(elfsym)<<8)
			}
		} else {
			return -1
		}

	case objabi.R_PCREL:
		if r.Siz == 4 {
			ld.Thearch.Lput(ld.R_386_PC32 | uint32(elfsym)<<8)
		} else {
			return -1
		}

	case objabi.R_TLS_LE:
		if r.Siz == 4 {
			ld.Thearch.Lput(ld.R_386_TLS_LE | uint32(elfsym)<<8)
		} else {
			return -1
		}

	case objabi.R_TLS_IE:
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

func machoreloc1(s *ld.Symbol, r *ld.Reloc, sectoff int64) int {
	var v uint32

	rs := r.Xsym

	if rs.Type == ld.SHOSTOBJ {
		if rs.Dynid < 0 {
			ld.Errorf(s, "reloc %d to non-macho symbol %s type=%d", r.Type, rs.Name, rs.Type)
			return -1
		}

		v = uint32(rs.Dynid)
		v |= 1 << 27 // external relocation
	} else {
		v = uint32(rs.Sect.Extnum)
		if v == 0 {
			ld.Errorf(s, "reloc %d to symbol %s in non-macho section %s type=%d", r.Type, rs.Name, rs.Sect.Name, rs.Type)
			return -1
		}
	}

	switch r.Type {
	default:
		return -1

	case objabi.R_ADDR:
		v |= ld.MACHO_GENERIC_RELOC_VANILLA << 28

	case objabi.R_CALL,
		objabi.R_PCREL:
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

func pereloc1(s *ld.Symbol, r *ld.Reloc, sectoff int64) bool {
	var v uint32

	rs := r.Xsym

	if rs.Dynid < 0 {
		ld.Errorf(s, "reloc %d to non-coff symbol %s type=%d", r.Type, rs.Name, rs.Type)
		return false
	}

	ld.Thearch.Lput(uint32(sectoff))
	ld.Thearch.Lput(uint32(rs.Dynid))

	switch r.Type {
	default:
		return false

	case objabi.R_DWARFREF:
		v = ld.IMAGE_REL_I386_SECREL

	case objabi.R_ADDR:
		v = ld.IMAGE_REL_I386_DIR32

	case objabi.R_CALL,
		objabi.R_PCREL:
		v = ld.IMAGE_REL_I386_REL32
	}

	ld.Thearch.Wput(uint16(v))

	return true
}

func archreloc(ctxt *ld.Link, r *ld.Reloc, s *ld.Symbol, val *int64) int {
	if ld.Linkmode == ld.LinkExternal {
		return -1
	}
	switch r.Type {
	case objabi.R_CONST:
		*val = r.Add
		return 0

	case objabi.R_GOTOFF:
		*val = ld.Symaddr(r.Sym) + r.Add - ld.Symaddr(ctxt.Syms.Lookup(".got", 0))
		return 0
	}

	return -1
}

func archrelocvariant(ctxt *ld.Link, r *ld.Reloc, s *ld.Symbol, t int64) int64 {
	log.Fatalf("unexpected relocation variant")
	return t
}

func elfsetupplt(ctxt *ld.Link) {
	plt := ctxt.Syms.Lookup(".plt", 0)
	got := ctxt.Syms.Lookup(".got.plt", 0)
	if plt.Size == 0 {
		// pushl got+4
		ld.Adduint8(ctxt, plt, 0xff)

		ld.Adduint8(ctxt, plt, 0x35)
		ld.Addaddrplus(ctxt, plt, got, 4)

		// jmp *got+8
		ld.Adduint8(ctxt, plt, 0xff)

		ld.Adduint8(ctxt, plt, 0x25)
		ld.Addaddrplus(ctxt, plt, got, 8)

		// zero pad
		ld.Adduint32(ctxt, plt, 0)

		// assume got->size == 0 too
		ld.Addaddrplus(ctxt, got, ctxt.Syms.Lookup(".dynamic", 0), 0)

		ld.Adduint32(ctxt, got, 0)
		ld.Adduint32(ctxt, got, 0)
	}
}

func addpltsym(ctxt *ld.Link, s *ld.Symbol) {
	if s.Plt >= 0 {
		return
	}

	ld.Adddynsym(ctxt, s)

	if ld.Iself {
		plt := ctxt.Syms.Lookup(".plt", 0)
		got := ctxt.Syms.Lookup(".got.plt", 0)
		rel := ctxt.Syms.Lookup(".rel.plt", 0)
		if plt.Size == 0 {
			elfsetupplt(ctxt)
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
	} else if ld.Headtype == objabi.Hdarwin {
		// Same laziness as in 6l.

		plt := ctxt.Syms.Lookup(".plt", 0)

		addgotsym(ctxt, s)

		ld.Adduint32(ctxt, ctxt.Syms.Lookup(".linkedit.plt", 0), uint32(s.Dynid))

		// jmpq *got+size(IP)
		s.Plt = int32(plt.Size)

		ld.Adduint8(ctxt, plt, 0xff)
		ld.Adduint8(ctxt, plt, 0x25)
		ld.Addaddrplus(ctxt, plt, ctxt.Syms.Lookup(".got", 0), int64(s.Got))
	} else {
		ld.Errorf(s, "addpltsym: unsupported binary format")
	}
}

func addgotsym(ctxt *ld.Link, s *ld.Symbol) {
	if s.Got >= 0 {
		return
	}

	ld.Adddynsym(ctxt, s)
	got := ctxt.Syms.Lookup(".got", 0)
	s.Got = int32(got.Size)
	ld.Adduint32(ctxt, got, 0)

	if ld.Iself {
		rel := ctxt.Syms.Lookup(".rel", 0)
		ld.Addaddrplus(ctxt, rel, got, int64(s.Got))
		ld.Adduint32(ctxt, rel, ld.ELF32_R_INFO(uint32(s.Dynid), ld.R_386_GLOB_DAT))
	} else if ld.Headtype == objabi.Hdarwin {
		ld.Adduint32(ctxt, ctxt.Syms.Lookup(".linkedit.got", 0), uint32(s.Dynid))
	} else {
		ld.Errorf(s, "addgotsym: unsupported binary format")
	}
}

func asmb(ctxt *ld.Link) {
	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f asmb\n", ld.Cputime())
	}

	if ld.Iself {
		ld.Asmbelfsetup()
	}

	sect := ld.Segtext.Sections[0]
	ld.Cseek(int64(sect.Vaddr - ld.Segtext.Vaddr + ld.Segtext.Fileoff))
	// 0xCC is INT $3 - breakpoint instruction
	ld.CodeblkPad(ctxt, int64(sect.Vaddr), int64(sect.Length), []byte{0xCC})
	for _, sect = range ld.Segtext.Sections[1:] {
		ld.Cseek(int64(sect.Vaddr - ld.Segtext.Vaddr + ld.Segtext.Fileoff))
		ld.Datblk(ctxt, int64(sect.Vaddr), int64(sect.Length))
	}

	if ld.Segrodata.Filelen > 0 {
		if ctxt.Debugvlog != 0 {
			ctxt.Logf("%5.2f rodatblk\n", ld.Cputime())
		}

		ld.Cseek(int64(ld.Segrodata.Fileoff))
		ld.Datblk(ctxt, int64(ld.Segrodata.Vaddr), int64(ld.Segrodata.Filelen))
	}
	if ld.Segrelrodata.Filelen > 0 {
		if ctxt.Debugvlog != 0 {
			ctxt.Logf("%5.2f relrodatblk\n", ld.Cputime())
		}
		ld.Cseek(int64(ld.Segrelrodata.Fileoff))
		ld.Datblk(ctxt, int64(ld.Segrelrodata.Vaddr), int64(ld.Segrelrodata.Filelen))
	}

	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f datblk\n", ld.Cputime())
	}

	ld.Cseek(int64(ld.Segdata.Fileoff))
	ld.Datblk(ctxt, int64(ld.Segdata.Vaddr), int64(ld.Segdata.Filelen))

	ld.Cseek(int64(ld.Segdwarf.Fileoff))
	ld.Dwarfblk(ctxt, int64(ld.Segdwarf.Vaddr), int64(ld.Segdwarf.Filelen))

	machlink := uint32(0)
	if ld.Headtype == objabi.Hdarwin {
		machlink = uint32(ld.Domacholink(ctxt))
	}

	ld.Symsize = 0
	ld.Spsize = 0
	ld.Lcsize = 0
	symo := uint32(0)
	if !*ld.FlagS {
		// TODO: rationalize
		if ctxt.Debugvlog != 0 {
			ctxt.Logf("%5.2f sym\n", ld.Cputime())
		}
		switch ld.Headtype {
		default:
			if ld.Iself {
				symo = uint32(ld.Segdwarf.Fileoff + ld.Segdwarf.Filelen)
				symo = uint32(ld.Rnd(int64(symo), int64(*ld.FlagRound)))
			}

		case objabi.Hplan9:
			symo = uint32(ld.Segdata.Fileoff + ld.Segdata.Filelen)

		case objabi.Hdarwin:
			symo = uint32(ld.Segdwarf.Fileoff + uint64(ld.Rnd(int64(ld.Segdwarf.Filelen), int64(*ld.FlagRound))) + uint64(machlink))

		case objabi.Hwindows:
			symo = uint32(ld.Segdwarf.Fileoff + ld.Segdwarf.Filelen)
			symo = uint32(ld.Rnd(int64(symo), ld.PEFILEALIGN))
		}

		ld.Cseek(int64(symo))
		switch ld.Headtype {
		default:
			if ld.Iself {
				if ctxt.Debugvlog != 0 {
					ctxt.Logf("%5.2f elfsym\n", ld.Cputime())
				}
				ld.Asmelfsym(ctxt)
				ld.Cflush()
				ld.Cwrite(ld.Elfstrdat)

				if ld.Linkmode == ld.LinkExternal {
					ld.Elfemitreloc(ctxt)
				}
			}

		case objabi.Hplan9:
			ld.Asmplan9sym(ctxt)
			ld.Cflush()

			sym := ctxt.Syms.Lookup("pclntab", 0)
			if sym != nil {
				ld.Lcsize = int32(len(sym.P))
				for i := 0; int32(i) < ld.Lcsize; i++ {
					ld.Cput(sym.P[i])
				}

				ld.Cflush()
			}

		case objabi.Hwindows:
			if ctxt.Debugvlog != 0 {
				ctxt.Logf("%5.2f dwarf\n", ld.Cputime())
			}

		case objabi.Hdarwin:
			if ld.Linkmode == ld.LinkExternal {
				ld.Machoemitreloc(ctxt)
			}
		}
	}

	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f headr\n", ld.Cputime())
	}
	ld.Cseek(0)
	switch ld.Headtype {
	default:
	case objabi.Hplan9: /* plan9 */
		magic := int32(4*11*11 + 7)

		ld.Lputb(uint32(magic))              /* magic */
		ld.Lputb(uint32(ld.Segtext.Filelen)) /* sizes */
		ld.Lputb(uint32(ld.Segdata.Filelen))
		ld.Lputb(uint32(ld.Segdata.Length - ld.Segdata.Filelen))
		ld.Lputb(uint32(ld.Symsize))          /* nsyms */
		ld.Lputb(uint32(ld.Entryvalue(ctxt))) /* va of entry */
		ld.Lputb(uint32(ld.Spsize))           /* sp offsets */
		ld.Lputb(uint32(ld.Lcsize))           /* line offsets */

	case objabi.Hdarwin:
		ld.Asmbmacho(ctxt)

	case objabi.Hlinux,
		objabi.Hfreebsd,
		objabi.Hnetbsd,
		objabi.Hopenbsd,
		objabi.Hnacl:
		ld.Asmbelf(ctxt, int64(symo))

	case objabi.Hwindows:
		ld.Asmbpe(ctxt)
	}

	ld.Cflush()
}
