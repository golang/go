// Inferno utils/6l/asm.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/6l/asm.c
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
	"cmd/link/internal/ld"
	"debug/elf"
	"log"
)

func PADDR(x uint32) uint32 {
	return x &^ 0x80000000
}

func Addcall(ctxt *ld.Link, s *ld.Symbol, t *ld.Symbol) int64 {
	s.Attr |= ld.AttrReachable
	i := s.Size
	s.Size += 4
	ld.Symgrow(s, s.Size)
	r := ld.Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Type = objabi.R_CALL
	r.Siz = 4
	return i + int64(r.Siz)
}

func gentext(ctxt *ld.Link) {
	if !ctxt.DynlinkingGo() {
		return
	}
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
	// 0000000000000000 <local.dso_init>:
	//    0:	48 8d 3d 00 00 00 00 	lea    0x0(%rip),%rdi        # 7 <local.dso_init+0x7>
	// 			3: R_X86_64_PC32	runtime.firstmoduledata-0x4
	o(0x48, 0x8d, 0x3d)
	ld.Addpcrelplus(ctxt, initfunc, ctxt.Moduledata, 0)
	//    7:	e8 00 00 00 00       	callq  c <local.dso_init+0xc>
	// 			8: R_X86_64_PLT32	runtime.addmoduledata-0x4
	o(0xe8)
	Addcall(ctxt, initfunc, addmoduledata)
	//    c:	c3                   	retq
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
	case 256 + ld.R_X86_64_PC32:
		if targ.Type == ld.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_X86_64_PC32 relocation for dynamic symbol %s", targ.Name)
		}
		if targ.Type == 0 || targ.Type == ld.SXREF {
			ld.Errorf(s, "unknown symbol %s in pcrel", targ.Name)
		}
		r.Type = objabi.R_PCREL
		r.Add += 4
		return true

	case 256 + ld.R_X86_64_PLT32:
		r.Type = objabi.R_PCREL
		r.Add += 4
		if targ.Type == ld.SDYNIMPORT {
			addpltsym(ctxt, targ)
			r.Sym = ctxt.Syms.Lookup(".plt", 0)
			r.Add += int64(targ.Plt)
		}

		return true

	case 256 + ld.R_X86_64_GOTPCREL, 256 + ld.R_X86_64_GOTPCRELX, 256 + ld.R_X86_64_REX_GOTPCRELX:
		if targ.Type != ld.SDYNIMPORT {
			// have symbol
			if r.Off >= 2 && s.P[r.Off-2] == 0x8b {
				// turn MOVQ of GOT entry into LEAQ of symbol itself
				s.P[r.Off-2] = 0x8d

				r.Type = objabi.R_PCREL
				r.Add += 4
				return true
			}
		}

		// fall back to using GOT and hope for the best (CMOV*)
		// TODO: just needs relocation, no need to put in .dynsym
		addgotsym(ctxt, targ)

		r.Type = objabi.R_PCREL
		r.Sym = ctxt.Syms.Lookup(".got", 0)
		r.Add += 4
		r.Add += int64(targ.Got)
		return true

	case 256 + ld.R_X86_64_64:
		if targ.Type == ld.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_X86_64_64 relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = objabi.R_ADDR
		return true

	// Handle relocations found in Mach-O object files.
	case 512 + ld.MACHO_X86_64_RELOC_UNSIGNED*2 + 0,
		512 + ld.MACHO_X86_64_RELOC_SIGNED*2 + 0,
		512 + ld.MACHO_X86_64_RELOC_BRANCH*2 + 0:
		// TODO: What is the difference between all these?
		r.Type = objabi.R_ADDR

		if targ.Type == ld.SDYNIMPORT {
			ld.Errorf(s, "unexpected reloc for dynamic symbol %s", targ.Name)
		}
		return true

	case 512 + ld.MACHO_X86_64_RELOC_BRANCH*2 + 1:
		if targ.Type == ld.SDYNIMPORT {
			addpltsym(ctxt, targ)
			r.Sym = ctxt.Syms.Lookup(".plt", 0)
			r.Add = int64(targ.Plt)
			r.Type = objabi.R_PCREL
			return true
		}
		fallthrough

		// fall through
	case 512 + ld.MACHO_X86_64_RELOC_UNSIGNED*2 + 1,
		512 + ld.MACHO_X86_64_RELOC_SIGNED*2 + 1,
		512 + ld.MACHO_X86_64_RELOC_SIGNED_1*2 + 1,
		512 + ld.MACHO_X86_64_RELOC_SIGNED_2*2 + 1,
		512 + ld.MACHO_X86_64_RELOC_SIGNED_4*2 + 1:
		r.Type = objabi.R_PCREL

		if targ.Type == ld.SDYNIMPORT {
			ld.Errorf(s, "unexpected pc-relative reloc for dynamic symbol %s", targ.Name)
		}
		return true

	case 512 + ld.MACHO_X86_64_RELOC_GOT_LOAD*2 + 1:
		if targ.Type != ld.SDYNIMPORT {
			// have symbol
			// turn MOVQ of GOT entry into LEAQ of symbol itself
			if r.Off < 2 || s.P[r.Off-2] != 0x8b {
				ld.Errorf(s, "unexpected GOT_LOAD reloc for non-dynamic symbol %s", targ.Name)
				return false
			}

			s.P[r.Off-2] = 0x8d
			r.Type = objabi.R_PCREL
			return true
		}
		fallthrough

		// fall through
	case 512 + ld.MACHO_X86_64_RELOC_GOT*2 + 1:
		if targ.Type != ld.SDYNIMPORT {
			ld.Errorf(s, "unexpected GOT reloc for non-dynamic symbol %s", targ.Name)
		}
		addgotsym(ctxt, targ)
		r.Type = objabi.R_PCREL
		r.Sym = ctxt.Syms.Lookup(".got", 0)
		r.Add += int64(targ.Got)
		return true
	}

	switch r.Type {
	case objabi.R_CALL,
		objabi.R_PCREL:
		if targ.Type != ld.SDYNIMPORT {
			// nothing to do, the relocation will be laid out in reloc
			return true
		}
		if ld.Headtype == objabi.Hwindows {
			// nothing to do, the relocation will be laid out in pereloc1
			return true
		} else {
			// for both ELF and Mach-O
			addpltsym(ctxt, targ)
			r.Sym = ctxt.Syms.Lookup(".plt", 0)
			r.Add = int64(targ.Plt)
			return true
		}

	case objabi.R_ADDR:
		if s.Type == ld.STEXT && ld.Iself {
			if ld.Headtype == objabi.Hsolaris {
				addpltsym(ctxt, targ)
				r.Sym = ctxt.Syms.Lookup(".plt", 0)
				r.Add += int64(targ.Plt)
				return true
			}
			// The code is asking for the address of an external
			// function. We provide it with the address of the
			// correspondent GOT symbol.
			addgotsym(ctxt, targ)

			r.Sym = ctxt.Syms.Lookup(".got", 0)
			r.Add += int64(targ.Got)
			return true
		}

		// Process dynamic relocations for the data sections.
		if ld.Buildmode == ld.BuildmodePIE && ld.Linkmode == ld.LinkInternal {
			// When internally linking, generate dynamic relocations
			// for all typical R_ADDR relocations. The exception
			// are those R_ADDR that are created as part of generating
			// the dynamic relocations and must be resolved statically.
			//
			// There are three phases relevant to understanding this:
			//
			//	dodata()  // we are here
			//	address() // symbol address assignment
			//	reloc()   // resolution of static R_ADDR relocs
			//
			// At this point symbol addresses have not been
			// assigned yet (as the final size of the .rela section
			// will affect the addresses), and so we cannot write
			// the Elf64_Rela.r_offset now. Instead we delay it
			// until after the 'address' phase of the linker is
			// complete. We do this via Addaddrplus, which creates
			// a new R_ADDR relocation which will be resolved in
			// the 'reloc' phase.
			//
			// These synthetic static R_ADDR relocs must be skipped
			// now, or else we will be caught in an infinite loop
			// of generating synthetic relocs for our synthetic
			// relocs.
			//
			// Furthermore, the rela sections contain dynamic
			// relocations with R_ADDR relocations on
			// Elf64_Rela.r_offset. This field should contain the
			// symbol offset as determined by reloc(), not the
			// final dynamically linked address as a dynamic
			// relocation would provide.
			switch s.Name {
			case ".dynsym", ".rela", ".rela.plt", ".got.plt", ".dynamic":
				return false
			}
		} else {
			// Either internally linking a static executable,
			// in which case we can resolve these relocations
			// statically in the 'reloc' phase, or externally
			// linking, in which case the relocation will be
			// prepared in the 'reloc' phase and passed to the
			// external linker in the 'asmb' phase.
			if s.Type != ld.SDATA && s.Type != ld.SRODATA {
				break
			}
		}

		if ld.Iself {
			// TODO: We generate a R_X86_64_64 relocation for every R_ADDR, even
			// though it would be more efficient (for the dynamic linker) if we
			// generated R_X86_RELATIVE instead.
			ld.Adddynsym(ctxt, targ)
			rela := ctxt.Syms.Lookup(".rela", 0)
			ld.Addaddrplus(ctxt, rela, s, int64(r.Off))
			if r.Siz == 8 {
				ld.Adduint64(ctxt, rela, ld.ELF64_R_INFO(uint32(targ.Dynid), ld.R_X86_64_64))
			} else {
				// TODO: never happens, remove.
				ld.Adduint64(ctxt, rela, ld.ELF64_R_INFO(uint32(targ.Dynid), ld.R_X86_64_32))
			}
			ld.Adduint64(ctxt, rela, uint64(r.Add))
			r.Type = 256 // ignore during relocsym
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
			ld.Adduint64(ctxt, got, 0)
			ld.Adduint32(ctxt, ctxt.Syms.Lookup(".linkedit.got", 0), uint32(targ.Dynid))
			r.Type = 256 // ignore during relocsym
			return true
		}

		if ld.Headtype == objabi.Hwindows {
			// nothing to do, the relocation will be laid out in pereloc1
			return true
		}
	}

	return false
}

func elfreloc1(ctxt *ld.Link, r *ld.Reloc, sectoff int64) int {
	ld.Thearch.Vput(uint64(sectoff))

	elfsym := r.Xsym.ElfsymForReloc()
	switch r.Type {
	default:
		return -1

	case objabi.R_ADDR:
		if r.Siz == 4 {
			ld.Thearch.Vput(ld.R_X86_64_32 | uint64(elfsym)<<32)
		} else if r.Siz == 8 {
			ld.Thearch.Vput(ld.R_X86_64_64 | uint64(elfsym)<<32)
		} else {
			return -1
		}

	case objabi.R_TLS_LE:
		if r.Siz == 4 {
			ld.Thearch.Vput(ld.R_X86_64_TPOFF32 | uint64(elfsym)<<32)
		} else {
			return -1
		}

	case objabi.R_TLS_IE:
		if r.Siz == 4 {
			ld.Thearch.Vput(ld.R_X86_64_GOTTPOFF | uint64(elfsym)<<32)
		} else {
			return -1
		}

	case objabi.R_CALL:
		if r.Siz == 4 {
			if r.Xsym.Type == ld.SDYNIMPORT {
				if ctxt.DynlinkingGo() {
					ld.Thearch.Vput(ld.R_X86_64_PLT32 | uint64(elfsym)<<32)
				} else {
					ld.Thearch.Vput(ld.R_X86_64_GOTPCREL | uint64(elfsym)<<32)
				}
			} else {
				ld.Thearch.Vput(ld.R_X86_64_PC32 | uint64(elfsym)<<32)
			}
		} else {
			return -1
		}

	case objabi.R_PCREL:
		if r.Siz == 4 {
			if r.Xsym.Type == ld.SDYNIMPORT && r.Xsym.ElfType == elf.STT_FUNC {
				ld.Thearch.Vput(ld.R_X86_64_PLT32 | uint64(elfsym)<<32)
			} else {
				ld.Thearch.Vput(ld.R_X86_64_PC32 | uint64(elfsym)<<32)
			}
		} else {
			return -1
		}

	case objabi.R_GOTPCREL:
		if r.Siz == 4 {
			ld.Thearch.Vput(ld.R_X86_64_GOTPCREL | uint64(elfsym)<<32)
		} else {
			return -1
		}
	}

	ld.Thearch.Vput(uint64(r.Xadd))
	return 0
}

func machoreloc1(s *ld.Symbol, r *ld.Reloc, sectoff int64) int {
	var v uint32

	rs := r.Xsym

	if rs.Type == ld.SHOSTOBJ || r.Type == objabi.R_PCREL || r.Type == objabi.R_GOTPCREL {
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
		v |= ld.MACHO_X86_64_RELOC_UNSIGNED << 28

	case objabi.R_CALL:
		v |= 1 << 24 // pc-relative bit
		v |= ld.MACHO_X86_64_RELOC_BRANCH << 28

		// NOTE: Only works with 'external' relocation. Forced above.
	case objabi.R_PCREL:
		v |= 1 << 24 // pc-relative bit
		v |= ld.MACHO_X86_64_RELOC_SIGNED << 28
	case objabi.R_GOTPCREL:
		v |= 1 << 24 // pc-relative bit
		v |= ld.MACHO_X86_64_RELOC_GOT_LOAD << 28
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
		v = ld.IMAGE_REL_AMD64_SECREL

	case objabi.R_ADDR:
		if r.Siz == 8 {
			v = ld.IMAGE_REL_AMD64_ADDR64
		} else {
			v = ld.IMAGE_REL_AMD64_ADDR32
		}

	case objabi.R_CALL,
		objabi.R_PCREL:
		v = ld.IMAGE_REL_AMD64_REL32
	}

	ld.Thearch.Wput(uint16(v))

	return true
}

func archreloc(ctxt *ld.Link, r *ld.Reloc, s *ld.Symbol, val *int64) int {
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
		// pushq got+8(IP)
		ld.Adduint8(ctxt, plt, 0xff)

		ld.Adduint8(ctxt, plt, 0x35)
		ld.Addpcrelplus(ctxt, plt, got, 8)

		// jmpq got+16(IP)
		ld.Adduint8(ctxt, plt, 0xff)

		ld.Adduint8(ctxt, plt, 0x25)
		ld.Addpcrelplus(ctxt, plt, got, 16)

		// nopl 0(AX)
		ld.Adduint32(ctxt, plt, 0x00401f0f)

		// assume got->size == 0 too
		ld.Addaddrplus(ctxt, got, ctxt.Syms.Lookup(".dynamic", 0), 0)

		ld.Adduint64(ctxt, got, 0)
		ld.Adduint64(ctxt, got, 0)
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
		rela := ctxt.Syms.Lookup(".rela.plt", 0)
		if plt.Size == 0 {
			elfsetupplt(ctxt)
		}

		// jmpq *got+size(IP)
		ld.Adduint8(ctxt, plt, 0xff)

		ld.Adduint8(ctxt, plt, 0x25)
		ld.Addpcrelplus(ctxt, plt, got, got.Size)

		// add to got: pointer to current pos in plt
		ld.Addaddrplus(ctxt, got, plt, plt.Size)

		// pushq $x
		ld.Adduint8(ctxt, plt, 0x68)

		ld.Adduint32(ctxt, plt, uint32((got.Size-24-8)/8))

		// jmpq .plt
		ld.Adduint8(ctxt, plt, 0xe9)

		ld.Adduint32(ctxt, plt, uint32(-(plt.Size + 4)))

		// rela
		ld.Addaddrplus(ctxt, rela, got, got.Size-8)

		ld.Adduint64(ctxt, rela, ld.ELF64_R_INFO(uint32(s.Dynid), ld.R_X86_64_JMP_SLOT))
		ld.Adduint64(ctxt, rela, 0)

		s.Plt = int32(plt.Size - 16)
	} else if ld.Headtype == objabi.Hdarwin {
		// To do lazy symbol lookup right, we're supposed
		// to tell the dynamic loader which library each
		// symbol comes from and format the link info
		// section just so. I'm too lazy (ha!) to do that
		// so for now we'll just use non-lazy pointers,
		// which don't need to be told which library to use.
		//
		// http://networkpx.blogspot.com/2009/09/about-lcdyldinfoonly-command.html
		// has details about what we're avoiding.

		addgotsym(ctxt, s)
		plt := ctxt.Syms.Lookup(".plt", 0)

		ld.Adduint32(ctxt, ctxt.Syms.Lookup(".linkedit.plt", 0), uint32(s.Dynid))

		// jmpq *got+size(IP)
		s.Plt = int32(plt.Size)

		ld.Adduint8(ctxt, plt, 0xff)
		ld.Adduint8(ctxt, plt, 0x25)
		ld.Addpcrelplus(ctxt, plt, ctxt.Syms.Lookup(".got", 0), int64(s.Got))
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
	ld.Adduint64(ctxt, got, 0)

	if ld.Iself {
		rela := ctxt.Syms.Lookup(".rela", 0)
		ld.Addaddrplus(ctxt, rela, got, int64(s.Got))
		ld.Adduint64(ctxt, rela, ld.ELF64_R_INFO(uint32(s.Dynid), ld.R_X86_64_GLOB_DAT))
		ld.Adduint64(ctxt, rela, 0)
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

	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f codeblk\n", ld.Cputime())
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

	machlink := int64(0)
	if ld.Headtype == objabi.Hdarwin {
		machlink = ld.Domacholink(ctxt)
	}

	switch ld.Headtype {
	default:
		ld.Errorf(nil, "unknown header type %v", ld.Headtype)
		fallthrough

	case objabi.Hplan9:
		break

	case objabi.Hdarwin:
		ld.Flag8 = true /* 64-bit addresses */

	case objabi.Hlinux,
		objabi.Hfreebsd,
		objabi.Hnetbsd,
		objabi.Hopenbsd,
		objabi.Hdragonfly,
		objabi.Hsolaris:
		ld.Flag8 = true /* 64-bit addresses */

	case objabi.Hnacl,
		objabi.Hwindows:
		break
	}

	ld.Symsize = 0
	ld.Spsize = 0
	ld.Lcsize = 0
	symo := int64(0)
	if !*ld.FlagS {
		if ctxt.Debugvlog != 0 {
			ctxt.Logf("%5.2f sym\n", ld.Cputime())
		}
		switch ld.Headtype {
		default:
		case objabi.Hplan9:
			*ld.FlagS = true
			symo = int64(ld.Segdata.Fileoff + ld.Segdata.Filelen)

		case objabi.Hdarwin:
			symo = int64(ld.Segdwarf.Fileoff + uint64(ld.Rnd(int64(ld.Segdwarf.Filelen), int64(*ld.FlagRound))) + uint64(machlink))

		case objabi.Hlinux,
			objabi.Hfreebsd,
			objabi.Hnetbsd,
			objabi.Hopenbsd,
			objabi.Hdragonfly,
			objabi.Hsolaris,
			objabi.Hnacl:
			symo = int64(ld.Segdwarf.Fileoff + ld.Segdwarf.Filelen)
			symo = ld.Rnd(symo, int64(*ld.FlagRound))

		case objabi.Hwindows:
			symo = int64(ld.Segdwarf.Fileoff + ld.Segdwarf.Filelen)
			symo = ld.Rnd(symo, ld.PEFILEALIGN)
		}

		ld.Cseek(symo)
		switch ld.Headtype {
		default:
			if ld.Iself {
				ld.Cseek(symo)
				ld.Asmelfsym(ctxt)
				ld.Cflush()
				ld.Cwrite(ld.Elfstrdat)

				if ctxt.Debugvlog != 0 {
					ctxt.Logf("%5.2f dwarf\n", ld.Cputime())
				}

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
		magic := int32(4*26*26 + 7)

		magic |= 0x00008000                  /* fat header */
		ld.Lputb(uint32(magic))              /* magic */
		ld.Lputb(uint32(ld.Segtext.Filelen)) /* sizes */
		ld.Lputb(uint32(ld.Segdata.Filelen))
		ld.Lputb(uint32(ld.Segdata.Length - ld.Segdata.Filelen))
		ld.Lputb(uint32(ld.Symsize)) /* nsyms */
		vl := ld.Entryvalue(ctxt)
		ld.Lputb(PADDR(uint32(vl))) /* va of entry */
		ld.Lputb(uint32(ld.Spsize)) /* sp offsets */
		ld.Lputb(uint32(ld.Lcsize)) /* line offsets */
		ld.Vputb(uint64(vl))        /* va of entry */

	case objabi.Hdarwin:
		ld.Asmbmacho(ctxt)

	case objabi.Hlinux,
		objabi.Hfreebsd,
		objabi.Hnetbsd,
		objabi.Hopenbsd,
		objabi.Hdragonfly,
		objabi.Hsolaris,
		objabi.Hnacl:
		ld.Asmbelf(ctxt, symo)

	case objabi.Hwindows:
		ld.Asmbpe(ctxt)
	}

	ld.Cflush()
}

func tlsIEtoLE(s *ld.Symbol, off, size int) {
	// Transform the PC-relative instruction into a constant load.
	// That is,
	//
	//	MOVQ X(IP), REG  ->  MOVQ $Y, REG
	//
	// To determine the instruction and register, we study the op codes.
	// Consult an AMD64 instruction encoding guide to decipher this.
	if off < 3 {
		log.Fatal("R_X86_64_GOTTPOFF reloc not preceded by MOVQ or ADDQ instruction")
	}
	op := s.P[off-3 : off]
	reg := op[2] >> 3

	if op[1] == 0x8b || reg == 4 {
		// MOVQ
		if op[0] == 0x4c {
			op[0] = 0x49
		} else if size == 4 && op[0] == 0x44 {
			op[0] = 0x41
		}
		if op[1] == 0x8b {
			op[1] = 0xc7
		} else {
			op[1] = 0x81 // special case for SP
		}
		op[2] = 0xc0 | reg
	} else {
		// An alternate op is ADDQ. This is handled by GNU gold,
		// but right now is not generated by the Go compiler:
		//	ADDQ X(IP), REG  ->  ADDQ $Y, REG
		// Consider adding support for it here.
		log.Fatalf("expected TLS IE op to be MOVQ, got %v", op)
	}
}
