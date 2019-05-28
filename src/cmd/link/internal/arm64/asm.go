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

package arm64

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/ld"
	"cmd/link/internal/sym"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"log"
)

func gentext(ctxt *ld.Link) {
	if !ctxt.DynlinkingGo() {
		return
	}
	addmoduledata := ctxt.Syms.Lookup("runtime.addmoduledata", 0)
	if addmoduledata.Type == sym.STEXT && ctxt.BuildMode != ld.BuildModePlugin {
		// we're linking a module containing the runtime -> no need for
		// an init function
		return
	}
	addmoduledata.Attr |= sym.AttrReachable
	initfunc := ctxt.Syms.Lookup("go.link.addmoduledata", 0)
	initfunc.Type = sym.STEXT
	initfunc.Attr |= sym.AttrLocal
	initfunc.Attr |= sym.AttrReachable
	o := func(op uint32) {
		initfunc.AddUint32(ctxt.Arch, op)
	}
	// 0000000000000000 <local.dso_init>:
	// 0:	90000000 	adrp	x0, 0 <runtime.firstmoduledata>
	// 	0: R_AARCH64_ADR_PREL_PG_HI21	local.moduledata
	// 4:	91000000 	add	x0, x0, #0x0
	// 	4: R_AARCH64_ADD_ABS_LO12_NC	local.moduledata
	o(0x90000000)
	o(0x91000000)
	rel := initfunc.AddRel()
	rel.Off = 0
	rel.Siz = 8
	rel.Sym = ctxt.Moduledata
	rel.Type = objabi.R_ADDRARM64

	// 8:	14000000 	b	0 <runtime.addmoduledata>
	// 	8: R_AARCH64_CALL26	runtime.addmoduledata
	o(0x14000000)
	rel = initfunc.AddRel()
	rel.Off = 8
	rel.Siz = 4
	rel.Sym = ctxt.Syms.Lookup("runtime.addmoduledata", 0)
	rel.Type = objabi.R_CALLARM64 // Really should be R_AARCH64_JUMP26 but doesn't seem to make any difference

	if ctxt.BuildMode == ld.BuildModePlugin {
		ctxt.Textp = append(ctxt.Textp, addmoduledata)
	}
	ctxt.Textp = append(ctxt.Textp, initfunc)
	initarray_entry := ctxt.Syms.Lookup("go.link.addmoduledatainit", 0)
	initarray_entry.Attr |= sym.AttrReachable
	initarray_entry.Attr |= sym.AttrLocal
	initarray_entry.Type = sym.SINITARR
	initarray_entry.AddAddr(ctxt.Arch, initfunc)
}

func adddynrel(ctxt *ld.Link, s *sym.Symbol, r *sym.Reloc) bool {
	targ := r.Sym

	switch r.Type {
	default:
		if r.Type >= objabi.ElfRelocOffset {
			ld.Errorf(s, "unexpected relocation type %d (%s)", r.Type, sym.RelocName(ctxt.Arch, r.Type))
			return false
		}

	// Handle relocations found in ELF object files.
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_PREL32):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_AARCH64_PREL32 relocation for dynamic symbol %s", targ.Name)
		}
		// TODO(mwhudson): the test of VisibilityHidden here probably doesn't make
		// sense and should be removed when someone has thought about it properly.
		if (targ.Type == 0 || targ.Type == sym.SXREF) && !targ.Attr.VisibilityHidden() {
			ld.Errorf(s, "unknown symbol %s in pcrel", targ.Name)
		}
		r.Type = objabi.R_PCREL
		r.Add += 4
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_PREL64):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_AARCH64_PREL64 relocation for dynamic symbol %s", targ.Name)
		}
		if targ.Type == 0 || targ.Type == sym.SXREF {
			ld.Errorf(s, "unknown symbol %s in pcrel", targ.Name)
		}
		r.Type = objabi.R_PCREL
		r.Add += 8
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_CALL26),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_JUMP26):
		if targ.Type == sym.SDYNIMPORT {
			addpltsym(ctxt, targ)
			r.Sym = ctxt.Syms.Lookup(".plt", 0)
			r.Add += int64(targ.Plt())
		}
		if (targ.Type == 0 || targ.Type == sym.SXREF) && !targ.Attr.VisibilityHidden() {
			ld.Errorf(s, "unknown symbol %s in callarm64", targ.Name)
		}
		r.Type = objabi.R_CALLARM64
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_ADR_GOT_PAGE),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_LD64_GOT_LO12_NC):
		if targ.Type != sym.SDYNIMPORT {
			// have symbol
			// TODO: turn LDR of GOT entry into ADR of symbol itself
		}

		// fall back to using GOT
		// TODO: just needs relocation, no need to put in .dynsym
		addgotsym(ctxt, targ)

		r.Type = objabi.R_ARM64_GOT
		r.Sym = ctxt.Syms.Lookup(".got", 0)
		r.Add += int64(targ.Got())
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_ADR_PREL_PG_HI21),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_ADD_ABS_LO12_NC):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected relocation for dynamic symbol %s", targ.Name)
		}
		if targ.Type == 0 || targ.Type == sym.SXREF {
			ld.Errorf(s, "unknown symbol %s", targ.Name)
		}
		r.Type = objabi.R_ARM64_PCREL
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_ABS64):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_AARCH64_ABS64 relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = objabi.R_ADDR
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_LDST8_ABS_LO12_NC):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = objabi.R_ARM64_LDST8
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_LDST32_ABS_LO12_NC):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = objabi.R_ARM64_LDST32
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_LDST64_ABS_LO12_NC):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = objabi.R_ARM64_LDST64
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_LDST128_ABS_LO12_NC):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = objabi.R_ARM64_LDST128
		return true
	}

	switch r.Type {
	case objabi.R_CALL,
		objabi.R_PCREL,
		objabi.R_CALLARM64:
		if targ.Type != sym.SDYNIMPORT {
			// nothing to do, the relocation will be laid out in reloc
			return true
		}
		if ctxt.LinkMode == ld.LinkExternal {
			// External linker will do this relocation.
			return true
		}

	case objabi.R_ADDR:
		if s.Type == sym.STEXT && ctxt.IsELF {
			// The code is asking for the address of an external
			// function. We provide it with the address of the
			// correspondent GOT symbol.
			addgotsym(ctxt, targ)

			r.Sym = ctxt.Syms.Lookup(".got", 0)
			r.Add += int64(targ.Got())
			return true
		}

		// Process dynamic relocations for the data sections.
		if ctxt.BuildMode == ld.BuildModePIE && ctxt.LinkMode == ld.LinkInternal {
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
			if s.Type != sym.SDATA && s.Type != sym.SRODATA {
				break
			}
		}

		if ctxt.IsELF {
			// TODO: We generate a R_AARCH64_ABS64 relocation for every R_ADDR, even
			// though it would be more efficient (for the dynamic linker) if we
			// generated R_AARCH64_RELATIVE instead.
			ld.Adddynsym(ctxt, targ)
			rela := ctxt.Syms.Lookup(".rela", 0)
			rela.AddAddrPlus(ctxt.Arch, s, int64(r.Off))
			if r.Siz == 8 {
				rela.AddUint64(ctxt.Arch, ld.ELF64_R_INFO(uint32(targ.Dynid), uint32(elf.R_AARCH64_ABS64)))
			} else {
				ld.Errorf(s, "unexpected relocation for dynamic symbol %s", targ.Name)
			}
			rela.AddUint64(ctxt.Arch, uint64(r.Add))
			r.Type = objabi.ElfRelocOffset // ignore during relocsym
			return true
		}
	}
	return false
}

func elfreloc1(ctxt *ld.Link, r *sym.Reloc, sectoff int64) bool {
	ctxt.Out.Write64(uint64(sectoff))

	elfsym := r.Xsym.ElfsymForReloc()
	switch r.Type {
	default:
		return false
	case objabi.R_ADDR:
		switch r.Siz {
		case 4:
			ctxt.Out.Write64(uint64(elf.R_AARCH64_ABS32) | uint64(elfsym)<<32)
		case 8:
			ctxt.Out.Write64(uint64(elf.R_AARCH64_ABS64) | uint64(elfsym)<<32)
		default:
			return false
		}
	case objabi.R_ADDRARM64:
		// two relocations: R_AARCH64_ADR_PREL_PG_HI21 and R_AARCH64_ADD_ABS_LO12_NC
		ctxt.Out.Write64(uint64(elf.R_AARCH64_ADR_PREL_PG_HI21) | uint64(elfsym)<<32)
		ctxt.Out.Write64(uint64(r.Xadd))
		ctxt.Out.Write64(uint64(sectoff + 4))
		ctxt.Out.Write64(uint64(elf.R_AARCH64_ADD_ABS_LO12_NC) | uint64(elfsym)<<32)
	case objabi.R_ARM64_TLS_LE:
		ctxt.Out.Write64(uint64(elf.R_AARCH64_TLSLE_MOVW_TPREL_G0) | uint64(elfsym)<<32)
	case objabi.R_ARM64_TLS_IE:
		ctxt.Out.Write64(uint64(elf.R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21) | uint64(elfsym)<<32)
		ctxt.Out.Write64(uint64(r.Xadd))
		ctxt.Out.Write64(uint64(sectoff + 4))
		ctxt.Out.Write64(uint64(elf.R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC) | uint64(elfsym)<<32)
	case objabi.R_ARM64_GOTPCREL:
		ctxt.Out.Write64(uint64(elf.R_AARCH64_ADR_GOT_PAGE) | uint64(elfsym)<<32)
		ctxt.Out.Write64(uint64(r.Xadd))
		ctxt.Out.Write64(uint64(sectoff + 4))
		ctxt.Out.Write64(uint64(elf.R_AARCH64_LD64_GOT_LO12_NC) | uint64(elfsym)<<32)
	case objabi.R_CALLARM64:
		if r.Siz != 4 {
			return false
		}
		ctxt.Out.Write64(uint64(elf.R_AARCH64_CALL26) | uint64(elfsym)<<32)

	}
	ctxt.Out.Write64(uint64(r.Xadd))

	return true
}

func machoreloc1(arch *sys.Arch, out *ld.OutBuf, s *sym.Symbol, r *sym.Reloc, sectoff int64) bool {
	var v uint32

	rs := r.Xsym

	if rs.Type == sym.SHOSTOBJ || r.Type == objabi.R_CALLARM64 || r.Type == objabi.R_ADDRARM64 {
		if rs.Dynid < 0 {
			ld.Errorf(s, "reloc %d (%s) to non-macho symbol %s type=%d (%s)", r.Type, sym.RelocName(arch, r.Type), rs.Name, rs.Type, rs.Type)
			return false
		}

		v = uint32(rs.Dynid)
		v |= 1 << 27 // external relocation
	} else {
		v = uint32(rs.Sect.Extnum)
		if v == 0 {
			ld.Errorf(s, "reloc %d (%s) to symbol %s in non-macho section %s type=%d (%s)", r.Type, sym.RelocName(arch, r.Type), rs.Name, rs.Sect.Name, rs.Type, rs.Type)
			return false
		}
	}

	switch r.Type {
	default:
		return false
	case objabi.R_ADDR:
		v |= ld.MACHO_ARM64_RELOC_UNSIGNED << 28
	case objabi.R_CALLARM64:
		if r.Xadd != 0 {
			ld.Errorf(s, "ld64 doesn't allow BR26 reloc with non-zero addend: %s+%d", rs.Name, r.Xadd)
		}

		v |= 1 << 24 // pc-relative bit
		v |= ld.MACHO_ARM64_RELOC_BRANCH26 << 28
	case objabi.R_ADDRARM64:
		r.Siz = 4
		// Two relocation entries: MACHO_ARM64_RELOC_PAGEOFF12 MACHO_ARM64_RELOC_PAGE21
		// if r.Xadd is non-zero, add two MACHO_ARM64_RELOC_ADDEND.
		if r.Xadd != 0 {
			out.Write32(uint32(sectoff + 4))
			out.Write32((ld.MACHO_ARM64_RELOC_ADDEND << 28) | (2 << 25) | uint32(r.Xadd&0xffffff))
		}
		out.Write32(uint32(sectoff + 4))
		out.Write32(v | (ld.MACHO_ARM64_RELOC_PAGEOFF12 << 28) | (2 << 25))
		if r.Xadd != 0 {
			out.Write32(uint32(sectoff))
			out.Write32((ld.MACHO_ARM64_RELOC_ADDEND << 28) | (2 << 25) | uint32(r.Xadd&0xffffff))
		}
		v |= 1 << 24 // pc-relative bit
		v |= ld.MACHO_ARM64_RELOC_PAGE21 << 28
	}

	switch r.Siz {
	default:
		return false
	case 1:
		v |= 0 << 25
	case 2:
		v |= 1 << 25
	case 4:
		v |= 2 << 25
	case 8:
		v |= 3 << 25
	}

	out.Write32(uint32(sectoff))
	out.Write32(v)
	return true
}

func archreloc(ctxt *ld.Link, r *sym.Reloc, s *sym.Symbol, val int64) (int64, bool) {
	if ctxt.LinkMode == ld.LinkExternal {
		switch r.Type {
		default:
			return val, false
		case objabi.R_ARM64_GOTPCREL:
			var o1, o2 uint32
			if ctxt.Arch.ByteOrder == binary.BigEndian {
				o1 = uint32(val >> 32)
				o2 = uint32(val)
			} else {
				o1 = uint32(val)
				o2 = uint32(val >> 32)
			}
			// Any relocation against a function symbol is redirected to
			// be against a local symbol instead (see putelfsym in
			// symtab.go) but unfortunately the system linker was buggy
			// when confronted with a R_AARCH64_ADR_GOT_PAGE relocation
			// against a local symbol until May 2015
			// (https://sourceware.org/bugzilla/show_bug.cgi?id=18270). So
			// we convert the adrp; ld64 + R_ARM64_GOTPCREL into adrp;
			// add + R_ADDRARM64.
			if !(r.Sym.IsFileLocal() || r.Sym.Attr.VisibilityHidden() || r.Sym.Attr.Local()) && r.Sym.Type == sym.STEXT && ctxt.DynlinkingGo() {
				if o2&0xffc00000 != 0xf9400000 {
					ld.Errorf(s, "R_ARM64_GOTPCREL against unexpected instruction %x", o2)
				}
				o2 = 0x91000000 | (o2 & 0x000003ff)
				r.Type = objabi.R_ADDRARM64
			}
			if ctxt.Arch.ByteOrder == binary.BigEndian {
				val = int64(o1)<<32 | int64(o2)
			} else {
				val = int64(o2)<<32 | int64(o1)
			}
			fallthrough
		case objabi.R_ADDRARM64:
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

			// Note: ld64 currently has a bug that any non-zero addend for BR26 relocation
			// will make the linking fail because it thinks the code is not PIC even though
			// the BR26 relocation should be fully resolved at link time.
			// That is the reason why the next if block is disabled. When the bug in ld64
			// is fixed, we can enable this block and also enable duff's device in cmd/7g.
			if false && ctxt.HeadType == objabi.Hdarwin {
				var o0, o1 uint32

				if ctxt.Arch.ByteOrder == binary.BigEndian {
					o0 = uint32(val >> 32)
					o1 = uint32(val)
				} else {
					o0 = uint32(val)
					o1 = uint32(val >> 32)
				}
				// Mach-O wants the addend to be encoded in the instruction
				// Note that although Mach-O supports ARM64_RELOC_ADDEND, it
				// can only encode 24-bit of signed addend, but the instructions
				// supports 33-bit of signed addend, so we always encode the
				// addend in place.
				o0 |= (uint32((r.Xadd>>12)&3) << 29) | (uint32((r.Xadd>>12>>2)&0x7ffff) << 5)
				o1 |= uint32(r.Xadd&0xfff) << 10
				r.Xadd = 0

				// when laid out, the instruction order must always be o1, o2.
				if ctxt.Arch.ByteOrder == binary.BigEndian {
					val = int64(o0)<<32 | int64(o1)
				} else {
					val = int64(o1)<<32 | int64(o0)
				}
			}

			return val, true
		case objabi.R_CALLARM64,
			objabi.R_ARM64_TLS_LE,
			objabi.R_ARM64_TLS_IE:
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

	case objabi.R_ADDRARM64:
		t := ld.Symaddr(r.Sym) + r.Add - ((s.Value + int64(r.Off)) &^ 0xfff)
		if t >= 1<<32 || t < -1<<32 {
			ld.Errorf(s, "program too large, address relocation distance = %d", t)
		}

		var o0, o1 uint32

		if ctxt.Arch.ByteOrder == binary.BigEndian {
			o0 = uint32(val >> 32)
			o1 = uint32(val)
		} else {
			o0 = uint32(val)
			o1 = uint32(val >> 32)
		}

		o0 |= (uint32((t>>12)&3) << 29) | (uint32((t>>12>>2)&0x7ffff) << 5)
		o1 |= uint32(t&0xfff) << 10

		// when laid out, the instruction order must always be o1, o2.
		if ctxt.Arch.ByteOrder == binary.BigEndian {
			return int64(o0)<<32 | int64(o1), true
		}
		return int64(o1)<<32 | int64(o0), true

	case objabi.R_ARM64_TLS_LE:
		r.Done = false
		if ctxt.HeadType == objabi.Hdarwin {
			ld.Errorf(s, "TLS reloc on unsupported OS %v", ctxt.HeadType)
		}
		// The TCB is two pointers. This is not documented anywhere, but is
		// de facto part of the ABI.
		v := r.Sym.Value + int64(2*ctxt.Arch.PtrSize)
		if v < 0 || v >= 32678 {
			ld.Errorf(s, "TLS offset out of range %d", v)
		}
		return val | (v << 5), true

	case objabi.R_ARM64_TLS_IE:
		if ctxt.BuildMode == ld.BuildModePIE && ctxt.IsELF {
			// We are linking the final executable, so we
			// can optimize any TLS IE relocation to LE.
			r.Done = false
			if ctxt.HeadType != objabi.Hlinux {
				ld.Errorf(s, "TLS reloc on unsupported OS %v", ctxt.HeadType)
			}

			// The TCB is two pointers. This is not documented anywhere, but is
			// de facto part of the ABI.
			v := ld.Symaddr(r.Sym) + int64(2*ctxt.Arch.PtrSize) + r.Add
			if v < 0 || v >= 32678 {
				ld.Errorf(s, "TLS offset out of range %d", v)
			}

			var o0, o1 uint32
			if ctxt.Arch.ByteOrder == binary.BigEndian {
				o0 = uint32(val >> 32)
				o1 = uint32(val)
			} else {
				o0 = uint32(val)
				o1 = uint32(val >> 32)
			}

			// R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21
			// turn ADRP to MOVZ
			o0 = 0xd2a00000 | uint32(o0&0x1f) | (uint32((v>>16)&0xffff) << 5)
			// R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC
			// turn LD64 to MOVK
			if v&3 != 0 {
				ld.Errorf(s, "invalid address: %x for relocation type: R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC", v)
			}
			o1 = 0xf2800000 | uint32(o1&0x1f) | (uint32(v&0xffff) << 5)

			// when laid out, the instruction order must always be o0, o1.
			if ctxt.Arch.ByteOrder == binary.BigEndian {
				return int64(o0)<<32 | int64(o1), true
			}
			return int64(o1)<<32 | int64(o0), true
		} else {
			log.Fatalf("cannot handle R_ARM64_TLS_IE (sym %s) when linking internally", s.Name)
		}

	case objabi.R_CALLARM64:
		var t int64
		if r.Sym.Type == sym.SDYNIMPORT {
			t = (ld.Symaddr(ctxt.Syms.Lookup(".plt", 0)) + r.Add) - (s.Value + int64(r.Off))
		} else {
			t = (ld.Symaddr(r.Sym) + r.Add) - (s.Value + int64(r.Off))
		}
		if t >= 1<<27 || t < -1<<27 {
			ld.Errorf(s, "program too large, call relocation distance = %d", t)
		}
		return val | ((t >> 2) & 0x03ffffff), true

	case objabi.R_ARM64_GOT:
		if s.P[r.Off+3]&0x9f == 0x90 {
			// R_AARCH64_ADR_GOT_PAGE
			// patch instruction: adrp
			t := ld.Symaddr(r.Sym) + r.Add - ((s.Value + int64(r.Off)) &^ 0xfff)
			if t >= 1<<32 || t < -1<<32 {
				ld.Errorf(s, "program too large, address relocation distance = %d", t)
			}
			var o0 uint32
			o0 |= (uint32((t>>12)&3) << 29) | (uint32((t>>12>>2)&0x7ffff) << 5)
			return val | int64(o0), true
		} else if s.P[r.Off+3] == 0xf9 {
			// R_AARCH64_LD64_GOT_LO12_NC
			// patch instruction: ldr
			t := ld.Symaddr(r.Sym) + r.Add - ((s.Value + int64(r.Off)) &^ 0xfff)
			if t&7 != 0 {
				ld.Errorf(s, "invalid address: %x for relocation type: R_AARCH64_LD64_GOT_LO12_NC", t)
			}
			var o1 uint32
			o1 |= uint32(t&0xfff) << (10 - 3)
			return val | int64(uint64(o1)), true
		} else {
			ld.Errorf(s, "unsupported instruction for %v R_GOTARM64", s.P[r.Off:r.Off+4])
		}

	case objabi.R_ARM64_PCREL:
		if s.P[r.Off+3]&0x9f == 0x90 {
			// R_AARCH64_ADR_PREL_PG_HI21
			// patch instruction: adrp
			t := ld.Symaddr(r.Sym) + r.Add - ((s.Value + int64(r.Off)) &^ 0xfff)
			if t >= 1<<32 || t < -1<<32 {
				ld.Errorf(s, "program too large, address relocation distance = %d", t)
			}
			o0 := (uint32((t>>12)&3) << 29) | (uint32((t>>12>>2)&0x7ffff) << 5)
			return val | int64(o0), true
		} else if s.P[r.Off+3]&0x91 == 0x91 {
			// R_AARCH64_ADD_ABS_LO12_NC
			// patch instruction: add
			t := ld.Symaddr(r.Sym) + r.Add - ((s.Value + int64(r.Off)) &^ 0xfff)
			o1 := uint32(t&0xfff) << 10
			return val | int64(o1), true
		} else {
			ld.Errorf(s, "unsupported instruction for %v R_PCRELARM64", s.P[r.Off:r.Off+4])
		}

	case objabi.R_ARM64_LDST8:
		t := ld.Symaddr(r.Sym) + r.Add - ((s.Value + int64(r.Off)) &^ 0xfff)
		o0 := uint32(t&0xfff) << 10
		return val | int64(o0), true

	case objabi.R_ARM64_LDST32:
		t := ld.Symaddr(r.Sym) + r.Add - ((s.Value + int64(r.Off)) &^ 0xfff)
		if t&3 != 0 {
			ld.Errorf(s, "invalid address: %x for relocation type: R_AARCH64_LDST32_ABS_LO12_NC", t)
		}
		o0 := (uint32(t&0xfff) >> 2) << 10
		return val | int64(o0), true

	case objabi.R_ARM64_LDST64:
		t := ld.Symaddr(r.Sym) + r.Add - ((s.Value + int64(r.Off)) &^ 0xfff)
		if t&7 != 0 {
			ld.Errorf(s, "invalid address: %x for relocation type: R_AARCH64_LDST64_ABS_LO12_NC", t)
		}
		o0 := (uint32(t&0xfff) >> 3) << 10
		return val | int64(o0), true

	case objabi.R_ARM64_LDST128:
		t := ld.Symaddr(r.Sym) + r.Add - ((s.Value + int64(r.Off)) &^ 0xfff)
		if t&15 != 0 {
			ld.Errorf(s, "invalid address: %x for relocation type: R_AARCH64_LDST128_ABS_LO12_NC", t)
		}
		o0 := (uint32(t&0xfff) >> 4) << 10
		return val | int64(o0), true
	}

	return val, false
}

func archrelocvariant(ctxt *ld.Link, r *sym.Reloc, s *sym.Symbol, t int64) int64 {
	log.Fatalf("unexpected relocation variant")
	return -1
}

func elfsetupplt(ctxt *ld.Link) {
	plt := ctxt.Syms.Lookup(".plt", 0)
	gotplt := ctxt.Syms.Lookup(".got.plt", 0)
	if plt.Size == 0 {
		// stp     x16, x30, [sp, #-16]!
		// identifying information
		plt.AddUint32(ctxt.Arch, 0xa9bf7bf0)

		// the following two instructions (adrp + ldr) load *got[2] into x17
		// adrp    x16, &got[0]
		plt.AddAddrPlus4(gotplt, 16)
		plt.SetUint32(ctxt.Arch, plt.Size-4, 0x90000010)
		plt.R[len(plt.R)-1].Type = objabi.R_ARM64_GOT

		// <imm> is the offset value of &got[2] to &got[0], the same below
		// ldr     x17, [x16, <imm>]
		plt.AddAddrPlus4(gotplt, 16)
		plt.SetUint32(ctxt.Arch, plt.Size-4, 0xf9400211)
		plt.R[len(plt.R)-1].Type = objabi.R_ARM64_GOT

		// add     x16, x16, <imm>
		plt.AddAddrPlus4(gotplt, 16)
		plt.SetUint32(ctxt.Arch, plt.Size-4, 0x91000210)
		plt.R[len(plt.R)-1].Type = objabi.R_ARM64_PCREL

		// br      x17
		plt.AddUint32(ctxt.Arch, 0xd61f0220)

		// 3 nop for place holder
		plt.AddUint32(ctxt.Arch, 0xd503201f)
		plt.AddUint32(ctxt.Arch, 0xd503201f)
		plt.AddUint32(ctxt.Arch, 0xd503201f)

		// check gotplt.size == 0
		if gotplt.Size != 0 {
			ld.Errorf(gotplt, "got.plt is not empty at the very beginning")
		}
		gotplt.AddAddrPlus(ctxt.Arch, ctxt.Syms.Lookup(".dynamic", 0), 0)

		gotplt.AddUint64(ctxt.Arch, 0)
		gotplt.AddUint64(ctxt.Arch, 0)
	}
}

func addpltsym(ctxt *ld.Link, s *sym.Symbol) {
	if s.Plt() >= 0 {
		return
	}

	ld.Adddynsym(ctxt, s)

	if ctxt.IsELF {
		plt := ctxt.Syms.Lookup(".plt", 0)
		gotplt := ctxt.Syms.Lookup(".got.plt", 0)
		rela := ctxt.Syms.Lookup(".rela.plt", 0)
		if plt.Size == 0 {
			elfsetupplt(ctxt)
		}

		// adrp    x16, &got.plt[0]
		plt.AddAddrPlus4(gotplt, gotplt.Size)
		plt.SetUint32(ctxt.Arch, plt.Size-4, 0x90000010)
		plt.R[len(plt.R)-1].Type = objabi.R_ARM64_GOT

		// <offset> is the offset value of &got.plt[n] to &got.plt[0]
		// ldr     x17, [x16, <offset>]
		plt.AddAddrPlus4(gotplt, gotplt.Size)
		plt.SetUint32(ctxt.Arch, plt.Size-4, 0xf9400211)
		plt.R[len(plt.R)-1].Type = objabi.R_ARM64_GOT

		// add     x16, x16, <offset>
		plt.AddAddrPlus4(gotplt, gotplt.Size)
		plt.SetUint32(ctxt.Arch, plt.Size-4, 0x91000210)
		plt.R[len(plt.R)-1].Type = objabi.R_ARM64_PCREL

		// br      x17
		plt.AddUint32(ctxt.Arch, 0xd61f0220)

		// add to got.plt: pointer to plt[0]
		gotplt.AddAddrPlus(ctxt.Arch, plt, 0)

		// rela
		rela.AddAddrPlus(ctxt.Arch, gotplt, gotplt.Size-8)
		rela.AddUint64(ctxt.Arch, ld.ELF64_R_INFO(uint32(s.Dynid), uint32(elf.R_AARCH64_JUMP_SLOT)))
		rela.AddUint64(ctxt.Arch, 0)

		s.SetPlt(int32(plt.Size - 16))
	} else {
		ld.Errorf(s, "addpltsym: unsupported binary format")
	}
}

func addgotsym(ctxt *ld.Link, s *sym.Symbol) {
	if s.Got() >= 0 {
		return
	}

	ld.Adddynsym(ctxt, s)
	got := ctxt.Syms.Lookup(".got", 0)
	s.SetGot(int32(got.Size))
	got.AddUint64(ctxt.Arch, 0)

	if ctxt.IsELF {
		rela := ctxt.Syms.Lookup(".rela", 0)
		rela.AddAddrPlus(ctxt.Arch, got, int64(s.Got()))
		rela.AddUint64(ctxt.Arch, ld.ELF64_R_INFO(uint32(s.Dynid), uint32(elf.R_AARCH64_GLOB_DAT)))
		rela.AddUint64(ctxt.Arch, 0)
	} else {
		ld.Errorf(s, "addgotsym: unsupported binary format")
	}
}

func asmb(ctxt *ld.Link) {
	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f asmb\n", ld.Cputime())
	}

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
		if ctxt.Debugvlog != 0 {
			ctxt.Logf("%5.2f rodatblk\n", ld.Cputime())
		}
		ctxt.Out.SeekSet(int64(ld.Segrodata.Fileoff))
		ld.Datblk(ctxt, int64(ld.Segrodata.Vaddr), int64(ld.Segrodata.Filelen))
	}
	if ld.Segrelrodata.Filelen > 0 {
		if ctxt.Debugvlog != 0 {
			ctxt.Logf("%5.2f relrodatblk\n", ld.Cputime())
		}
		ctxt.Out.SeekSet(int64(ld.Segrelrodata.Fileoff))
		ld.Datblk(ctxt, int64(ld.Segrelrodata.Vaddr), int64(ld.Segrelrodata.Filelen))
	}

	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f datblk\n", ld.Cputime())
	}

	ctxt.Out.SeekSet(int64(ld.Segdata.Fileoff))
	ld.Datblk(ctxt, int64(ld.Segdata.Vaddr), int64(ld.Segdata.Filelen))

	ctxt.Out.SeekSet(int64(ld.Segdwarf.Fileoff))
	ld.Dwarfblk(ctxt, int64(ld.Segdwarf.Vaddr), int64(ld.Segdwarf.Filelen))
}

func asmb2(ctxt *ld.Link) {
	machlink := uint32(0)
	if ctxt.HeadType == objabi.Hdarwin {
		machlink = uint32(ld.Domacholink(ctxt))
	}

	/* output symbol table */
	ld.Symsize = 0

	ld.Lcsize = 0
	symo := uint32(0)
	if !*ld.FlagS {
		// TODO: rationalize
		if ctxt.Debugvlog != 0 {
			ctxt.Logf("%5.2f sym\n", ld.Cputime())
		}
		switch ctxt.HeadType {
		default:
			if ctxt.IsELF {
				symo = uint32(ld.Segdwarf.Fileoff + ld.Segdwarf.Filelen)
				symo = uint32(ld.Rnd(int64(symo), int64(*ld.FlagRound)))
			}

		case objabi.Hplan9:
			symo = uint32(ld.Segdata.Fileoff + ld.Segdata.Filelen)

		case objabi.Hdarwin:
			symo = uint32(ld.Segdwarf.Fileoff + uint64(ld.Rnd(int64(ld.Segdwarf.Filelen), int64(*ld.FlagRound))) + uint64(machlink))
		}

		ctxt.Out.SeekSet(int64(symo))
		switch ctxt.HeadType {
		default:
			if ctxt.IsELF {
				if ctxt.Debugvlog != 0 {
					ctxt.Logf("%5.2f elfsym\n", ld.Cputime())
				}
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

		case objabi.Hdarwin:
			if ctxt.LinkMode == ld.LinkExternal {
				ld.Machoemitreloc(ctxt)
			}
		}
	}

	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f header\n", ld.Cputime())
	}
	ctxt.Out.SeekSet(0)
	switch ctxt.HeadType {
	default:
	case objabi.Hplan9: /* plan 9 */
		ctxt.Out.Write32(0x647)                      /* magic */
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
		objabi.Hopenbsd,
		objabi.Hnacl:
		ld.Asmbelf(ctxt, int64(symo))

	case objabi.Hdarwin:
		ld.Asmbmacho(ctxt)
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
