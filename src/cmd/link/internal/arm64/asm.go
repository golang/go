// Inferno utils/5l/asm.c
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/5l/asm.c
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
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/elf"
	"fmt"
	"log"
)

func gentext(ctxt *ld.Link, ldr *loader.Loader) {
	initfunc, addmoduledata := ld.PrepareAddmoduledata(ctxt)
	if initfunc == nil {
		return
	}

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
	rel, _ := initfunc.AddRel(objabi.R_ADDRARM64)
	rel.SetOff(0)
	rel.SetSiz(8)
	rel.SetSym(ctxt.Moduledata)

	// 8:	14000000 	b	0 <runtime.addmoduledata>
	// 	8: R_AARCH64_CALL26	runtime.addmoduledata
	o(0x14000000)
	rel2, _ := initfunc.AddRel(objabi.R_CALLARM64)
	rel2.SetOff(8)
	rel2.SetSiz(4)
	rel2.SetSym(addmoduledata)
}

func adddynrel(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s loader.Sym, r loader.Reloc, rIdx int) bool {
	targ := r.Sym()
	var targType sym.SymKind
	if targ != 0 {
		targType = ldr.SymType(targ)
	}

	const pcrel = 1
	switch r.Type() {
	default:
		if r.Type() >= objabi.ElfRelocOffset {
			ldr.Errorf(s, "unexpected relocation type %d (%s)", r.Type(), sym.RelocName(target.Arch, r.Type()))
			return false
		}

	// Handle relocations found in ELF object files.
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_PREL32):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected R_AARCH64_PREL32 relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		// TODO(mwhudson): the test of VisibilityHidden here probably doesn't make
		// sense and should be removed when someone has thought about it properly.
		if (targType == 0 || targType == sym.SXREF) && !ldr.AttrVisibilityHidden(targ) {
			ldr.Errorf(s, "unknown symbol %s in pcrel", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		su.SetRelocAdd(rIdx, r.Add()+4)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_PREL64):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected R_AARCH64_PREL64 relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		if targType == 0 || targType == sym.SXREF {
			ldr.Errorf(s, "unknown symbol %s in pcrel", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		su.SetRelocAdd(rIdx, r.Add()+8)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_CALL26),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_JUMP26):
		if targType == sym.SDYNIMPORT {
			addpltsym(target, ldr, syms, targ)
			su := ldr.MakeSymbolUpdater(s)
			su.SetRelocSym(rIdx, syms.PLT)
			su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymPlt(targ)))
		}
		if (targType == 0 || targType == sym.SXREF) && !ldr.AttrVisibilityHidden(targ) {
			ldr.Errorf(s, "unknown symbol %s in callarm64", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_CALLARM64)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_ADR_GOT_PAGE),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_LD64_GOT_LO12_NC):
		if targType != sym.SDYNIMPORT {
			// have symbol
			// TODO: turn LDR of GOT entry into ADR of symbol itself
		}

		// fall back to using GOT
		// TODO: just needs relocation, no need to put in .dynsym
		ld.AddGotSym(target, ldr, syms, targ, uint32(elf.R_AARCH64_GLOB_DAT))
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ARM64_GOT)
		su.SetRelocSym(rIdx, syms.GOT)
		su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymGot(targ)))
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_ADR_PREL_PG_HI21),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_ADD_ABS_LO12_NC):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		if targType == 0 || targType == sym.SXREF {
			ldr.Errorf(s, "unknown symbol %s", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ARM64_PCREL)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_ABS64):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected R_AARCH64_ABS64 relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ADDR)
		if target.IsPIE() && target.IsInternal() {
			// For internal linking PIE, this R_ADDR relocation cannot
			// be resolved statically. We need to generate a dynamic
			// relocation. Let the code below handle it.
			break
		}
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_LDST8_ABS_LO12_NC):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ARM64_LDST8)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_LDST16_ABS_LO12_NC):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ARM64_LDST16)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_LDST32_ABS_LO12_NC):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ARM64_LDST32)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_LDST64_ABS_LO12_NC):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ARM64_LDST64)

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_LDST128_ABS_LO12_NC):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ARM64_LDST128)
		return true

	// Handle relocations found in Mach-O object files.
	case objabi.MachoRelocOffset + ld.MACHO_ARM64_RELOC_UNSIGNED*2:
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected reloc for dynamic symbol %s", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ADDR)
		if target.IsPIE() && target.IsInternal() {
			// For internal linking PIE, this R_ADDR relocation cannot
			// be resolved statically. We need to generate a dynamic
			// relocation. Let the code below handle it.
			break
		}
		return true

	case objabi.MachoRelocOffset + ld.MACHO_ARM64_RELOC_BRANCH26*2 + pcrel:
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_CALLARM64)
		if targType == sym.SDYNIMPORT {
			addpltsym(target, ldr, syms, targ)
			su.SetRelocSym(rIdx, syms.PLT)
			su.SetRelocAdd(rIdx, int64(ldr.SymPlt(targ)))
		}
		return true

	case objabi.MachoRelocOffset + ld.MACHO_ARM64_RELOC_PAGE21*2 + pcrel,
		objabi.MachoRelocOffset + ld.MACHO_ARM64_RELOC_PAGEOFF12*2:
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ARM64_PCREL)
		return true

	case objabi.MachoRelocOffset + ld.MACHO_ARM64_RELOC_GOT_LOAD_PAGE21*2 + pcrel,
		objabi.MachoRelocOffset + ld.MACHO_ARM64_RELOC_GOT_LOAD_PAGEOFF12*2:
		if targType != sym.SDYNIMPORT {
			// have symbol
			// turn MOVD sym@GOT (adrp+ldr) into MOVD $sym (adrp+add)
			data := ldr.Data(s)
			off := r.Off()
			if int(off+3) >= len(data) {
				ldr.Errorf(s, "unexpected GOT_LOAD reloc for non-dynamic symbol %s", ldr.SymName(targ))
				return false
			}
			o := target.Arch.ByteOrder.Uint32(data[off:])
			su := ldr.MakeSymbolUpdater(s)
			switch {
			case (o>>24)&0x9f == 0x90: // adrp
				// keep instruction unchanged, change relocation type below
			case o>>24 == 0xf9: // ldr
				// rewrite to add
				o = (0x91 << 24) | (o & (1<<22 - 1))
				su.MakeWritable()
				su.SetUint32(target.Arch, int64(off), o)
			default:
				ldr.Errorf(s, "unexpected GOT_LOAD reloc for non-dynamic symbol %s", ldr.SymName(targ))
				return false
			}
			su.SetRelocType(rIdx, objabi.R_ARM64_PCREL)
			return true
		}
		ld.AddGotSym(target, ldr, syms, targ, 0)
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ARM64_GOT)
		su.SetRelocSym(rIdx, syms.GOT)
		su.SetRelocAdd(rIdx, int64(ldr.SymGot(targ)))
		return true
	}

	// Reread the reloc to incorporate any changes in type above.
	relocs := ldr.Relocs(s)
	r = relocs.At(rIdx)

	switch r.Type() {
	case objabi.R_CALL,
		objabi.R_PCREL,
		objabi.R_CALLARM64:
		if targType != sym.SDYNIMPORT {
			// nothing to do, the relocation will be laid out in reloc
			return true
		}
		if target.IsExternal() {
			// External linker will do this relocation.
			return true
		}
		// Internal linking.
		if r.Add() != 0 {
			ldr.Errorf(s, "PLT call with non-zero addend (%v)", r.Add())
		}
		// Build a PLT entry and change the relocation target to that entry.
		addpltsym(target, ldr, syms, targ)
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocSym(rIdx, syms.PLT)
		su.SetRelocAdd(rIdx, int64(ldr.SymPlt(targ)))
		return true

	case objabi.R_ADDR:
		if ldr.SymType(s) == sym.STEXT && target.IsElf() {
			// The code is asking for the address of an external
			// function. We provide it with the address of the
			// correspondent GOT symbol.
			ld.AddGotSym(target, ldr, syms, targ, uint32(elf.R_AARCH64_GLOB_DAT))
			su := ldr.MakeSymbolUpdater(s)
			su.SetRelocSym(rIdx, syms.GOT)
			su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymGot(targ)))
			return true
		}

		// Process dynamic relocations for the data sections.
		if target.IsPIE() && target.IsInternal() {
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
			switch ldr.SymName(s) {
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
			if ldr.SymType(s) != sym.SDATA && ldr.SymType(s) != sym.SRODATA {
				break
			}
		}

		if target.IsElf() {
			// Generate R_AARCH64_RELATIVE relocations for best
			// efficiency in the dynamic linker.
			//
			// As noted above, symbol addresses have not been
			// assigned yet, so we can't generate the final reloc
			// entry yet. We ultimately want:
			//
			// r_offset = s + r.Off
			// r_info = R_AARCH64_RELATIVE
			// r_addend = targ + r.Add
			//
			// The dynamic linker will set *offset = base address +
			// addend.
			//
			// AddAddrPlus is used for r_offset and r_addend to
			// generate new R_ADDR relocations that will update
			// these fields in the 'reloc' phase.
			rela := ldr.MakeSymbolUpdater(syms.Rela)
			rela.AddAddrPlus(target.Arch, s, int64(r.Off()))
			if r.Siz() == 8 {
				rela.AddUint64(target.Arch, elf.R_INFO(0, uint32(elf.R_AARCH64_RELATIVE)))
			} else {
				ldr.Errorf(s, "unexpected relocation for dynamic symbol %s", ldr.SymName(targ))
			}
			rela.AddAddrPlus(target.Arch, targ, int64(r.Add()))
			// Not mark r done here. So we still apply it statically,
			// so in the file content we'll also have the right offset
			// to the relocation target. So it can be examined statically
			// (e.g. go version).
			return true
		}

		if target.IsDarwin() {
			// Mach-O relocations are a royal pain to lay out.
			// They use a compact stateful bytecode representation.
			// Here we record what are needed and encode them later.
			ld.MachoAddRebase(s, int64(r.Off()))
			// Not mark r done here. So we still apply it statically,
			// so in the file content we'll also have the right offset
			// to the relocation target. So it can be examined statically
			// (e.g. go version).
			return true
		}

	case objabi.R_ARM64_GOTPCREL:
		if target.IsExternal() {
			// External linker will do this relocation.
			return true
		}
		if targType != sym.SDYNIMPORT {
			ldr.Errorf(s, "R_ARM64_GOTPCREL target is not SDYNIMPORT symbol: %v", ldr.SymName(targ))
		}
		if r.Add() != 0 {
			ldr.Errorf(s, "R_ARM64_GOTPCREL with non-zero addend (%v)", r.Add())
		}
		if target.IsElf() {
			ld.AddGotSym(target, ldr, syms, targ, uint32(elf.R_AARCH64_GLOB_DAT))
		} else {
			ld.AddGotSym(target, ldr, syms, targ, 0)
		}
		// turn into two relocations, one for each instruction.
		su := ldr.MakeSymbolUpdater(s)
		r.SetType(objabi.R_ARM64_GOT)
		r.SetSiz(4)
		r.SetSym(syms.GOT)
		r.SetAdd(int64(ldr.SymGot(targ)))
		r2, _ := su.AddRel(objabi.R_ARM64_GOT)
		r2.SetSiz(4)
		r2.SetOff(r.Off() + 4)
		r2.SetSym(syms.GOT)
		r2.SetAdd(int64(ldr.SymGot(targ)))
		return true
	}
	return false
}

func elfreloc1(ctxt *ld.Link, out *ld.OutBuf, ldr *loader.Loader, s loader.Sym, r loader.ExtReloc, ri int, sectoff int64) bool {
	out.Write64(uint64(sectoff))

	elfsym := ld.ElfSymForReloc(ctxt, r.Xsym)
	siz := r.Size
	switch r.Type {
	default:
		return false
	case objabi.R_ADDR, objabi.R_DWARFSECREF:
		switch siz {
		case 4:
			out.Write64(uint64(elf.R_AARCH64_ABS32) | uint64(elfsym)<<32)
		case 8:
			out.Write64(uint64(elf.R_AARCH64_ABS64) | uint64(elfsym)<<32)
		default:
			return false
		}
	case objabi.R_ADDRARM64:
		// two relocations: R_AARCH64_ADR_PREL_PG_HI21 and R_AARCH64_ADD_ABS_LO12_NC
		out.Write64(uint64(elf.R_AARCH64_ADR_PREL_PG_HI21) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))
		out.Write64(uint64(sectoff + 4))
		out.Write64(uint64(elf.R_AARCH64_ADD_ABS_LO12_NC) | uint64(elfsym)<<32)
	case objabi.R_ARM64_TLS_LE:
		out.Write64(uint64(elf.R_AARCH64_TLSLE_MOVW_TPREL_G0) | uint64(elfsym)<<32)
	case objabi.R_ARM64_TLS_IE:
		out.Write64(uint64(elf.R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))
		out.Write64(uint64(sectoff + 4))
		out.Write64(uint64(elf.R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC) | uint64(elfsym)<<32)
	case objabi.R_ARM64_GOTPCREL:
		out.Write64(uint64(elf.R_AARCH64_ADR_GOT_PAGE) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))
		out.Write64(uint64(sectoff + 4))
		out.Write64(uint64(elf.R_AARCH64_LD64_GOT_LO12_NC) | uint64(elfsym)<<32)
	case objabi.R_CALLARM64:
		if siz != 4 {
			return false
		}
		out.Write64(uint64(elf.R_AARCH64_CALL26) | uint64(elfsym)<<32)

	}
	out.Write64(uint64(r.Xadd))

	return true
}

// sign-extends from 21, 24-bit.
func signext21(x int64) int64 { return x << (64 - 21) >> (64 - 21) }
func signext24(x int64) int64 { return x << (64 - 24) >> (64 - 24) }

func machoreloc1(arch *sys.Arch, out *ld.OutBuf, ldr *loader.Loader, s loader.Sym, r loader.ExtReloc, sectoff int64) bool {
	var v uint32

	rs := r.Xsym
	rt := r.Type
	siz := r.Size
	xadd := r.Xadd

	if xadd != signext24(xadd) {
		// If the relocation target would overflow the addend, then target
		// a linker-manufactured label symbol with a smaller addend instead.
		label := ldr.Lookup(offsetLabelName(ldr, rs, xadd/machoRelocLimit*machoRelocLimit), ldr.SymVersion(rs))
		if label != 0 {
			xadd = ldr.SymValue(rs) + xadd - ldr.SymValue(label)
			rs = label
		}
		if xadd != signext24(xadd) {
			ldr.Errorf(s, "internal error: relocation addend overflow: %s+0x%x", ldr.SymName(rs), xadd)
		}
	}

	if ldr.SymType(rs) == sym.SHOSTOBJ || rt == objabi.R_CALLARM64 || rt == objabi.R_ADDRARM64 || rt == objabi.R_ARM64_GOTPCREL {
		if ldr.SymDynid(rs) < 0 {
			ldr.Errorf(s, "reloc %d (%s) to non-macho symbol %s type=%d (%s)", rt, sym.RelocName(arch, rt), ldr.SymName(rs), ldr.SymType(rs), ldr.SymType(rs))
			return false
		}

		v = uint32(ldr.SymDynid(rs))
		v |= 1 << 27 // external relocation
	} else {
		v = uint32(ldr.SymSect(rs).Extnum)
		if v == 0 {
			ldr.Errorf(s, "reloc %d (%s) to symbol %s in non-macho section %s type=%d (%s)", rt, sym.RelocName(arch, rt), ldr.SymName(rs), ldr.SymSect(rs).Name, ldr.SymType(rs), ldr.SymType(rs))
			return false
		}
	}

	switch rt {
	default:
		return false
	case objabi.R_ADDR:
		v |= ld.MACHO_ARM64_RELOC_UNSIGNED << 28
	case objabi.R_CALLARM64:
		if xadd != 0 {
			ldr.Errorf(s, "ld64 doesn't allow BR26 reloc with non-zero addend: %s+%d", ldr.SymName(rs), xadd)
		}

		v |= 1 << 24 // pc-relative bit
		v |= ld.MACHO_ARM64_RELOC_BRANCH26 << 28
	case objabi.R_ADDRARM64:
		siz = 4
		// Two relocation entries: MACHO_ARM64_RELOC_PAGEOFF12 MACHO_ARM64_RELOC_PAGE21
		// if r.Xadd is non-zero, add two MACHO_ARM64_RELOC_ADDEND.
		if r.Xadd != 0 {
			out.Write32(uint32(sectoff + 4))
			out.Write32((ld.MACHO_ARM64_RELOC_ADDEND << 28) | (2 << 25) | uint32(xadd&0xffffff))
		}
		out.Write32(uint32(sectoff + 4))
		out.Write32(v | (ld.MACHO_ARM64_RELOC_PAGEOFF12 << 28) | (2 << 25))
		if r.Xadd != 0 {
			out.Write32(uint32(sectoff))
			out.Write32((ld.MACHO_ARM64_RELOC_ADDEND << 28) | (2 << 25) | uint32(xadd&0xffffff))
		}
		v |= 1 << 24 // pc-relative bit
		v |= ld.MACHO_ARM64_RELOC_PAGE21 << 28
	case objabi.R_ARM64_GOTPCREL:
		siz = 4
		// Two relocation entries: MACHO_ARM64_RELOC_GOT_LOAD_PAGEOFF12 MACHO_ARM64_RELOC_GOT_LOAD_PAGE21
		// if r.Xadd is non-zero, add two MACHO_ARM64_RELOC_ADDEND.
		if r.Xadd != 0 {
			out.Write32(uint32(sectoff + 4))
			out.Write32((ld.MACHO_ARM64_RELOC_ADDEND << 28) | (2 << 25) | uint32(xadd&0xffffff))
		}
		out.Write32(uint32(sectoff + 4))
		out.Write32(v | (ld.MACHO_ARM64_RELOC_GOT_LOAD_PAGEOFF12 << 28) | (2 << 25))
		if r.Xadd != 0 {
			out.Write32(uint32(sectoff))
			out.Write32((ld.MACHO_ARM64_RELOC_ADDEND << 28) | (2 << 25) | uint32(xadd&0xffffff))
		}
		v |= 1 << 24 // pc-relative bit
		v |= ld.MACHO_ARM64_RELOC_GOT_LOAD_PAGE21 << 28
	}

	switch siz {
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

func pereloc1(arch *sys.Arch, out *ld.OutBuf, ldr *loader.Loader, s loader.Sym, r loader.ExtReloc, sectoff int64) bool {
	rs := r.Xsym
	rt := r.Type

	if rt == objabi.R_ADDRARM64 && r.Xadd != signext21(r.Xadd) {
		// If the relocation target would overflow the addend, then target
		// a linker-manufactured label symbol with a smaller addend instead.
		label := ldr.Lookup(offsetLabelName(ldr, rs, r.Xadd/peRelocLimit*peRelocLimit), ldr.SymVersion(rs))
		if label == 0 {
			ldr.Errorf(s, "invalid relocation: %v %s+0x%x", rt, ldr.SymName(rs), r.Xadd)
			return false
		}
		rs = label
	}
	if rt == objabi.R_CALLARM64 && r.Xadd != 0 {
		label := ldr.Lookup(offsetLabelName(ldr, rs, r.Xadd), ldr.SymVersion(rs))
		if label == 0 {
			ldr.Errorf(s, "invalid relocation: %v %s+0x%x", rt, ldr.SymName(rs), r.Xadd)
			return false
		}
		rs = label
	}
	symdynid := ldr.SymDynid(rs)
	if symdynid < 0 {
		ldr.Errorf(s, "reloc %d (%s) to non-coff symbol %s type=%d (%s)", rt, sym.RelocName(arch, rt), ldr.SymName(rs), ldr.SymType(rs), ldr.SymType(rs))
		return false
	}

	switch rt {
	default:
		return false

	case objabi.R_DWARFSECREF:
		out.Write32(uint32(sectoff))
		out.Write32(uint32(symdynid))
		out.Write16(ld.IMAGE_REL_ARM64_SECREL)

	case objabi.R_ADDR:
		out.Write32(uint32(sectoff))
		out.Write32(uint32(symdynid))
		if r.Size == 8 {
			out.Write16(ld.IMAGE_REL_ARM64_ADDR64)
		} else {
			out.Write16(ld.IMAGE_REL_ARM64_ADDR32)
		}

	case objabi.R_ADDRARM64:
		// Note: r.Xadd has been taken care of below, in archreloc.
		out.Write32(uint32(sectoff))
		out.Write32(uint32(symdynid))
		out.Write16(ld.IMAGE_REL_ARM64_PAGEBASE_REL21)

		out.Write32(uint32(sectoff + 4))
		out.Write32(uint32(symdynid))
		out.Write16(ld.IMAGE_REL_ARM64_PAGEOFFSET_12A)

	case objabi.R_CALLARM64:
		// Note: r.Xadd has been taken care of above, by using a label pointing into the middle of the function.
		out.Write32(uint32(sectoff))
		out.Write32(uint32(symdynid))
		out.Write16(ld.IMAGE_REL_ARM64_BRANCH26)
	}

	return true
}

func archreloc(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, r loader.Reloc, s loader.Sym, val int64) (int64, int, bool) {
	const noExtReloc = 0
	const isOk = true

	rs := r.Sym()

	if target.IsExternal() {
		nExtReloc := 0
		switch rt := r.Type(); rt {
		default:
		case objabi.R_ARM64_GOTPCREL,
			objabi.R_ADDRARM64:

			// set up addend for eventual relocation via outer symbol.
			rs, off := ld.FoldSubSymbolOffset(ldr, rs)
			xadd := r.Add() + off
			rst := ldr.SymType(rs)
			if rst != sym.SHOSTOBJ && rst != sym.SDYNIMPORT && ldr.SymSect(rs) == nil {
				ldr.Errorf(s, "missing section for %s", ldr.SymName(rs))
			}

			nExtReloc = 2 // need two ELF/Mach-O relocations. see elfreloc1/machoreloc1
			if target.IsDarwin() && xadd != 0 {
				nExtReloc = 4 // need another two relocations for non-zero addend
			}

			if target.IsWindows() {
				var o0, o1 uint32
				if target.IsBigEndian() {
					o0 = uint32(val >> 32)
					o1 = uint32(val)
				} else {
					o0 = uint32(val)
					o1 = uint32(val >> 32)
				}

				// The first instruction (ADRP) has a 21-bit immediate field,
				// and the second (ADD) has a 12-bit immediate field.
				// The first instruction is only for high bits, but to get the carry bits right we have
				// to put the full addend, including the bottom 12 bits again.
				// That limits the distance of any addend to only 21 bits.
				// But we assume that LDRP's top bit will be interpreted as a sign bit,
				// so we only use 20 bits.
				// pereloc takes care of introducing new symbol labels
				// every megabyte for longer relocations.
				xadd := uint32(xadd)
				o0 |= (xadd&3)<<29 | (xadd&0xffffc)<<3
				o1 |= (xadd & 0xfff) << 10

				if target.IsBigEndian() {
					val = int64(o0)<<32 | int64(o1)
				} else {
					val = int64(o1)<<32 | int64(o0)
				}
			}

			return val, nExtReloc, isOk
		case objabi.R_CALLARM64,
			objabi.R_ARM64_TLS_LE,
			objabi.R_ARM64_TLS_IE:
			nExtReloc = 1
			if rt == objabi.R_ARM64_TLS_IE {
				nExtReloc = 2 // need two ELF relocations. see elfreloc1
			}
			return val, nExtReloc, isOk

		case objabi.R_ADDR:
			if target.IsWindows() && r.Add() != 0 {
				if r.Siz() == 8 {
					val = r.Add()
				} else if target.IsBigEndian() {
					val = int64(uint32(val)) | int64(r.Add())<<32
				} else {
					val = val>>32<<32 | int64(uint32(r.Add()))
				}
				return val, 1, true
			}
		}
	}

	switch r.Type() {
	case objabi.R_ADDRARM64:
		t := ldr.SymAddr(rs) + r.Add() - ((ldr.SymValue(s) + int64(r.Off())) &^ 0xfff)
		if t >= 1<<32 || t < -1<<32 {
			ldr.Errorf(s, "program too large, address relocation distance = %d", t)
		}

		var o0, o1 uint32

		if target.IsBigEndian() {
			o0 = uint32(val >> 32)
			o1 = uint32(val)
		} else {
			o0 = uint32(val)
			o1 = uint32(val >> 32)
		}

		o0 |= (uint32((t>>12)&3) << 29) | (uint32((t>>12>>2)&0x7ffff) << 5)
		o1 |= uint32(t&0xfff) << 10

		// when laid out, the instruction order must always be o1, o2.
		if target.IsBigEndian() {
			return int64(o0)<<32 | int64(o1), noExtReloc, true
		}
		return int64(o1)<<32 | int64(o0), noExtReloc, true

	case objabi.R_ARM64_TLS_LE:
		if target.IsDarwin() {
			ldr.Errorf(s, "TLS reloc on unsupported OS %v", target.HeadType)
		}
		// The TCB is two pointers. This is not documented anywhere, but is
		// de facto part of the ABI.
		v := ldr.SymValue(rs) + int64(2*target.Arch.PtrSize)
		if v < 0 || v >= 32678 {
			ldr.Errorf(s, "TLS offset out of range %d", v)
		}
		return val | (v << 5), noExtReloc, true

	case objabi.R_ARM64_TLS_IE:
		if target.IsPIE() && target.IsElf() {
			// We are linking the final executable, so we
			// can optimize any TLS IE relocation to LE.

			if !target.IsLinux() {
				ldr.Errorf(s, "TLS reloc on unsupported OS %v", target.HeadType)
			}

			// The TCB is two pointers. This is not documented anywhere, but is
			// de facto part of the ABI.
			v := ldr.SymAddr(rs) + int64(2*target.Arch.PtrSize) + r.Add()
			if v < 0 || v >= 32678 {
				ldr.Errorf(s, "TLS offset out of range %d", v)
			}

			var o0, o1 uint32
			if target.IsBigEndian() {
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
				ldr.Errorf(s, "invalid address: %x for relocation type: R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC", v)
			}
			o1 = 0xf2800000 | uint32(o1&0x1f) | (uint32(v&0xffff) << 5)

			// when laid out, the instruction order must always be o0, o1.
			if target.IsBigEndian() {
				return int64(o0)<<32 | int64(o1), noExtReloc, isOk
			}
			return int64(o1)<<32 | int64(o0), noExtReloc, isOk
		} else {
			log.Fatalf("cannot handle R_ARM64_TLS_IE (sym %s) when linking internally", ldr.SymName(s))
		}

	case objabi.R_CALLARM64:
		var t int64
		if ldr.SymType(rs) == sym.SDYNIMPORT {
			t = (ldr.SymAddr(syms.PLT) + r.Add()) - (ldr.SymValue(s) + int64(r.Off()))
		} else {
			t = (ldr.SymAddr(rs) + r.Add()) - (ldr.SymValue(s) + int64(r.Off()))
		}
		if t >= 1<<27 || t < -1<<27 {
			ldr.Errorf(s, "program too large, call relocation distance = %d", t)
		}
		return val | ((t >> 2) & 0x03ffffff), noExtReloc, true

	case objabi.R_ARM64_GOT:
		if (val>>24)&0x9f == 0x90 {
			// R_AARCH64_ADR_GOT_PAGE
			// patch instruction: adrp
			t := ldr.SymAddr(rs) + r.Add() - ((ldr.SymValue(s) + int64(r.Off())) &^ 0xfff)
			if t >= 1<<32 || t < -1<<32 {
				ldr.Errorf(s, "program too large, address relocation distance = %d", t)
			}
			var o0 uint32
			o0 |= (uint32((t>>12)&3) << 29) | (uint32((t>>12>>2)&0x7ffff) << 5)
			return val | int64(o0), noExtReloc, isOk
		} else if val>>24 == 0xf9 {
			// R_AARCH64_LD64_GOT_LO12_NC
			// patch instruction: ldr
			t := ldr.SymAddr(rs) + r.Add() - ((ldr.SymValue(s) + int64(r.Off())) &^ 0xfff)
			if t&7 != 0 {
				ldr.Errorf(s, "invalid address: %x for relocation type: R_AARCH64_LD64_GOT_LO12_NC", t)
			}
			var o1 uint32
			o1 |= uint32(t&0xfff) << (10 - 3)
			return val | int64(uint64(o1)), noExtReloc, isOk
		} else {
			ldr.Errorf(s, "unsupported instruction for %x R_GOTARM64", val)
		}

	case objabi.R_ARM64_PCREL:
		if (val>>24)&0x9f == 0x90 {
			// R_AARCH64_ADR_PREL_PG_HI21
			// patch instruction: adrp
			t := ldr.SymAddr(rs) + r.Add() - ((ldr.SymValue(s) + int64(r.Off())) &^ 0xfff)
			if t >= 1<<32 || t < -1<<32 {
				ldr.Errorf(s, "program too large, address relocation distance = %d", t)
			}
			o0 := (uint32((t>>12)&3) << 29) | (uint32((t>>12>>2)&0x7ffff) << 5)
			return val | int64(o0), noExtReloc, isOk
		} else if (val>>24)&0x9f == 0x91 {
			// ELF R_AARCH64_ADD_ABS_LO12_NC or Mach-O ARM64_RELOC_PAGEOFF12
			// patch instruction: add
			t := ldr.SymAddr(rs) + r.Add() - ((ldr.SymValue(s) + int64(r.Off())) &^ 0xfff)
			o1 := uint32(t&0xfff) << 10
			return val | int64(o1), noExtReloc, isOk
		} else if (val>>24)&0x3b == 0x39 {
			// Mach-O ARM64_RELOC_PAGEOFF12
			// patch ldr/str(b/h/w/d/q) (integer or vector) instructions, which have different scaling factors.
			// Mach-O uses same relocation type for them.
			shift := uint32(val) >> 30
			if shift == 0 && (val>>20)&0x048 == 0x048 { // 128-bit vector load
				shift = 4
			}
			t := ldr.SymAddr(rs) + r.Add() - ((ldr.SymValue(s) + int64(r.Off())) &^ 0xfff)
			if t&(1<<shift-1) != 0 {
				ldr.Errorf(s, "invalid address: %x for relocation type: ARM64_RELOC_PAGEOFF12", t)
			}
			o1 := (uint32(t&0xfff) >> shift) << 10
			return val | int64(o1), noExtReloc, isOk
		} else {
			ldr.Errorf(s, "unsupported instruction for %x R_ARM64_PCREL", val)
		}

	case objabi.R_ARM64_LDST8:
		t := ldr.SymAddr(rs) + r.Add() - ((ldr.SymValue(s) + int64(r.Off())) &^ 0xfff)
		o0 := uint32(t&0xfff) << 10
		return val | int64(o0), noExtReloc, true

	case objabi.R_ARM64_LDST16:
		t := ldr.SymAddr(rs) + r.Add() - ((ldr.SymValue(s) + int64(r.Off())) &^ 0xfff)
		if t&1 != 0 {
			ldr.Errorf(s, "invalid address: %x for relocation type: R_AARCH64_LDST16_ABS_LO12_NC", t)
		}
		o0 := (uint32(t&0xfff) >> 1) << 10
		return val | int64(o0), noExtReloc, true

	case objabi.R_ARM64_LDST32:
		t := ldr.SymAddr(rs) + r.Add() - ((ldr.SymValue(s) + int64(r.Off())) &^ 0xfff)
		if t&3 != 0 {
			ldr.Errorf(s, "invalid address: %x for relocation type: R_AARCH64_LDST32_ABS_LO12_NC", t)
		}
		o0 := (uint32(t&0xfff) >> 2) << 10
		return val | int64(o0), noExtReloc, true

	case objabi.R_ARM64_LDST64:
		t := ldr.SymAddr(rs) + r.Add() - ((ldr.SymValue(s) + int64(r.Off())) &^ 0xfff)
		if t&7 != 0 {
			ldr.Errorf(s, "invalid address: %x for relocation type: R_AARCH64_LDST64_ABS_LO12_NC", t)
		}
		o0 := (uint32(t&0xfff) >> 3) << 10
		return val | int64(o0), noExtReloc, true

	case objabi.R_ARM64_LDST128:
		t := ldr.SymAddr(rs) + r.Add() - ((ldr.SymValue(s) + int64(r.Off())) &^ 0xfff)
		if t&15 != 0 {
			ldr.Errorf(s, "invalid address: %x for relocation type: R_AARCH64_LDST128_ABS_LO12_NC", t)
		}
		o0 := (uint32(t&0xfff) >> 4) << 10
		return val | int64(o0), noExtReloc, true
	}

	return val, 0, false
}

func archrelocvariant(*ld.Target, *loader.Loader, loader.Reloc, sym.RelocVariant, loader.Sym, int64, []byte) int64 {
	log.Fatalf("unexpected relocation variant")
	return -1
}

func extreloc(target *ld.Target, ldr *loader.Loader, r loader.Reloc, s loader.Sym) (loader.ExtReloc, bool) {
	switch rt := r.Type(); rt {
	case objabi.R_ARM64_GOTPCREL,
		objabi.R_ADDRARM64:
		rr := ld.ExtrelocViaOuterSym(ldr, r, s)

		// Note: ld64 currently has a bug that any non-zero addend for BR26 relocation
		// will make the linking fail because it thinks the code is not PIC even though
		// the BR26 relocation should be fully resolved at link time.
		// That is the reason why the next if block is disabled. When the bug in ld64
		// is fixed, we can enable this block and also enable duff's device in cmd/7g.
		if false && target.IsDarwin() {
			// Mach-O wants the addend to be encoded in the instruction
			// Note that although Mach-O supports ARM64_RELOC_ADDEND, it
			// can only encode 24-bit of signed addend, but the instructions
			// supports 33-bit of signed addend, so we always encode the
			// addend in place.
			rr.Xadd = 0
		}
		return rr, true
	case objabi.R_CALLARM64,
		objabi.R_ARM64_TLS_LE,
		objabi.R_ARM64_TLS_IE:
		return ld.ExtrelocSimple(ldr, r), true
	}
	return loader.ExtReloc{}, false
}

func elfsetupplt(ctxt *ld.Link, plt, gotplt *loader.SymbolBuilder, dynamic loader.Sym) {
	if plt.Size() == 0 {
		// stp     x16, x30, [sp, #-16]!
		// identifying information
		plt.AddUint32(ctxt.Arch, 0xa9bf7bf0)

		// the following two instructions (adrp + ldr) load *got[2] into x17
		// adrp    x16, &got[0]
		plt.AddSymRef(ctxt.Arch, gotplt.Sym(), 16, objabi.R_ARM64_GOT, 4)
		plt.SetUint32(ctxt.Arch, plt.Size()-4, 0x90000010)

		// <imm> is the offset value of &got[2] to &got[0], the same below
		// ldr     x17, [x16, <imm>]
		plt.AddSymRef(ctxt.Arch, gotplt.Sym(), 16, objabi.R_ARM64_GOT, 4)
		plt.SetUint32(ctxt.Arch, plt.Size()-4, 0xf9400211)

		// add     x16, x16, <imm>
		plt.AddSymRef(ctxt.Arch, gotplt.Sym(), 16, objabi.R_ARM64_PCREL, 4)
		plt.SetUint32(ctxt.Arch, plt.Size()-4, 0x91000210)

		// br      x17
		plt.AddUint32(ctxt.Arch, 0xd61f0220)

		// 3 nop for place holder
		plt.AddUint32(ctxt.Arch, 0xd503201f)
		plt.AddUint32(ctxt.Arch, 0xd503201f)
		plt.AddUint32(ctxt.Arch, 0xd503201f)

		// check gotplt.size == 0
		if gotplt.Size() != 0 {
			ctxt.Errorf(gotplt.Sym(), "got.plt is not empty at the very beginning")
		}
		gotplt.AddAddrPlus(ctxt.Arch, dynamic, 0)

		gotplt.AddUint64(ctxt.Arch, 0)
		gotplt.AddUint64(ctxt.Arch, 0)
	}
}

func addpltsym(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s loader.Sym) {
	if ldr.SymPlt(s) >= 0 {
		return
	}

	ld.Adddynsym(ldr, target, syms, s)

	if target.IsElf() {
		plt := ldr.MakeSymbolUpdater(syms.PLT)
		gotplt := ldr.MakeSymbolUpdater(syms.GOTPLT)
		rela := ldr.MakeSymbolUpdater(syms.RelaPLT)
		if plt.Size() == 0 {
			panic("plt is not set up")
		}

		// adrp    x16, &got.plt[0]
		plt.AddAddrPlus4(target.Arch, gotplt.Sym(), gotplt.Size())
		plt.SetUint32(target.Arch, plt.Size()-4, 0x90000010)
		relocs := plt.Relocs()
		plt.SetRelocType(relocs.Count()-1, objabi.R_ARM64_GOT)

		// <offset> is the offset value of &got.plt[n] to &got.plt[0]
		// ldr     x17, [x16, <offset>]
		plt.AddAddrPlus4(target.Arch, gotplt.Sym(), gotplt.Size())
		plt.SetUint32(target.Arch, plt.Size()-4, 0xf9400211)
		relocs = plt.Relocs()
		plt.SetRelocType(relocs.Count()-1, objabi.R_ARM64_GOT)

		// add     x16, x16, <offset>
		plt.AddAddrPlus4(target.Arch, gotplt.Sym(), gotplt.Size())
		plt.SetUint32(target.Arch, plt.Size()-4, 0x91000210)
		relocs = plt.Relocs()
		plt.SetRelocType(relocs.Count()-1, objabi.R_ARM64_PCREL)

		// br      x17
		plt.AddUint32(target.Arch, 0xd61f0220)

		// add to got.plt: pointer to plt[0]
		gotplt.AddAddrPlus(target.Arch, plt.Sym(), 0)

		// rela
		rela.AddAddrPlus(target.Arch, gotplt.Sym(), gotplt.Size()-8)
		sDynid := ldr.SymDynid(s)

		rela.AddUint64(target.Arch, elf.R_INFO(uint32(sDynid), uint32(elf.R_AARCH64_JUMP_SLOT)))
		rela.AddUint64(target.Arch, 0)

		ldr.SetPlt(s, int32(plt.Size()-16))
	} else if target.IsDarwin() {
		ld.AddGotSym(target, ldr, syms, s, 0)

		sDynid := ldr.SymDynid(s)
		lep := ldr.MakeSymbolUpdater(syms.LinkEditPLT)
		lep.AddUint32(target.Arch, uint32(sDynid))

		plt := ldr.MakeSymbolUpdater(syms.PLT)
		ldr.SetPlt(s, int32(plt.Size()))

		// adrp x16, GOT
		plt.AddUint32(target.Arch, 0x90000010)
		r, _ := plt.AddRel(objabi.R_ARM64_GOT)
		r.SetOff(int32(plt.Size() - 4))
		r.SetSiz(4)
		r.SetSym(syms.GOT)
		r.SetAdd(int64(ldr.SymGot(s)))

		// ldr x17, [x16, <offset>]
		plt.AddUint32(target.Arch, 0xf9400211)
		r, _ = plt.AddRel(objabi.R_ARM64_GOT)
		r.SetOff(int32(plt.Size() - 4))
		r.SetSiz(4)
		r.SetSym(syms.GOT)
		r.SetAdd(int64(ldr.SymGot(s)))

		// br x17
		plt.AddUint32(target.Arch, 0xd61f0220)
	} else {
		ldr.Errorf(s, "addpltsym: unsupported binary format")
	}
}

const (
	machoRelocLimit = 1 << 23
	peRelocLimit    = 1 << 20
)

func gensymlate(ctxt *ld.Link, ldr *loader.Loader) {
	// When external linking on darwin, Mach-O relocation has only signed 24-bit
	// addend. For large symbols, we generate "label" symbols in the middle, so
	// that relocations can target them with smaller addends.
	// On Windows, we only get 21 bits, again (presumably) signed.
	if !ctxt.IsDarwin() && !ctxt.IsWindows() || !ctxt.IsExternal() {
		return
	}

	limit := int64(machoRelocLimit)
	if ctxt.IsWindows() {
		limit = peRelocLimit
	}

	if ctxt.IsDarwin() {
		big := false
		for _, seg := range ld.Segments {
			if seg.Length >= machoRelocLimit {
				big = true
				break
			}
		}
		if !big {
			return // skip work if nothing big
		}
	}

	// addLabelSyms adds "label" symbols at s+limit, s+2*limit, etc.
	addLabelSyms := func(s loader.Sym, limit, sz int64) {
		v := ldr.SymValue(s)
		for off := limit; off < sz; off += limit {
			p := ldr.LookupOrCreateSym(offsetLabelName(ldr, s, off), ldr.SymVersion(s))
			ldr.SetAttrReachable(p, true)
			ldr.SetSymValue(p, v+off)
			ldr.SetSymSect(p, ldr.SymSect(s))
			if ctxt.IsDarwin() {
				ld.AddMachoSym(ldr, p)
			} else if ctxt.IsWindows() {
				ld.AddPELabelSym(ldr, p)
			} else {
				panic("missing case in gensymlate")
			}
			// fmt.Printf("gensymlate %s %x\n", ldr.SymName(p), ldr.SymValue(p))
		}
	}

	for s, n := loader.Sym(1), loader.Sym(ldr.NSym()); s < n; s++ {
		if !ldr.AttrReachable(s) {
			continue
		}
		if ldr.SymType(s) == sym.STEXT {
			if ctxt.IsDarwin() || ctxt.IsWindows() {
				// Cannot relocate into middle of function.
				// Generate symbol names for every offset we need in duffcopy/duffzero (only 64 each).
				switch ldr.SymName(s) {
				case "runtime.duffcopy":
					addLabelSyms(s, 8, 8*64)
				case "runtime.duffzero":
					addLabelSyms(s, 4, 4*64)
				}
			}
			continue // we don't target the middle of other functions
		}
		sz := ldr.SymSize(s)
		if sz <= limit {
			continue
		}
		addLabelSyms(s, limit, sz)
	}

	// Also for carrier symbols (for which SymSize is 0)
	for _, ss := range ld.CarrierSymByType {
		if ss.Sym != 0 && ss.Size > limit {
			addLabelSyms(ss.Sym, limit, ss.Size)
		}
	}
}

// offsetLabelName returns the name of the "label" symbol used for a
// relocation targeting s+off. The label symbols is used on Darwin/Windows
// when external linking, so that the addend fits in a Mach-O/PE relocation.
func offsetLabelName(ldr *loader.Loader, s loader.Sym, off int64) string {
	if off>>20<<20 == off {
		return fmt.Sprintf("%s+%dMB", ldr.SymExtname(s), off>>20)
	}
	return fmt.Sprintf("%s+%d", ldr.SymExtname(s), off)
}

// Convert the direct jump relocation r to refer to a trampoline if the target is too far
func trampoline(ctxt *ld.Link, ldr *loader.Loader, ri int, rs, s loader.Sym) {
	relocs := ldr.Relocs(s)
	r := relocs.At(ri)
	const pcrel = 1
	switch r.Type() {
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_CALL26),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_JUMP26),
		objabi.MachoRelocOffset + ld.MACHO_ARM64_RELOC_BRANCH26*2 + pcrel:
		// Host object relocations that will be turned into a PLT call.
		// The PLT may be too far. Insert a trampoline for them.
		fallthrough
	case objabi.R_CALLARM64:
		var t int64
		// ldr.SymValue(rs) == 0 indicates a cross-package jump to a function that is not yet
		// laid out. Conservatively use a trampoline. This should be rare, as we lay out packages
		// in dependency order.
		if ldr.SymValue(rs) != 0 {
			t = ldr.SymValue(rs) + r.Add() - (ldr.SymValue(s) + int64(r.Off()))
		}
		if t >= 1<<27 || t < -1<<27 || ldr.SymValue(rs) == 0 || (*ld.FlagDebugTramp > 1 && (ldr.SymPkg(s) == "" || ldr.SymPkg(s) != ldr.SymPkg(rs))) {
			// direct call too far, need to insert trampoline.
			// look up existing trampolines first. if we found one within the range
			// of direct call, we can reuse it. otherwise create a new one.
			var tramp loader.Sym
			for i := 0; ; i++ {
				oName := ldr.SymName(rs)
				name := oName + fmt.Sprintf("%+x-tramp%d", r.Add(), i)
				tramp = ldr.LookupOrCreateSym(name, int(ldr.SymVersion(rs)))
				ldr.SetAttrReachable(tramp, true)
				if ldr.SymType(tramp) == sym.SDYNIMPORT {
					// don't reuse trampoline defined in other module
					continue
				}
				if oName == "runtime.deferreturn" {
					ldr.SetIsDeferReturnTramp(tramp, true)
				}
				if ldr.SymValue(tramp) == 0 {
					// either the trampoline does not exist -- we need to create one,
					// or found one the address which is not assigned -- this will be
					// laid down immediately after the current function. use this one.
					break
				}

				t = ldr.SymValue(tramp) - (ldr.SymValue(s) + int64(r.Off()))
				if t >= -1<<27 && t < 1<<27 {
					// found an existing trampoline that is not too far
					// we can just use it
					break
				}
			}
			if ldr.SymType(tramp) == 0 {
				// trampoline does not exist, create one
				trampb := ldr.MakeSymbolUpdater(tramp)
				ctxt.AddTramp(trampb)
				if ldr.SymType(rs) == sym.SDYNIMPORT {
					if r.Add() != 0 {
						ctxt.Errorf(s, "nonzero addend for DYNIMPORT call: %v+%d", ldr.SymName(rs), r.Add())
					}
					gentrampgot(ctxt, ldr, trampb, rs)
				} else {
					gentramp(ctxt, ldr, trampb, rs, r.Add())
				}
			}
			// modify reloc to point to tramp, which will be resolved later
			sb := ldr.MakeSymbolUpdater(s)
			relocs := sb.Relocs()
			r := relocs.At(ri)
			r.SetSym(tramp)
			r.SetAdd(0) // clear the offset embedded in the instruction
		}
	default:
		ctxt.Errorf(s, "trampoline called with non-jump reloc: %d (%s)", r.Type(), sym.RelocName(ctxt.Arch, r.Type()))
	}
}

// generate a trampoline to target+offset.
func gentramp(ctxt *ld.Link, ldr *loader.Loader, tramp *loader.SymbolBuilder, target loader.Sym, offset int64) {
	tramp.SetSize(12) // 3 instructions
	P := make([]byte, tramp.Size())
	o1 := uint32(0x90000010) // adrp x16, target
	o2 := uint32(0x91000210) // add x16, pc-relative-offset
	o3 := uint32(0xd61f0200) // br x16
	ctxt.Arch.ByteOrder.PutUint32(P, o1)
	ctxt.Arch.ByteOrder.PutUint32(P[4:], o2)
	ctxt.Arch.ByteOrder.PutUint32(P[8:], o3)
	tramp.SetData(P)

	r, _ := tramp.AddRel(objabi.R_ADDRARM64)
	r.SetSiz(8)
	r.SetSym(target)
	r.SetAdd(offset)
}

// generate a trampoline to target+offset for a DYNIMPORT symbol via GOT.
func gentrampgot(ctxt *ld.Link, ldr *loader.Loader, tramp *loader.SymbolBuilder, target loader.Sym) {
	tramp.SetSize(12) // 3 instructions
	P := make([]byte, tramp.Size())
	o1 := uint32(0x90000010) // adrp x16, target@GOT
	o2 := uint32(0xf9400210) // ldr x16, [x16, offset]
	o3 := uint32(0xd61f0200) // br x16
	ctxt.Arch.ByteOrder.PutUint32(P, o1)
	ctxt.Arch.ByteOrder.PutUint32(P[4:], o2)
	ctxt.Arch.ByteOrder.PutUint32(P[8:], o3)
	tramp.SetData(P)

	r, _ := tramp.AddRel(objabi.R_ARM64_GOTPCREL)
	r.SetSiz(8)
	r.SetSym(target)
}
