// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loong64

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

	// Emit the following function:
	//
	//	local.dso_init:
	//		la.pcrel $a0, local.moduledata
	//		b runtime.addmoduledata

	//	0000000000000000 <local.dso_init>:
	//	0:	1a000004	pcalau12i	$a0, 0
	//				0: R_LARCH_PCALA_HI20	local.moduledata
	o(0x1a000004)
	rel, _ := initfunc.AddRel(objabi.R_LOONG64_ADDR_HI)
	rel.SetOff(0)
	rel.SetSiz(4)
	rel.SetSym(ctxt.Moduledata)

	//	4:	02c00084	addi.d	$a0, $a0, 0
	//				4: R_LARCH_PCALA_LO12	local.moduledata
	o(0x02c00084)
	rel2, _ := initfunc.AddRel(objabi.R_LOONG64_ADDR_LO)
	rel2.SetOff(4)
	rel2.SetSiz(4)
	rel2.SetSym(ctxt.Moduledata)

	//	8:	50000000	b	0
	//				8: R_LARCH_B26	runtime.addmoduledata
	o(0x50000000)
	rel3, _ := initfunc.AddRel(objabi.R_CALLLOONG64)
	rel3.SetOff(8)
	rel3.SetSiz(4)
	rel3.SetSym(addmoduledata)
}

func adddynrel(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s loader.Sym, r loader.Reloc, rIdx int) bool {
	targ := r.Sym()
	var targType sym.SymKind
	if targ != 0 {
		targType = ldr.SymType(targ)
	}

	switch r.Type() {
	default:
		if r.Type() >= objabi.ElfRelocOffset {
			ldr.Errorf(s, "adddynrel: unexpected reloction type %d (%s)", r.Type(), sym.RelocName(target.Arch, r.Type()))
			return false
		}

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_64):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected R_LARCH_64 relocation for dynamic symbol %s", ldr.SymName(targ))
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

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_B26):
		if targType == sym.SDYNIMPORT {
			addpltsym(target, ldr, syms, targ)
			su := ldr.MakeSymbolUpdater(s)
			su.SetRelocSym(rIdx, syms.PLT)
			su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymPlt(targ)))
		}
		if targType == 0 || targType == sym.SXREF {
			ldr.Errorf(s, "unknown symbol %s in callloong64", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_CALLLOONG64)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_GOT_PC_HI20),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_GOT_PC_LO12):
		if targType != sym.SDYNIMPORT {
			// TODO: turn LDR of GOT entry into ADR of symbol itself
		}

		ld.AddGotSym(target, ldr, syms, targ, uint32(elf.R_LARCH_64))
		su := ldr.MakeSymbolUpdater(s)
		if r.Type() == objabi.ElfRelocOffset+objabi.RelocType(elf.R_LARCH_GOT_PC_HI20) {
			su.SetRelocType(rIdx, objabi.R_LOONG64_ADDR_HI)
		} else {
			su.SetRelocType(rIdx, objabi.R_LOONG64_ADDR_LO)
		}
		su.SetRelocSym(rIdx, syms.GOT)
		su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymGot(targ)))
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_PCALA_HI20),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_PCALA_LO12):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		if targType == 0 || targType == sym.SXREF {
			ldr.Errorf(s, "unknown symbol %s", ldr.SymName(targ))
		}

		su := ldr.MakeSymbolUpdater(s)
		if r.Type() == objabi.ElfRelocOffset+objabi.RelocType(elf.R_LARCH_PCALA_HI20) {
			su.SetRelocType(rIdx, objabi.R_LOONG64_ADDR_HI)
		} else {
			su.SetRelocType(rIdx, objabi.R_LOONG64_ADDR_LO)
		}
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_ADD64),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_SUB64):
		su := ldr.MakeSymbolUpdater(s)
		if r.Type() == objabi.ElfRelocOffset+objabi.RelocType(elf.R_LARCH_ADD64) {
			su.SetRelocType(rIdx, objabi.R_LOONG64_ADD64)
		} else {
			su.SetRelocType(rIdx, objabi.R_LOONG64_SUB64)
		}
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_B16),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_B21):
		if targType == sym.SDYNIMPORT {
			addpltsym(target, ldr, syms, targ)
			su := ldr.MakeSymbolUpdater(s)
			su.SetRelocSym(rIdx, syms.PLT)
			su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymPlt(targ)))
		}
		if targType == 0 || targType == sym.SXREF {
			ldr.Errorf(s, "unknown symbol %s in R_JMPxxLOONG64", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		if r.Type() == objabi.ElfRelocOffset+objabi.RelocType(elf.R_LARCH_B16) {
			su.SetRelocType(rIdx, objabi.R_JMP16LOONG64)
		} else {
			su.SetRelocType(rIdx, objabi.R_JMP21LOONG64)
		}
		return true
	}

	relocs := ldr.Relocs(s)
	r = relocs.At(rIdx)

	switch r.Type() {
	case objabi.R_CALLLOONG64:
		if targType != sym.SDYNIMPORT {
			return true
		}
		if target.IsExternal() {
			return true
		}

		// Internal linking.
		if r.Add() != 0 {
			ldr.Errorf(s, "PLT call with no-zero addend (%v)", r.Add())
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
			ld.AddGotSym(target, ldr, syms, targ, uint32(elf.R_LARCH_64))
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
			// Generate R_LARCH_RELATIVE relocations for best
			// efficiency in the dynamic linker.
			//
			// As noted above, symbol addresses have not been
			// assigned yet, so we can't generate the final reloc
			// entry yet. We ultimately want:
			//
			// r_offset = s + r.Off
			// r_info = R_LARCH_RELATIVE
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
				rela.AddUint64(target.Arch, elf.R_INFO(0, uint32(elf.R_LARCH_RELATIVE)))
			} else {
				ldr.Errorf(s, "unexpected relocation for dynamic symbol %s", ldr.SymName(targ))
			}
			rela.AddAddrPlus(target.Arch, targ, r.Add())
			return true
		}

	case objabi.R_LOONG64_GOT_HI,
		objabi.R_LOONG64_GOT_LO:
		ld.AddGotSym(target, ldr, syms, targ, uint32(elf.R_LARCH_64))
		su := ldr.MakeSymbolUpdater(s)
		if r.Type() == objabi.R_LOONG64_GOT_HI {
			su.SetRelocType(rIdx, objabi.R_LOONG64_ADDR_HI)
		} else {
			su.SetRelocType(rIdx, objabi.R_LOONG64_ADDR_LO)
		}
		su.SetRelocSym(rIdx, syms.GOT)
		su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymGot(targ)))
		return true
	}
	return false
}

func elfsetupplt(ctxt *ld.Link, ldr *loader.Loader, plt, gotplt *loader.SymbolBuilder, dynamic loader.Sym) {
	if plt.Size() == 0 {
		// pcalau12i $r14, imm
		plt.AddSymRef(ctxt.Arch, gotplt.Sym(), 0, objabi.R_LOONG64_ADDR_HI, 4)
		plt.SetUint32(ctxt.Arch, plt.Size()-4, 0x1a00000e)

		// sub.d $r13, $r13, $r15
		plt.AddUint32(ctxt.Arch, 0x0011bdad)

		// ld.d $r15, $r14, imm
		plt.AddSymRef(ctxt.Arch, gotplt.Sym(), 0, objabi.R_LOONG64_ADDR_LO, 4)
		plt.SetUint32(ctxt.Arch, plt.Size()-4, 0x28c001cf)

		// addi.d $r13, $r13, -40
		plt.AddUint32(ctxt.Arch, 0x02ff61ad)

		// addi.d $r12, $r14, imm
		plt.AddSymRef(ctxt.Arch, gotplt.Sym(), 0, objabi.R_LOONG64_ADDR_LO, 4)
		plt.SetUint32(ctxt.Arch, plt.Size()-4, 0x2c001cc)

		// srli.d $r13, $r13, 1
		plt.AddUint32(ctxt.Arch, 0x004505ad)

		// ld.d $r12, $r12, 8
		plt.AddUint32(ctxt.Arch, 0x28c0218c)

		// jirl $r0, $r15, 0
		plt.AddUint32(ctxt.Arch, 0x4c0001e0)

		// check gotplt.size == 0
		if gotplt.Size() != 0 {
			ctxt.Errorf(gotplt.Sym(), "got.plt is not empty at the very beginning")
		}

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

		// pcalau12i $r15, imm
		plt.AddAddrPlus4(target.Arch, gotplt.Sym(), gotplt.Size())
		plt.SetUint32(target.Arch, plt.Size()-4, 0x1a00000f)
		relocs := plt.Relocs()
		plt.SetRelocType(relocs.Count()-1, objabi.R_LOONG64_ADDR_HI)

		// ld.d $r15, $r15, imm
		plt.AddAddrPlus4(target.Arch, gotplt.Sym(), gotplt.Size())
		plt.SetUint32(target.Arch, plt.Size()-4, 0x28c001ef)
		relocs = plt.Relocs()
		plt.SetRelocType(relocs.Count()-1, objabi.R_LOONG64_ADDR_LO)

		// pcaddu12i $r13, 0
		plt.AddUint32(target.Arch, 0x1c00000d)

		// jirl r0, r15, 0
		plt.AddUint32(target.Arch, 0x4c0001e0)

		// add to got.plt: pointer to plt[0]
		gotplt.AddAddrPlus(target.Arch, plt.Sym(), 0)

		// rela
		rela.AddAddrPlus(target.Arch, gotplt.Sym(), gotplt.Size()-8)
		sDynid := ldr.SymDynid(s)
		rela.AddUint64(target.Arch, elf.R_INFO(uint32(sDynid), uint32(elf.R_LARCH_JUMP_SLOT)))
		rela.AddUint64(target.Arch, 0)

		ldr.SetPlt(s, int32(plt.Size()-16))
	} else {
		ldr.Errorf(s, "addpltsym: unsupport binary format")
	}
}

func elfreloc1(ctxt *ld.Link, out *ld.OutBuf, ldr *loader.Loader, s loader.Sym, r loader.ExtReloc, ri int, sectoff int64) bool {
	// loong64 ELF relocation (endian neutral)
	//		offset     uint64
	//		symreloc   uint64  // The high 32-bit is the symbol, the low 32-bit is the relocation type.
	//		addend     int64

	elfsym := ld.ElfSymForReloc(ctxt, r.Xsym)
	switch r.Type {
	default:
		return false
	case objabi.R_ADDR, objabi.R_DWARFSECREF:
		switch r.Size {
		case 4:
			out.Write64(uint64(sectoff))
			out.Write64(uint64(elf.R_LARCH_32) | uint64(elfsym)<<32)
			out.Write64(uint64(r.Xadd))
		case 8:
			out.Write64(uint64(sectoff))
			out.Write64(uint64(elf.R_LARCH_64) | uint64(elfsym)<<32)
			out.Write64(uint64(r.Xadd))
		default:
			return false
		}
	case objabi.R_LOONG64_TLS_LE_LO:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_TLS_LE_LO12) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))

	case objabi.R_LOONG64_TLS_LE_HI:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_TLS_LE_HI20) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))

	case objabi.R_CALLLOONG64:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_B26) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))

	case objabi.R_LOONG64_TLS_IE_HI:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_TLS_IE_PC_HI20) | uint64(elfsym)<<32)
		out.Write64(uint64(0x0))

	case objabi.R_LOONG64_TLS_IE_LO:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_TLS_IE_PC_LO12) | uint64(elfsym)<<32)
		out.Write64(uint64(0x0))

	case objabi.R_LOONG64_ADDR_LO:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_PCALA_LO12) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))

	case objabi.R_LOONG64_ADDR_HI:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_PCALA_HI20) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))

	case objabi.R_LOONG64_GOT_HI:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_GOT_PC_HI20) | uint64(elfsym)<<32)
		out.Write64(uint64(0x0))

	case objabi.R_LOONG64_GOT_LO:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_GOT_PC_LO12) | uint64(elfsym)<<32)
		out.Write64(uint64(0x0))
	}

	return true
}

func machoreloc1(*sys.Arch, *ld.OutBuf, *loader.Loader, loader.Sym, loader.ExtReloc, int64) bool {
	return false
}

func archreloc(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, r loader.Reloc, s loader.Sym, val int64) (o int64, nExtReloc int, ok bool) {
	rs := r.Sym()
	if target.IsExternal() {
		switch r.Type() {
		default:
			return val, 0, false
		case objabi.R_LOONG64_ADDR_HI,
			objabi.R_LOONG64_ADDR_LO:
			// set up addend for eventual relocation via outer symbol.
			rs, _ := ld.FoldSubSymbolOffset(ldr, rs)
			rst := ldr.SymType(rs)
			if rst != sym.SHOSTOBJ && rst != sym.SDYNIMPORT && ldr.SymSect(rs) == nil {
				ldr.Errorf(s, "missing section for %s", ldr.SymName(rs))
			}
			return val, 1, true
		case objabi.R_LOONG64_TLS_LE_HI,
			objabi.R_LOONG64_TLS_LE_LO,
			objabi.R_CALLLOONG64,
			objabi.R_JMPLOONG64,
			objabi.R_LOONG64_TLS_IE_HI,
			objabi.R_LOONG64_TLS_IE_LO,
			objabi.R_LOONG64_GOT_HI,
			objabi.R_LOONG64_GOT_LO:
			return val, 1, true
		}
	}

	const isOk = true
	const noExtReloc = 0

	switch r.Type() {
	case objabi.R_CONST:
		return r.Add(), noExtReloc, isOk
	case objabi.R_GOTOFF:
		return ldr.SymValue(r.Sym()) + r.Add() - ldr.SymValue(syms.GOT), noExtReloc, isOk
	case objabi.R_LOONG64_ADDR_HI,
		objabi.R_LOONG64_ADDR_LO:
		pc := ldr.SymValue(s) + int64(r.Off())
		t := calculatePCAlignedReloc(r.Type(), ldr.SymAddr(rs)+r.Add(), pc)
		if r.Type() == objabi.R_LOONG64_ADDR_LO {
			return val&0xffc003ff | (t << 10), noExtReloc, isOk
		}
		return val&0xfe00001f | (t << 5), noExtReloc, isOk
	case objabi.R_LOONG64_TLS_LE_HI,
		objabi.R_LOONG64_TLS_LE_LO:
		t := ldr.SymAddr(rs) + r.Add()
		if r.Type() == objabi.R_LOONG64_TLS_LE_LO {
			return val&0xffc003ff | ((t & 0xfff) << 10), noExtReloc, isOk
		}
		return val&0xfe00001f | (((t) >> 12 << 5) & 0x1ffffe0), noExtReloc, isOk
	case objabi.R_CALLLOONG64,
		objabi.R_JMPLOONG64:
		pc := ldr.SymValue(s) + int64(r.Off())
		t := ldr.SymAddr(rs) + r.Add() - pc
		return val&0xfc000000 | (((t >> 2) & 0xffff) << 10) | (((t >> 2) & 0x3ff0000) >> 16), noExtReloc, isOk

	case objabi.R_JMP16LOONG64,
		objabi.R_JMP21LOONG64:
		pc := ldr.SymValue(s) + int64(r.Off())
		t := ldr.SymAddr(rs) + r.Add() - pc
		if r.Type() == objabi.R_JMP16LOONG64 {
			return val&0xfc0003ff | (((t >> 2) & 0xffff) << 10), noExtReloc, isOk
		}
		return val&0xfc0003e0 | (((t >> 2) & 0xffff) << 10) | (((t >> 2) & 0x1f0000) >> 16), noExtReloc, isOk

	case objabi.R_LOONG64_TLS_IE_HI,
		objabi.R_LOONG64_TLS_IE_LO:
		if target.IsPIE() && target.IsElf() {
			if !target.IsLinux() {
				ldr.Errorf(s, "TLS reloc on unsupported OS %v", target.HeadType)
			}
			t := ldr.SymAddr(rs) + r.Add()
			if r.Type() == objabi.R_LOONG64_TLS_IE_HI {
				// pcalau12i -> lu12i.w
				return (0x14000000 | (val & 0x1f) | ((t >> 12) << 5)), noExtReloc, isOk
			}
			// ld.d -> ori
			return (0x03800000 | (val & 0x3ff) | ((t & 0xfff) << 10)), noExtReloc, isOk
		} else {
			log.Fatalf("cannot handle R_LOONG64_TLS_IE_x (sym %s) when linking internally", ldr.SymName(rs))
		}

	case objabi.R_LOONG64_ADD64, objabi.R_LOONG64_SUB64:
		if r.Type() == objabi.R_LOONG64_ADD64 {
			return val + ldr.SymAddr(rs) + r.Add(), noExtReloc, isOk
		}
		return val - (ldr.SymAddr(rs) + r.Add()), noExtReloc, isOk
	}

	return val, 0, false
}

func archrelocvariant(*ld.Target, *loader.Loader, loader.Reloc, sym.RelocVariant, loader.Sym, int64, []byte) int64 {
	return -1
}

func extreloc(target *ld.Target, ldr *loader.Loader, r loader.Reloc, s loader.Sym) (loader.ExtReloc, bool) {
	switch r.Type() {
	case objabi.R_LOONG64_ADDR_HI,
		objabi.R_LOONG64_ADDR_LO,
		objabi.R_LOONG64_GOT_HI,
		objabi.R_LOONG64_GOT_LO:
		return ld.ExtrelocViaOuterSym(ldr, r, s), true

	case objabi.R_LOONG64_TLS_LE_HI,
		objabi.R_LOONG64_TLS_LE_LO,
		objabi.R_CONST,
		objabi.R_GOTOFF,
		objabi.R_CALLLOONG64,
		objabi.R_JMPLOONG64,
		objabi.R_LOONG64_TLS_IE_HI,
		objabi.R_LOONG64_TLS_IE_LO:
		return ld.ExtrelocSimple(ldr, r), true
	}
	return loader.ExtReloc{}, false
}

func isRequestingLowPageBits(t objabi.RelocType) bool {
	switch t {
	case objabi.R_LOONG64_ADDR_LO:
		return true
	}
	return false
}

// Calculates the value to put into the immediate slot, according to the
// desired relocation type, target and PC.
// The value to use varies based on the reloc type. Namely, the absolute low
// bits of the target are to be used for the low part, while the page-aligned
// offset is to be used for the higher part. A "page" here is not related to
// the system's actual page size, but rather a fixed 12-bit range (designed to
// cooperate with ADDI/LD/ST's 12-bit immediates).
func calculatePCAlignedReloc(t objabi.RelocType, tgt int64, pc int64) int64 {
	if isRequestingLowPageBits(t) {
		// corresponding immediate field is 12 bits wide
		return tgt & 0xfff
	}

	pageDelta := (tgt >> 12) - (pc >> 12)
	if tgt&0xfff >= 0x800 {
		// adjust for sign-extended addition of the low bits
		pageDelta += 1
	}
	// corresponding immediate field is 20 bits wide
	return pageDelta & 0xfffff
}

// Convert the direct jump relocation r to refer to a trampoline if the target is too far.
func trampoline(ctxt *ld.Link, ldr *loader.Loader, ri int, rs, s loader.Sym) {
	relocs := ldr.Relocs(s)
	r := relocs.At(ri)
	switch r.Type() {
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_B26):
		// Nothing to do.
		// The plt symbol has not been added. If we add tramp
		// here, plt will not work.
	case objabi.R_CALLLOONG64:
		var t int64
		// ldr.SymValue(rs) == 0 indicates a cross-package jump to a function that is not yet
		// laid out. Conservatively use a trampoline. This should be rare, as we lay out packages
		// in dependency order.
		if ldr.SymValue(rs) != 0 {
			t = ldr.SymValue(rs) + r.Add() - (ldr.SymValue(s) + int64(r.Off()))
		}
		if t >= 1<<27 || t < -1<<27 || ldr.SymValue(rs) == 0 || (*ld.FlagDebugTramp > 1 && (ldr.SymPkg(s) == "" || ldr.SymPkg(s) != ldr.SymPkg(rs))) {
			// direct call too far need to insert trampoline.
			// look up existing trampolines first. if we found one within the range
			// of direct call, we can reuse it. otherwise create a new one.
			var tramp loader.Sym
			for i := 0; ; i++ {
				oName := ldr.SymName(rs)
				name := oName + fmt.Sprintf("%+x-tramp%d", r.Add(), i)
				tramp = ldr.LookupOrCreateSym(name, ldr.SymVersion(rs))
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
					// we can just use it.
					break
				}
			}
			if ldr.SymType(tramp) == 0 {
				// trampoline does not exist, create one
				trampb := ldr.MakeSymbolUpdater(tramp)
				ctxt.AddTramp(trampb, ldr.SymType(s))
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

	o1 := uint32(0x1a00001e) // pcalau12i $r30, 0
	ctxt.Arch.ByteOrder.PutUint32(P, o1)
	r1, _ := tramp.AddRel(objabi.R_LOONG64_ADDR_HI)
	r1.SetOff(0)
	r1.SetSiz(4)
	r1.SetSym(target)
	r1.SetAdd(offset)

	o2 := uint32(0x02c003de) // addi.d $r30, $r30, 0
	ctxt.Arch.ByteOrder.PutUint32(P[4:], o2)
	r2, _ := tramp.AddRel(objabi.R_LOONG64_ADDR_LO)
	r2.SetOff(4)
	r2.SetSiz(4)
	r2.SetSym(target)
	r2.SetAdd(offset)

	o3 := uint32(0x4c0003c0) // jirl $r0, $r30, 0
	ctxt.Arch.ByteOrder.PutUint32(P[8:], o3)

	tramp.SetData(P)
}

func gentrampgot(ctxt *ld.Link, ldr *loader.Loader, tramp *loader.SymbolBuilder, target loader.Sym) {
	tramp.SetSize(12) // 3 instructions
	P := make([]byte, tramp.Size())

	o1 := uint32(0x1a00001e) // pcalau12i $r30, 0
	ctxt.Arch.ByteOrder.PutUint32(P, o1)
	r1, _ := tramp.AddRel(objabi.R_LOONG64_GOT_HI)
	r1.SetOff(0)
	r1.SetSiz(4)
	r1.SetSym(target)

	o2 := uint32(0x28c003de) // ld.d $r30, $r30, 0
	ctxt.Arch.ByteOrder.PutUint32(P[4:], o2)
	r2, _ := tramp.AddRel(objabi.R_LOONG64_GOT_LO)
	r2.SetOff(4)
	r2.SetSiz(4)
	r2.SetSym(target)

	o3 := uint32(0x4c0003c0) // jirl $r0, $r30, 0
	ctxt.Arch.ByteOrder.PutUint32(P[8:], o3)

	tramp.SetData(P)
}
