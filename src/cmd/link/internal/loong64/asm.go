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

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_B26),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_CALL36):
		if targType == sym.SDYNIMPORT {
			addpltsym(target, ldr, syms, targ)
			su := ldr.MakeSymbolUpdater(s)
			su.SetRelocSym(rIdx, syms.PLT)
			su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymPlt(targ)))
		}
		if targType == 0 || targType == sym.SXREF {
			ldr.Errorf(s, "unknown symbol %s in callloong64", ldr.SymName(targ))
		}
		relocType := objabi.R_CALLLOONG64
		if r.Type() == objabi.ElfRelocOffset+objabi.RelocType(elf.R_LARCH_CALL36) {
			relocType = objabi.R_LOONG64_CALL36
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, relocType)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_GOT_PC_HI20),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_GOT_PC_LO12),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_GOT64_PC_HI12),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_GOT64_PC_LO20):
		if targType != sym.SDYNIMPORT {
			// TODO: turn LDR of GOT entry into ADR of symbol itself
		}

		ld.AddGotSym(target, ldr, syms, targ, uint32(elf.R_LARCH_64))
		su := ldr.MakeSymbolUpdater(s)

		var relocType objabi.RelocType
		switch r.Type() - objabi.ElfRelocOffset {
		case objabi.RelocType(elf.R_LARCH_GOT_PC_HI20):
			relocType = objabi.R_LOONG64_ADDR_HI
		case objabi.RelocType(elf.R_LARCH_GOT_PC_LO12):
			relocType = objabi.R_LOONG64_ADDR_LO
		case objabi.RelocType(elf.R_LARCH_GOT64_PC_HI12):
			relocType = objabi.R_LOONG64_ADDR64_HI
		case objabi.RelocType(elf.R_LARCH_GOT64_PC_LO20):
			relocType = objabi.R_LOONG64_ADDR64_LO
		}

		su.SetRelocType(rIdx, relocType)
		su.SetRelocSym(rIdx, syms.GOT)
		su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymGot(targ)))
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_PCALA_HI20),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_PCALA_LO12),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_PCALA64_HI12),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_PCALA64_LO20),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_PCREL20_S2):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		if targType == 0 || targType == sym.SXREF {
			ldr.Errorf(s, "unknown symbol %s", ldr.SymName(targ))
		}

		var relocType objabi.RelocType
		switch r.Type() - objabi.ElfRelocOffset {
		case objabi.RelocType(elf.R_LARCH_PCALA_HI20):
			relocType = objabi.R_LOONG64_ADDR_HI
		case objabi.RelocType(elf.R_LARCH_PCALA_LO12):
			relocType = objabi.R_LOONG64_ADDR_LO
		case objabi.RelocType(elf.R_LARCH_PCALA64_HI12):
			relocType = objabi.R_LOONG64_ADDR64_HI
		case objabi.RelocType(elf.R_LARCH_PCALA64_LO20):
			relocType = objabi.R_LOONG64_ADDR64_LO
		case objabi.RelocType(elf.R_LARCH_PCREL20_S2):
			relocType = objabi.R_LOONG64_ADDR_PCREL20_S2
		}

		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, relocType)
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
		objabi.R_LOONG64_GOT_LO,
		objabi.R_LOONG64_GOT64_HI,
		objabi.R_LOONG64_GOT64_LO:
		ld.AddGotSym(target, ldr, syms, targ, uint32(elf.R_LARCH_64))
		su := ldr.MakeSymbolUpdater(s)

		var relocType objabi.RelocType
		switch r.Type() {
		case objabi.R_LOONG64_GOT_HI:
			relocType = objabi.R_LOONG64_ADDR_HI
		case objabi.R_LOONG64_GOT_LO:
			relocType = objabi.R_LOONG64_ADDR_LO
		case objabi.R_LOONG64_GOT64_HI:
			relocType = objabi.R_LOONG64_ADDR64_HI
		case objabi.R_LOONG64_GOT64_LO:
			relocType = objabi.R_LOONG64_ADDR64_LO
		}

		su.SetRelocType(rIdx, relocType)
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

	case objabi.R_LOONG64_CALL36:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_CALL36) | uint64(elfsym)<<32)
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

	case objabi.R_LOONG64_ADDR64_LO:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_PCALA64_LO20) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))

	case objabi.R_LOONG64_ADDR64_HI:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_PCALA64_HI12) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))

	case objabi.R_LOONG64_ADDR_PCREL20_S2:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_PCREL20_S2) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))

	case objabi.R_LOONG64_GOT_HI:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_GOT_PC_HI20) | uint64(elfsym)<<32)
		out.Write64(uint64(0x0))

	case objabi.R_LOONG64_GOT_LO:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_GOT_PC_LO12) | uint64(elfsym)<<32)
		out.Write64(uint64(0x0))

	case objabi.R_LOONG64_GOT64_HI:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_GOT64_PC_HI12) | uint64(elfsym)<<32)
		out.Write64(uint64(0x0))

	case objabi.R_LOONG64_GOT64_LO:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_GOT64_PC_LO20) | uint64(elfsym)<<32)
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
			objabi.R_LOONG64_ADDR_LO,
			objabi.R_LOONG64_ADDR64_HI,
			objabi.R_LOONG64_ADDR64_LO,
			objabi.R_LOONG64_ADDR_PCREL20_S2:
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
			objabi.R_LOONG64_CALL36,
			objabi.R_LOONG64_TLS_IE_HI,
			objabi.R_LOONG64_TLS_IE_LO,
			objabi.R_LOONG64_GOT_HI,
			objabi.R_LOONG64_GOT_LO,
			objabi.R_LOONG64_GOT64_HI,
			objabi.R_LOONG64_GOT64_LO:
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
		objabi.R_LOONG64_ADDR_LO,
		objabi.R_LOONG64_ADDR64_HI,
		objabi.R_LOONG64_ADDR64_LO:
		pc := ldr.SymValue(s) + int64(r.Off())
		var t int64

		switch r.Type() {
		case objabi.R_LOONG64_ADDR_LO:
			// pcalau12i
			t = pc32RelocBits(r.Type(), ldr.SymAddr(rs)+r.Add(), pc)
			o = val&0xffc003ff | (t << 10)
		case objabi.R_LOONG64_ADDR_HI:
			// addi.w/addi.d
			t = pc32RelocBits(r.Type(), ldr.SymAddr(rs)+r.Add(), pc)
			o = val&0xfe00001f | (t << 5)
		case objabi.R_LOONG64_ADDR64_LO:
			// lu32i.d
			t = pc64RelocBits(r.Type(), ldr.SymAddr(rs)+r.Add(), pc-8)
			o = val&0xfe00001f | (t << 5)
		case objabi.R_LOONG64_ADDR64_HI:
			// lu52i.d
			t = pc64RelocBits(r.Type(), ldr.SymAddr(rs)+r.Add(), pc-12)
			o = val&0xffc003ff | (t << 10)
		}

		return o, noExtReloc, isOk

	case objabi.R_LOONG64_ADDR_PCREL20_S2:
		pc := ldr.SymValue(s) + int64(r.Off())
		t := (ldr.SymAddr(rs) + r.Add() - pc) >> 2
		if t < -1<<21 || t >= 1<<21 {
			ldr.Errorf(s, "reloc: R_LOONG64_ADDR_PCREL20_S2, offset out of range %d", t)
		}
		// pcaddi
		return val&0xfe00001f | ((t & 0xfffff) << 5), noExtReloc, isOk

	case objabi.R_LOONG64_TLS_LE_HI,
		objabi.R_LOONG64_TLS_LE_LO:
		t := ldr.SymAddr(rs) + r.Add()
		if t < -1<<31 || t >= 1<<31 {
			ldr.Errorf(s, "reloc: R_LOONG64_TLS_LE_HI/LO, TLS offset out of range %d", t)
		}
		if r.Type() == objabi.R_LOONG64_TLS_LE_LO {
			// ori
			return val&0xffc003ff | ((t & 0xfff) << 10), noExtReloc, isOk
		}
		// lu12i.w
		return val&0xfe00001f | (((t) >> 12 << 5) & 0x1ffffe0), noExtReloc, isOk

	case objabi.R_CALLLOONG64:
		pc := ldr.SymValue(s) + int64(r.Off())
		t := ldr.SymAddr(rs) + r.Add() - pc
		if t < -1<<27 || t >= 1<<27 {
			ldr.Errorf(s, "reloc: R_CALLLOONG64, program too large, call relocation distance = %d", t)
		}
		// bl
		return val&0xfc000000 | (((t >> 2) & 0xffff) << 10) | (((t >> 2) & 0x3ff0000) >> 16), noExtReloc, isOk

	case objabi.R_LOONG64_CALL36:
		pc := ldr.SymValue(s) + int64(r.Off())
		t := (ldr.SymAddr(rs) + r.Add() - pc) >> 2
		if t < -1<<37 || t >= 1<<37 {
			ldr.Errorf(s, "reloc: R_LOONG64_CALL36, program too large, call relocation distance = %d", t)
		}
		// val is pcaddu18i (lower half) + jirl (upper half)
		pcaddu18i := (val & 0xfe00001f) | (((t + 0x8000) >> 16) << 5)
		jirl := ((val >> 32) & 0xfc0003ff) | ((t & 0xffff) << 10)
		return pcaddu18i | (jirl << 32), noExtReloc, isOk

	case objabi.R_JMP16LOONG64,
		objabi.R_JMP21LOONG64:
		pc := ldr.SymValue(s) + int64(r.Off())
		t := ldr.SymAddr(rs) + r.Add() - pc
		if r.Type() == objabi.R_JMP16LOONG64 {
			if t < -1<<17 || t >= 1<<17 {
				ldr.Errorf(s, "reloc: R_JMP16LOONG64, program too large, jmp relocation distance = %d", t)
			}
			// beq/bne/blt[u]/bge[u]
			return val&0xfc0003ff | (((t >> 2) & 0xffff) << 10), noExtReloc, isOk
		}
		if t < -1<<22 || t >= 1<<22 {
			ldr.Errorf(s, "reloc: R_JMP21LOONG64, program too large, call relocation distance = %d", t)
		}
		// beqz/bnez
		return val&0xfc0003e0 | (((t >> 2) & 0xffff) << 10) | (((t >> 2) & 0x1f0000) >> 16), noExtReloc, isOk

	case objabi.R_LOONG64_TLS_IE_HI,
		objabi.R_LOONG64_TLS_IE_LO:
		if target.IsPIE() && target.IsElf() {
			if !target.IsLinux() {
				ldr.Errorf(s, "TLS reloc on unsupported OS %v", target.HeadType)
			}
			t := ldr.SymAddr(rs) + r.Add()
			if t < -1<<31 || t >= 1<<31 {
				ldr.Errorf(s, "reloc: R_LOONG64_TLS_IE_HI/LO, TLS offset out of range %d", t)
			}
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
		objabi.R_LOONG64_ADDR64_HI,
		objabi.R_LOONG64_ADDR64_LO,
		objabi.R_LOONG64_GOT_HI,
		objabi.R_LOONG64_GOT_LO,
		objabi.R_LOONG64_GOT64_HI,
		objabi.R_LOONG64_GOT64_LO:
		return ld.ExtrelocViaOuterSym(ldr, r, s), true

	case objabi.R_LOONG64_TLS_LE_HI,
		objabi.R_LOONG64_TLS_LE_LO,
		objabi.R_CONST,
		objabi.R_GOTOFF,
		objabi.R_CALLLOONG64,
		objabi.R_LOONG64_TLS_IE_HI,
		objabi.R_LOONG64_TLS_IE_LO:
		return ld.ExtrelocSimple(ldr, r), true
	}
	return loader.ExtReloc{}, false
}

// Comments from copying binutils/bfd/elfnn-loongarch.c
//
// For example: pc is 0x11000010000100, symbol is 0x1812348ffff812
//
// offset = (0x1812348ffff812 & ~0xfff) - (0x11000010000100 & ~0xfff)
// offset = 0x712347ffff000
//
// lo12: 0x1812348ffff812 & 0xfff = 0x812
// hi20: 0x7ffff + 0x1(lo12 > 0x7ff) = 0x80000
// lo20: 0x71234 - 0x1(lo12 > 0x7ff) + 0x1(hi20 > 0x7ffff)
// hi12: 0x0
//
// pcalau12i $t1, hi20 (0x80000)
//
//	$t1 = 0x11000010000100 + sign-extend(0x80000 << 12)
//	    = 0x11000010000100 + 0xffffffff80000000
//	    = 0x10ffff90000000
//
// addi.d $t0, $zero, lo12 (0x812)
//
//	$t0 = 0xfffffffffffff812 (if lo12 > 0x7ff, because sign-extend,
//	lo20 need to sub 0x1)
//
// lu32i.d $t0, lo20 (0x71234)
//
//	$t0 = {0x71234, 0xfffff812}
//	    = 0x71234fffff812
//
// lu52i.d $t0, hi12 (0x0)
//
//	$t0 = {0x0, 0x71234fffff812}
//	    = 0x71234fffff812
func pc32RelocBits(reloc objabi.RelocType, tgt int64, pc int64) int64 {
	lo12 := tgt & 0xfff
	if reloc == objabi.R_LOONG64_ADDR_LO {
		return lo12
	}

	off := (tgt & ^0xfff) - (pc & ^0xfff)
	if lo12 >= 0x800 {
		off += 0x1000
	}

	// objabi.R_LOONG64_ADDR_HI
	return (off >> 12) & 0xfffff
}

func pc64RelocBits(reloc objabi.RelocType, tgt int64, pc int64) int64 {
	lo12 := tgt & 0xfff
	off := (tgt & ^0xfff) - (pc & ^0xfff)

	if lo12 >= 0x800 {
		off += (0x1000 - 0x100000000)
	}

	if (off & 0x80000000) != 0 {
		off += 0x100000000
	}

	if reloc == objabi.R_LOONG64_ADDR64_LO {
		return (off >> 32) & 0xfffff
	}

	// objabi.R_LOONG64_ADDR64_HI
	return (off >> 52) & 0xfff
}

// Convert the direct jump relocation r to refer to a trampoline if the target is too far.
func trampoline(ctxt *ld.Link, ldr *loader.Loader, ri int, rs, s loader.Sym) {
	relocs := ldr.Relocs(s)
	r := relocs.At(ri)
	switch r.Type() {
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_LARCH_B26), objabi.R_CALLLOONG64:
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

	o1 := uint32(0x1a000014) // pcalau12i $r20, 0
	ctxt.Arch.ByteOrder.PutUint32(P, o1)
	r1, _ := tramp.AddRel(objabi.R_LOONG64_ADDR_HI)
	r1.SetOff(0)
	r1.SetSiz(4)
	r1.SetSym(target)
	r1.SetAdd(offset)

	o2 := uint32(0x02c00294) // addi.d $r20, $r20, 0
	ctxt.Arch.ByteOrder.PutUint32(P[4:], o2)
	r2, _ := tramp.AddRel(objabi.R_LOONG64_ADDR_LO)
	r2.SetOff(4)
	r2.SetSiz(4)
	r2.SetSym(target)
	r2.SetAdd(offset)

	o3 := uint32(0x4c000280) // jirl $r0, $r20, 0
	ctxt.Arch.ByteOrder.PutUint32(P[8:], o3)

	tramp.SetData(P)
}

func gentrampgot(ctxt *ld.Link, ldr *loader.Loader, tramp *loader.SymbolBuilder, target loader.Sym) {
	tramp.SetSize(12) // 3 instructions
	P := make([]byte, tramp.Size())

	o1 := uint32(0x1a000014) // pcalau12i $r20, 0
	ctxt.Arch.ByteOrder.PutUint32(P, o1)
	r1, _ := tramp.AddRel(objabi.R_LOONG64_GOT_HI)
	r1.SetOff(0)
	r1.SetSiz(4)
	r1.SetSym(target)

	o2 := uint32(0x28c00294) // ld.d $r20, $r20, 0
	ctxt.Arch.ByteOrder.PutUint32(P[4:], o2)
	r2, _ := tramp.AddRel(objabi.R_LOONG64_GOT_LO)
	r2.SetOff(4)
	r2.SetSiz(4)
	r2.SetSym(target)

	o3 := uint32(0x4c000280) // jirl $r0, $r20, 0
	ctxt.Arch.ByteOrder.PutUint32(P[8:], o3)

	tramp.SetData(P)
}
