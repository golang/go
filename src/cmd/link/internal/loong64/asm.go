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
	"log"
)

func gentext(ctxt *ld.Link, ldr *loader.Loader) {}

func adddynrel(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s loader.Sym, r loader.Reloc, rIdx int) bool {
	log.Fatalf("adddynrel not implemented")
	return false
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
	case objabi.R_ADDRLOONG64TLS:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_PUSH_TLS_TPREL) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))

		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_PUSH_ABSOLUTE))
		out.Write64(uint64(0xfff))

		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_AND))
		out.Write64(uint64(0x0))

		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_POP_32_U_10_12))
		out.Write64(uint64(0x0))

	case objabi.R_ADDRLOONG64TLSU:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_PUSH_TLS_TPREL) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))

		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_PUSH_ABSOLUTE))
		out.Write64(uint64(0xc))

		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_SR))
		out.Write64(uint64(0x0))

		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_POP_32_S_5_20) | uint64(0)<<32)
		out.Write64(uint64(0x0))

	case objabi.R_CALLLOONG64:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_PUSH_PLT_PCREL) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))

		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_POP_32_S_0_10_10_16_S2))
		out.Write64(uint64(0x0))
	// The pcaddu12i + addi.d instructions is used to obtain address of a symbol on Loong64.
	// The low 12-bit of the symbol address need to be added. The addi.d instruction have
	// signed 12-bit immediate operand. The 0x800 (addr+U12 <=> addr+0x800+S12) is introduced
	// to do sign extending from 12 bits. The 0x804 is 0x800 + 4, 4 is instruction bit
	// width on Loong64 and is used to correct the PC of the addi.d instruction.
	case objabi.R_ADDRLOONG64:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_PUSH_PCREL) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd + 0x4))

		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_PUSH_PCREL) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd + 0x804))

		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_PUSH_ABSOLUTE))
		out.Write64(uint64(0xc))

		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_SR))
		out.Write64(uint64(0x0))

		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_PUSH_ABSOLUTE))
		out.Write64(uint64(0xc))

		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_SL))
		out.Write64(uint64(0x0))

		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_SUB))
		out.Write64(uint64(0x0))

		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_POP_32_S_10_12))
		out.Write64(uint64(0x0))

	case objabi.R_ADDRLOONG64U:
		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_PUSH_PCREL) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd + 0x800))

		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_PUSH_ABSOLUTE))
		out.Write64(uint64(0xc))

		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_SR))
		out.Write64(uint64(0x0))

		out.Write64(uint64(sectoff))
		out.Write64(uint64(elf.R_LARCH_SOP_POP_32_S_5_20) | uint64(0)<<32)
		out.Write64(uint64(0x0))
	}

	return true
}

func elfsetupplt(ctxt *ld.Link, plt, gotplt *loader.SymbolBuilder, dynamic loader.Sym) {
	return
}

func machoreloc1(*sys.Arch, *ld.OutBuf, *loader.Loader, loader.Sym, loader.ExtReloc, int64) bool {
	return false
}

func archreloc(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, r loader.Reloc, s loader.Sym, val int64) (o int64, nExtReloc int, ok bool) {
	rs := r.Sym()
	if target.IsExternal() {
		nExtReloc := 0
		switch r.Type() {
		default:
			return val, 0, false
		case objabi.R_ADDRLOONG64,
			objabi.R_ADDRLOONG64U:
			// set up addend for eventual relocation via outer symbol.
			rs, _ := ld.FoldSubSymbolOffset(ldr, rs)
			rst := ldr.SymType(rs)
			if rst != sym.SHOSTOBJ && rst != sym.SDYNIMPORT && ldr.SymSect(rs) == nil {
				ldr.Errorf(s, "missing section for %s", ldr.SymName(rs))
			}
			nExtReloc = 8 // need 8 ELF relocations. see elfreloc1
			if r.Type() == objabi.R_ADDRLOONG64U {
				nExtReloc = 4
			}
			return val, nExtReloc, true
		case objabi.R_ADDRLOONG64TLS,
			objabi.R_ADDRLOONG64TLSU,
			objabi.R_CALLLOONG64,
			objabi.R_JMPLOONG64:
			nExtReloc = 4
			if r.Type() == objabi.R_CALLLOONG64 || r.Type() == objabi.R_JMPLOONG64 {
				nExtReloc = 2
			}
			return val, nExtReloc, true
		}
	}

	const isOk = true
	const noExtReloc = 0

	switch r.Type() {
	case objabi.R_CONST:
		return r.Add(), noExtReloc, isOk
	case objabi.R_GOTOFF:
		return ldr.SymValue(r.Sym()) + r.Add() - ldr.SymValue(syms.GOT), noExtReloc, isOk
	case objabi.R_ADDRLOONG64,
		objabi.R_ADDRLOONG64U:
		pc := ldr.SymValue(s) + int64(r.Off())
		t := ldr.SymAddr(rs) + r.Add() - pc
		if r.Type() == objabi.R_ADDRLOONG64 {
			return int64(val&0xffc003ff | (((t + 4 - ((t + 4 + 1<<11) >> 12 << 12)) << 10) & 0x3ffc00)), noExtReloc, isOk
		}
		return int64(val&0xfe00001f | (((t + 1<<11) >> 12 << 5) & 0x1ffffe0)), noExtReloc, isOk
	case objabi.R_ADDRLOONG64TLS,
		objabi.R_ADDRLOONG64TLSU:
		t := ldr.SymAddr(rs) + r.Add()
		if r.Type() == objabi.R_ADDRLOONG64TLS {
			return int64(val&0xffc003ff | ((t & 0xfff) << 10)), noExtReloc, isOk
		}
		return int64(val&0xfe00001f | (((t) >> 12 << 5) & 0x1ffffe0)), noExtReloc, isOk
	case objabi.R_CALLLOONG64,
		objabi.R_JMPLOONG64:
		pc := ldr.SymValue(s) + int64(r.Off())
		t := ldr.SymAddr(rs) + r.Add() - pc
		return int64(val&0xfc000000 | (((t >> 2) & 0xffff) << 10) | (((t >> 2) & 0x3ff0000) >> 16)), noExtReloc, isOk
	}

	return val, 0, false
}

func archrelocvariant(*ld.Target, *loader.Loader, loader.Reloc, sym.RelocVariant, loader.Sym, int64, []byte) int64 {
	return -1
}

func extreloc(target *ld.Target, ldr *loader.Loader, r loader.Reloc, s loader.Sym) (loader.ExtReloc, bool) {
	switch r.Type() {
	case objabi.R_ADDRLOONG64,
		objabi.R_ADDRLOONG64U:
		return ld.ExtrelocViaOuterSym(ldr, r, s), true

	case objabi.R_ADDRLOONG64TLS,
		objabi.R_ADDRLOONG64TLSU,
		objabi.R_CONST,
		objabi.R_GOTOFF,
		objabi.R_CALLLOONG64,
		objabi.R_JMPLOONG64:
		return ld.ExtrelocSimple(ldr, r), true
	}
	return loader.ExtReloc{}, false
}
