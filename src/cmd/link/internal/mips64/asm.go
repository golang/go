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

package mips64

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/ld"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/elf"
)

var (
	// dtOffsets contains offsets for entries within the .dynamic section.
	// These are used to fix up symbol values once they are known.
	dtOffsets map[elf.DynTag]int64

	// dynSymCount contains the number of entries in the .dynsym section.
	// This is used to populate the DT_MIPS_SYMTABNO entry in the .dynamic
	// section.
	dynSymCount uint64

	// gotLocalCount contains the number of local global offset table
	// entries. This is used to populate the DT_MIPS_LOCAL_GOTNO entry in
	// the .dynamic section.
	gotLocalCount uint64

	// gotSymIndex contains the index of the first dynamic symbol table
	// entry that corresponds to an entry in the global offset table.
	// This is used to populate the DT_MIPS_GOTSYM entry in the .dynamic
	// section.
	gotSymIndex uint64
)

func gentext(ctxt *ld.Link, ldr *loader.Loader) {
	if *ld.FlagD || ctxt.Target.IsExternal() {
		return
	}

	dynamic := ldr.MakeSymbolUpdater(ctxt.ArchSyms.Dynamic)

	ld.Elfwritedynent(ctxt.Arch, dynamic, elf.DT_MIPS_RLD_VERSION, 1)
	ld.Elfwritedynent(ctxt.Arch, dynamic, elf.DT_MIPS_BASE_ADDRESS, 0)

	// elfsetupplt should have been called and gotLocalCount should now
	// have its correct value.
	if gotLocalCount == 0 {
		ctxt.Errorf(0, "internal error: elfsetupplt has not been called")
	}
	ld.Elfwritedynent(ctxt.Arch, dynamic, elf.DT_MIPS_LOCAL_GOTNO, gotLocalCount)

	// DT_* entries have to exist prior to elfdynhash(), which finalises the
	// table by adding DT_NULL. However, the values for the following entries
	// are not know until after dynreloc() has completed. Add the symbols now,
	// then update their values prior to code generation.
	dts := []elf.DynTag{
		elf.DT_MIPS_SYMTABNO,
		elf.DT_MIPS_GOTSYM,
	}
	dtOffsets = make(map[elf.DynTag]int64)
	for _, dt := range dts {
		ld.Elfwritedynent(ctxt.Arch, dynamic, dt, 0)
		dtOffsets[dt] = dynamic.Size() - 8
	}
}

func adddynrel(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s loader.Sym, r loader.Reloc, rIdx int) bool {
	targ := r.Sym()
	var targType sym.SymKind
	if targ != 0 {
		targType = ldr.SymType(targ)
	}

	if r.Type() >= objabi.ElfRelocOffset {
		ldr.Errorf(s, "unexpected relocation type %d (%s)", r.Type(), sym.RelocName(target.Arch, r.Type()))
		return false
	}

	switch r.Type() {
	case objabi.R_CALLMIPS, objabi.R_JMPMIPS:
		if targType != sym.SDYNIMPORT {
			// Nothing to do, the relocation will be laid out in reloc
			return true
		}
		if target.IsExternal() {
			// External linker will do this relocation.
			return true
		}

		// Internal linking, build a PLT entry and change the relocation
		// target to that entry.
		if r.Add() != 0 {
			ldr.Errorf(s, "PLT call with non-zero addend (%v)", r.Add())
		}
		addpltsym(target, ldr, syms, targ)
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocSym(rIdx, syms.PLT)
		su.SetRelocAdd(rIdx, int64(ldr.SymPlt(targ)))
		return true
	}

	return false
}

func elfreloc1(ctxt *ld.Link, out *ld.OutBuf, ldr *loader.Loader, s loader.Sym, r loader.ExtReloc, ri int, sectoff int64) bool {

	// mips64 ELF relocation (endian neutral)
	//		offset	uint64
	//		sym		uint32
	//		ssym	uint8
	//		type3	uint8
	//		type2	uint8
	//		type	uint8
	//		addend	int64

	addend := r.Xadd

	out.Write64(uint64(sectoff))

	elfsym := ld.ElfSymForReloc(ctxt, r.Xsym)
	out.Write32(uint32(elfsym))
	out.Write8(0)
	out.Write8(0)
	out.Write8(0)
	switch r.Type {
	default:
		return false
	case objabi.R_ADDR, objabi.R_DWARFSECREF:
		switch r.Size {
		case 4:
			out.Write8(uint8(elf.R_MIPS_32))
		case 8:
			out.Write8(uint8(elf.R_MIPS_64))
		default:
			return false
		}
	case objabi.R_ADDRMIPS:
		out.Write8(uint8(elf.R_MIPS_LO16))
	case objabi.R_ADDRMIPSU:
		out.Write8(uint8(elf.R_MIPS_HI16))
	case objabi.R_ADDRMIPSTLS:
		out.Write8(uint8(elf.R_MIPS_TLS_TPREL_LO16))
		if ctxt.Target.IsOpenbsd() {
			// OpenBSD mips64 does not currently offset TLS by 0x7000,
			// as such we need to add this back to get the correct offset
			// via the external linker.
			addend += 0x7000
		}
	case objabi.R_CALLMIPS,
		objabi.R_JMPMIPS:
		out.Write8(uint8(elf.R_MIPS_26))
	}
	out.Write64(uint64(addend))

	return true
}

func elfsetupplt(ctxt *ld.Link, ldr *loader.Loader, plt, gotplt *loader.SymbolBuilder, dynamic loader.Sym) {
	if plt.Size() != 0 {
		return
	}

	// Load resolver address from got[0] into r25.
	plt.AddSymRef(ctxt.Arch, gotplt.Sym(), 0, objabi.R_ADDRMIPSU, 4)
	plt.SetUint32(ctxt.Arch, plt.Size()-4, 0x3c0e0000) // lui   $14, %hi(&GOTPLT[0])
	plt.AddSymRef(ctxt.Arch, gotplt.Sym(), 0, objabi.R_ADDRMIPS, 4)
	plt.SetUint32(ctxt.Arch, plt.Size()-4, 0xddd90000) // ld    $25, %lo(&GOTPLT[0])($14)

	// Load return address into r15, the index of the got.plt entry into r24, then
	// JALR to the resolver. The address of the got.plt entry is currently in r24,
	// which we have to turn into an index.
	plt.AddSymRef(ctxt.Arch, gotplt.Sym(), 0, objabi.R_ADDRMIPS, 4)
	plt.SetUint32(ctxt.Arch, plt.Size()-4, 0x25ce0000) // addiu $14, $14, %lo(&GOTPLT[0])
	plt.AddUint32(ctxt.Arch, 0x030ec023)               // subu  $24, $24, $14
	plt.AddUint32(ctxt.Arch, 0x03e07825)               // move  $15, $31
	plt.AddUint32(ctxt.Arch, 0x0018c0c2)               // srl   $24, $24, 3
	plt.AddUint32(ctxt.Arch, 0x0320f809)               // jalr  $25
	plt.AddUint32(ctxt.Arch, 0x2718fffe)               // subu  $24, $24, 2

	if gotplt.Size() != 0 {
		ctxt.Errorf(gotplt.Sym(), "got.plt is not empty")
	}

	// Reserve got[0] for resolver address (populated by dynamic loader).
	gotplt.AddUint32(ctxt.Arch, 0)
	gotplt.AddUint32(ctxt.Arch, 0)
	gotLocalCount++

	// Reserve got[1] for ELF object pointer (populated by dynamic loader).
	gotplt.AddUint32(ctxt.Arch, 0)
	gotplt.AddUint32(ctxt.Arch, 0)
	gotLocalCount++
}

func addpltsym(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s loader.Sym) {
	if ldr.SymPlt(s) >= 0 {
		return
	}

	dynamic := ldr.MakeSymbolUpdater(syms.Dynamic)

	const dynSymEntrySize = 20
	if gotSymIndex == 0 {
		// Compute and update GOT symbol index.
		gotSymIndex = uint64(ldr.SymSize(syms.DynSym) / dynSymEntrySize)
		dynamic.SetUint(target.Arch, dtOffsets[elf.DT_MIPS_GOTSYM], gotSymIndex)
	}
	if dynSymCount == 0 {
		dynSymCount = uint64(ldr.SymSize(syms.DynSym) / dynSymEntrySize)
	}

	ld.Adddynsym(ldr, target, syms, s)
	dynSymCount++

	if !target.IsElf() {
		ldr.Errorf(s, "addpltsym: unsupported binary format")
	}

	plt := ldr.MakeSymbolUpdater(syms.PLT)
	gotplt := ldr.MakeSymbolUpdater(syms.GOTPLT)
	if plt.Size() == 0 {
		panic("plt is not set up")
	}

	// Load got.plt entry into r25.
	plt.AddSymRef(target.Arch, gotplt.Sym(), gotplt.Size(), objabi.R_ADDRMIPSU, 4)
	plt.SetUint32(target.Arch, plt.Size()-4, 0x3c0f0000) // lui   $15, %hi(.got.plt entry)
	plt.AddSymRef(target.Arch, gotplt.Sym(), gotplt.Size(), objabi.R_ADDRMIPS, 4)
	plt.SetUint32(target.Arch, plt.Size()-4, 0xddf90000) // ld    $25, %lo(.got.plt entry)($15)

	// Load address of got.plt entry into r24 and JALR to address in r25.
	plt.AddUint32(target.Arch, 0x03200008) // jr  $25
	plt.AddSymRef(target.Arch, gotplt.Sym(), gotplt.Size(), objabi.R_ADDRMIPS, 4)
	plt.SetUint32(target.Arch, plt.Size()-4, 0x65f80000) // daddiu $24, $15, %lo(.got.plt entry)

	// Add pointer to plt[0] to got.plt
	gotplt.AddAddrPlus(target.Arch, plt.Sym(), 0)

	ldr.SetPlt(s, int32(plt.Size()-16))

	// Update dynamic symbol count.
	dynamic.SetUint(target.Arch, dtOffsets[elf.DT_MIPS_SYMTABNO], dynSymCount)
}

func machoreloc1(*sys.Arch, *ld.OutBuf, *loader.Loader, loader.Sym, loader.ExtReloc, int64) bool {
	return false
}

func archreloc(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, r loader.Reloc, s loader.Sym, val int64) (o int64, nExtReloc int, ok bool) {
	if target.IsExternal() {
		switch r.Type() {
		default:
			return val, 0, false

		case objabi.R_ADDRMIPS,
			objabi.R_ADDRMIPSU,
			objabi.R_ADDRMIPSTLS,
			objabi.R_CALLMIPS,
			objabi.R_JMPMIPS:
			return val, 1, true
		}
	}

	const isOk = true
	const noExtReloc = 0
	rs := r.Sym()
	switch r.Type() {
	case objabi.R_ADDRMIPS,
		objabi.R_ADDRMIPSU:
		t := ldr.SymValue(rs) + r.Add()
		if r.Type() == objabi.R_ADDRMIPS {
			return int64(val&0xffff0000 | t&0xffff), noExtReloc, isOk
		}
		return int64(val&0xffff0000 | ((t+1<<15)>>16)&0xffff), noExtReloc, isOk
	case objabi.R_ADDRMIPSTLS:
		// thread pointer is at 0x7000 offset from the start of TLS data area
		t := ldr.SymValue(rs) + r.Add() - 0x7000
		if target.IsOpenbsd() {
			// OpenBSD mips64 does not currently offset TLS by 0x7000,
			// as such we need to add this back to get the correct offset.
			t += 0x7000
		}
		if t < -32768 || t >= 32678 {
			ldr.Errorf(s, "TLS offset out of range %d", t)
		}
		return int64(val&0xffff0000 | t&0xffff), noExtReloc, isOk
	case objabi.R_CALLMIPS,
		objabi.R_JMPMIPS:
		// Low 26 bits = (S + A) >> 2
		t := ldr.SymValue(rs) + r.Add()
		return int64(val&0xfc000000 | (t>>2)&^0xfc000000), noExtReloc, isOk
	}

	return val, 0, false
}

func archrelocvariant(*ld.Target, *loader.Loader, loader.Reloc, sym.RelocVariant, loader.Sym, int64, []byte) int64 {
	return -1
}

func extreloc(target *ld.Target, ldr *loader.Loader, r loader.Reloc, s loader.Sym) (loader.ExtReloc, bool) {
	switch r.Type() {
	case objabi.R_ADDRMIPS,
		objabi.R_ADDRMIPSU:
		return ld.ExtrelocViaOuterSym(ldr, r, s), true

	case objabi.R_ADDRMIPSTLS,
		objabi.R_CALLMIPS,
		objabi.R_JMPMIPS:
		return ld.ExtrelocSimple(ldr, r), true
	}
	return loader.ExtReloc{}, false
}
