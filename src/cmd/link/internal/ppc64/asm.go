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

package ppc64

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/ld"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"log"
	"strings"
)

func genplt(ctxt *ld.Link, ldr *loader.Loader) {
	// The ppc64 ABI PLT has similar concepts to other
	// architectures, but is laid out quite differently. When we
	// see an R_PPC64_REL24 relocation to a dynamic symbol
	// (indicating that the call needs to go through the PLT), we
	// generate up to three stubs and reserve a PLT slot.
	//
	// 1) The call site will be bl x; nop (where the relocation
	//    applies to the bl).  We rewrite this to bl x_stub; ld
	//    r2,24(r1).  The ld is necessary because x_stub will save
	//    r2 (the TOC pointer) at 24(r1) (the "TOC save slot").
	//
	// 2) We reserve space for a pointer in the .plt section (once
	//    per referenced dynamic function).  .plt is a data
	//    section filled solely by the dynamic linker (more like
	//    .plt.got on other architectures).  Initially, the
	//    dynamic linker will fill each slot with a pointer to the
	//    corresponding x@plt entry point.
	//
	// 3) We generate the "call stub" x_stub (once per dynamic
	//    function/object file pair).  This saves the TOC in the
	//    TOC save slot, reads the function pointer from x's .plt
	//    slot and calls it like any other global entry point
	//    (including setting r12 to the function address).
	//
	// 4) We generate the "symbol resolver stub" x@plt (once per
	//    dynamic function).  This is solely a branch to the glink
	//    resolver stub.
	//
	// 5) We generate the glink resolver stub (only once).  This
	//    computes which symbol resolver stub we came through and
	//    invokes the dynamic resolver via a pointer provided by
	//    the dynamic linker. This will patch up the .plt slot to
	//    point directly at the function so future calls go
	//    straight from the call stub to the real function, and
	//    then call the function.

	// NOTE: It's possible we could make ppc64 closer to other
	// architectures: ppc64's .plt is like .plt.got on other
	// platforms and ppc64's .glink is like .plt on other
	// platforms.

	// Find all R_PPC64_REL24 relocations that reference dynamic
	// imports. Reserve PLT entries for these symbols and
	// generate call stubs. The call stubs need to live in .text,
	// which is why we need to do this pass this early.
	//
	// This assumes "case 1" from the ABI, where the caller needs
	// us to save and restore the TOC pointer.
	var stubs []loader.Sym
	for _, s := range ctxt.Textp {
		relocs := ldr.Relocs(s)
		for i := 0; i < relocs.Count(); i++ {
			r := relocs.At(i)
			if r.Type() != objabi.ElfRelocOffset+objabi.RelocType(elf.R_PPC64_REL24) || ldr.SymType(r.Sym()) != sym.SDYNIMPORT {
				continue
			}

			// Reserve PLT entry and generate symbol
			// resolver
			addpltsym(ctxt, ldr, r.Sym())

			// Generate call stub. Important to note that we're looking
			// up the stub using the same version as the parent symbol (s),
			// needed so that symtoc() will select the right .TOC. symbol
			// when processing the stub.  In older versions of the linker
			// this was done by setting stub.Outer to the parent, but
			// if the stub has the right version initially this is not needed.
			n := fmt.Sprintf("%s.%s", ldr.SymName(s), ldr.SymName(r.Sym()))
			stub := ldr.CreateSymForUpdate(n, ldr.SymVersion(s))
			if stub.Size() == 0 {
				stubs = append(stubs, stub.Sym())
				gencallstub(ctxt, ldr, 1, stub, r.Sym())
			}

			// Update the relocation to use the call stub
			r.SetSym(stub.Sym())

			// Make the symbol writeable so we can fixup toc.
			su := ldr.MakeSymbolUpdater(s)
			su.MakeWritable()
			p := su.Data()

			// Check for toc restore slot (a nop), and replace with toc restore.
			var nop uint32
			if len(p) >= int(r.Off()+8) {
				nop = ctxt.Arch.ByteOrder.Uint32(p[r.Off()+4:])
			}
			if nop != 0x60000000 {
				ldr.Errorf(s, "Symbol %s is missing toc restoration slot at offset %d", ldr.SymName(s), r.Off()+4)
			}
			const o1 = 0xe8410018 // ld r2,24(r1)
			ctxt.Arch.ByteOrder.PutUint32(p[r.Off()+4:], o1)
		}
	}
	// Put call stubs at the beginning (instead of the end).
	// So when resolving the relocations to calls to the stubs,
	// the addresses are known and trampolines can be inserted
	// when necessary.
	ctxt.Textp = append(stubs, ctxt.Textp...)
}

func genaddmoduledata(ctxt *ld.Link, ldr *loader.Loader) {
	initfunc, addmoduledata := ld.PrepareAddmoduledata(ctxt)
	if initfunc == nil {
		return
	}

	o := func(op uint32) {
		initfunc.AddUint32(ctxt.Arch, op)
	}

	// addis r2, r12, .TOC.-func@ha
	toc := ctxt.DotTOC[0]
	rel1, _ := initfunc.AddRel(objabi.R_ADDRPOWER_PCREL)
	rel1.SetOff(0)
	rel1.SetSiz(8)
	rel1.SetSym(toc)
	o(0x3c4c0000)
	// addi r2, r2, .TOC.-func@l
	o(0x38420000)
	// mflr r31
	o(0x7c0802a6)
	// stdu r31, -32(r1)
	o(0xf801ffe1)
	// addis r3, r2, local.moduledata@got@ha
	var tgt loader.Sym
	if s := ldr.Lookup("local.moduledata", 0); s != 0 {
		tgt = s
	} else if s := ldr.Lookup("local.pluginmoduledata", 0); s != 0 {
		tgt = s
	} else {
		tgt = ldr.LookupOrCreateSym("runtime.firstmoduledata", 0)
	}
	rel2, _ := initfunc.AddRel(objabi.R_ADDRPOWER_GOT)
	rel2.SetOff(int32(initfunc.Size()))
	rel2.SetSiz(8)
	rel2.SetSym(tgt)
	o(0x3c620000)
	// ld r3, local.moduledata@got@l(r3)
	o(0xe8630000)
	// bl runtime.addmoduledata
	rel3, _ := initfunc.AddRel(objabi.R_CALLPOWER)
	rel3.SetOff(int32(initfunc.Size()))
	rel3.SetSiz(4)
	rel3.SetSym(addmoduledata)
	o(0x48000001)
	// nop
	o(0x60000000)
	// ld r31, 0(r1)
	o(0xe8010000)
	// mtlr r31
	o(0x7c0803a6)
	// addi r1,r1,32
	o(0x38210020)
	// blr
	o(0x4e800020)
}

func gentext(ctxt *ld.Link, ldr *loader.Loader) {
	if ctxt.DynlinkingGo() {
		genaddmoduledata(ctxt, ldr)
	}

	if ctxt.LinkMode == ld.LinkInternal {
		genplt(ctxt, ldr)
	}
}

// Construct a call stub in stub that calls symbol targ via its PLT
// entry.
func gencallstub(ctxt *ld.Link, ldr *loader.Loader, abicase int, stub *loader.SymbolBuilder, targ loader.Sym) {
	if abicase != 1 {
		// If we see R_PPC64_TOCSAVE or R_PPC64_REL24_NOTOC
		// relocations, we'll need to implement cases 2 and 3.
		log.Fatalf("gencallstub only implements case 1 calls")
	}

	plt := ctxt.PLT

	stub.SetType(sym.STEXT)

	// Save TOC pointer in TOC save slot
	stub.AddUint32(ctxt.Arch, 0xf8410018) // std r2,24(r1)

	// Load the function pointer from the PLT.
	rel, ri1 := stub.AddRel(objabi.R_POWER_TOC)
	rel.SetOff(int32(stub.Size()))
	rel.SetSiz(2)
	rel.SetAdd(int64(ldr.SymPlt(targ)))
	rel.SetSym(plt)
	if ctxt.Arch.ByteOrder == binary.BigEndian {
		rel.SetOff(rel.Off() + int32(rel.Siz()))
	}
	ldr.SetRelocVariant(stub.Sym(), int(ri1), sym.RV_POWER_HA)
	stub.AddUint32(ctxt.Arch, 0x3d820000) // addis r12,r2,targ@plt@toc@ha

	rel2, ri2 := stub.AddRel(objabi.R_POWER_TOC)
	rel2.SetOff(int32(stub.Size()))
	rel2.SetSiz(2)
	rel2.SetAdd(int64(ldr.SymPlt(targ)))
	rel2.SetSym(plt)
	if ctxt.Arch.ByteOrder == binary.BigEndian {
		rel2.SetOff(rel2.Off() + int32(rel2.Siz()))
	}
	ldr.SetRelocVariant(stub.Sym(), int(ri2), sym.RV_POWER_LO)
	stub.AddUint32(ctxt.Arch, 0xe98c0000) // ld r12,targ@plt@toc@l(r12)

	// Jump to the loaded pointer
	stub.AddUint32(ctxt.Arch, 0x7d8903a6) // mtctr r12
	stub.AddUint32(ctxt.Arch, 0x4e800420) // bctr
}

func adddynrel(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s loader.Sym, r loader.Reloc, rIdx int) bool {
	if target.IsElf() {
		return addelfdynrel(target, ldr, syms, s, r, rIdx)
	} else if target.IsAIX() {
		return ld.Xcoffadddynrel(target, ldr, syms, s, r, rIdx)
	}
	return false
}

func addelfdynrel(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s loader.Sym, r loader.Reloc, rIdx int) bool {
	targ := r.Sym()
	var targType sym.SymKind
	if targ != 0 {
		targType = ldr.SymType(targ)
	}

	switch r.Type() {
	default:
		if r.Type() >= objabi.ElfRelocOffset {
			ldr.Errorf(s, "unexpected relocation type %d (%s)", r.Type(), sym.RelocName(target.Arch, r.Type()))
			return false
		}

		// Handle relocations found in ELF object files.
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL24):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_CALLPOWER)

		// This is a local call, so the caller isn't setting
		// up r12 and r2 is the same for the caller and
		// callee. Hence, we need to go to the local entry
		// point.  (If we don't do this, the callee will try
		// to use r12 to compute r2.)
		su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymLocalentry(targ))*4)

		if targType == sym.SDYNIMPORT {
			// Should have been handled in elfsetupplt
			ldr.Errorf(s, "unexpected R_PPC64_REL24 for dyn import")
		}

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC_REL32):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		su.SetRelocAdd(rIdx, r.Add()+4)

		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected R_PPC_REL32 for dyn import")
		}

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_ADDR64):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ADDR)
		if targType == sym.SDYNIMPORT {
			// These happen in .toc sections
			ld.Adddynsym(ldr, target, syms, targ)

			rela := ldr.MakeSymbolUpdater(syms.Rela)
			rela.AddAddrPlus(target.Arch, s, int64(r.Off()))
			rela.AddUint64(target.Arch, elf.R_INFO(uint32(ldr.SymDynid(targ)), uint32(elf.R_PPC64_ADDR64)))
			rela.AddUint64(target.Arch, uint64(r.Add()))
			su.SetRelocType(rIdx, objabi.ElfRelocOffset) // ignore during relocsym
		} else if target.IsPIE() && target.IsInternal() {
			// For internal linking PIE, this R_ADDR relocation cannot
			// be resolved statically. We need to generate a dynamic
			// relocation. Let the code below handle it.
			break
		}
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_POWER_TOC)
		ldr.SetRelocVariant(s, rIdx, sym.RV_POWER_LO|sym.RV_CHECK_OVERFLOW)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16_LO):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_POWER_TOC)
		ldr.SetRelocVariant(s, rIdx, sym.RV_POWER_LO)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16_HA):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_POWER_TOC)
		ldr.SetRelocVariant(s, rIdx, sym.RV_POWER_HA|sym.RV_CHECK_OVERFLOW)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16_HI):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_POWER_TOC)
		ldr.SetRelocVariant(s, rIdx, sym.RV_POWER_HI|sym.RV_CHECK_OVERFLOW)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16_DS):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_POWER_TOC)
		ldr.SetRelocVariant(s, rIdx, sym.RV_POWER_DS|sym.RV_CHECK_OVERFLOW)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16_LO_DS):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_POWER_TOC)
		ldr.SetRelocVariant(s, rIdx, sym.RV_POWER_DS)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL16_LO):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		ldr.SetRelocVariant(s, rIdx, sym.RV_POWER_LO)
		su.SetRelocAdd(rIdx, r.Add()+2) // Compensate for relocation size of 2
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL16_HI):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		ldr.SetRelocVariant(s, rIdx, sym.RV_POWER_HI|sym.RV_CHECK_OVERFLOW)
		su.SetRelocAdd(rIdx, r.Add()+2)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL16_HA):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		ldr.SetRelocVariant(s, rIdx, sym.RV_POWER_HA|sym.RV_CHECK_OVERFLOW)
		su.SetRelocAdd(rIdx, r.Add()+2)
		return true
	}

	// Handle references to ELF symbols from our own object files.
	relocs := ldr.Relocs(s)
	r = relocs.At(rIdx)

	switch r.Type() {
	case objabi.R_ADDR:
		if ldr.SymType(s) == sym.STEXT {
			log.Fatalf("R_ADDR relocation in text symbol %s is unsupported\n", ldr.SymName(s))
		}
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
		// Generate R_PPC64_RELATIVE relocations for best
		// efficiency in the dynamic linker.
		//
		// As noted above, symbol addresses have not been
		// assigned yet, so we can't generate the final reloc
		// entry yet. We ultimately want:
		//
		// r_offset = s + r.Off
		// r_info = R_PPC64_RELATIVE
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
			rela.AddUint64(target.Arch, elf.R_INFO(0, uint32(elf.R_PPC64_RELATIVE)))
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

	return false
}

func xcoffreloc1(arch *sys.Arch, out *ld.OutBuf, ldr *loader.Loader, s loader.Sym, r loader.ExtReloc, sectoff int64) bool {
	rs := r.Xsym

	emitReloc := func(v uint16, off uint64) {
		out.Write64(uint64(sectoff) + off)
		out.Write32(uint32(ldr.SymDynid(rs)))
		out.Write16(v)
	}

	var v uint16
	switch r.Type {
	default:
		return false
	case objabi.R_ADDR, objabi.R_DWARFSECREF:
		v = ld.XCOFF_R_POS
		if r.Size == 4 {
			v |= 0x1F << 8
		} else {
			v |= 0x3F << 8
		}
		emitReloc(v, 0)
	case objabi.R_ADDRPOWER_TOCREL:
	case objabi.R_ADDRPOWER_TOCREL_DS:
		emitReloc(ld.XCOFF_R_TOCU|(0x0F<<8), 2)
		emitReloc(ld.XCOFF_R_TOCL|(0x0F<<8), 6)
	case objabi.R_POWER_TLS_LE:
		// This only supports 16b relocations.  It is fixed up in archreloc.
		emitReloc(ld.XCOFF_R_TLS_LE|0x0F<<8, 2)
	case objabi.R_CALLPOWER:
		if r.Size != 4 {
			return false
		}
		emitReloc(ld.XCOFF_R_RBR|0x19<<8, 0)
	case objabi.R_XCOFFREF:
		emitReloc(ld.XCOFF_R_REF|0x3F<<8, 0)
	}
	return true

}

func elfreloc1(ctxt *ld.Link, out *ld.OutBuf, ldr *loader.Loader, s loader.Sym, r loader.ExtReloc, ri int, sectoff int64) bool {
	// Beware that bit0~bit15 start from the third byte of a instruction in Big-Endian machines.
	rt := r.Type
	if rt == objabi.R_ADDR || rt == objabi.R_POWER_TLS || rt == objabi.R_CALLPOWER {
	} else {
		if ctxt.Arch.ByteOrder == binary.BigEndian {
			sectoff += 2
		}
	}
	out.Write64(uint64(sectoff))

	elfsym := ld.ElfSymForReloc(ctxt, r.Xsym)
	switch rt {
	default:
		return false
	case objabi.R_ADDR, objabi.R_DWARFSECREF:
		switch r.Size {
		case 4:
			out.Write64(uint64(elf.R_PPC64_ADDR32) | uint64(elfsym)<<32)
		case 8:
			out.Write64(uint64(elf.R_PPC64_ADDR64) | uint64(elfsym)<<32)
		default:
			return false
		}
	case objabi.R_POWER_TLS:
		out.Write64(uint64(elf.R_PPC64_TLS) | uint64(elfsym)<<32)
	case objabi.R_POWER_TLS_LE:
		out.Write64(uint64(elf.R_PPC64_TPREL16_HA) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))
		out.Write64(uint64(sectoff + 4))
		out.Write64(uint64(elf.R_PPC64_TPREL16_LO) | uint64(elfsym)<<32)
	case objabi.R_POWER_TLS_IE:
		out.Write64(uint64(elf.R_PPC64_GOT_TPREL16_HA) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))
		out.Write64(uint64(sectoff + 4))
		out.Write64(uint64(elf.R_PPC64_GOT_TPREL16_LO_DS) | uint64(elfsym)<<32)
	case objabi.R_ADDRPOWER:
		out.Write64(uint64(elf.R_PPC64_ADDR16_HA) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))
		out.Write64(uint64(sectoff + 4))
		out.Write64(uint64(elf.R_PPC64_ADDR16_LO) | uint64(elfsym)<<32)
	case objabi.R_ADDRPOWER_DS:
		out.Write64(uint64(elf.R_PPC64_ADDR16_HA) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))
		out.Write64(uint64(sectoff + 4))
		out.Write64(uint64(elf.R_PPC64_ADDR16_LO_DS) | uint64(elfsym)<<32)
	case objabi.R_ADDRPOWER_GOT:
		out.Write64(uint64(elf.R_PPC64_GOT16_HA) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))
		out.Write64(uint64(sectoff + 4))
		out.Write64(uint64(elf.R_PPC64_GOT16_LO_DS) | uint64(elfsym)<<32)
	case objabi.R_ADDRPOWER_PCREL:
		out.Write64(uint64(elf.R_PPC64_REL16_HA) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))
		out.Write64(uint64(sectoff + 4))
		out.Write64(uint64(elf.R_PPC64_REL16_LO) | uint64(elfsym)<<32)
		r.Xadd += 4
	case objabi.R_ADDRPOWER_TOCREL:
		out.Write64(uint64(elf.R_PPC64_TOC16_HA) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))
		out.Write64(uint64(sectoff + 4))
		out.Write64(uint64(elf.R_PPC64_TOC16_LO) | uint64(elfsym)<<32)
	case objabi.R_ADDRPOWER_TOCREL_DS:
		out.Write64(uint64(elf.R_PPC64_TOC16_HA) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))
		out.Write64(uint64(sectoff + 4))
		out.Write64(uint64(elf.R_PPC64_TOC16_LO_DS) | uint64(elfsym)<<32)
	case objabi.R_CALLPOWER:
		if r.Size != 4 {
			return false
		}
		out.Write64(uint64(elf.R_PPC64_REL24) | uint64(elfsym)<<32)

	}
	out.Write64(uint64(r.Xadd))

	return true
}

func elfsetupplt(ctxt *ld.Link, plt, got *loader.SymbolBuilder, dynamic loader.Sym) {
	if plt.Size() == 0 {
		// The dynamic linker stores the address of the
		// dynamic resolver and the DSO identifier in the two
		// doublewords at the beginning of the .plt section
		// before the PLT array. Reserve space for these.
		plt.SetSize(16)
	}
}

func machoreloc1(*sys.Arch, *ld.OutBuf, *loader.Loader, loader.Sym, loader.ExtReloc, int64) bool {
	return false
}

// Return the value of .TOC. for symbol s
func symtoc(ldr *loader.Loader, syms *ld.ArchSyms, s loader.Sym) int64 {
	v := ldr.SymVersion(s)
	if out := ldr.OuterSym(s); out != 0 {
		v = ldr.SymVersion(out)
	}

	toc := syms.DotTOC[v]
	if toc == 0 {
		ldr.Errorf(s, "TOC-relative relocation in object without .TOC.")
		return 0
	}

	return ldr.SymValue(toc)
}

// archreloctoc relocates a TOC relative symbol.
func archreloctoc(ldr *loader.Loader, target *ld.Target, syms *ld.ArchSyms, r loader.Reloc, s loader.Sym, val int64) int64 {
	rs := r.Sym()
	var o1, o2 uint32
	var t int64
	useAddi := false

	if target.IsBigEndian() {
		o1 = uint32(val >> 32)
		o2 = uint32(val)
	} else {
		o1 = uint32(val)
		o2 = uint32(val >> 32)
	}

	// On AIX, TOC data accesses are always made indirectly against R2 (a sequence of addis+ld+load/store). If the
	// The target of the load is known, the sequence can be written into addis+addi+load/store. On Linux,
	// TOC data accesses are always made directly against R2 (e.g addis+load/store).
	if target.IsAIX() {
		if !strings.HasPrefix(ldr.SymName(rs), "TOC.") {
			ldr.Errorf(s, "archreloctoc called for a symbol without TOC anchor")
		}
		relocs := ldr.Relocs(rs)
		tarSym := relocs.At(0).Sym()

		if target.IsInternal() && tarSym != 0 && ldr.AttrReachable(tarSym) && ldr.SymSect(tarSym).Seg == &ld.Segdata {
			t = ldr.SymValue(tarSym) + r.Add() - ldr.SymValue(syms.TOC)
			// change ld to addi in the second instruction
			o2 = (o2 & 0x03FF0000) | 0xE<<26
			useAddi = true
		} else {
			t = ldr.SymValue(rs) + r.Add() - ldr.SymValue(syms.TOC)
		}
	} else {
		t = ldr.SymValue(rs) + r.Add() - symtoc(ldr, syms, s)
	}

	if t != int64(int32(t)) {
		ldr.Errorf(s, "TOC relocation for %s is too big to relocate %s: 0x%x", ldr.SymName(s), rs, t)
	}

	if t&0x8000 != 0 {
		t += 0x10000
	}

	o1 |= uint32((t >> 16) & 0xFFFF)

	switch r.Type() {
	case objabi.R_ADDRPOWER_TOCREL_DS:
		if useAddi {
			o2 |= uint32(t) & 0xFFFF
		} else {
			if t&3 != 0 {
				ldr.Errorf(s, "bad DS reloc for %s: %d", ldr.SymName(s), ldr.SymValue(rs))
			}
			o2 |= uint32(t) & 0xFFFC
		}
	case objabi.R_ADDRPOWER_TOCREL:
		o2 |= uint32(t) & 0xffff
	default:
		return -1
	}

	if target.IsBigEndian() {
		return int64(o1)<<32 | int64(o2)
	}
	return int64(o2)<<32 | int64(o1)
}

// archrelocaddr relocates a symbol address.
// This code is for linux only.
func archrelocaddr(ldr *loader.Loader, target *ld.Target, syms *ld.ArchSyms, r loader.Reloc, s loader.Sym, val int64) int64 {
	rs := r.Sym()
	if target.IsAIX() {
		ldr.Errorf(s, "archrelocaddr called for %s relocation\n", ldr.SymName(rs))
	}
	var o1, o2 uint32
	if target.IsBigEndian() {
		o1 = uint32(val >> 32)
		o2 = uint32(val)
	} else {
		o1 = uint32(val)
		o2 = uint32(val >> 32)
	}

	// We are spreading a 31-bit address across two instructions, putting the
	// high (adjusted) part in the low 16 bits of the first instruction and the
	// low part in the low 16 bits of the second instruction, or, in the DS case,
	// bits 15-2 (inclusive) of the address into bits 15-2 of the second
	// instruction (it is an error in this case if the low 2 bits of the address
	// are non-zero).

	t := ldr.SymAddr(rs) + r.Add()
	if t < 0 || t >= 1<<31 {
		ldr.Errorf(s, "relocation for %s is too big (>=2G): 0x%x", ldr.SymName(s), ldr.SymValue(rs))
	}
	if t&0x8000 != 0 {
		t += 0x10000
	}

	switch r.Type() {
	case objabi.R_ADDRPOWER:
		o1 |= (uint32(t) >> 16) & 0xffff
		o2 |= uint32(t) & 0xffff
	case objabi.R_ADDRPOWER_DS:
		o1 |= (uint32(t) >> 16) & 0xffff
		if t&3 != 0 {
			ldr.Errorf(s, "bad DS reloc for %s: %d", ldr.SymName(s), ldr.SymValue(rs))
		}
		o2 |= uint32(t) & 0xfffc
	default:
		return -1
	}

	if target.IsBigEndian() {
		return int64(o1)<<32 | int64(o2)
	}
	return int64(o2)<<32 | int64(o1)
}

// Determine if the code was compiled so that the TOC register R2 is initialized and maintained
func r2Valid(ctxt *ld.Link) bool {
	switch ctxt.BuildMode {
	case ld.BuildModeCArchive, ld.BuildModeCShared, ld.BuildModePIE, ld.BuildModeShared, ld.BuildModePlugin:
		return true
	}
	// -linkshared option
	return ctxt.IsSharedGoLink()
}

// resolve direct jump relocation r in s, and add trampoline if necessary
func trampoline(ctxt *ld.Link, ldr *loader.Loader, ri int, rs, s loader.Sym) {

	// Trampolines are created if the branch offset is too large and the linker cannot insert a call stub to handle it.
	// For internal linking, trampolines are always created for long calls.
	// For external linking, the linker can insert a call stub to handle a long call, but depends on having the TOC address in
	// r2.  For those build modes with external linking where the TOC address is not maintained in r2, trampolines must be created.
	if ctxt.IsExternal() && r2Valid(ctxt) {
		// The TOC pointer is valid. The external linker will insert trampolines.
		return
	}

	relocs := ldr.Relocs(s)
	r := relocs.At(ri)
	var t int64
	// ldr.SymValue(rs) == 0 indicates a cross-package jump to a function that is not yet
	// laid out. Conservatively use a trampoline. This should be rare, as we lay out packages
	// in dependency order.
	if ldr.SymValue(rs) != 0 {
		t = ldr.SymValue(rs) + r.Add() - (ldr.SymValue(s) + int64(r.Off()))
	}
	switch r.Type() {
	case objabi.R_CALLPOWER:

		// If branch offset is too far then create a trampoline.

		if (ctxt.IsExternal() && ldr.SymSect(s) != ldr.SymSect(rs)) || (ctxt.IsInternal() && int64(int32(t<<6)>>6) != t) || ldr.SymValue(rs) == 0 || (*ld.FlagDebugTramp > 1 && ldr.SymPkg(s) != ldr.SymPkg(rs)) {
			var tramp loader.Sym
			for i := 0; ; i++ {

				// Using r.Add as part of the name is significant in functions like duffzero where the call
				// target is at some offset within the function.  Calls to duff+8 and duff+256 must appear as
				// distinct trampolines.

				oName := ldr.SymName(rs)
				name := oName
				if r.Add() == 0 {
					name += fmt.Sprintf("-tramp%d", i)
				} else {
					name += fmt.Sprintf("%+x-tramp%d", r.Add(), i)
				}

				// Look up the trampoline in case it already exists

				tramp = ldr.LookupOrCreateSym(name, int(ldr.SymVersion(rs)))
				if oName == "runtime.deferreturn" {
					ldr.SetIsDeferReturnTramp(tramp, true)
				}
				if ldr.SymValue(tramp) == 0 {
					break
				}

				t = ldr.SymValue(tramp) + r.Add() - (ldr.SymValue(s) + int64(r.Off()))

				// With internal linking, the trampoline can be used if it is not too far.
				// With external linking, the trampoline must be in this section for it to be reused.
				if (ctxt.IsInternal() && int64(int32(t<<6)>>6) == t) || (ctxt.IsExternal() && ldr.SymSect(s) == ldr.SymSect(tramp)) {
					break
				}
			}
			if ldr.SymType(tramp) == 0 {
				trampb := ldr.MakeSymbolUpdater(tramp)
				ctxt.AddTramp(trampb)
				gentramp(ctxt, ldr, trampb, rs, r.Add())
			}
			sb := ldr.MakeSymbolUpdater(s)
			relocs := sb.Relocs()
			r := relocs.At(ri)
			r.SetSym(tramp)
			r.SetAdd(0) // This was folded into the trampoline target address
		}
	default:
		ctxt.Errorf(s, "trampoline called with non-jump reloc: %d (%s)", r.Type(), sym.RelocName(ctxt.Arch, r.Type()))
	}
}

func gentramp(ctxt *ld.Link, ldr *loader.Loader, tramp *loader.SymbolBuilder, target loader.Sym, offset int64) {
	tramp.SetSize(16) // 4 instructions
	P := make([]byte, tramp.Size())
	var o1, o2 uint32

	if ctxt.IsAIX() {
		// On AIX, the address is retrieved with a TOC symbol.
		// For internal linking, the "Linux" way might still be used.
		// However, all text symbols are accessed with a TOC symbol as
		// text relocations aren't supposed to be possible.
		// So, keep using the external linking way to be more AIX friendly.
		o1 = uint32(0x3c000000) | 12<<21 | 2<<16  // addis r12,  r2, toctargetaddr hi
		o2 = uint32(0xe8000000) | 12<<21 | 12<<16 // ld    r12, r12, toctargetaddr lo

		toctramp := ldr.CreateSymForUpdate("TOC."+ldr.SymName(tramp.Sym()), 0)
		toctramp.SetType(sym.SXCOFFTOC)
		toctramp.AddAddrPlus(ctxt.Arch, target, offset)

		r, _ := tramp.AddRel(objabi.R_ADDRPOWER_TOCREL_DS)
		r.SetOff(0)
		r.SetSiz(8) // generates 2 relocations: HA + LO
		r.SetSym(toctramp.Sym())
	} else {
		// Used for default build mode for an executable
		// Address of the call target is generated using
		// relocation and doesn't depend on r2 (TOC).
		o1 = uint32(0x3c000000) | 12<<21          // lis  r12,targetaddr hi
		o2 = uint32(0x38000000) | 12<<21 | 12<<16 // addi r12,r12,targetaddr lo

		t := ldr.SymValue(target)
		if t == 0 || r2Valid(ctxt) || ctxt.IsExternal() {
			// Target address is unknown, generate relocations
			r, _ := tramp.AddRel(objabi.R_ADDRPOWER)
			if r2Valid(ctxt) {
				// Use a TOC relative address if R2 holds the TOC pointer
				o1 |= uint32(2 << 16) // Transform lis r31,ha into addis r31,r2,ha
				r.SetType(objabi.R_ADDRPOWER_TOCREL)
			}
			r.SetOff(0)
			r.SetSiz(8) // generates 2 relocations: HA + LO
			r.SetSym(target)
			r.SetAdd(offset)
		} else {
			// The target address is known, resolve it
			t += offset
			o1 |= (uint32(t) + 0x8000) >> 16 // HA
			o2 |= uint32(t) & 0xFFFF         // LO
		}
	}

	o3 := uint32(0x7c0903a6) | 12<<21 // mtctr r12
	o4 := uint32(0x4e800420)          // bctr
	ctxt.Arch.ByteOrder.PutUint32(P, o1)
	ctxt.Arch.ByteOrder.PutUint32(P[4:], o2)
	ctxt.Arch.ByteOrder.PutUint32(P[8:], o3)
	ctxt.Arch.ByteOrder.PutUint32(P[12:], o4)
	tramp.SetData(P)
}

func archreloc(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, r loader.Reloc, s loader.Sym, val int64) (relocatedOffset int64, nExtReloc int, ok bool) {
	rs := r.Sym()
	if target.IsExternal() {
		// On AIX, relocations (except TLS ones) must be also done to the
		// value with the current addresses.
		switch rt := r.Type(); rt {
		default:
			if !target.IsAIX() {
				return val, nExtReloc, false
			}
		case objabi.R_POWER_TLS:
			nExtReloc = 1
			return val, nExtReloc, true
		case objabi.R_POWER_TLS_LE, objabi.R_POWER_TLS_IE:
			if target.IsAIX() && rt == objabi.R_POWER_TLS_LE {
				// Fixup val, an addis/addi pair of instructions, which generate a 32b displacement
				// from the threadpointer (R13), into a 16b relocation. XCOFF only supports 16b
				// TLS LE relocations. Likewise, verify this is an addis/addi sequence.
				const expectedOpcodes = 0x3C00000038000000
				const expectedOpmasks = 0xFC000000FC000000
				if uint64(val)&expectedOpmasks != expectedOpcodes {
					ldr.Errorf(s, "relocation for %s+%d is not an addis/addi pair: %16x", ldr.SymName(rs), r.Off(), uint64(val))
				}
				nval := (int64(uint32(0x380d0000)) | val&0x03e00000) << 32 // addi rX, r13, $0
				nval |= int64(0x60000000)                                  // nop
				val = nval
				nExtReloc = 1
			} else {
				nExtReloc = 2
			}
			return val, nExtReloc, true
		case objabi.R_ADDRPOWER,
			objabi.R_ADDRPOWER_DS,
			objabi.R_ADDRPOWER_TOCREL,
			objabi.R_ADDRPOWER_TOCREL_DS,
			objabi.R_ADDRPOWER_GOT,
			objabi.R_ADDRPOWER_PCREL:
			nExtReloc = 2 // need two ELF relocations, see elfreloc1
			if !target.IsAIX() {
				return val, nExtReloc, true
			}
		case objabi.R_CALLPOWER:
			nExtReloc = 1
			if !target.IsAIX() {
				return val, nExtReloc, true
			}
		}
	}

	switch r.Type() {
	case objabi.R_ADDRPOWER_TOCREL, objabi.R_ADDRPOWER_TOCREL_DS:
		return archreloctoc(ldr, target, syms, r, s, val), nExtReloc, true
	case objabi.R_ADDRPOWER, objabi.R_ADDRPOWER_DS:
		return archrelocaddr(ldr, target, syms, r, s, val), nExtReloc, true
	case objabi.R_CALLPOWER:
		// Bits 6 through 29 = (S + A - P) >> 2

		t := ldr.SymValue(rs) + r.Add() - (ldr.SymValue(s) + int64(r.Off()))

		tgtName := ldr.SymName(rs)

		// If we are linking PIE or shared code, all golang generated object files have an extra 2 instruction prologue
		// to regenerate the TOC pointer from R12.  The exception are two special case functions tested below.  Note,
		// local call offsets for externally generated objects are accounted for when converting into golang relocs.
		if !ldr.AttrExternal(rs) && ldr.AttrShared(rs) && tgtName != "runtime.duffzero" && tgtName != "runtime.duffcopy" {
			// Furthermore, only apply the offset if the target looks like the start of a function call.
			if r.Add() == 0 && ldr.SymType(rs) == sym.STEXT {
				t += 8
			}
		}

		if t&3 != 0 {
			ldr.Errorf(s, "relocation for %s+%d is not aligned: %d", ldr.SymName(rs), r.Off(), t)
		}
		// If branch offset is too far then create a trampoline.

		if int64(int32(t<<6)>>6) != t {
			ldr.Errorf(s, "direct call too far: %s %x", ldr.SymName(rs), t)
		}
		return val | int64(uint32(t)&^0xfc000003), nExtReloc, true
	case objabi.R_POWER_TOC: // S + A - .TOC.
		return ldr.SymValue(rs) + r.Add() - symtoc(ldr, syms, s), nExtReloc, true

	case objabi.R_ADDRPOWER_PCREL: // S + A - P
		t := ldr.SymValue(rs) + r.Add() - (ldr.SymValue(s) + int64(r.Off()))
		ha := uint16(((t + 0x8000) >> 16) & 0xFFFF)
		l := uint16(t)
		if target.IsBigEndian() {
			val |= int64(l)
			val |= int64(ha) << 32
		} else {
			val |= int64(ha)
			val |= int64(l) << 32
		}
		return val, nExtReloc, true

	case objabi.R_POWER_TLS:
		const OP_ADD = 31<<26 | 266<<1
		const MASK_OP_ADD = 0x3F<<26 | 0x1FF<<1
		if val&MASK_OP_ADD != OP_ADD {
			ldr.Errorf(s, "R_POWER_TLS reloc only supports XO form ADD, not %08X", val)
		}
		// Verify RB is R13 in ADD RA,RB,RT.
		if (val>>11)&0x1F != 13 {
			// If external linking is made to support this, it may expect the linker to rewrite RB.
			ldr.Errorf(s, "R_POWER_TLS reloc requires R13 in RB (%08X).", uint32(val))
		}
		return val, nExtReloc, true

	case objabi.R_POWER_TLS_IE:
		// Convert TLS_IE relocation to TLS_LE if supported.
		if !(target.IsPIE() && target.IsElf()) {
			log.Fatalf("cannot handle R_POWER_TLS_IE (sym %s) when linking non-PIE, non-ELF binaries internally", ldr.SymName(s))
		}

		// We are an ELF binary, we can safely convert to TLS_LE from:
		// addis to, r2, x@got@tprel@ha
		// ld to, to, x@got@tprel@l(to)
		//
		// to TLS_LE by converting to:
		// addis to, r0, x@tprel@ha
		// addi to, to, x@tprel@l(to)

		const OP_ADDI = 14 << 26
		const OP_MASK = 0x3F << 26
		const OP_RA_MASK = 0x1F << 16
		uval := uint64(val)
		// convert r2 to r0, and ld to addi
		if target.IsBigEndian() {
			uval = uval &^ (OP_RA_MASK << 32)
			uval = (uval &^ OP_MASK) | OP_ADDI
		} else {
			uval = uval &^ (OP_RA_MASK)
			uval = (uval &^ (OP_MASK << 32)) | (OP_ADDI << 32)
		}
		val = int64(uval)
		// Treat this like an R_POWER_TLS_LE relocation now.
		fallthrough

	case objabi.R_POWER_TLS_LE:
		// The thread pointer points 0x7000 bytes after the start of the
		// thread local storage area as documented in section "3.7.2 TLS
		// Runtime Handling" of "Power Architecture 64-Bit ELF V2 ABI
		// Specification".
		v := ldr.SymValue(rs) - 0x7000
		if target.IsAIX() {
			// On AIX, the thread pointer points 0x7800 bytes after
			// the TLS.
			v -= 0x800
		}

		var o1, o2 uint32
		if int64(int32(v)) != v {
			ldr.Errorf(s, "TLS offset out of range %d", v)
		}
		if target.IsBigEndian() {
			o1 = uint32(val >> 32)
			o2 = uint32(val)
		} else {
			o1 = uint32(val)
			o2 = uint32(val >> 32)
		}

		o1 |= uint32(((v + 0x8000) >> 16) & 0xFFFF)
		o2 |= uint32(v & 0xFFFF)

		if target.IsBigEndian() {
			return int64(o1)<<32 | int64(o2), nExtReloc, true
		}
		return int64(o2)<<32 | int64(o1), nExtReloc, true
	}

	return val, nExtReloc, false
}

func archrelocvariant(target *ld.Target, ldr *loader.Loader, r loader.Reloc, rv sym.RelocVariant, s loader.Sym, t int64, p []byte) (relocatedOffset int64) {
	rs := r.Sym()
	switch rv & sym.RV_TYPE_MASK {
	default:
		ldr.Errorf(s, "unexpected relocation variant %d", rv)
		fallthrough

	case sym.RV_NONE:
		return t

	case sym.RV_POWER_LO:
		if rv&sym.RV_CHECK_OVERFLOW != 0 {
			// Whether to check for signed or unsigned
			// overflow depends on the instruction
			var o1 uint32
			if target.IsBigEndian() {
				o1 = binary.BigEndian.Uint32(p[r.Off()-2:])

			} else {
				o1 = binary.LittleEndian.Uint32(p[r.Off():])
			}
			switch o1 >> 26 {
			case 24, // ori
				26, // xori
				28: // andi
				if t>>16 != 0 {
					goto overflow
				}

			default:
				if int64(int16(t)) != t {
					goto overflow
				}
			}
		}

		return int64(int16(t))

	case sym.RV_POWER_HA:
		t += 0x8000
		fallthrough

		// Fallthrough
	case sym.RV_POWER_HI:
		t >>= 16

		if rv&sym.RV_CHECK_OVERFLOW != 0 {
			// Whether to check for signed or unsigned
			// overflow depends on the instruction
			var o1 uint32
			if target.IsBigEndian() {
				o1 = binary.BigEndian.Uint32(p[r.Off()-2:])
			} else {
				o1 = binary.LittleEndian.Uint32(p[r.Off():])
			}
			switch o1 >> 26 {
			case 25, // oris
				27, // xoris
				29: // andis
				if t>>16 != 0 {
					goto overflow
				}

			default:
				if int64(int16(t)) != t {
					goto overflow
				}
			}
		}

		return int64(int16(t))

	case sym.RV_POWER_DS:
		var o1 uint32
		if target.IsBigEndian() {
			o1 = uint32(binary.BigEndian.Uint16(p[r.Off():]))
		} else {
			o1 = uint32(binary.LittleEndian.Uint16(p[r.Off():]))
		}
		if t&3 != 0 {
			ldr.Errorf(s, "relocation for %s+%d is not aligned: %d", ldr.SymName(rs), r.Off(), t)
		}
		if (rv&sym.RV_CHECK_OVERFLOW != 0) && int64(int16(t)) != t {
			goto overflow
		}
		return int64(o1)&0x3 | int64(int16(t))
	}

overflow:
	ldr.Errorf(s, "relocation for %s+%d is too big: %d", ldr.SymName(rs), r.Off(), t)
	return t
}

func extreloc(target *ld.Target, ldr *loader.Loader, r loader.Reloc, s loader.Sym) (loader.ExtReloc, bool) {
	switch r.Type() {
	case objabi.R_POWER_TLS, objabi.R_POWER_TLS_LE, objabi.R_POWER_TLS_IE, objabi.R_CALLPOWER:
		return ld.ExtrelocSimple(ldr, r), true
	case objabi.R_ADDRPOWER,
		objabi.R_ADDRPOWER_DS,
		objabi.R_ADDRPOWER_TOCREL,
		objabi.R_ADDRPOWER_TOCREL_DS,
		objabi.R_ADDRPOWER_GOT,
		objabi.R_ADDRPOWER_PCREL:
		return ld.ExtrelocViaOuterSym(ldr, r, s), true
	}
	return loader.ExtReloc{}, false
}

func addpltsym(ctxt *ld.Link, ldr *loader.Loader, s loader.Sym) {
	if ldr.SymPlt(s) >= 0 {
		return
	}

	ld.Adddynsym(ldr, &ctxt.Target, &ctxt.ArchSyms, s)

	if ctxt.IsELF {
		plt := ldr.MakeSymbolUpdater(ctxt.PLT)
		rela := ldr.MakeSymbolUpdater(ctxt.RelaPLT)
		if plt.Size() == 0 {
			panic("plt is not set up")
		}

		// Create the glink resolver if necessary
		glink := ensureglinkresolver(ctxt, ldr)

		// Write symbol resolver stub (just a branch to the
		// glink resolver stub)
		rel, _ := glink.AddRel(objabi.R_CALLPOWER)
		rel.SetOff(int32(glink.Size()))
		rel.SetSiz(4)
		rel.SetSym(glink.Sym())
		glink.AddUint32(ctxt.Arch, 0x48000000) // b .glink

		// In the ppc64 ABI, the dynamic linker is responsible
		// for writing the entire PLT.  We just need to
		// reserve 8 bytes for each PLT entry and generate a
		// JMP_SLOT dynamic relocation for it.
		//
		// TODO(austin): ABI v1 is different
		ldr.SetPlt(s, int32(plt.Size()))

		plt.Grow(plt.Size() + 8)
		plt.SetSize(plt.Size() + 8)

		rela.AddAddrPlus(ctxt.Arch, plt.Sym(), int64(ldr.SymPlt(s)))
		rela.AddUint64(ctxt.Arch, elf.R_INFO(uint32(ldr.SymDynid(s)), uint32(elf.R_PPC64_JMP_SLOT)))
		rela.AddUint64(ctxt.Arch, 0)
	} else {
		ctxt.Errorf(s, "addpltsym: unsupported binary format")
	}
}

// Generate the glink resolver stub if necessary and return the .glink section
func ensureglinkresolver(ctxt *ld.Link, ldr *loader.Loader) *loader.SymbolBuilder {
	glink := ldr.CreateSymForUpdate(".glink", 0)
	if glink.Size() != 0 {
		return glink
	}

	// This is essentially the resolver from the ppc64 ELFv2 ABI.
	// At entry, r12 holds the address of the symbol resolver stub
	// for the target routine and the argument registers hold the
	// arguments for the target routine.
	//
	// PC-rel offsets are computed once the final codesize of the
	// resolver is known.
	//
	// This stub is PIC, so first get the PC of label 1 into r11.
	glink.AddUint32(ctxt.Arch, 0x7c0802a6) // mflr r0
	glink.AddUint32(ctxt.Arch, 0x429f0005) // bcl 20,31,1f
	glink.AddUint32(ctxt.Arch, 0x7d6802a6) // 1: mflr r11
	glink.AddUint32(ctxt.Arch, 0x7c0803a6) // mtlr r0

	// Compute the .plt array index from the entry point address
	// into r0. This is computed relative to label 1 above.
	glink.AddUint32(ctxt.Arch, 0x38000000) // li r0,-(res_0-1b)
	glink.AddUint32(ctxt.Arch, 0x7c006214) // add r0,r0,r12
	glink.AddUint32(ctxt.Arch, 0x7c0b0050) // sub r0,r0,r11
	glink.AddUint32(ctxt.Arch, 0x7800f082) // srdi r0,r0,2

	// Load the PC-rel offset of ".plt - 1b", and add it to 1b.
	// This is stored after this stub and before the resolvers.
	glink.AddUint32(ctxt.Arch, 0xe98b0000) // ld r12,res_0-1b-8(r11)
	glink.AddUint32(ctxt.Arch, 0x7d6b6214) // add r11,r11,r12

	// Load r12 = dynamic resolver address and r11 = DSO
	// identifier from the first two doublewords of the PLT.
	glink.AddUint32(ctxt.Arch, 0xe98b0000) // ld r12,0(r11)
	glink.AddUint32(ctxt.Arch, 0xe96b0008) // ld r11,8(r11)

	// Jump to the dynamic resolver
	glink.AddUint32(ctxt.Arch, 0x7d8903a6) // mtctr r12
	glink.AddUint32(ctxt.Arch, 0x4e800420) // bctr

	// Store the PC-rel offset to the PLT
	r, _ := glink.AddRel(objabi.R_PCREL)
	r.SetSym(ctxt.PLT)
	r.SetSiz(8)
	r.SetOff(int32(glink.Size()))
	r.SetAdd(glink.Size())        // Adjust the offset to be relative to label 1 above.
	glink.AddUint64(ctxt.Arch, 0) // The offset to the PLT.

	// Resolve PC-rel offsets above now the final size of the stub is known.
	res0m1b := glink.Size() - 8 // res_0 - 1b
	glink.SetUint32(ctxt.Arch, 16, 0x38000000|uint32(uint16(-res0m1b)))
	glink.SetUint32(ctxt.Arch, 32, 0xe98b0000|uint32(uint16(res0m1b-8)))

	// The symbol resolvers must immediately follow.
	//   res_0:

	// Add DT_PPC64_GLINK .dynamic entry, which points to 32 bytes
	// before the first symbol resolver stub.
	du := ldr.MakeSymbolUpdater(ctxt.Dynamic)
	ld.Elfwritedynentsymplus(ctxt, du, elf.DT_PPC64_GLINK, glink.Sym(), glink.Size()-32)

	return glink
}
