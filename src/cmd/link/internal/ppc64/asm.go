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

package ppc64

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/ld"
	"cmd/link/internal/sym"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"log"
	"strings"
)

func genplt(ctxt *ld.Link) {
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
	var stubs []*sym.Symbol
	for _, s := range ctxt.Textp {
		for i := range s.R {
			r := &s.R[i]
			if r.Type != objabi.ElfRelocOffset+objabi.RelocType(elf.R_PPC64_REL24) || r.Sym.Type != sym.SDYNIMPORT {
				continue
			}

			// Reserve PLT entry and generate symbol
			// resolver
			addpltsym(ctxt, r.Sym)

			// Generate call stub
			n := fmt.Sprintf("%s.%s", s.Name, r.Sym.Name)

			stub := ctxt.Syms.Lookup(n, 0)
			if s.Attr.Reachable() {
				stub.Attr |= sym.AttrReachable
			}
			if stub.Size == 0 {
				// Need outer to resolve .TOC.
				stub.Outer = s
				stubs = append(stubs, stub)
				gencallstub(ctxt, 1, stub, r.Sym)
			}

			// Update the relocation to use the call stub
			r.Sym = stub

			// Restore TOC after bl. The compiler put a
			// nop here for us to overwrite.
			const o1 = 0xe8410018 // ld r2,24(r1)
			ctxt.Arch.ByteOrder.PutUint32(s.P[r.Off+4:], o1)
		}
	}
	// Put call stubs at the beginning (instead of the end).
	// So when resolving the relocations to calls to the stubs,
	// the addresses are known and trampolines can be inserted
	// when necessary.
	ctxt.Textp = append(stubs, ctxt.Textp...)
}

func genaddmoduledata(ctxt *ld.Link) {
	addmoduledata := ctxt.Syms.ROLookup("runtime.addmoduledata", sym.SymVerABI0)
	if addmoduledata.Type == sym.STEXT && ctxt.BuildMode != ld.BuildModePlugin {
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
	// addis r2, r12, .TOC.-func@ha
	rel := initfunc.AddRel()
	rel.Off = int32(initfunc.Size)
	rel.Siz = 8
	rel.Sym = ctxt.Syms.Lookup(".TOC.", 0)
	rel.Sym.Attr |= sym.AttrReachable
	rel.Type = objabi.R_ADDRPOWER_PCREL
	o(0x3c4c0000)
	// addi r2, r2, .TOC.-func@l
	o(0x38420000)
	// mflr r31
	o(0x7c0802a6)
	// stdu r31, -32(r1)
	o(0xf801ffe1)
	// addis r3, r2, local.moduledata@got@ha
	rel = initfunc.AddRel()
	rel.Off = int32(initfunc.Size)
	rel.Siz = 8
	if s := ctxt.Syms.ROLookup("local.moduledata", 0); s != nil {
		rel.Sym = s
	} else if s := ctxt.Syms.ROLookup("local.pluginmoduledata", 0); s != nil {
		rel.Sym = s
	} else {
		rel.Sym = ctxt.Syms.Lookup("runtime.firstmoduledata", 0)
	}
	rel.Sym.Attr |= sym.AttrReachable
	rel.Sym.Attr |= sym.AttrLocal
	rel.Type = objabi.R_ADDRPOWER_GOT
	o(0x3c620000)
	// ld r3, local.moduledata@got@l(r3)
	o(0xe8630000)
	// bl runtime.addmoduledata
	rel = initfunc.AddRel()
	rel.Off = int32(initfunc.Size)
	rel.Siz = 4
	rel.Sym = addmoduledata
	rel.Type = objabi.R_CALLPOWER
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

	if ctxt.BuildMode == ld.BuildModePlugin {
		ctxt.Textp = append(ctxt.Textp, addmoduledata)
	}
	initarray_entry := ctxt.Syms.Lookup("go.link.addmoduledatainit", 0)
	ctxt.Textp = append(ctxt.Textp, initfunc)
	initarray_entry.Attr |= sym.AttrReachable
	initarray_entry.Attr |= sym.AttrLocal
	initarray_entry.Type = sym.SINITARR
	initarray_entry.AddAddr(ctxt.Arch, initfunc)
}

func gentext(ctxt *ld.Link) {
	if ctxt.DynlinkingGo() {
		genaddmoduledata(ctxt)
	}

	if ctxt.LinkMode == ld.LinkInternal {
		genplt(ctxt)
	}
}

// Construct a call stub in stub that calls symbol targ via its PLT
// entry.
func gencallstub(ctxt *ld.Link, abicase int, stub *sym.Symbol, targ *sym.Symbol) {
	if abicase != 1 {
		// If we see R_PPC64_TOCSAVE or R_PPC64_REL24_NOTOC
		// relocations, we'll need to implement cases 2 and 3.
		log.Fatalf("gencallstub only implements case 1 calls")
	}

	plt := ctxt.Syms.Lookup(".plt", 0)

	stub.Type = sym.STEXT

	// Save TOC pointer in TOC save slot
	stub.AddUint32(ctxt.Arch, 0xf8410018) // std r2,24(r1)

	// Load the function pointer from the PLT.
	r := stub.AddRel()

	r.Off = int32(stub.Size)
	r.Sym = plt
	r.Add = int64(targ.Plt())
	r.Siz = 2
	if ctxt.Arch.ByteOrder == binary.BigEndian {
		r.Off += int32(r.Siz)
	}
	r.Type = objabi.R_POWER_TOC
	r.Variant = sym.RV_POWER_HA
	stub.AddUint32(ctxt.Arch, 0x3d820000) // addis r12,r2,targ@plt@toc@ha
	r = stub.AddRel()
	r.Off = int32(stub.Size)
	r.Sym = plt
	r.Add = int64(targ.Plt())
	r.Siz = 2
	if ctxt.Arch.ByteOrder == binary.BigEndian {
		r.Off += int32(r.Siz)
	}
	r.Type = objabi.R_POWER_TOC
	r.Variant = sym.RV_POWER_LO
	stub.AddUint32(ctxt.Arch, 0xe98c0000) // ld r12,targ@plt@toc@l(r12)

	// Jump to the loaded pointer
	stub.AddUint32(ctxt.Arch, 0x7d8903a6) // mtctr r12
	stub.AddUint32(ctxt.Arch, 0x4e800420) // bctr
}

func adddynrel(ctxt *ld.Link, s *sym.Symbol, r *sym.Reloc) bool {
	if ctxt.IsELF {
		return addelfdynrel(ctxt, s, r)
	} else if ctxt.HeadType == objabi.Haix {
		return ld.Xcoffadddynrel(ctxt, s, r)
	}
	return false
}
func addelfdynrel(ctxt *ld.Link, s *sym.Symbol, r *sym.Reloc) bool {
	targ := r.Sym
	r.InitExt()

	switch r.Type {
	default:
		if r.Type >= objabi.ElfRelocOffset {
			ld.Errorf(s, "unexpected relocation type %d (%s)", r.Type, sym.RelocName(ctxt.Arch, r.Type))
			return false
		}

		// Handle relocations found in ELF object files.
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL24):
		r.Type = objabi.R_CALLPOWER

		// This is a local call, so the caller isn't setting
		// up r12 and r2 is the same for the caller and
		// callee. Hence, we need to go to the local entry
		// point.  (If we don't do this, the callee will try
		// to use r12 to compute r2.)
		r.Add += int64(r.Sym.Localentry()) * 4

		if targ.Type == sym.SDYNIMPORT {
			// Should have been handled in elfsetupplt
			ld.Errorf(s, "unexpected R_PPC64_REL24 for dyn import")
		}

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC_REL32):
		r.Type = objabi.R_PCREL
		r.Add += 4

		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_PPC_REL32 for dyn import")
		}

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_ADDR64):
		r.Type = objabi.R_ADDR
		if targ.Type == sym.SDYNIMPORT {
			// These happen in .toc sections
			ld.Adddynsym(ctxt, targ)

			rela := ctxt.Syms.Lookup(".rela", 0)
			rela.AddAddrPlus(ctxt.Arch, s, int64(r.Off))
			rela.AddUint64(ctxt.Arch, ld.ELF64_R_INFO(uint32(targ.Dynid), uint32(elf.R_PPC64_ADDR64)))
			rela.AddUint64(ctxt.Arch, uint64(r.Add))
			r.Type = objabi.ElfRelocOffset // ignore during relocsym
		}

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16):
		r.Type = objabi.R_POWER_TOC
		r.Variant = sym.RV_POWER_LO | sym.RV_CHECK_OVERFLOW
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16_LO):
		r.Type = objabi.R_POWER_TOC
		r.Variant = sym.RV_POWER_LO
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16_HA):
		r.Type = objabi.R_POWER_TOC
		r.Variant = sym.RV_POWER_HA | sym.RV_CHECK_OVERFLOW
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16_HI):
		r.Type = objabi.R_POWER_TOC
		r.Variant = sym.RV_POWER_HI | sym.RV_CHECK_OVERFLOW
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16_DS):
		r.Type = objabi.R_POWER_TOC
		r.Variant = sym.RV_POWER_DS | sym.RV_CHECK_OVERFLOW
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16_LO_DS):
		r.Type = objabi.R_POWER_TOC
		r.Variant = sym.RV_POWER_DS
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL16_LO):
		r.Type = objabi.R_PCREL
		r.Variant = sym.RV_POWER_LO
		r.Add += 2 // Compensate for relocation size of 2
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL16_HI):
		r.Type = objabi.R_PCREL
		r.Variant = sym.RV_POWER_HI | sym.RV_CHECK_OVERFLOW
		r.Add += 2
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL16_HA):
		r.Type = objabi.R_PCREL
		r.Variant = sym.RV_POWER_HA | sym.RV_CHECK_OVERFLOW
		r.Add += 2
		return true
	}

	// Handle references to ELF symbols from our own object files.
	if targ.Type != sym.SDYNIMPORT {
		return true
	}

	// TODO(austin): Translate our relocations to ELF

	return false
}

func xcoffreloc1(arch *sys.Arch, out *ld.OutBuf, s *sym.Symbol, r *sym.Reloc, sectoff int64) bool {
	rs := r.Xsym

	emitReloc := func(v uint16, off uint64) {
		out.Write64(uint64(sectoff) + off)
		out.Write32(uint32(rs.Dynid))
		out.Write16(v)
	}

	var v uint16
	switch r.Type {
	default:
		return false
	case objabi.R_ADDR:
		v = ld.XCOFF_R_POS
		if r.Siz == 4 {
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
		emitReloc(ld.XCOFF_R_TLS_LE|0x0F<<8, 2)
	case objabi.R_CALLPOWER:
		if r.Siz != 4 {
			return false
		}
		emitReloc(ld.XCOFF_R_RBR|0x19<<8, 0)
	case objabi.R_XCOFFREF:
		emitReloc(ld.XCOFF_R_REF|0x3F<<8, 0)

	}
	return true

}

func elfreloc1(ctxt *ld.Link, r *sym.Reloc, sectoff int64) bool {
	// Beware that bit0~bit15 start from the third byte of a instruction in Big-Endian machines.
	if r.Type == objabi.R_ADDR || r.Type == objabi.R_POWER_TLS || r.Type == objabi.R_CALLPOWER {
	} else {
		if ctxt.Arch.ByteOrder == binary.BigEndian {
			sectoff += 2
		}
	}
	ctxt.Out.Write64(uint64(sectoff))

	elfsym := r.Xsym.ElfsymForReloc()
	switch r.Type {
	default:
		return false
	case objabi.R_ADDR:
		switch r.Siz {
		case 4:
			ctxt.Out.Write64(uint64(elf.R_PPC64_ADDR32) | uint64(elfsym)<<32)
		case 8:
			ctxt.Out.Write64(uint64(elf.R_PPC64_ADDR64) | uint64(elfsym)<<32)
		default:
			return false
		}
	case objabi.R_POWER_TLS:
		ctxt.Out.Write64(uint64(elf.R_PPC64_TLS) | uint64(elfsym)<<32)
	case objabi.R_POWER_TLS_LE:
		ctxt.Out.Write64(uint64(elf.R_PPC64_TPREL16) | uint64(elfsym)<<32)
	case objabi.R_POWER_TLS_IE:
		ctxt.Out.Write64(uint64(elf.R_PPC64_GOT_TPREL16_HA) | uint64(elfsym)<<32)
		ctxt.Out.Write64(uint64(r.Xadd))
		ctxt.Out.Write64(uint64(sectoff + 4))
		ctxt.Out.Write64(uint64(elf.R_PPC64_GOT_TPREL16_LO_DS) | uint64(elfsym)<<32)
	case objabi.R_ADDRPOWER:
		ctxt.Out.Write64(uint64(elf.R_PPC64_ADDR16_HA) | uint64(elfsym)<<32)
		ctxt.Out.Write64(uint64(r.Xadd))
		ctxt.Out.Write64(uint64(sectoff + 4))
		ctxt.Out.Write64(uint64(elf.R_PPC64_ADDR16_LO) | uint64(elfsym)<<32)
	case objabi.R_ADDRPOWER_DS:
		ctxt.Out.Write64(uint64(elf.R_PPC64_ADDR16_HA) | uint64(elfsym)<<32)
		ctxt.Out.Write64(uint64(r.Xadd))
		ctxt.Out.Write64(uint64(sectoff + 4))
		ctxt.Out.Write64(uint64(elf.R_PPC64_ADDR16_LO_DS) | uint64(elfsym)<<32)
	case objabi.R_ADDRPOWER_GOT:
		ctxt.Out.Write64(uint64(elf.R_PPC64_GOT16_HA) | uint64(elfsym)<<32)
		ctxt.Out.Write64(uint64(r.Xadd))
		ctxt.Out.Write64(uint64(sectoff + 4))
		ctxt.Out.Write64(uint64(elf.R_PPC64_GOT16_LO_DS) | uint64(elfsym)<<32)
	case objabi.R_ADDRPOWER_PCREL:
		ctxt.Out.Write64(uint64(elf.R_PPC64_REL16_HA) | uint64(elfsym)<<32)
		ctxt.Out.Write64(uint64(r.Xadd))
		ctxt.Out.Write64(uint64(sectoff + 4))
		ctxt.Out.Write64(uint64(elf.R_PPC64_REL16_LO) | uint64(elfsym)<<32)
		r.Xadd += 4
	case objabi.R_ADDRPOWER_TOCREL:
		ctxt.Out.Write64(uint64(elf.R_PPC64_TOC16_HA) | uint64(elfsym)<<32)
		ctxt.Out.Write64(uint64(r.Xadd))
		ctxt.Out.Write64(uint64(sectoff + 4))
		ctxt.Out.Write64(uint64(elf.R_PPC64_TOC16_LO) | uint64(elfsym)<<32)
	case objabi.R_ADDRPOWER_TOCREL_DS:
		ctxt.Out.Write64(uint64(elf.R_PPC64_TOC16_HA) | uint64(elfsym)<<32)
		ctxt.Out.Write64(uint64(r.Xadd))
		ctxt.Out.Write64(uint64(sectoff + 4))
		ctxt.Out.Write64(uint64(elf.R_PPC64_TOC16_LO_DS) | uint64(elfsym)<<32)
	case objabi.R_CALLPOWER:
		if r.Siz != 4 {
			return false
		}
		ctxt.Out.Write64(uint64(elf.R_PPC64_REL24) | uint64(elfsym)<<32)

	}
	ctxt.Out.Write64(uint64(r.Xadd))

	return true
}

func elfsetupplt(ctxt *ld.Link) {
	plt := ctxt.Syms.Lookup(".plt", 0)
	if plt.Size == 0 {
		// The dynamic linker stores the address of the
		// dynamic resolver and the DSO identifier in the two
		// doublewords at the beginning of the .plt section
		// before the PLT array. Reserve space for these.
		plt.Size = 16
	}
}

func machoreloc1(arch *sys.Arch, out *ld.OutBuf, s *sym.Symbol, r *sym.Reloc, sectoff int64) bool {
	return false
}

// Return the value of .TOC. for symbol s
func symtoc(ctxt *ld.Link, s *sym.Symbol) int64 {
	var toc *sym.Symbol

	if s.Outer != nil {
		toc = ctxt.Syms.ROLookup(".TOC.", int(s.Outer.Version))
	} else {
		toc = ctxt.Syms.ROLookup(".TOC.", int(s.Version))
	}

	if toc == nil {
		ld.Errorf(s, "TOC-relative relocation in object without .TOC.")
		return 0
	}

	return toc.Value
}

// archreloctoc relocates a TOC relative symbol.
// If the symbol pointed by this TOC relative symbol is in .data or .bss, the
// default load instruction can be changed to an addi instruction and the
// symbol address can be used directly.
// This code is for AIX only.
func archreloctoc(ctxt *ld.Link, r *sym.Reloc, s *sym.Symbol, val int64) int64 {
	if ctxt.HeadType == objabi.Hlinux {
		ld.Errorf(s, "archrelocaddr called for %s relocation\n", r.Sym.Name)
	}
	var o1, o2 uint32

	o1 = uint32(val >> 32)
	o2 = uint32(val)

	var t int64
	useAddi := false
	const prefix = "TOC."
	var tarSym *sym.Symbol
	if strings.HasPrefix(r.Sym.Name, prefix) {
		tarSym = r.Sym.R[0].Sym
	} else {
		ld.Errorf(s, "archreloctoc called for a symbol without TOC anchor")
	}

	if ctxt.LinkMode == ld.LinkInternal && tarSym != nil && tarSym.Attr.Reachable() && (tarSym.Sect.Seg == &ld.Segdata) {
		t = ld.Symaddr(tarSym) + r.Add - ctxt.Syms.ROLookup("TOC", 0).Value
		// change ld to addi in the second instruction
		o2 = (o2 & 0x03FF0000) | 0xE<<26
		useAddi = true
	} else {
		t = ld.Symaddr(r.Sym) + r.Add - ctxt.Syms.ROLookup("TOC", 0).Value
	}

	if t != int64(int32(t)) {
		ld.Errorf(s, "TOC relocation for %s is too big to relocate %s: 0x%x", s.Name, r.Sym, t)
	}

	if t&0x8000 != 0 {
		t += 0x10000
	}

	o1 |= uint32((t >> 16) & 0xFFFF)

	switch r.Type {
	case objabi.R_ADDRPOWER_TOCREL_DS:
		if useAddi {
			o2 |= uint32(t) & 0xFFFF
		} else {
			if t&3 != 0 {
				ld.Errorf(s, "bad DS reloc for %s: %d", s.Name, ld.Symaddr(r.Sym))
			}
			o2 |= uint32(t) & 0xFFFC
		}
	default:
		return -1
	}

	return int64(o1)<<32 | int64(o2)
}

// archrelocaddr relocates a symbol address.
// This code is for AIX only.
func archrelocaddr(ctxt *ld.Link, r *sym.Reloc, s *sym.Symbol, val int64) int64 {
	if ctxt.HeadType == objabi.Haix {
		ld.Errorf(s, "archrelocaddr called for %s relocation\n", r.Sym.Name)
	}
	var o1, o2 uint32
	if ctxt.Arch.ByteOrder == binary.BigEndian {
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

	t := ld.Symaddr(r.Sym) + r.Add
	if t < 0 || t >= 1<<31 {
		ld.Errorf(s, "relocation for %s is too big (>=2G): 0x%x", s.Name, ld.Symaddr(r.Sym))
	}
	if t&0x8000 != 0 {
		t += 0x10000
	}

	switch r.Type {
	case objabi.R_ADDRPOWER:
		o1 |= (uint32(t) >> 16) & 0xffff
		o2 |= uint32(t) & 0xffff
	case objabi.R_ADDRPOWER_DS:
		o1 |= (uint32(t) >> 16) & 0xffff
		if t&3 != 0 {
			ld.Errorf(s, "bad DS reloc for %s: %d", s.Name, ld.Symaddr(r.Sym))
		}
		o2 |= uint32(t) & 0xfffc
	default:
		return -1
	}

	if ctxt.Arch.ByteOrder == binary.BigEndian {
		return int64(o1)<<32 | int64(o2)
	}
	return int64(o2)<<32 | int64(o1)
}

// resolve direct jump relocation r in s, and add trampoline if necessary
func trampoline(ctxt *ld.Link, r *sym.Reloc, s *sym.Symbol) {

	// Trampolines are created if the branch offset is too large and the linker cannot insert a call stub to handle it.
	// For internal linking, trampolines are always created for long calls.
	// For external linking, the linker can insert a call stub to handle a long call, but depends on having the TOC address in
	// r2.  For those build modes with external linking where the TOC address is not maintained in r2, trampolines must be created.
	if ctxt.LinkMode == ld.LinkExternal && (ctxt.DynlinkingGo() || ctxt.BuildMode == ld.BuildModeCArchive || ctxt.BuildMode == ld.BuildModeCShared || ctxt.BuildMode == ld.BuildModePIE) {
		// No trampolines needed since r2 contains the TOC
		return
	}

	t := ld.Symaddr(r.Sym) + r.Add - (s.Value + int64(r.Off))
	switch r.Type {
	case objabi.R_CALLPOWER:

		// If branch offset is too far then create a trampoline.

		if (ctxt.LinkMode == ld.LinkExternal && s.Sect != r.Sym.Sect) || (ctxt.LinkMode == ld.LinkInternal && int64(int32(t<<6)>>6) != t) || (*ld.FlagDebugTramp > 1 && s.File != r.Sym.File) {
			var tramp *sym.Symbol
			for i := 0; ; i++ {

				// Using r.Add as part of the name is significant in functions like duffzero where the call
				// target is at some offset within the function.  Calls to duff+8 and duff+256 must appear as
				// distinct trampolines.

				name := r.Sym.Name
				if r.Add == 0 {
					name = name + fmt.Sprintf("-tramp%d", i)
				} else {
					name = name + fmt.Sprintf("%+x-tramp%d", r.Add, i)
				}

				// Look up the trampoline in case it already exists

				tramp = ctxt.Syms.Lookup(name, int(r.Sym.Version))
				if tramp.Value == 0 {
					break
				}

				t = ld.Symaddr(tramp) + r.Add - (s.Value + int64(r.Off))

				// With internal linking, the trampoline can be used if it is not too far.
				// With external linking, the trampoline must be in this section for it to be reused.
				if (ctxt.LinkMode == ld.LinkInternal && int64(int32(t<<6)>>6) == t) || (ctxt.LinkMode == ld.LinkExternal && s.Sect == tramp.Sect) {
					break
				}
			}
			if tramp.Type == 0 {
				if ctxt.DynlinkingGo() || ctxt.BuildMode == ld.BuildModeCArchive || ctxt.BuildMode == ld.BuildModeCShared || ctxt.BuildMode == ld.BuildModePIE {
					// Should have returned for above cases
					ld.Errorf(s, "unexpected trampoline for shared or dynamic linking\n")
				} else {
					ctxt.AddTramp(tramp)
					gentramp(ctxt, tramp, r.Sym, r.Add)
				}
			}
			r.Sym = tramp
			r.Add = 0 // This was folded into the trampoline target address
			r.Done = false
		}
	default:
		ld.Errorf(s, "trampoline called with non-jump reloc: %d (%s)", r.Type, sym.RelocName(ctxt.Arch, r.Type))
	}
}

func gentramp(ctxt *ld.Link, tramp, target *sym.Symbol, offset int64) {
	tramp.Size = 16 // 4 instructions
	tramp.P = make([]byte, tramp.Size)
	t := ld.Symaddr(target) + offset
	var o1, o2 uint32

	if ctxt.HeadType == objabi.Haix {
		// On AIX, the address is retrieved with a TOC symbol.
		// For internal linking, the "Linux" way might still be used.
		// However, all text symbols are accessed with a TOC symbol as
		// text relocations aren't supposed to be possible.
		// So, keep using the external linking way to be more AIX friendly.
		o1 = uint32(0x3fe20000) // lis r2, toctargetaddr hi
		o2 = uint32(0xebff0000) // ld r31, toctargetaddr lo

		toctramp := ctxt.Syms.Lookup("TOC."+tramp.Name, 0)
		toctramp.Type = sym.SXCOFFTOC
		toctramp.Attr |= sym.AttrReachable
		toctramp.AddAddr(ctxt.Arch, target)

		tr := tramp.AddRel()
		tr.Off = 0
		tr.Type = objabi.R_ADDRPOWER_TOCREL_DS
		tr.Siz = 8 // generates 2 relocations:  HA + LO
		tr.Sym = toctramp
		tr.Add = offset
	} else {
		// Used for default build mode for an executable
		// Address of the call target is generated using
		// relocation and doesn't depend on r2 (TOC).
		o1 = uint32(0x3fe00000) // lis r31,targetaddr hi
		o2 = uint32(0x3bff0000) // addi r31,targetaddr lo

		// With external linking, the target address must be
		// relocated using LO and HA
		if ctxt.LinkMode == ld.LinkExternal {
			tr := tramp.AddRel()
			tr.Off = 0
			tr.Type = objabi.R_ADDRPOWER
			tr.Siz = 8 // generates 2 relocations:  HA + LO
			tr.Sym = target
			tr.Add = offset

		} else {
			// adjustment needed if lo has sign bit set
			// when using addi to compute address
			val := uint32((t & 0xffff0000) >> 16)
			if t&0x8000 != 0 {
				val += 1
			}
			o1 |= val                // hi part of addr
			o2 |= uint32(t & 0xffff) // lo part of addr
		}
	}

	o3 := uint32(0x7fe903a6) // mtctr r31
	o4 := uint32(0x4e800420) // bctr
	ctxt.Arch.ByteOrder.PutUint32(tramp.P, o1)
	ctxt.Arch.ByteOrder.PutUint32(tramp.P[4:], o2)
	ctxt.Arch.ByteOrder.PutUint32(tramp.P[8:], o3)
	ctxt.Arch.ByteOrder.PutUint32(tramp.P[12:], o4)
}

func archreloc(ctxt *ld.Link, r *sym.Reloc, s *sym.Symbol, val int64) (int64, bool) {
	if ctxt.LinkMode == ld.LinkExternal {
		// On AIX, relocations (except TLS ones) must be also done to the
		// value with the current addresses.
		switch r.Type {
		default:
			if ctxt.HeadType != objabi.Haix {
				return val, false
			}
		case objabi.R_POWER_TLS, objabi.R_POWER_TLS_LE, objabi.R_POWER_TLS_IE:
			r.Done = false
			// check Outer is nil, Type is TLSBSS?
			r.Xadd = r.Add
			r.Xsym = r.Sym
			return val, true
		case objabi.R_ADDRPOWER,
			objabi.R_ADDRPOWER_DS,
			objabi.R_ADDRPOWER_TOCREL,
			objabi.R_ADDRPOWER_TOCREL_DS,
			objabi.R_ADDRPOWER_GOT,
			objabi.R_ADDRPOWER_PCREL:
			r.Done = false

			// set up addend for eventual relocation via outer symbol.
			rs := r.Sym
			r.Xadd = r.Add
			for rs.Outer != nil {
				r.Xadd += ld.Symaddr(rs) - ld.Symaddr(rs.Outer)
				rs = rs.Outer
			}

			if rs.Type != sym.SHOSTOBJ && rs.Type != sym.SDYNIMPORT && rs.Type != sym.SUNDEFEXT && rs.Sect == nil {
				ld.Errorf(s, "missing section for %s", rs.Name)
			}
			r.Xsym = rs

			if ctxt.HeadType != objabi.Haix {
				return val, true
			}
		case objabi.R_CALLPOWER:
			r.Done = false
			r.Xsym = r.Sym
			r.Xadd = r.Add
			if ctxt.HeadType != objabi.Haix {
				return val, true
			}
		}
	}

	switch r.Type {
	case objabi.R_CONST:
		return r.Add, true
	case objabi.R_GOTOFF:
		return ld.Symaddr(r.Sym) + r.Add - ld.Symaddr(ctxt.Syms.Lookup(".got", 0)), true
	case objabi.R_ADDRPOWER_TOCREL, objabi.R_ADDRPOWER_TOCREL_DS:
		return archreloctoc(ctxt, r, s, val), true
	case objabi.R_ADDRPOWER, objabi.R_ADDRPOWER_DS:
		return archrelocaddr(ctxt, r, s, val), true
	case objabi.R_CALLPOWER:
		// Bits 6 through 29 = (S + A - P) >> 2

		t := ld.Symaddr(r.Sym) + r.Add - (s.Value + int64(r.Off))

		if t&3 != 0 {
			ld.Errorf(s, "relocation for %s+%d is not aligned: %d", r.Sym.Name, r.Off, t)
		}
		// If branch offset is too far then create a trampoline.

		if int64(int32(t<<6)>>6) != t {
			ld.Errorf(s, "direct call too far: %s %x", r.Sym.Name, t)
		}
		return val | int64(uint32(t)&^0xfc000003), true
	case objabi.R_POWER_TOC: // S + A - .TOC.
		return ld.Symaddr(r.Sym) + r.Add - symtoc(ctxt, s), true

	case objabi.R_POWER_TLS_LE:
		// The thread pointer points 0x7000 bytes after the start of the
		// thread local storage area as documented in section "3.7.2 TLS
		// Runtime Handling" of "Power Architecture 64-Bit ELF V2 ABI
		// Specification".
		v := r.Sym.Value - 0x7000
		if ctxt.HeadType == objabi.Haix {
			// On AIX, the thread pointer points 0x7800 bytes after
			// the TLS.
			v -= 0x800
		}
		if int64(int16(v)) != v {
			ld.Errorf(s, "TLS offset out of range %d", v)
		}
		return (val &^ 0xffff) | (v & 0xffff), true
	}

	return val, false
}

func archrelocvariant(ctxt *ld.Link, r *sym.Reloc, s *sym.Symbol, t int64) int64 {
	switch r.Variant & sym.RV_TYPE_MASK {
	default:
		ld.Errorf(s, "unexpected relocation variant %d", r.Variant)
		fallthrough

	case sym.RV_NONE:
		return t

	case sym.RV_POWER_LO:
		if r.Variant&sym.RV_CHECK_OVERFLOW != 0 {
			// Whether to check for signed or unsigned
			// overflow depends on the instruction
			var o1 uint32
			if ctxt.Arch.ByteOrder == binary.BigEndian {
				o1 = binary.BigEndian.Uint32(s.P[r.Off-2:])
			} else {
				o1 = binary.LittleEndian.Uint32(s.P[r.Off:])
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

		if r.Variant&sym.RV_CHECK_OVERFLOW != 0 {
			// Whether to check for signed or unsigned
			// overflow depends on the instruction
			var o1 uint32
			if ctxt.Arch.ByteOrder == binary.BigEndian {
				o1 = binary.BigEndian.Uint32(s.P[r.Off-2:])
			} else {
				o1 = binary.LittleEndian.Uint32(s.P[r.Off:])
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
		if ctxt.Arch.ByteOrder == binary.BigEndian {
			o1 = uint32(binary.BigEndian.Uint16(s.P[r.Off:]))
		} else {
			o1 = uint32(binary.LittleEndian.Uint16(s.P[r.Off:]))
		}
		if t&3 != 0 {
			ld.Errorf(s, "relocation for %s+%d is not aligned: %d", r.Sym.Name, r.Off, t)
		}
		if (r.Variant&sym.RV_CHECK_OVERFLOW != 0) && int64(int16(t)) != t {
			goto overflow
		}
		return int64(o1)&0x3 | int64(int16(t))
	}

overflow:
	ld.Errorf(s, "relocation for %s+%d is too big: %d", r.Sym.Name, r.Off, t)
	return t
}

func addpltsym(ctxt *ld.Link, s *sym.Symbol) {
	if s.Plt() >= 0 {
		return
	}

	ld.Adddynsym(ctxt, s)

	if ctxt.IsELF {
		plt := ctxt.Syms.Lookup(".plt", 0)
		rela := ctxt.Syms.Lookup(".rela.plt", 0)
		if plt.Size == 0 {
			elfsetupplt(ctxt)
		}

		// Create the glink resolver if necessary
		glink := ensureglinkresolver(ctxt)

		// Write symbol resolver stub (just a branch to the
		// glink resolver stub)
		r := glink.AddRel()

		r.Sym = glink
		r.Off = int32(glink.Size)
		r.Siz = 4
		r.Type = objabi.R_CALLPOWER
		glink.AddUint32(ctxt.Arch, 0x48000000) // b .glink

		// In the ppc64 ABI, the dynamic linker is responsible
		// for writing the entire PLT.  We just need to
		// reserve 8 bytes for each PLT entry and generate a
		// JMP_SLOT dynamic relocation for it.
		//
		// TODO(austin): ABI v1 is different
		s.SetPlt(int32(plt.Size))

		plt.Size += 8

		rela.AddAddrPlus(ctxt.Arch, plt, int64(s.Plt()))
		rela.AddUint64(ctxt.Arch, ld.ELF64_R_INFO(uint32(s.Dynid), uint32(elf.R_PPC64_JMP_SLOT)))
		rela.AddUint64(ctxt.Arch, 0)
	} else {
		ld.Errorf(s, "addpltsym: unsupported binary format")
	}
}

// Generate the glink resolver stub if necessary and return the .glink section
func ensureglinkresolver(ctxt *ld.Link) *sym.Symbol {
	glink := ctxt.Syms.Lookup(".glink", 0)
	if glink.Size != 0 {
		return glink
	}

	// This is essentially the resolver from the ppc64 ELF ABI.
	// At entry, r12 holds the address of the symbol resolver stub
	// for the target routine and the argument registers hold the
	// arguments for the target routine.
	//
	// This stub is PIC, so first get the PC of label 1 into r11.
	// Other things will be relative to this.
	glink.AddUint32(ctxt.Arch, 0x7c0802a6) // mflr r0
	glink.AddUint32(ctxt.Arch, 0x429f0005) // bcl 20,31,1f
	glink.AddUint32(ctxt.Arch, 0x7d6802a6) // 1: mflr r11
	glink.AddUint32(ctxt.Arch, 0x7c0803a6) // mtlf r0

	// Compute the .plt array index from the entry point address.
	// Because this is PIC, everything is relative to label 1b (in
	// r11):
	//   r0 = ((r12 - r11) - (res_0 - r11)) / 4 = (r12 - res_0) / 4
	glink.AddUint32(ctxt.Arch, 0x3800ffd0) // li r0,-(res_0-1b)=-48
	glink.AddUint32(ctxt.Arch, 0x7c006214) // add r0,r0,r12
	glink.AddUint32(ctxt.Arch, 0x7c0b0050) // sub r0,r0,r11
	glink.AddUint32(ctxt.Arch, 0x7800f082) // srdi r0,r0,2

	// r11 = address of the first byte of the PLT
	r := glink.AddRel()

	r.Off = int32(glink.Size)
	r.Sym = ctxt.Syms.Lookup(".plt", 0)
	r.Siz = 8
	r.Type = objabi.R_ADDRPOWER

	glink.AddUint32(ctxt.Arch, 0x3d600000) // addis r11,0,.plt@ha
	glink.AddUint32(ctxt.Arch, 0x396b0000) // addi r11,r11,.plt@l

	// Load r12 = dynamic resolver address and r11 = DSO
	// identifier from the first two doublewords of the PLT.
	glink.AddUint32(ctxt.Arch, 0xe98b0000) // ld r12,0(r11)
	glink.AddUint32(ctxt.Arch, 0xe96b0008) // ld r11,8(r11)

	// Jump to the dynamic resolver
	glink.AddUint32(ctxt.Arch, 0x7d8903a6) // mtctr r12
	glink.AddUint32(ctxt.Arch, 0x4e800420) // bctr

	// The symbol resolvers must immediately follow.
	//   res_0:

	// Add DT_PPC64_GLINK .dynamic entry, which points to 32 bytes
	// before the first symbol resolver stub.
	s := ctxt.Syms.Lookup(".dynamic", 0)

	ld.Elfwritedynentsymplus(ctxt, s, ld.DT_PPC64_GLINK, glink, glink.Size-32)

	return glink
}

func asmb(ctxt *ld.Link) {
	if ctxt.IsELF {
		ld.Asmbelfsetup()
	}

	for _, sect := range ld.Segtext.Sections {
		ctxt.Out.SeekSet(int64(sect.Vaddr - ld.Segtext.Vaddr + ld.Segtext.Fileoff))
		// Handle additional text sections with Codeblk
		if sect.Name == ".text" {
			ld.Codeblk(ctxt, int64(sect.Vaddr), int64(sect.Length))
		} else {
			ld.Datblk(ctxt, int64(sect.Vaddr), int64(sect.Length))
		}
	}

	if ld.Segrodata.Filelen > 0 {
		ctxt.Out.SeekSet(int64(ld.Segrodata.Fileoff))
		ld.Datblk(ctxt, int64(ld.Segrodata.Vaddr), int64(ld.Segrodata.Filelen))
	}
	if ld.Segrelrodata.Filelen > 0 {
		ctxt.Out.SeekSet(int64(ld.Segrelrodata.Fileoff))
		ld.Datblk(ctxt, int64(ld.Segrelrodata.Vaddr), int64(ld.Segrelrodata.Filelen))
	}

	ctxt.Out.SeekSet(int64(ld.Segdata.Fileoff))
	ld.Datblk(ctxt, int64(ld.Segdata.Vaddr), int64(ld.Segdata.Filelen))

	ctxt.Out.SeekSet(int64(ld.Segdwarf.Fileoff))
	ld.Dwarfblk(ctxt, int64(ld.Segdwarf.Vaddr), int64(ld.Segdwarf.Filelen))
}

func asmb2(ctxt *ld.Link) {
	/* output symbol table */
	ld.Symsize = 0

	ld.Lcsize = 0
	symo := uint32(0)
	if !*ld.FlagS {
		// TODO: rationalize
		switch ctxt.HeadType {
		default:
			if ctxt.IsELF {
				symo = uint32(ld.Segdwarf.Fileoff + ld.Segdwarf.Filelen)
				symo = uint32(ld.Rnd(int64(symo), int64(*ld.FlagRound)))
			}

		case objabi.Hplan9:
			symo = uint32(ld.Segdata.Fileoff + ld.Segdata.Filelen)

		case objabi.Haix:
			// Nothing to do
		}

		ctxt.Out.SeekSet(int64(symo))
		switch ctxt.HeadType {
		default:
			if ctxt.IsELF {
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

		case objabi.Haix:
			// symtab must be added once sections have been created in ld.Asmbxcoff
			ctxt.Out.Flush()
		}
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
		objabi.Hopenbsd:
		ld.Asmbelf(ctxt, int64(symo))

	case objabi.Haix:
		fileoff := uint32(ld.Segdwarf.Fileoff + ld.Segdwarf.Filelen)
		fileoff = uint32(ld.Rnd(int64(fileoff), int64(*ld.FlagRound)))
		ld.Asmbxcoff(ctxt, int64(fileoff))
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
