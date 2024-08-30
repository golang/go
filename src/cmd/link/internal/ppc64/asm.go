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
	"internal/buildcfg"
	"log"
	"strconv"
	"strings"
)

// The build configuration supports PC-relative instructions and relocations (limited to tested targets).
var hasPCrel = buildcfg.GOPPC64 >= 10 && buildcfg.GOOS == "linux"

const (
	// For genstub, the type of stub required by the caller.
	STUB_TOC = iota
	STUB_PCREL
)

var stubStrs = []string{
	STUB_TOC:   "_callstub_toc",
	STUB_PCREL: "_callstub_pcrel",
}

const (
	OP_TOCRESTORE    = 0xe8410018 // ld r2,24(r1)
	OP_TOCSAVE       = 0xf8410018 // std r2,24(r1)
	OP_NOP           = 0x60000000 // nop
	OP_BL            = 0x48000001 // bl 0
	OP_BCTR          = 0x4e800420 // bctr
	OP_BCTRL         = 0x4e800421 // bctrl
	OP_BCL           = 0x40000001 // bcl
	OP_ADDI          = 0x38000000 // addi
	OP_ADDIS         = 0x3c000000 // addis
	OP_LD            = 0xe8000000 // ld
	OP_PLA_PFX       = 0x06100000 // pla (prefix instruction word)
	OP_PLA_SFX       = 0x38000000 // pla (suffix instruction word)
	OP_PLD_PFX_PCREL = 0x04100000 // pld (prefix instruction word, R=1)
	OP_PLD_SFX       = 0xe4000000 // pld (suffix instruction word)
	OP_MFLR          = 0x7c0802a6 // mflr
	OP_MTLR          = 0x7c0803a6 // mtlr
	OP_MFCTR         = 0x7c0902a6 // mfctr
	OP_MTCTR         = 0x7c0903a6 // mtctr

	OP_ADDIS_R12_R2  = OP_ADDIS | 12<<21 | 2<<16  // addis r12,r2,0
	OP_ADDIS_R12_R12 = OP_ADDIS | 12<<21 | 12<<16 // addis  r12,r12,0
	OP_ADDI_R12_R12  = OP_ADDI | 12<<21 | 12<<16  // addi  r12,r12,0
	OP_PLD_SFX_R12   = OP_PLD_SFX | 12<<21        // pld   r12,0 (suffix instruction word)
	OP_PLA_SFX_R12   = OP_PLA_SFX | 12<<21        // pla   r12,0 (suffix instruction word)
	OP_LIS_R12       = OP_ADDIS | 12<<21          // lis r12,0
	OP_LD_R12_R12    = OP_LD | 12<<21 | 12<<16    // ld r12,0(r12)
	OP_MTCTR_R12     = OP_MTCTR | 12<<21          // mtctr r12
	OP_MFLR_R12      = OP_MFLR | 12<<21           // mflr r12
	OP_MFLR_R0       = OP_MFLR | 0<<21            // mflr r0
	OP_MTLR_R0       = OP_MTLR | 0<<21            // mtlr r0

	// This is a special, preferred form of bcl to obtain the next
	// instruction address (NIA, aka PC+4) in LR.
	OP_BCL_NIA = OP_BCL | 20<<21 | 31<<16 | 1<<2 // bcl 20,31,$+4

	// Masks to match opcodes
	MASK_PLD_PFX  = 0xfff70000
	MASK_PLD_SFX  = 0xfc1f0000 // Also checks RA = 0 if check value is OP_PLD_SFX.
	MASK_PLD_RT   = 0x03e00000 // Extract RT from the pld suffix.
	MASK_OP_LD    = 0xfc000003
	MASK_OP_ADDIS = 0xfc000000
)

// Generate a stub to call between TOC and NOTOC functions. See genpltstub for more details about calling stubs.
// This is almost identical to genpltstub, except the location of the target symbol is known at link time.
func genstub(ctxt *ld.Link, ldr *loader.Loader, r loader.Reloc, ri int, s loader.Sym, stubType int) (ssym loader.Sym, firstUse bool) {
	addendStr := ""
	if r.Add() != 0 {
		addendStr = fmt.Sprintf("%+d", r.Add())
	}

	stubName := fmt.Sprintf("%s%s.%s", stubStrs[stubType], addendStr, ldr.SymName(r.Sym()))
	stub := ldr.CreateSymForUpdate(stubName, 0)
	firstUse = stub.Size() == 0
	if firstUse {
		switch stubType {
		// A call from a function using a TOC pointer.
		case STUB_TOC:
			stub.AddUint32(ctxt.Arch, OP_TOCSAVE) // std r2,24(r1)
			stub.AddSymRef(ctxt.Arch, r.Sym(), r.Add(), objabi.R_ADDRPOWER_TOCREL_DS, 8)
			stub.SetUint32(ctxt.Arch, stub.Size()-8, OP_ADDIS_R12_R2) // addis r12,r2,targ@toc@ha
			stub.SetUint32(ctxt.Arch, stub.Size()-4, OP_ADDI_R12_R12) // addi  r12,targ@toc@l(r12)

		// A call from PC relative function.
		case STUB_PCREL:
			if buildcfg.GOPPC64 >= 10 {
				// Set up address of targ in r12, PCrel
				stub.AddSymRef(ctxt.Arch, r.Sym(), r.Add(), objabi.R_ADDRPOWER_PCREL34, 8)
				stub.SetUint32(ctxt.Arch, stub.Size()-8, OP_PLA_PFX)
				stub.SetUint32(ctxt.Arch, stub.Size()-4, OP_PLA_SFX_R12) // pla r12, r
			} else {
				// The target may not be a P10. Generate a P8 compatible stub.
				stub.AddUint32(ctxt.Arch, OP_MFLR_R0)  // mflr r0
				stub.AddUint32(ctxt.Arch, OP_BCL_NIA)  // bcl 20,31,1f
				stub.AddUint32(ctxt.Arch, OP_MFLR_R12) // 1: mflr r12  (r12 is the address of this instruction)
				stub.AddUint32(ctxt.Arch, OP_MTLR_R0)  // mtlr r0
				stub.AddSymRef(ctxt.Arch, r.Sym(), r.Add()+8, objabi.R_ADDRPOWER_PCREL, 8)
				stub.SetUint32(ctxt.Arch, stub.Size()-8, OP_ADDIS_R12_R12) // addis r12,(r - 1b) + 8
				stub.SetUint32(ctxt.Arch, stub.Size()-4, OP_ADDI_R12_R12)  // addi  r12,(r - 1b) + 12
			}
		}
		// Jump to the loaded pointer
		stub.AddUint32(ctxt.Arch, OP_MTCTR_R12) // mtctr r12
		stub.AddUint32(ctxt.Arch, OP_BCTR)      // bctr
		stub.SetType(sym.STEXT)
	}

	// Update the relocation to use the call stub
	su := ldr.MakeSymbolUpdater(s)
	su.SetRelocSym(ri, stub.Sym())

	// Rewrite the TOC restore slot (a nop) if the caller uses a TOC pointer.
	switch stubType {
	case STUB_TOC:
		rewritetoinsn(&ctxt.Target, ldr, su, int64(r.Off()+4), 0xFFFFFFFF, OP_NOP, OP_TOCRESTORE)
	}

	return stub.Sym(), firstUse
}

func genpltstub(ctxt *ld.Link, ldr *loader.Loader, r loader.Reloc, ri int, s loader.Sym) (sym loader.Sym, firstUse bool) {
	// The ppc64 ABI PLT has similar concepts to other
	// architectures, but is laid out quite differently. When we
	// see a relocation to a dynamic symbol (indicating that the
	// call needs to go through the PLT), we generate up to three
	// stubs and reserve a PLT slot.
	//
	// 1) The call site is a "bl x" where genpltstub rewrites it to
	//    "bl x_stub". Depending on the properties of the caller
	//    (see ELFv2 1.5 4.2.5.3), a nop may be expected immediately
	//    after the bl. This nop is rewritten to ld r2,24(r1) to
	//    restore the toc pointer saved by x_stub.
	//
	// 2) We reserve space for a pointer in the .plt section (once
	//    per referenced dynamic function).  .plt is a data
	//    section filled solely by the dynamic linker (more like
	//    .plt.got on other architectures).  Initially, the
	//    dynamic linker will fill each slot with a pointer to the
	//    corresponding x@plt entry point.
	//
	// 3) We generate a "call stub" x_stub based on the properties
	//    of the caller.
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

	// Find all relocations that reference dynamic imports.
	// Reserve PLT entries for these symbols and generate call
	// stubs. The call stubs need to live in .text, which is why we
	// need to do this pass this early.

	// Reserve PLT entry and generate symbol resolver
	addpltsym(ctxt, ldr, r.Sym())

	// The stub types are described in gencallstub.
	stubType := 0
	stubTypeStr := ""

	// For now, the choice of call stub type is determined by whether
	// the caller maintains a TOC pointer in R2. A TOC pointer implies
	// we can always generate a position independent stub.
	//
	// For dynamic calls made from an external object, a caller maintains
	// a TOC pointer only when an R_PPC64_REL24 relocation is used.
	// An R_PPC64_REL24_NOTOC relocation does not use or maintain
	// a TOC pointer, and almost always implies a Power10 target.
	//
	// For dynamic calls made from a Go caller, a TOC relative stub is
	// always needed when a TOC pointer is maintained (specifically, if
	// the Go caller is PIC, and cannot use PCrel instructions).
	if (r.Type() == objabi.ElfRelocOffset+objabi.RelocType(elf.R_PPC64_REL24)) || (!ldr.AttrExternal(s) && ldr.AttrShared(s) && !hasPCrel) {
		stubTypeStr = "_tocrel"
		stubType = 1
	} else {
		stubTypeStr = "_notoc"
		stubType = 3
	}
	n := fmt.Sprintf("_pltstub%s.%s", stubTypeStr, ldr.SymName(r.Sym()))

	// When internal linking, all text symbols share the same toc pointer.
	stub := ldr.CreateSymForUpdate(n, 0)
	firstUse = stub.Size() == 0
	if firstUse {
		gencallstub(ctxt, ldr, stubType, stub, r.Sym())
	}

	// Update the relocation to use the call stub
	su := ldr.MakeSymbolUpdater(s)
	su.SetRelocSym(ri, stub.Sym())

	// A type 1 call must restore the toc pointer after the call.
	if stubType == 1 {
		su.MakeWritable()
		p := su.Data()

		// Check for a toc pointer restore slot (a nop), and rewrite to restore the toc pointer.
		var nop uint32
		if len(p) >= int(r.Off()+8) {
			nop = ctxt.Arch.ByteOrder.Uint32(p[r.Off()+4:])
		}
		if nop != OP_NOP {
			ldr.Errorf(s, "Symbol %s is missing toc restoration slot at offset %d", ldr.SymName(s), r.Off()+4)
		}
		ctxt.Arch.ByteOrder.PutUint32(p[r.Off()+4:], OP_TOCRESTORE)
	}

	return stub.Sym(), firstUse
}

// Scan relocs and generate PLT stubs and generate/fixup ABI defined functions created by the linker.
func genstubs(ctxt *ld.Link, ldr *loader.Loader) {
	var stubs []loader.Sym
	var abifuncs []loader.Sym
	for _, s := range ctxt.Textp {
		relocs := ldr.Relocs(s)
		for i := 0; i < relocs.Count(); i++ {
			switch r := relocs.At(i); r.Type() {
			case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL24), objabi.R_CALLPOWER:
				switch ldr.SymType(r.Sym()) {
				case sym.SDYNIMPORT:
					// This call goes through the PLT, generate and call through a PLT stub.
					if sym, firstUse := genpltstub(ctxt, ldr, r, i, s); firstUse {
						stubs = append(stubs, sym)
					}

				case sym.SXREF:
					// Is this an ELF ABI defined function which is (in practice)
					// generated by the linker to save/restore callee save registers?
					// These are defined similarly for both PPC64 ELF and ELFv2.
					targName := ldr.SymName(r.Sym())
					if strings.HasPrefix(targName, "_save") || strings.HasPrefix(targName, "_rest") {
						if sym, firstUse := rewriteABIFuncReloc(ctxt, ldr, targName, r); firstUse {
							abifuncs = append(abifuncs, sym)
						}
					}
				case sym.STEXT:
					targ := r.Sym()
					if (ldr.AttrExternal(targ) && ldr.SymLocalentry(targ) != 1) || !ldr.AttrExternal(targ) {
						// All local symbols share the same TOC pointer. This caller has a valid TOC
						// pointer in R2. Calls into a Go symbol preserve R2. No call stub is needed.
					} else {
						// This caller has a TOC pointer. The callee might clobber it. R2 needs to be saved
						// and restored.
						if sym, firstUse := genstub(ctxt, ldr, r, i, s, STUB_TOC); firstUse {
							stubs = append(stubs, sym)
						}
					}
				}

			case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL24_P9NOTOC):
				// This can be treated identically to R_PPC64_REL24_NOTOC, as stubs are determined by
				// GOPPC64 and -buildmode.
				fallthrough
			case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL24_NOTOC):
				switch ldr.SymType(r.Sym()) {
				case sym.SDYNIMPORT:
					// This call goes through the PLT, generate and call through a PLT stub.
					if sym, firstUse := genpltstub(ctxt, ldr, r, i, s); firstUse {
						stubs = append(stubs, sym)
					}

				case sym.SXREF:
					// TODO: This is not supported yet.
					ldr.Errorf(s, "Unsupported NOTOC external reference call into %s", ldr.SymName(r.Sym()))

				case sym.STEXT:
					targ := r.Sym()
					if (ldr.AttrExternal(targ) && ldr.SymLocalentry(targ) <= 1) || (!ldr.AttrExternal(targ) && (!ldr.AttrShared(targ) || hasPCrel)) {
						// This is NOTOC to NOTOC call (st_other is 0 or 1). No call stub is needed.
					} else {
						// This is a NOTOC to TOC function. Generate a calling stub.
						if sym, firstUse := genstub(ctxt, ldr, r, i, s, STUB_PCREL); firstUse {
							stubs = append(stubs, sym)
						}
					}
				}

			// Handle objects compiled with -fno-plt. Rewrite local calls to avoid indirect calling.
			// These are 0 sized relocs. They mark the mtctr r12, or bctrl + ld r2,24(r1).
			case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_PLTSEQ):
				if ldr.SymType(r.Sym()) == sym.STEXT {
					// This should be an mtctr instruction. Turn it into a nop.
					su := ldr.MakeSymbolUpdater(s)
					const MASK_OP_MTCTR = 63<<26 | 0x3FF<<11 | 0x1FF<<1
					rewritetonop(&ctxt.Target, ldr, su, int64(r.Off()), MASK_OP_MTCTR, OP_MTCTR)
				}
			case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_PLTCALL):
				if ldr.SymType(r.Sym()) == sym.STEXT {
					// This relocation should point to a bctrl followed by a ld r2, 24(41)
					// Convert the bctrl into a bl.
					su := ldr.MakeSymbolUpdater(s)
					rewritetoinsn(&ctxt.Target, ldr, su, int64(r.Off()), 0xFFFFFFFF, OP_BCTRL, OP_BL)

					// Turn this reloc into an R_CALLPOWER, and convert the TOC restore into a nop.
					su.SetRelocType(i, objabi.R_CALLPOWER)
					localEoffset := int64(ldr.SymLocalentry(r.Sym()))
					if localEoffset == 1 {
						ldr.Errorf(s, "Unsupported NOTOC call to %s", ldr.SymName(r.Sym()))
					}
					su.SetRelocAdd(i, r.Add()+localEoffset)
					r.SetSiz(4)
					rewritetonop(&ctxt.Target, ldr, su, int64(r.Off()+4), 0xFFFFFFFF, OP_TOCRESTORE)
				}
			}
		}
	}

	// Append any usage of the go versions of ELF save/restore
	// functions to the end of the callstub list to minimize
	// chances a trampoline might be needed.
	stubs = append(stubs, abifuncs...)

	// Put stubs at the beginning (instead of the end).
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

	// Write a function to load this module's local.moduledata. This is shared code.
	//
	// package link
	// void addmoduledata() {
	//	runtime.addmoduledata(local.moduledata)
	// }

	if !hasPCrel {
		// Regenerate TOC from R12 (the address of this function).
		sz := initfunc.AddSymRef(ctxt.Arch, ctxt.DotTOC[0], 0, objabi.R_ADDRPOWER_PCREL, 8)
		initfunc.SetUint32(ctxt.Arch, sz-8, 0x3c4c0000) // addis r2, r12, .TOC.-func@ha
		initfunc.SetUint32(ctxt.Arch, sz-4, 0x38420000) // addi r2, r2, .TOC.-func@l
	}

	// This is Go ABI. Stack a frame and save LR.
	o(OP_MFLR_R0) // mflr r0
	o(0xf801ffe1) // stdu r0, -32(r1)

	// Get the moduledata pointer from GOT and put into R3.
	var tgt loader.Sym
	if s := ldr.Lookup("local.moduledata", 0); s != 0 {
		tgt = s
	} else if s := ldr.Lookup("local.pluginmoduledata", 0); s != 0 {
		tgt = s
	} else {
		tgt = ldr.LookupOrCreateSym("runtime.firstmoduledata", 0)
	}

	if !hasPCrel {
		sz := initfunc.AddSymRef(ctxt.Arch, tgt, 0, objabi.R_ADDRPOWER_GOT, 8)
		initfunc.SetUint32(ctxt.Arch, sz-8, 0x3c620000) // addis r3, r2, local.moduledata@got@ha
		initfunc.SetUint32(ctxt.Arch, sz-4, 0xe8630000) // ld r3, local.moduledata@got@l(r3)
	} else {
		sz := initfunc.AddSymRef(ctxt.Arch, tgt, 0, objabi.R_ADDRPOWER_GOT_PCREL34, 8)
		// Note, this is prefixed instruction. It must not cross a 64B boundary.
		// It is doubleworld aligned here, so it will never cross (this function is 16B aligned, minimum).
		initfunc.SetUint32(ctxt.Arch, sz-8, OP_PLD_PFX_PCREL)
		initfunc.SetUint32(ctxt.Arch, sz-4, OP_PLD_SFX|(3<<21)) // pld r3, local.moduledata@got@pcrel
	}

	// Call runtime.addmoduledata
	sz := initfunc.AddSymRef(ctxt.Arch, addmoduledata, 0, objabi.R_CALLPOWER, 4)
	initfunc.SetUint32(ctxt.Arch, sz-4, OP_BL) // bl runtime.addmoduledata
	o(OP_NOP)                                  // nop (for TOC restore)

	// Pop stack frame and return.
	o(0xe8010000) // ld r0, 0(r1)
	o(OP_MTLR_R0) // mtlr r0
	o(0x38210020) // addi r1,r1,32
	o(0x4e800020) // blr
}

// Rewrite ELF (v1 or v2) calls to _savegpr0_n, _savegpr1_n, _savefpr_n, _restfpr_n, _savevr_m, or
// _restvr_m (14<=n<=31, 20<=m<=31). Redirect them to runtime.elf_restgpr0+(n-14)*4,
// runtime.elf_restvr+(m-20)*8, and similar.
//
// These functions are defined in the ELFv2 ABI (generated when using gcc -Os option) to save and
// restore callee-saved registers (as defined in the PPC64 ELF ABIs) from registers n or m to 31 of
// the named type. R12 and R0 are sometimes used in exceptional ways described in the ABI.
//
// Final note, this is only needed when linking internally. The external linker will generate these
// functions if they are used.
func rewriteABIFuncReloc(ctxt *ld.Link, ldr *loader.Loader, tname string, r loader.Reloc) (sym loader.Sym, firstUse bool) {
	s := strings.Split(tname, "_")
	// A valid call will split like {"", "savegpr0", "20"}
	if len(s) != 3 {
		return 0, false // Not an abi func.
	}
	minReg := 14 // _savegpr0_{n}, _savegpr1_{n}, _savefpr_{n}, 14 <= n <= 31
	offMul := 4  // 1 instruction per register op.
	switch s[1] {
	case "savegpr0", "savegpr1", "savefpr":
	case "restgpr0", "restgpr1", "restfpr":
	case "savevr", "restvr":
		minReg = 20 // _savevr_{n} or _restvr_{n}, 20 <= n <= 31
		offMul = 8  // 2 instructions per register op.
	default:
		return 0, false // Not an abi func
	}
	n, e := strconv.Atoi(s[2])
	if e != nil || n < minReg || n > 31 || r.Add() != 0 {
		return 0, false // Invalid register number, or non-zero addend. Not an abi func.
	}

	// tname is a valid relocation to an ABI defined register save/restore function. Re-relocate
	// them to a go version of these functions in runtime/asm_ppc64x.s
	ts := ldr.LookupOrCreateSym("runtime.elf_"+s[1], 0)
	r.SetSym(ts)
	r.SetAdd(int64((n - minReg) * offMul))
	firstUse = !ldr.AttrReachable(ts)
	if firstUse {
		// This function only becomes reachable now. It has been dropped from
		// the text section (it was unreachable until now), it needs included.
		ldr.SetAttrReachable(ts, true)
	}
	return ts, firstUse
}

func gentext(ctxt *ld.Link, ldr *loader.Loader) {
	if ctxt.DynlinkingGo() {
		genaddmoduledata(ctxt, ldr)
	}

	if ctxt.LinkMode == ld.LinkInternal {
		genstubs(ctxt, ldr)
	}
}

// Create a calling stub. The stubType maps directly to the properties listed in the ELFv2 1.5
// section 4.2.5.3.
//
// There are 3 cases today (as paraphrased from the ELFv2 document):
//
//  1. R2 holds the TOC pointer on entry. The call stub must save R2 into the ELFv2 TOC stack save slot.
//
//  2. R2 holds the TOC pointer on entry. The caller has already saved R2 to the TOC stack save slot.
//
//  3. R2 does not hold the TOC pointer on entry. The caller has no expectations of R2.
//
// Go only needs case 1 and 3 today. Go symbols which have AttrShare set could use case 2, but case 1 always
// works in those cases too.
func gencallstub(ctxt *ld.Link, ldr *loader.Loader, stubType int, stub *loader.SymbolBuilder, targ loader.Sym) {
	plt := ctxt.PLT
	stub.SetType(sym.STEXT)

	switch stubType {
	case 1:
		// Save TOC, then load targ address from PLT using TOC.
		stub.AddUint32(ctxt.Arch, OP_TOCSAVE) // std r2,24(r1)
		stub.AddSymRef(ctxt.Arch, plt, int64(ldr.SymPlt(targ)), objabi.R_ADDRPOWER_TOCREL_DS, 8)
		stub.SetUint32(ctxt.Arch, stub.Size()-8, OP_ADDIS_R12_R2) // addis r12,r2,targ@plt@toc@ha
		stub.SetUint32(ctxt.Arch, stub.Size()-4, OP_LD_R12_R12)   // ld r12,targ@plt@toc@l(r12)
	case 3:
		// No TOC needs to be saved, but the stub may need to position-independent.
		if buildcfg.GOPPC64 >= 10 {
			// Power10 is supported, load targ address into r12 using PCrel load.
			stub.AddSymRef(ctxt.Arch, plt, int64(ldr.SymPlt(targ)), objabi.R_ADDRPOWER_PCREL34, 8)
			stub.SetUint32(ctxt.Arch, stub.Size()-8, OP_PLD_PFX_PCREL)
			stub.SetUint32(ctxt.Arch, stub.Size()-4, OP_PLD_SFX_R12) // pld r12, targ@plt
		} else if !isLinkingPIC(ctxt) {
			// This stub doesn't need to be PIC. Load targ address from the PLT via its absolute address.
			stub.AddSymRef(ctxt.Arch, plt, int64(ldr.SymPlt(targ)), objabi.R_ADDRPOWER_DS, 8)
			stub.SetUint32(ctxt.Arch, stub.Size()-8, OP_LIS_R12)    // lis r12,targ@plt@ha
			stub.SetUint32(ctxt.Arch, stub.Size()-4, OP_LD_R12_R12) // ld r12,targ@plt@l(r12)
		} else {
			// Generate a PIC stub. This is ugly as the stub must determine its location using
			// POWER8 or older instruction. These stubs are likely the combination of using
			// GOPPC64 < 8 and linking external objects built with CFLAGS="... -mcpu=power10 ..."
			stub.AddUint32(ctxt.Arch, OP_MFLR_R0)  // mflr r0
			stub.AddUint32(ctxt.Arch, OP_BCL_NIA)  // bcl 20,31,1f
			stub.AddUint32(ctxt.Arch, OP_MFLR_R12) // 1: mflr r12  (r12 is the address of this instruction)
			stub.AddUint32(ctxt.Arch, OP_MTLR_R0)  // mtlr r0
			stub.AddSymRef(ctxt.Arch, plt, int64(ldr.SymPlt(targ))+8, objabi.R_ADDRPOWER_PCREL, 8)
			stub.SetUint32(ctxt.Arch, stub.Size()-8, OP_ADDIS_R12_R12) // addis r12,(targ@plt - 1b) + 8
			stub.SetUint32(ctxt.Arch, stub.Size()-4, OP_ADDI_R12_R12)  // addi  r12,(targ@plt - 1b) + 12
			stub.AddUint32(ctxt.Arch, OP_LD_R12_R12)                   // ld r12, 0(r12)
		}
	default:
		log.Fatalf("gencallstub does not support ELFv2 ABI property %d", stubType)
	}

	// Jump to the loaded pointer
	stub.AddUint32(ctxt.Arch, OP_MTCTR_R12) // mtctr r12
	stub.AddUint32(ctxt.Arch, OP_BCTR)      // bctr
}

// Rewrite the instruction at offset into newinsn. Also, verify the
// existing instruction under mask matches the check value.
func rewritetoinsn(target *ld.Target, ldr *loader.Loader, su *loader.SymbolBuilder, offset int64, mask, check, newinsn uint32) {
	su.MakeWritable()
	op := target.Arch.ByteOrder.Uint32(su.Data()[offset:])
	if op&mask != check {
		ldr.Errorf(su.Sym(), "Rewrite offset 0x%x to 0x%08X failed check (0x%08X&0x%08X != 0x%08X)", offset, newinsn, op, mask, check)
	}
	su.SetUint32(target.Arch, offset, newinsn)
}

// Rewrite the instruction at offset into a hardware nop instruction. Also, verify the
// existing instruction under mask matches the check value.
func rewritetonop(target *ld.Target, ldr *loader.Loader, su *loader.SymbolBuilder, offset int64, mask, check uint32) {
	rewritetoinsn(target, ldr, su, offset, mask, check, OP_NOP)
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
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL24_NOTOC),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL24_P9NOTOC):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_CALLPOWER)

		if targType == sym.SDYNIMPORT {
			// Should have been handled in elfsetupplt
			ldr.Errorf(s, "unexpected R_PPC64_REL24_NOTOC/R_PPC64_REL24_P9NOTOC for dyn import")
		}
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL24):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_CALLPOWER)

		// This is a local call, so the caller isn't setting
		// up r12 and r2 is the same for the caller and
		// callee. Hence, we need to go to the local entry
		// point.  (If we don't do this, the callee will try
		// to use r12 to compute r2.)
		localEoffset := int64(ldr.SymLocalentry(targ))
		if localEoffset == 1 {
			ldr.Errorf(s, "Unsupported NOTOC call to %s", targ)
		}
		su.SetRelocAdd(rIdx, r.Add()+localEoffset)

		if targType == sym.SDYNIMPORT {
			// Should have been handled in genstubs
			ldr.Errorf(s, "unexpected R_PPC64_REL24 for dyn import")
		}

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_PCREL34):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ADDRPOWER_PCREL34)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_GOT_PCREL34):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ADDRPOWER_PCREL34)
		if targType != sym.STEXT {
			ld.AddGotSym(target, ldr, syms, targ, uint32(elf.R_PPC64_GLOB_DAT))
			su.SetRelocSym(rIdx, syms.GOT)
			su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymGot(targ)))
		} else {
			// The address of targ is known at link time. Rewrite to "pla rt,targ" from "pld rt,targ@got"
			rewritetoinsn(target, ldr, su, int64(r.Off()), MASK_PLD_PFX, OP_PLD_PFX_PCREL, OP_PLA_PFX)
			pla_sfx := target.Arch.ByteOrder.Uint32(su.Data()[r.Off()+4:])&MASK_PLD_RT | OP_PLA_SFX
			rewritetoinsn(target, ldr, su, int64(r.Off()+4), MASK_PLD_SFX, OP_PLD_SFX, pla_sfx)
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

	// When compiling with gcc's -fno-plt option (no PLT), the following code and relocation
	// sequences may be present to call an external function:
	//
	//   1. addis Rx,foo@R_PPC64_PLT16_HA
	//   2. ld 12,foo@R_PPC64_PLT16_LO_DS(Rx)
	//   3. mtctr 12 ; foo@R_PPC64_PLTSEQ
	//   4. bctrl ; foo@R_PPC64_PLTCALL
	//   5. ld r2,24(r1)
	//
	// Note, 5 is required to follow the R_PPC64_PLTCALL. Similarly, relocations targeting
	// instructions 3 and 4 are zero sized informational relocations.
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_PLT16_HA),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_PLT16_LO_DS):
		su := ldr.MakeSymbolUpdater(s)
		isPLT16_LO_DS := r.Type() == objabi.ElfRelocOffset+objabi.RelocType(elf.R_PPC64_PLT16_LO_DS)
		if isPLT16_LO_DS {
			ldr.SetRelocVariant(s, rIdx, sym.RV_POWER_DS)
		} else {
			ldr.SetRelocVariant(s, rIdx, sym.RV_POWER_HA|sym.RV_CHECK_OVERFLOW)
		}
		su.SetRelocType(rIdx, objabi.R_POWER_TOC)
		if targType == sym.SDYNIMPORT {
			// This is an external symbol, make space in the GOT and retarget the reloc.
			ld.AddGotSym(target, ldr, syms, targ, uint32(elf.R_PPC64_GLOB_DAT))
			su.SetRelocSym(rIdx, syms.GOT)
			su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymGot(targ)))
		} else if targType == sym.STEXT {
			if isPLT16_LO_DS {
				// Expect an ld opcode to nop
				rewritetonop(target, ldr, su, int64(r.Off()), MASK_OP_LD, OP_LD)
			} else {
				// Expect an addis opcode to nop
				rewritetonop(target, ldr, su, int64(r.Off()), MASK_OP_ADDIS, OP_ADDIS)
			}
			// And we can ignore this reloc now.
			su.SetRelocType(rIdx, objabi.ElfRelocOffset)
		} else {
			ldr.Errorf(s, "unexpected PLT relocation target symbol type %s", targType.String())
		}
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
	// Beware that bit0~bit15 start from the third byte of an instruction in Big-Endian machines.
	rt := r.Type
	if rt == objabi.R_ADDR || rt == objabi.R_POWER_TLS || rt == objabi.R_CALLPOWER || rt == objabi.R_DWARFSECREF {
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
	case objabi.R_ADDRPOWER_D34:
		out.Write64(uint64(elf.R_PPC64_D34) | uint64(elfsym)<<32)
	case objabi.R_ADDRPOWER_PCREL34:
		out.Write64(uint64(elf.R_PPC64_PCREL34) | uint64(elfsym)<<32)
	case objabi.R_POWER_TLS:
		out.Write64(uint64(elf.R_PPC64_TLS) | uint64(elfsym)<<32)
	case objabi.R_POWER_TLS_LE:
		out.Write64(uint64(elf.R_PPC64_TPREL16_HA) | uint64(elfsym)<<32)
		out.Write64(uint64(r.Xadd))
		out.Write64(uint64(sectoff + 4))
		out.Write64(uint64(elf.R_PPC64_TPREL16_LO) | uint64(elfsym)<<32)
	case objabi.R_POWER_TLS_LE_TPREL34:
		out.Write64(uint64(elf.R_PPC64_TPREL34) | uint64(elfsym)<<32)
	case objabi.R_POWER_TLS_IE_PCREL34:
		out.Write64(uint64(elf.R_PPC64_GOT_TPREL_PCREL34) | uint64(elfsym)<<32)
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
	case objabi.R_ADDRPOWER_GOT_PCREL34:
		out.Write64(uint64(elf.R_PPC64_GOT_PCREL34) | uint64(elfsym)<<32)
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
		if !hasPCrel {
			out.Write64(uint64(elf.R_PPC64_REL24) | uint64(elfsym)<<32)
		} else {
			// TOC is not used in PCrel compiled Go code.
			out.Write64(uint64(elf.R_PPC64_REL24_NOTOC) | uint64(elfsym)<<32)
		}

	}
	out.Write64(uint64(r.Xadd))

	return true
}

func elfsetupplt(ctxt *ld.Link, ldr *loader.Loader, plt, got *loader.SymbolBuilder, dynamic loader.Sym) {
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
	o1, o2 := unpackInstPair(target, val)

	// Verify resulting address fits within a 31 bit (2GB) address space.
	// This is a restriction arising  from the usage of lis (HA) + d-form
	// (LO) instruction sequences used to implement absolute relocations
	// on PPC64 prior to ISA 3.1 (P10). For consistency, maintain this
	// restriction for ISA 3.1 unless it becomes problematic.
	t := ldr.SymAddr(rs) + r.Add()
	if t < 0 || t >= 1<<31 {
		ldr.Errorf(s, "relocation for %s is too big (>=2G): 0x%x", ldr.SymName(s), ldr.SymValue(rs))
	}

	// Note, relocations imported from external objects may not have cleared bits
	// within a relocatable field. They need cleared before applying the relocation.
	switch r.Type() {
	case objabi.R_ADDRPOWER_PCREL34:
		// S + A - P
		t -= (ldr.SymValue(s) + int64(r.Off()))
		o1 &^= 0x3ffff
		o2 &^= 0x0ffff
		o1 |= computePrefix34HI(t)
		o2 |= computeLO(int32(t))
	case objabi.R_ADDRPOWER_D34:
		o1 &^= 0x3ffff
		o2 &^= 0x0ffff
		o1 |= computePrefix34HI(t)
		o2 |= computeLO(int32(t))
	case objabi.R_ADDRPOWER:
		o1 &^= 0xffff
		o2 &^= 0xffff
		o1 |= computeHA(int32(t))
		o2 |= computeLO(int32(t))
	case objabi.R_ADDRPOWER_DS:
		o1 &^= 0xffff
		o2 &^= 0xfffc
		o1 |= computeHA(int32(t))
		o2 |= computeLO(int32(t))
		if t&3 != 0 {
			ldr.Errorf(s, "bad DS reloc for %s: %d", ldr.SymName(s), ldr.SymValue(rs))
		}
	default:
		return -1
	}

	return packInstPair(target, o1, o2)
}

// Determine if the code was compiled so that the TOC register R2 is initialized and maintained.
func r2Valid(ctxt *ld.Link) bool {
	return isLinkingPIC(ctxt)
}

// Determine if this is linking a position-independent binary.
func isLinkingPIC(ctxt *ld.Link) bool {
	switch ctxt.BuildMode {
	case ld.BuildModeCArchive, ld.BuildModeCShared, ld.BuildModePIE, ld.BuildModeShared, ld.BuildModePlugin:
		return true
	}
	// -linkshared option
	return ctxt.IsSharedGoLink()
}

// resolve direct jump relocation r in s, and add trampoline if necessary.
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
				// Note, the trampoline is always called directly. The addend of the original relocation is accounted for in the
				// trampoline itself.
				t = ldr.SymValue(tramp) - (ldr.SymValue(s) + int64(r.Off()))

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

	// ELFv2 save/restore functions use R0/R12 in special ways, therefore trampolines
	// as generated here will not always work correctly.
	if strings.HasPrefix(ldr.SymName(target), "runtime.elf_") {
		log.Fatalf("Internal linker does not support trampolines to ELFv2 ABI"+
			" register save/restore function %s", ldr.SymName(target))
	}

	if ctxt.IsAIX() {
		// On AIX, the address is retrieved with a TOC symbol.
		// For internal linking, the "Linux" way might still be used.
		// However, all text symbols are accessed with a TOC symbol as
		// text relocations aren't supposed to be possible.
		// So, keep using the external linking way to be more AIX friendly.
		o1 = uint32(OP_ADDIS_R12_R2) // addis r12,  r2, toctargetaddr hi
		o2 = uint32(OP_LD_R12_R12)   // ld    r12, r12, toctargetaddr lo

		toctramp := ldr.CreateSymForUpdate("TOC."+ldr.SymName(tramp.Sym()), 0)
		toctramp.SetType(sym.SXCOFFTOC)
		toctramp.AddAddrPlus(ctxt.Arch, target, offset)

		r, _ := tramp.AddRel(objabi.R_ADDRPOWER_TOCREL_DS)
		r.SetOff(0)
		r.SetSiz(8) // generates 2 relocations: HA + LO
		r.SetSym(toctramp.Sym())
	} else if hasPCrel {
		// pla r12, addr (PCrel). This works for static or PIC, with or without a valid TOC pointer.
		o1 = uint32(OP_PLA_PFX)
		o2 = uint32(OP_PLA_SFX_R12) // pla r12, addr

		// The trampoline's position is not known yet, insert a relocation.
		r, _ := tramp.AddRel(objabi.R_ADDRPOWER_PCREL34)
		r.SetOff(0)
		r.SetSiz(8) // This spans 2 words.
		r.SetSym(target)
		r.SetAdd(offset)
	} else {
		// Used for default build mode for an executable
		// Address of the call target is generated using
		// relocation and doesn't depend on r2 (TOC).
		o1 = uint32(OP_LIS_R12)      // lis  r12,targetaddr hi
		o2 = uint32(OP_ADDI_R12_R12) // addi r12,r12,targetaddr lo

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

	o3 := uint32(OP_MTCTR_R12) // mtctr r12
	o4 := uint32(OP_BCTR)      // bctr
	ctxt.Arch.ByteOrder.PutUint32(P, o1)
	ctxt.Arch.ByteOrder.PutUint32(P[4:], o2)
	ctxt.Arch.ByteOrder.PutUint32(P[8:], o3)
	ctxt.Arch.ByteOrder.PutUint32(P[12:], o4)
	tramp.SetData(P)
}

// Unpack a pair of 32 bit instruction words from
// a 64 bit relocation into instN and instN+1 in endian order.
func unpackInstPair(target *ld.Target, r int64) (uint32, uint32) {
	if target.IsBigEndian() {
		return uint32(r >> 32), uint32(r)
	}
	return uint32(r), uint32(r >> 32)
}

// Pack a pair of 32 bit instruction words o1, o2 into 64 bit relocation
// in endian order.
func packInstPair(target *ld.Target, o1, o2 uint32) int64 {
	if target.IsBigEndian() {
		return (int64(o1) << 32) | int64(o2)
	}
	return int64(o1) | (int64(o2) << 32)
}

// Compute the high-adjusted value (always a signed 32b value) per the ELF ABI.
// The returned value is always 0 <= x <= 0xFFFF.
func computeHA(val int32) uint32 {
	return uint32(uint16((val + 0x8000) >> 16))
}

// Compute the low value (the lower 16 bits of any 32b value) per the ELF ABI.
// The returned value is always 0 <= x <= 0xFFFF.
func computeLO(val int32) uint32 {
	return uint32(uint16(val))
}

// Compute the high 18 bits of a signed 34b constant. Used to pack the high 18 bits
// of a prefix34 relocation field. This assumes the input is already restricted to
// 34 bits.
func computePrefix34HI(val int64) uint32 {
	return uint32((val >> 16) & 0x3FFFF)
}

func computeTLSLEReloc(target *ld.Target, ldr *loader.Loader, rs, s loader.Sym) int64 {
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

	if int64(int32(v)) != v {
		ldr.Errorf(s, "TLS offset out of range %d", v)
	}
	return v
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
		case objabi.R_POWER_TLS, objabi.R_POWER_TLS_IE_PCREL34, objabi.R_POWER_TLS_LE_TPREL34, objabi.R_ADDRPOWER_GOT_PCREL34:
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
				nval |= int64(OP_NOP)                                      // nop
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
		case objabi.R_CALLPOWER, objabi.R_ADDRPOWER_D34, objabi.R_ADDRPOWER_PCREL34:
			nExtReloc = 1
			if !target.IsAIX() {
				return val, nExtReloc, true
			}
		}
	}

	switch r.Type() {
	case objabi.R_ADDRPOWER_TOCREL, objabi.R_ADDRPOWER_TOCREL_DS:
		return archreloctoc(ldr, target, syms, r, s, val), nExtReloc, true
	case objabi.R_ADDRPOWER, objabi.R_ADDRPOWER_DS, objabi.R_ADDRPOWER_D34, objabi.R_ADDRPOWER_PCREL34:
		return archrelocaddr(ldr, target, syms, r, s, val), nExtReloc, true
	case objabi.R_CALLPOWER:
		// Bits 6 through 29 = (S + A - P) >> 2

		t := ldr.SymValue(rs) + r.Add() - (ldr.SymValue(s) + int64(r.Off()))

		tgtName := ldr.SymName(rs)

		// If we are linking PIE or shared code, non-PCrel golang generated object files have an extra 2 instruction prologue
		// to regenerate the TOC pointer from R12.  The exception are two special case functions tested below.  Note,
		// local call offsets for externally generated objects are accounted for when converting into golang relocs.
		if !hasPCrel && !ldr.AttrExternal(rs) && ldr.AttrShared(rs) && tgtName != "runtime.duffzero" && tgtName != "runtime.duffcopy" {
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
		ha, l := unpackInstPair(target, val)
		l |= computeLO(int32(t))
		ha |= computeHA(int32(t))
		return packInstPair(target, ha, l), nExtReloc, true

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

		const OP_MASK = 0x3F << 26
		const OP_RA_MASK = 0x1F << 16
		// convert r2 to r0, and ld to addi
		mask := packInstPair(target, OP_RA_MASK, OP_MASK)
		addi_op := packInstPair(target, 0, OP_ADDI)
		val &^= mask
		val |= addi_op
		fallthrough

	case objabi.R_POWER_TLS_LE:
		v := computeTLSLEReloc(target, ldr, rs, s)
		o1, o2 := unpackInstPair(target, val)
		o1 |= computeHA(int32(v))
		o2 |= computeLO(int32(v))
		return packInstPair(target, o1, o2), nExtReloc, true

	case objabi.R_POWER_TLS_IE_PCREL34:
		// Convert TLS_IE relocation to TLS_LE if supported.
		if !(target.IsPIE() && target.IsElf()) {
			log.Fatalf("cannot handle R_POWER_TLS_IE (sym %s) when linking non-PIE, non-ELF binaries internally", ldr.SymName(s))
		}

		// We are an ELF binary, we can safely convert to TLS_LE_TPREL34 from:
		// pld rX, x@got@tprel@pcrel
		//
		// to TLS_LE_TPREL32 by converting to:
		// pla rX, x@tprel

		const OP_MASK_PFX = 0xFFFFFFFF        // Discard prefix word
		const OP_MASK = (0x3F << 26) | 0xFFFF // Preserve RT, RA
		const OP_PFX = 1<<26 | 2<<24
		const OP_PLA = 14 << 26
		mask := packInstPair(target, OP_MASK_PFX, OP_MASK)
		pla_op := packInstPair(target, OP_PFX, OP_PLA)
		val &^= mask
		val |= pla_op
		fallthrough

	case objabi.R_POWER_TLS_LE_TPREL34:
		v := computeTLSLEReloc(target, ldr, rs, s)
		o1, o2 := unpackInstPair(target, val)
		o1 |= computePrefix34HI(v)
		o2 |= computeLO(int32(v))
		return packInstPair(target, o1, o2), nExtReloc, true
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
	case objabi.R_POWER_TLS, objabi.R_POWER_TLS_LE, objabi.R_POWER_TLS_IE, objabi.R_POWER_TLS_IE_PCREL34, objabi.R_POWER_TLS_LE_TPREL34, objabi.R_CALLPOWER:
		return ld.ExtrelocSimple(ldr, r), true
	case objabi.R_ADDRPOWER,
		objabi.R_ADDRPOWER_DS,
		objabi.R_ADDRPOWER_TOCREL,
		objabi.R_ADDRPOWER_TOCREL_DS,
		objabi.R_ADDRPOWER_GOT,
		objabi.R_ADDRPOWER_GOT_PCREL34,
		objabi.R_ADDRPOWER_PCREL,
		objabi.R_ADDRPOWER_D34,
		objabi.R_ADDRPOWER_PCREL34:
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

// Generate the glink resolver stub if necessary and return the .glink section.
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
	glink.AddUint32(ctxt.Arch, OP_MFLR_R0) // mflr r0
	glink.AddUint32(ctxt.Arch, OP_BCL_NIA) // bcl 20,31,1f
	glink.AddUint32(ctxt.Arch, 0x7d6802a6) // 1: mflr r11
	glink.AddUint32(ctxt.Arch, OP_MTLR_R0) // mtlr r0

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
	glink.AddUint32(ctxt.Arch, OP_MTCTR_R12) // mtctr r12
	glink.AddUint32(ctxt.Arch, OP_BCTR)      // bctr

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
