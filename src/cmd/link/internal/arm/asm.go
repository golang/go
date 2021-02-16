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

package arm

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

// This assembler:
//
//         .align 2
// local.dso_init:
//         ldr r0, .Lmoduledata
// .Lloadfrom:
//         ldr r0, [r0]
//         b runtime.addmoduledata@plt
// .align 2
// .Lmoduledata:
//         .word local.moduledata(GOT_PREL) + (. - (.Lloadfrom + 4))
// assembles to:
//
// 00000000 <local.dso_init>:
//    0:        e59f0004        ldr     r0, [pc, #4]    ; c <local.dso_init+0xc>
//    4:        e5900000        ldr     r0, [r0]
//    8:        eafffffe        b       0 <runtime.addmoduledata>
//                      8: R_ARM_JUMP24 runtime.addmoduledata
//    c:        00000004        .word   0x00000004
//                      c: R_ARM_GOT_PREL       local.moduledata

func gentext(ctxt *ld.Link, ldr *loader.Loader) {
	initfunc, addmoduledata := ld.PrepareAddmoduledata(ctxt)
	if initfunc == nil {
		return
	}

	o := func(op uint32) {
		initfunc.AddUint32(ctxt.Arch, op)
	}
	o(0xe59f0004)
	o(0xe08f0000)

	o(0xeafffffe)
	rel, _ := initfunc.AddRel(objabi.R_CALLARM)
	rel.SetOff(8)
	rel.SetSiz(4)
	rel.SetSym(addmoduledata)
	rel.SetAdd(0xeafffffe) // vomit

	o(0x00000000)

	rel2, _ := initfunc.AddRel(objabi.R_PCREL)
	rel2.SetOff(12)
	rel2.SetSiz(4)
	rel2.SetSym(ctxt.Moduledata)
	rel2.SetAdd(4)
}

// Preserve highest 8 bits of a, and do addition to lower 24-bit
// of a and b; used to adjust ARM branch instruction's target
func braddoff(a int32, b int32) int32 {
	return int32((uint32(a))&0xff000000 | 0x00ffffff&uint32(a+b))
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
			ldr.Errorf(s, "unexpected relocation type %d (%s)", r.Type(), sym.RelocName(target.Arch, r.Type()))
			return false
		}

		// Handle relocations found in ELF object files.
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_PLT32):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_CALLARM)

		if targType == sym.SDYNIMPORT {
			addpltsym(target, ldr, syms, targ)
			su.SetRelocSym(rIdx, syms.PLT)
			su.SetRelocAdd(rIdx, int64(braddoff(int32(r.Add()), ldr.SymPlt(targ)/4)))
		}

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_THM_PC22): // R_ARM_THM_CALL
		ld.Exitf("R_ARM_THM_CALL, are you using -marm?")
		return false

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_GOT32): // R_ARM_GOT_BREL
		if targType != sym.SDYNIMPORT {
			addgotsyminternal(target, ldr, syms, targ)
		} else {
			ld.AddGotSym(target, ldr, syms, targ, uint32(elf.R_ARM_GLOB_DAT))
		}

		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_CONST) // write r->add during relocsym
		su.SetRelocSym(rIdx, 0)
		su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymGot(targ)))
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_GOT_PREL): // GOT(nil) + A - nil
		if targType != sym.SDYNIMPORT {
			addgotsyminternal(target, ldr, syms, targ)
		} else {
			ld.AddGotSym(target, ldr, syms, targ, uint32(elf.R_ARM_GLOB_DAT))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		su.SetRelocSym(rIdx, syms.GOT)
		su.SetRelocAdd(rIdx, r.Add()+4+int64(ldr.SymGot(targ)))
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_GOTOFF): // R_ARM_GOTOFF32
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_GOTOFF)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_GOTPC): // R_ARM_BASE_PREL
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		su.SetRelocSym(rIdx, syms.GOT)
		su.SetRelocAdd(rIdx, r.Add()+4)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_CALL):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_CALLARM)
		if targType == sym.SDYNIMPORT {
			addpltsym(target, ldr, syms, targ)
			su.SetRelocSym(rIdx, syms.PLT)
			su.SetRelocAdd(rIdx, int64(braddoff(int32(r.Add()), ldr.SymPlt(targ)/4)))
		}
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_REL32): // R_ARM_REL32
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		su.SetRelocAdd(rIdx, r.Add()+4)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_ABS32):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected R_ARM_ABS32 relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ADDR)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_PC24),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_JUMP24):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_CALLARM)
		if targType == sym.SDYNIMPORT {
			addpltsym(target, ldr, syms, targ)
			su.SetRelocSym(rIdx, syms.PLT)
			su.SetRelocAdd(rIdx, int64(braddoff(int32(r.Add()), ldr.SymPlt(targ)/4)))
		}

		return true
	}

	// Handle references to ELF symbols from our own object files.
	if targType != sym.SDYNIMPORT {
		return true
	}

	// Reread the reloc to incorporate any changes in type above.
	relocs := ldr.Relocs(s)
	r = relocs.At(rIdx)

	switch r.Type() {
	case objabi.R_CALLARM:
		if target.IsExternal() {
			// External linker will do this relocation.
			return true
		}
		addpltsym(target, ldr, syms, targ)
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocSym(rIdx, syms.PLT)
		su.SetRelocAdd(rIdx, int64(braddoff(int32(r.Add()), ldr.SymPlt(targ)/4))) // TODO: don't use r.Add for instruction bytes (issue 19811)
		return true

	case objabi.R_ADDR:
		if ldr.SymType(s) != sym.SDATA {
			break
		}
		if target.IsElf() {
			ld.Adddynsym(ldr, target, syms, targ)
			rel := ldr.MakeSymbolUpdater(syms.Rel)
			rel.AddAddrPlus(target.Arch, s, int64(r.Off()))
			rel.AddUint32(target.Arch, elf.R_INFO32(uint32(ldr.SymDynid(targ)), uint32(elf.R_ARM_GLOB_DAT))) // we need a nil + A dynamic reloc
			su := ldr.MakeSymbolUpdater(s)
			su.SetRelocType(rIdx, objabi.R_CONST) // write r->add during relocsym
			su.SetRelocSym(rIdx, 0)
			return true
		}
	}

	return false
}

func elfreloc1(ctxt *ld.Link, out *ld.OutBuf, ldr *loader.Loader, s loader.Sym, r loader.ExtReloc, ri int, sectoff int64) bool {
	out.Write32(uint32(sectoff))

	elfsym := ld.ElfSymForReloc(ctxt, r.Xsym)
	siz := r.Size
	switch r.Type {
	default:
		return false
	case objabi.R_ADDR, objabi.R_DWARFSECREF:
		if siz == 4 {
			out.Write32(uint32(elf.R_ARM_ABS32) | uint32(elfsym)<<8)
		} else {
			return false
		}
	case objabi.R_PCREL:
		if siz == 4 {
			out.Write32(uint32(elf.R_ARM_REL32) | uint32(elfsym)<<8)
		} else {
			return false
		}
	case objabi.R_CALLARM:
		if siz == 4 {
			relocs := ldr.Relocs(s)
			r := relocs.At(ri)
			if r.Add()&0xff000000 == 0xeb000000 { // BL // TODO: using r.Add here is bad (issue 19811)
				out.Write32(uint32(elf.R_ARM_CALL) | uint32(elfsym)<<8)
			} else {
				out.Write32(uint32(elf.R_ARM_JUMP24) | uint32(elfsym)<<8)
			}
		} else {
			return false
		}
	case objabi.R_TLS_LE:
		out.Write32(uint32(elf.R_ARM_TLS_LE32) | uint32(elfsym)<<8)
	case objabi.R_TLS_IE:
		out.Write32(uint32(elf.R_ARM_TLS_IE32) | uint32(elfsym)<<8)
	case objabi.R_GOTPCREL:
		if siz == 4 {
			out.Write32(uint32(elf.R_ARM_GOT_PREL) | uint32(elfsym)<<8)
		} else {
			return false
		}
	}

	return true
}

func elfsetupplt(ctxt *ld.Link, plt, got *loader.SymbolBuilder, dynamic loader.Sym) {
	if plt.Size() == 0 {
		// str lr, [sp, #-4]!
		plt.AddUint32(ctxt.Arch, 0xe52de004)

		// ldr lr, [pc, #4]
		plt.AddUint32(ctxt.Arch, 0xe59fe004)

		// add lr, pc, lr
		plt.AddUint32(ctxt.Arch, 0xe08fe00e)

		// ldr pc, [lr, #8]!
		plt.AddUint32(ctxt.Arch, 0xe5bef008)

		// .word &GLOBAL_OFFSET_TABLE[0] - .
		plt.AddPCRelPlus(ctxt.Arch, got.Sym(), 4)

		// the first .plt entry requires 3 .plt.got entries
		got.AddUint32(ctxt.Arch, 0)

		got.AddUint32(ctxt.Arch, 0)
		got.AddUint32(ctxt.Arch, 0)
	}
}

func machoreloc1(*sys.Arch, *ld.OutBuf, *loader.Loader, loader.Sym, loader.ExtReloc, int64) bool {
	return false
}

func pereloc1(arch *sys.Arch, out *ld.OutBuf, ldr *loader.Loader, s loader.Sym, r loader.ExtReloc, sectoff int64) bool {
	rs := r.Xsym
	rt := r.Type

	if ldr.SymDynid(rs) < 0 {
		ldr.Errorf(s, "reloc %d (%s) to non-coff symbol %s type=%d (%s)", rt, sym.RelocName(arch, rt), ldr.SymName(rs), ldr.SymType(rs), ldr.SymType(rs))
		return false
	}

	out.Write32(uint32(sectoff))
	out.Write32(uint32(ldr.SymDynid(rs)))

	var v uint32
	switch rt {
	default:
		// unsupported relocation type
		return false

	case objabi.R_DWARFSECREF:
		v = ld.IMAGE_REL_ARM_SECREL

	case objabi.R_ADDR:
		v = ld.IMAGE_REL_ARM_ADDR32
	}

	out.Write16(uint16(v))

	return true
}

// sign extend a 24-bit integer
func signext24(x int64) int32 {
	return (int32(x) << 8) >> 8
}

// encode an immediate in ARM's imm12 format. copied from ../../../internal/obj/arm/asm5.go
func immrot(v uint32) uint32 {
	for i := 0; i < 16; i++ {
		if v&^0xff == 0 {
			return uint32(i<<8) | v | 1<<25
		}
		v = v<<2 | v>>30
	}
	return 0
}

// Convert the direct jump relocation r to refer to a trampoline if the target is too far
func trampoline(ctxt *ld.Link, ldr *loader.Loader, ri int, rs, s loader.Sym) {
	relocs := ldr.Relocs(s)
	r := relocs.At(ri)
	switch r.Type() {
	case objabi.R_CALLARM:
		var t int64
		// ldr.SymValue(rs) == 0 indicates a cross-package jump to a function that is not yet
		// laid out. Conservatively use a trampoline. This should be rare, as we lay out packages
		// in dependency order.
		if ldr.SymValue(rs) != 0 {
			// r.Add is the instruction
			// low 24-bit encodes the target address
			t = (ldr.SymValue(rs) + int64(signext24(r.Add()&0xffffff)*4) - (ldr.SymValue(s) + int64(r.Off()))) / 4
		}
		if t > 0x7fffff || t < -0x800000 || ldr.SymValue(rs) == 0 || (*ld.FlagDebugTramp > 1 && ldr.SymPkg(s) != ldr.SymPkg(rs)) {
			// direct call too far, need to insert trampoline.
			// look up existing trampolines first. if we found one within the range
			// of direct call, we can reuse it. otherwise create a new one.
			offset := (signext24(r.Add()&0xffffff) + 2) * 4
			var tramp loader.Sym
			for i := 0; ; i++ {
				oName := ldr.SymName(rs)
				name := oName + fmt.Sprintf("%+d-tramp%d", offset, i)
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

				t = (ldr.SymValue(tramp) - 8 - (ldr.SymValue(s) + int64(r.Off()))) / 4
				if t >= -0x800000 && t < 0x7fffff {
					// found an existing trampoline that is not too far
					// we can just use it
					break
				}
			}
			if ldr.SymType(tramp) == 0 {
				// trampoline does not exist, create one
				trampb := ldr.MakeSymbolUpdater(tramp)
				ctxt.AddTramp(trampb)
				if ctxt.DynlinkingGo() {
					if immrot(uint32(offset)) == 0 {
						ctxt.Errorf(s, "odd offset in dynlink direct call: %v+%d", ldr.SymName(rs), offset)
					}
					gentrampdyn(ctxt.Arch, trampb, rs, int64(offset))
				} else if ctxt.BuildMode == ld.BuildModeCArchive || ctxt.BuildMode == ld.BuildModeCShared || ctxt.BuildMode == ld.BuildModePIE {
					gentramppic(ctxt.Arch, trampb, rs, int64(offset))
				} else {
					gentramp(ctxt.Arch, ctxt.LinkMode, ldr, trampb, rs, int64(offset))
				}
			}
			// modify reloc to point to tramp, which will be resolved later
			sb := ldr.MakeSymbolUpdater(s)
			relocs := sb.Relocs()
			r := relocs.At(ri)
			r.SetSym(tramp)
			r.SetAdd(r.Add()&0xff000000 | 0xfffffe) // clear the offset embedded in the instruction
		}
	default:
		ctxt.Errorf(s, "trampoline called with non-jump reloc: %d (%s)", r.Type(), sym.RelocName(ctxt.Arch, r.Type()))
	}
}

// generate a trampoline to target+offset
func gentramp(arch *sys.Arch, linkmode ld.LinkMode, ldr *loader.Loader, tramp *loader.SymbolBuilder, target loader.Sym, offset int64) {
	tramp.SetSize(12) // 3 instructions
	P := make([]byte, tramp.Size())
	t := ldr.SymValue(target) + offset
	o1 := uint32(0xe5900000 | 11<<12 | 15<<16) // MOVW (R15), R11 // R15 is actual pc + 8
	o2 := uint32(0xe12fff10 | 11)              // JMP  (R11)
	o3 := uint32(t)                            // WORD $target
	arch.ByteOrder.PutUint32(P, o1)
	arch.ByteOrder.PutUint32(P[4:], o2)
	arch.ByteOrder.PutUint32(P[8:], o3)
	tramp.SetData(P)

	if linkmode == ld.LinkExternal || ldr.SymValue(target) == 0 {
		r, _ := tramp.AddRel(objabi.R_ADDR)
		r.SetOff(8)
		r.SetSiz(4)
		r.SetSym(target)
		r.SetAdd(offset)
	}
}

// generate a trampoline to target+offset in position independent code
func gentramppic(arch *sys.Arch, tramp *loader.SymbolBuilder, target loader.Sym, offset int64) {
	tramp.SetSize(16) // 4 instructions
	P := make([]byte, tramp.Size())
	o1 := uint32(0xe5900000 | 11<<12 | 15<<16 | 4)  // MOVW 4(R15), R11 // R15 is actual pc + 8
	o2 := uint32(0xe0800000 | 11<<12 | 15<<16 | 11) // ADD R15, R11, R11
	o3 := uint32(0xe12fff10 | 11)                   // JMP  (R11)
	o4 := uint32(0)                                 // WORD $(target-pc) // filled in with relocation
	arch.ByteOrder.PutUint32(P, o1)
	arch.ByteOrder.PutUint32(P[4:], o2)
	arch.ByteOrder.PutUint32(P[8:], o3)
	arch.ByteOrder.PutUint32(P[12:], o4)
	tramp.SetData(P)

	r, _ := tramp.AddRel(objabi.R_PCREL)
	r.SetOff(12)
	r.SetSiz(4)
	r.SetSym(target)
	r.SetAdd(offset + 4)
}

// generate a trampoline to target+offset in dynlink mode (using GOT)
func gentrampdyn(arch *sys.Arch, tramp *loader.SymbolBuilder, target loader.Sym, offset int64) {
	tramp.SetSize(20)                               // 5 instructions
	o1 := uint32(0xe5900000 | 11<<12 | 15<<16 | 8)  // MOVW 8(R15), R11 // R15 is actual pc + 8
	o2 := uint32(0xe0800000 | 11<<12 | 15<<16 | 11) // ADD R15, R11, R11
	o3 := uint32(0xe5900000 | 11<<12 | 11<<16)      // MOVW (R11), R11
	o4 := uint32(0xe12fff10 | 11)                   // JMP  (R11)
	o5 := uint32(0)                                 // WORD $target@GOT // filled in with relocation
	o6 := uint32(0)
	if offset != 0 {
		// insert an instruction to add offset
		tramp.SetSize(24) // 6 instructions
		o6 = o5
		o5 = o4
		o4 = 0xe2800000 | 11<<12 | 11<<16 | immrot(uint32(offset)) // ADD $offset, R11, R11
		o1 = uint32(0xe5900000 | 11<<12 | 15<<16 | 12)             // MOVW 12(R15), R11
	}
	P := make([]byte, tramp.Size())
	arch.ByteOrder.PutUint32(P, o1)
	arch.ByteOrder.PutUint32(P[4:], o2)
	arch.ByteOrder.PutUint32(P[8:], o3)
	arch.ByteOrder.PutUint32(P[12:], o4)
	arch.ByteOrder.PutUint32(P[16:], o5)
	if offset != 0 {
		arch.ByteOrder.PutUint32(P[20:], o6)
	}
	tramp.SetData(P)

	r, _ := tramp.AddRel(objabi.R_GOTPCREL)
	r.SetOff(16)
	r.SetSiz(4)
	r.SetSym(target)
	r.SetAdd(8)
	if offset != 0 {
		// increase reloc offset by 4 as we inserted an ADD instruction
		r.SetOff(20)
		r.SetAdd(12)
	}
}

func archreloc(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, r loader.Reloc, s loader.Sym, val int64) (o int64, nExtReloc int, ok bool) {
	rs := r.Sym()
	rs = ldr.ResolveABIAlias(rs)
	if target.IsExternal() {
		switch r.Type() {
		case objabi.R_CALLARM:
			// set up addend for eventual relocation via outer symbol.
			_, off := ld.FoldSubSymbolOffset(ldr, rs)
			xadd := int64(signext24(r.Add()&0xffffff))*4 + off
			if xadd/4 > 0x7fffff || xadd/4 < -0x800000 {
				ldr.Errorf(s, "direct call too far %d", xadd/4)
			}
			return int64(braddoff(int32(0xff000000&uint32(r.Add())), int32(0xffffff&uint32(xadd/4)))), 1, true
		}
		return -1, 0, false
	}

	const isOk = true
	const noExtReloc = 0
	switch r.Type() {
	// The following three arch specific relocations are only for generation of
	// Linux/ARM ELF's PLT entry (3 assembler instruction)
	case objabi.R_PLT0: // add ip, pc, #0xXX00000
		if ldr.SymValue(syms.GOTPLT) < ldr.SymValue(syms.PLT) {
			ldr.Errorf(s, ".got.plt should be placed after .plt section.")
		}
		return 0xe28fc600 + (0xff & (int64(uint32(ldr.SymValue(rs)-(ldr.SymValue(syms.PLT)+int64(r.Off()))+r.Add())) >> 20)), noExtReloc, isOk
	case objabi.R_PLT1: // add ip, ip, #0xYY000
		return 0xe28cca00 + (0xff & (int64(uint32(ldr.SymValue(rs)-(ldr.SymValue(syms.PLT)+int64(r.Off()))+r.Add()+4)) >> 12)), noExtReloc, isOk
	case objabi.R_PLT2: // ldr pc, [ip, #0xZZZ]!
		return 0xe5bcf000 + (0xfff & int64(uint32(ldr.SymValue(rs)-(ldr.SymValue(syms.PLT)+int64(r.Off()))+r.Add()+8))), noExtReloc, isOk
	case objabi.R_CALLARM: // bl XXXXXX or b YYYYYY
		// r.Add is the instruction
		// low 24-bit encodes the target address
		t := (ldr.SymValue(rs) + int64(signext24(r.Add()&0xffffff)*4) - (ldr.SymValue(s) + int64(r.Off()))) / 4
		if t > 0x7fffff || t < -0x800000 {
			ldr.Errorf(s, "direct call too far: %s %x", ldr.SymName(rs), t)
		}
		return int64(braddoff(int32(0xff000000&uint32(r.Add())), int32(0xffffff&t))), noExtReloc, isOk
	}

	return val, 0, false
}

func archrelocvariant(*ld.Target, *loader.Loader, loader.Reloc, sym.RelocVariant, loader.Sym, int64) int64 {
	log.Fatalf("unexpected relocation variant")
	return -1
}

func extreloc(target *ld.Target, ldr *loader.Loader, r loader.Reloc, s loader.Sym) (loader.ExtReloc, bool) {
	rs := ldr.ResolveABIAlias(r.Sym())
	var rr loader.ExtReloc
	switch r.Type() {
	case objabi.R_CALLARM:
		// set up addend for eventual relocation via outer symbol.
		rs, off := ld.FoldSubSymbolOffset(ldr, rs)
		rr.Xadd = int64(signext24(r.Add()&0xffffff))*4 + off
		rst := ldr.SymType(rs)
		if rst != sym.SHOSTOBJ && rst != sym.SDYNIMPORT && rst != sym.SUNDEFEXT && ldr.SymSect(rs) == nil {
			ldr.Errorf(s, "missing section for %s", ldr.SymName(rs))
		}
		rr.Xsym = rs
		rr.Type = r.Type()
		rr.Size = r.Siz()
		return rr, true
	}
	return rr, false
}

func addpltreloc(ldr *loader.Loader, plt *loader.SymbolBuilder, got *loader.SymbolBuilder, s loader.Sym, typ objabi.RelocType) {
	r, _ := plt.AddRel(typ)
	r.SetSym(got.Sym())
	r.SetOff(int32(plt.Size()))
	r.SetSiz(4)
	r.SetAdd(int64(ldr.SymGot(s)) - 8)

	plt.SetReachable(true)
	plt.SetSize(plt.Size() + 4)
	plt.Grow(plt.Size())
}

func addpltsym(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s loader.Sym) {
	if ldr.SymPlt(s) >= 0 {
		return
	}

	ld.Adddynsym(ldr, target, syms, s)

	if target.IsElf() {
		plt := ldr.MakeSymbolUpdater(syms.PLT)
		got := ldr.MakeSymbolUpdater(syms.GOTPLT)
		rel := ldr.MakeSymbolUpdater(syms.RelPLT)
		if plt.Size() == 0 {
			panic("plt is not set up")
		}

		// .got entry
		ldr.SetGot(s, int32(got.Size()))

		// In theory, all GOT should point to the first PLT entry,
		// Linux/ARM's dynamic linker will do that for us, but FreeBSD/ARM's
		// dynamic linker won't, so we'd better do it ourselves.
		got.AddAddrPlus(target.Arch, plt.Sym(), 0)

		// .plt entry, this depends on the .got entry
		ldr.SetPlt(s, int32(plt.Size()))

		addpltreloc(ldr, plt, got, s, objabi.R_PLT0) // add lr, pc, #0xXX00000
		addpltreloc(ldr, plt, got, s, objabi.R_PLT1) // add lr, lr, #0xYY000
		addpltreloc(ldr, plt, got, s, objabi.R_PLT2) // ldr pc, [lr, #0xZZZ]!

		// rel
		rel.AddAddrPlus(target.Arch, got.Sym(), int64(ldr.SymGot(s)))

		rel.AddUint32(target.Arch, elf.R_INFO32(uint32(ldr.SymDynid(s)), uint32(elf.R_ARM_JUMP_SLOT)))
	} else {
		ldr.Errorf(s, "addpltsym: unsupported binary format")
	}
}

func addgotsyminternal(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s loader.Sym) {
	if ldr.SymGot(s) >= 0 {
		return
	}

	got := ldr.MakeSymbolUpdater(syms.GOT)
	ldr.SetGot(s, int32(got.Size()))
	got.AddAddrPlus(target.Arch, s, 0)

	if target.IsElf() {
	} else {
		ldr.Errorf(s, "addgotsyminternal: unsupported binary format")
	}
}
