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
	"sync"
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

func gentext2(ctxt *ld.Link, ldr *loader.Loader) {
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
	rel := loader.Reloc{
		Off:  8,
		Size: 4,
		Type: objabi.R_CALLARM,
		Sym:  addmoduledata,
		Add:  0xeafffffe, // vomit
	}
	initfunc.AddReloc(rel)

	o(0x00000000)

	rel2 := loader.Reloc{
		Off:  12,
		Size: 4,
		Type: objabi.R_PCREL,
		Sym:  ctxt.Moduledata2,
		Add:  4,
	}
	initfunc.AddReloc(rel2)
}

// Preserve highest 8 bits of a, and do addition to lower 24-bit
// of a and b; used to adjust ARM branch instruction's target
func braddoff(a int32, b int32) int32 {
	return int32((uint32(a))&0xff000000 | 0x00ffffff&uint32(a+b))
}

func adddynrel2(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s loader.Sym, r loader.Reloc2, rIdx int) bool {

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
			addpltsym2(target, ldr, syms, targ)
			su.SetRelocSym(rIdx, syms.PLT2)
			su.SetRelocAdd(rIdx, int64(braddoff(int32(r.Add()), ldr.SymPlt(targ)/4)))
		}

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_THM_PC22): // R_ARM_THM_CALL
		ld.Exitf("R_ARM_THM_CALL, are you using -marm?")
		return false

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_GOT32): // R_ARM_GOT_BREL
		if targType != sym.SDYNIMPORT {
			addgotsyminternal2(target, ldr, syms, targ)
		} else {
			addgotsym2(target, ldr, syms, targ)
		}

		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_CONST) // write r->add during relocsym
		su.SetRelocSym(rIdx, 0)
		su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymGot(targ)))
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_GOT_PREL): // GOT(nil) + A - nil
		if targType != sym.SDYNIMPORT {
			addgotsyminternal2(target, ldr, syms, targ)
		} else {
			addgotsym2(target, ldr, syms, targ)
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		su.SetRelocSym(rIdx, syms.GOT2)
		su.SetRelocAdd(rIdx, r.Add()+4+int64(ldr.SymGot(targ)))
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_GOTOFF): // R_ARM_GOTOFF32
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_GOTOFF)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_GOTPC): // R_ARM_BASE_PREL
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		su.SetRelocSym(rIdx, syms.GOT2)
		su.SetRelocAdd(rIdx, r.Add()+4)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_CALL):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_CALLARM)
		if targType == sym.SDYNIMPORT {
			addpltsym2(target, ldr, syms, targ)
			su.SetRelocSym(rIdx, syms.PLT2)
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
			addpltsym2(target, ldr, syms, targ)
			su.SetRelocSym(rIdx, syms.PLT2)
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
	r = relocs.At2(rIdx)

	switch r.Type() {
	case objabi.R_CALLARM:
		if target.IsExternal() {
			// External linker will do this relocation.
			return true
		}
		addpltsym2(target, ldr, syms, targ)
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocSym(rIdx, syms.PLT2)
		su.SetRelocAdd(rIdx, int64(ldr.SymPlt(targ)))
		return true

	case objabi.R_ADDR:
		if ldr.SymType(s) != sym.SDATA {
			break
		}
		if target.IsElf() {
			ld.Adddynsym2(ldr, target, syms, targ)
			rel := ldr.MakeSymbolUpdater(syms.Rel2)
			rel.AddAddrPlus(target.Arch, s, int64(r.Off()))
			rel.AddUint32(target.Arch, ld.ELF32_R_INFO(uint32(ldr.SymDynid(targ)), uint32(elf.R_ARM_GLOB_DAT))) // we need a nil + A dynamic reloc
			su := ldr.MakeSymbolUpdater(s)
			su.SetRelocType(rIdx, objabi.R_CONST) // write r->add during relocsym
			su.SetRelocSym(rIdx, 0)
			return true
		}
	}

	return false
}

func elfreloc1(ctxt *ld.Link, r *sym.Reloc, sectoff int64) bool {
	ctxt.Out.Write32(uint32(sectoff))

	elfsym := ld.ElfSymForReloc(ctxt, r.Xsym)
	switch r.Type {
	default:
		return false
	case objabi.R_ADDR, objabi.R_DWARFSECREF:
		if r.Siz == 4 {
			ctxt.Out.Write32(uint32(elf.R_ARM_ABS32) | uint32(elfsym)<<8)
		} else {
			return false
		}
	case objabi.R_PCREL:
		if r.Siz == 4 {
			ctxt.Out.Write32(uint32(elf.R_ARM_REL32) | uint32(elfsym)<<8)
		} else {
			return false
		}
	case objabi.R_CALLARM:
		if r.Siz == 4 {
			if r.Add&0xff000000 == 0xeb000000 { // BL
				ctxt.Out.Write32(uint32(elf.R_ARM_CALL) | uint32(elfsym)<<8)
			} else {
				ctxt.Out.Write32(uint32(elf.R_ARM_JUMP24) | uint32(elfsym)<<8)
			}
		} else {
			return false
		}
	case objabi.R_TLS_LE:
		ctxt.Out.Write32(uint32(elf.R_ARM_TLS_LE32) | uint32(elfsym)<<8)
	case objabi.R_TLS_IE:
		ctxt.Out.Write32(uint32(elf.R_ARM_TLS_IE32) | uint32(elfsym)<<8)
	case objabi.R_GOTPCREL:
		if r.Siz == 4 {
			ctxt.Out.Write32(uint32(elf.R_ARM_GOT_PREL) | uint32(elfsym)<<8)
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

func machoreloc1(arch *sys.Arch, out *ld.OutBuf, s *sym.Symbol, r *sym.Reloc, sectoff int64) bool {
	return false
}

func pereloc1(arch *sys.Arch, out *ld.OutBuf, s *sym.Symbol, r *sym.Reloc, sectoff int64) bool {
	rs := r.Xsym

	if rs.Dynid < 0 {
		ld.Errorf(s, "reloc %d (%s) to non-coff symbol %s type=%d (%s)", r.Type, sym.RelocName(arch, r.Type), rs.Name, rs.Type, rs.Type)
		return false
	}

	out.Write32(uint32(sectoff))
	out.Write32(uint32(rs.Dynid))

	var v uint32
	switch r.Type {
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
	r := relocs.At2(ri)
	switch r.Type() {
	case objabi.R_CALLARM:
		// r.Add is the instruction
		// low 24-bit encodes the target address
		t := (ldr.SymValue(rs) + int64(signext24(r.Add()&0xffffff)*4) - (ldr.SymValue(s) + int64(r.Off()))) / 4
		if t > 0x7fffff || t < -0x800000 || (*ld.FlagDebugTramp > 1 && ldr.SymPkg(s) != ldr.SymPkg(rs)) {
			// direct call too far, need to insert trampoline.
			// look up existing trampolines first. if we found one within the range
			// of direct call, we can reuse it. otherwise create a new one.
			offset := (signext24(r.Add()&0xffffff) + 2) * 4
			var tramp loader.Sym
			for i := 0; ; i++ {
				oName := ldr.SymName(rs)
				name := oName + fmt.Sprintf("%+d-tramp%d", offset, i)
				tramp = ldr.LookupOrCreateSym(name, int(ldr.SymVersion(rs)))
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
			r := relocs.At2(ri)
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

	if linkmode == ld.LinkExternal {
		r := loader.Reloc{
			Off:  8,
			Type: objabi.R_ADDR,
			Size: 4,
			Sym:  target,
			Add:  offset,
		}
		tramp.AddReloc(r)
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

	r := loader.Reloc{
		Off:  12,
		Type: objabi.R_PCREL,
		Size: 4,
		Sym:  target,
		Add:  offset + 4,
	}
	tramp.AddReloc(r)
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

	r := loader.Reloc{
		Off:  16,
		Type: objabi.R_GOTPCREL,
		Size: 4,
		Sym:  target,
		Add:  8,
	}
	if offset != 0 {
		// increase reloc offset by 4 as we inserted an ADD instruction
		r.Off = 20
		r.Add = 12
	}
	tramp.AddReloc(r)
}

func archreloc(target *ld.Target, syms *ld.ArchSyms, r *sym.Reloc, s *sym.Symbol, val int64) (int64, bool) {
	if target.IsExternal() {
		switch r.Type {
		case objabi.R_CALLARM:
			r.Done = false

			// set up addend for eventual relocation via outer symbol.
			rs := r.Sym

			r.Xadd = int64(signext24(r.Add & 0xffffff))
			r.Xadd *= 4
			for rs.Outer != nil {
				r.Xadd += ld.Symaddr(rs) - ld.Symaddr(rs.Outer)
				rs = rs.Outer
			}

			if rs.Type != sym.SHOSTOBJ && rs.Type != sym.SDYNIMPORT && rs.Type != sym.SUNDEFEXT && rs.Sect == nil {
				ld.Errorf(s, "missing section for %s", rs.Name)
			}
			r.Xsym = rs

			// ld64 for arm seems to want the symbol table to contain offset
			// into the section rather than pseudo virtual address that contains
			// the section load address.
			// we need to compensate that by removing the instruction's address
			// from addend.
			if target.IsDarwin() {
				r.Xadd -= ld.Symaddr(s) + int64(r.Off)
			}

			if r.Xadd/4 > 0x7fffff || r.Xadd/4 < -0x800000 {
				ld.Errorf(s, "direct call too far %d", r.Xadd/4)
			}

			return int64(braddoff(int32(0xff000000&uint32(r.Add)), int32(0xffffff&uint32(r.Xadd/4)))), true
		}

		return -1, false
	}

	switch r.Type {
	case objabi.R_CONST:
		return r.Add, true
	case objabi.R_GOTOFF:
		return ld.Symaddr(r.Sym) + r.Add - ld.Symaddr(syms.GOT), true

	// The following three arch specific relocations are only for generation of
	// Linux/ARM ELF's PLT entry (3 assembler instruction)
	case objabi.R_PLT0: // add ip, pc, #0xXX00000
		if ld.Symaddr(syms.GOTPLT) < ld.Symaddr(syms.PLT) {
			ld.Errorf(s, ".got.plt should be placed after .plt section.")
		}
		return 0xe28fc600 + (0xff & (int64(uint32(ld.Symaddr(r.Sym)-(ld.Symaddr(syms.PLT)+int64(r.Off))+r.Add)) >> 20)), true
	case objabi.R_PLT1: // add ip, ip, #0xYY000
		return 0xe28cca00 + (0xff & (int64(uint32(ld.Symaddr(r.Sym)-(ld.Symaddr(syms.PLT)+int64(r.Off))+r.Add+4)) >> 12)), true
	case objabi.R_PLT2: // ldr pc, [ip, #0xZZZ]!
		return 0xe5bcf000 + (0xfff & int64(uint32(ld.Symaddr(r.Sym)-(ld.Symaddr(syms.PLT)+int64(r.Off))+r.Add+8))), true
	case objabi.R_CALLARM: // bl XXXXXX or b YYYYYY
		// r.Add is the instruction
		// low 24-bit encodes the target address
		t := (ld.Symaddr(r.Sym) + int64(signext24(r.Add&0xffffff)*4) - (s.Value + int64(r.Off))) / 4
		if t > 0x7fffff || t < -0x800000 {
			ld.Errorf(s, "direct call too far: %s %x", r.Sym.Name, t)
		}
		return int64(braddoff(int32(0xff000000&uint32(r.Add)), int32(0xffffff&t))), true
	}

	return val, false
}

func archrelocvariant(target *ld.Target, syms *ld.ArchSyms, r *sym.Reloc, s *sym.Symbol, t int64) int64 {
	log.Fatalf("unexpected relocation variant")
	return t
}

func addpltreloc2(ldr *loader.Loader, plt *loader.SymbolBuilder, got *loader.SymbolBuilder, s loader.Sym, typ objabi.RelocType) {
	r, _ := plt.AddRel(typ)
	r.SetSym(got.Sym())
	r.SetOff(int32(plt.Size()))
	r.SetSiz(4)
	r.SetAdd(int64(ldr.SymGot(s)) - 8)

	plt.SetReachable(true)
	plt.SetSize(plt.Size() + 4)
	plt.Grow(plt.Size())
}

func addpltsym2(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s loader.Sym) {
	if ldr.SymPlt(s) >= 0 {
		return
	}

	ld.Adddynsym2(ldr, target, syms, s)

	if target.IsElf() {
		plt := ldr.MakeSymbolUpdater(syms.PLT2)
		got := ldr.MakeSymbolUpdater(syms.GOTPLT2)
		rel := ldr.MakeSymbolUpdater(syms.RelPLT2)
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

		addpltreloc2(ldr, plt, got, s, objabi.R_PLT0) // add lr, pc, #0xXX00000
		addpltreloc2(ldr, plt, got, s, objabi.R_PLT1) // add lr, lr, #0xYY000
		addpltreloc2(ldr, plt, got, s, objabi.R_PLT2) // ldr pc, [lr, #0xZZZ]!

		// rel
		rel.AddAddrPlus(target.Arch, got.Sym(), int64(ldr.SymGot(s)))

		rel.AddUint32(target.Arch, ld.ELF32_R_INFO(uint32(ldr.SymDynid(s)), uint32(elf.R_ARM_JUMP_SLOT)))
	} else {
		ldr.Errorf(s, "addpltsym: unsupported binary format")
	}
}

func addgotsyminternal2(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s loader.Sym) {
	if ldr.SymGot(s) >= 0 {
		return
	}

	got := ldr.MakeSymbolUpdater(syms.GOT2)
	ldr.SetGot(s, int32(got.Size()))
	got.AddAddrPlus(target.Arch, s, 0)

	if target.IsElf() {
	} else {
		ldr.Errorf(s, "addgotsyminternal: unsupported binary format")
	}
}

func addgotsym2(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s loader.Sym) {
	if ldr.SymGot(s) >= 0 {
		return
	}

	ld.Adddynsym2(ldr, target, syms, s)
	got := ldr.MakeSymbolUpdater(syms.GOT2)
	ldr.SetGot(s, int32(got.Size()))
	got.AddUint64(target.Arch, 0)

	if target.IsElf() {
		rel := ldr.MakeSymbolUpdater(syms.Rel2)
		rel.AddAddrPlus(target.Arch, got.Sym(), int64(ldr.SymGot(s)))
		rel.AddUint32(target.Arch, ld.ELF32_R_INFO(uint32(ldr.SymDynid(s)), uint32(elf.R_ARM_GLOB_DAT)))
	} else {
		ldr.Errorf(s, "addgotsym: unsupported binary format")
	}
}

func asmb(ctxt *ld.Link, _ *loader.Loader) {
	if ctxt.IsELF {
		ld.Asmbelfsetup()
	}

	var wg sync.WaitGroup
	sect := ld.Segtext.Sections[0]
	offset := sect.Vaddr - ld.Segtext.Vaddr + ld.Segtext.Fileoff
	ld.WriteParallel(&wg, ld.Codeblk, ctxt, offset, sect.Vaddr, sect.Length)

	for _, sect := range ld.Segtext.Sections[1:] {
		offset := sect.Vaddr - ld.Segtext.Vaddr + ld.Segtext.Fileoff
		ld.WriteParallel(&wg, ld.Datblk, ctxt, offset, sect.Vaddr, sect.Length)
	}

	if ld.Segrodata.Filelen > 0 {
		ld.WriteParallel(&wg, ld.Datblk, ctxt, ld.Segrodata.Fileoff, ld.Segrodata.Vaddr, ld.Segrodata.Filelen)
	}

	if ld.Segrelrodata.Filelen > 0 {
		ld.WriteParallel(&wg, ld.Datblk, ctxt, ld.Segrelrodata.Fileoff, ld.Segrelrodata.Vaddr, ld.Segrelrodata.Filelen)
	}

	ld.WriteParallel(&wg, ld.Datblk, ctxt, ld.Segdata.Fileoff, ld.Segdata.Vaddr, ld.Segdata.Filelen)

	ld.WriteParallel(&wg, ld.Dwarfblk, ctxt, ld.Segdwarf.Fileoff, ld.Segdwarf.Vaddr, ld.Segdwarf.Filelen)
	wg.Wait()
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

		case objabi.Hwindows:
			symo = uint32(ld.Segdwarf.Fileoff + ld.Segdwarf.Filelen)
			symo = uint32(ld.Rnd(int64(symo), ld.PEFILEALIGN))
		}

		ctxt.Out.SeekSet(int64(symo))
		switch ctxt.HeadType {
		default:
			if ctxt.IsELF {
				ld.Asmelfsym(ctxt)
				ctxt.Out.Write(ld.Elfstrdat)

				if ctxt.LinkMode == ld.LinkExternal {
					ld.Elfemitreloc(ctxt)
				}
			}

		case objabi.Hplan9:
			ld.Asmplan9sym(ctxt)

			sym := ctxt.Syms.Lookup("pclntab", 0)
			if sym != nil {
				ld.Lcsize = int32(len(sym.P))
				ctxt.Out.Write(sym.P)
			}

		case objabi.Hwindows:
			// Do nothing
		}
	}

	ctxt.Out.SeekSet(0)
	switch ctxt.HeadType {
	default:
	case objabi.Hplan9: /* plan 9 */
		ctxt.Out.Write32b(0x647)                      /* magic */
		ctxt.Out.Write32b(uint32(ld.Segtext.Filelen)) /* sizes */
		ctxt.Out.Write32b(uint32(ld.Segdata.Filelen))
		ctxt.Out.Write32b(uint32(ld.Segdata.Length - ld.Segdata.Filelen))
		ctxt.Out.Write32b(uint32(ld.Symsize))          /* nsyms */
		ctxt.Out.Write32b(uint32(ld.Entryvalue(ctxt))) /* va of entry */
		ctxt.Out.Write32b(0)
		ctxt.Out.Write32b(uint32(ld.Lcsize))

	case objabi.Hlinux,
		objabi.Hfreebsd,
		objabi.Hnetbsd,
		objabi.Hopenbsd:
		ld.Asmbelf(ctxt, int64(symo))

	case objabi.Hwindows:
		ld.Asmbpe(ctxt)
	}

	if *ld.FlagC {
		fmt.Printf("textsize=%d\n", ld.Segtext.Filelen)
		fmt.Printf("datsize=%d\n", ld.Segdata.Filelen)
		fmt.Printf("bsssize=%d\n", ld.Segdata.Length-ld.Segdata.Filelen)
		fmt.Printf("symsize=%d\n", ld.Symsize)
		fmt.Printf("lcsize=%d\n", ld.Lcsize)
		fmt.Printf("total=%d\n", ld.Segtext.Filelen+ld.Segdata.Length+uint64(ld.Symsize)+uint64(ld.Lcsize))
	}
}
