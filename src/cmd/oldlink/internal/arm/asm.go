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
	"cmd/oldlink/internal/ld"
	"cmd/oldlink/internal/sym"
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

func gentext(ctxt *ld.Link) {
	if !ctxt.DynlinkingGo() {
		return
	}
	addmoduledata := ctxt.Syms.Lookup("runtime.addmoduledata", 0)
	if addmoduledata.Type == sym.STEXT && ctxt.BuildMode != ld.BuildModePlugin {
		// we're linking a module containing the runtime -> no need for
		// an init function
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
	o(0xe59f0004)
	o(0xe08f0000)

	o(0xeafffffe)
	rel := initfunc.AddRel()
	rel.Off = 8
	rel.Siz = 4
	rel.Sym = ctxt.Syms.Lookup("runtime.addmoduledata", 0)
	rel.Type = objabi.R_CALLARM
	rel.Add = 0xeafffffe // vomit

	o(0x00000000)
	rel = initfunc.AddRel()
	rel.Off = 12
	rel.Siz = 4
	rel.Sym = ctxt.Moduledata
	rel.Type = objabi.R_PCREL
	rel.Add = 4

	if ctxt.BuildMode == ld.BuildModePlugin {
		ctxt.Textp = append(ctxt.Textp, addmoduledata)
	}
	ctxt.Textp = append(ctxt.Textp, initfunc)
	initarray_entry := ctxt.Syms.Lookup("go.link.addmoduledatainit", 0)
	initarray_entry.Attr |= sym.AttrReachable
	initarray_entry.Attr |= sym.AttrLocal
	initarray_entry.Type = sym.SINITARR
	initarray_entry.AddAddr(ctxt.Arch, initfunc)
}

// Preserve highest 8 bits of a, and do addition to lower 24-bit
// of a and b; used to adjust ARM branch instruction's target
func braddoff(a int32, b int32) int32 {
	return int32((uint32(a))&0xff000000 | 0x00ffffff&uint32(a+b))
}

func adddynrel(ctxt *ld.Link, s *sym.Symbol, r *sym.Reloc) bool {
	targ := r.Sym

	switch r.Type {
	default:
		if r.Type >= objabi.ElfRelocOffset {
			ld.Errorf(s, "unexpected relocation type %d (%s)", r.Type, sym.RelocName(ctxt.Arch, r.Type))
			return false
		}

		// Handle relocations found in ELF object files.
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_PLT32):
		r.Type = objabi.R_CALLARM

		if targ.Type == sym.SDYNIMPORT {
			addpltsym(ctxt, targ)
			r.Sym = ctxt.Syms.Lookup(".plt", 0)
			r.Add = int64(braddoff(int32(r.Add), targ.Plt()/4))
		}

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_THM_PC22): // R_ARM_THM_CALL
		ld.Exitf("R_ARM_THM_CALL, are you using -marm?")
		return false

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_GOT32): // R_ARM_GOT_BREL
		if targ.Type != sym.SDYNIMPORT {
			addgotsyminternal(ctxt, targ)
		} else {
			addgotsym(ctxt, targ)
		}

		r.Type = objabi.R_CONST // write r->add during relocsym
		r.Sym = nil
		r.Add += int64(targ.Got())
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_GOT_PREL): // GOT(nil) + A - nil
		if targ.Type != sym.SDYNIMPORT {
			addgotsyminternal(ctxt, targ)
		} else {
			addgotsym(ctxt, targ)
		}

		r.Type = objabi.R_PCREL
		r.Sym = ctxt.Syms.Lookup(".got", 0)
		r.Add += int64(targ.Got()) + 4
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_GOTOFF): // R_ARM_GOTOFF32
		r.Type = objabi.R_GOTOFF

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_GOTPC): // R_ARM_BASE_PREL
		r.Type = objabi.R_PCREL

		r.Sym = ctxt.Syms.Lookup(".got", 0)
		r.Add += 4
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_CALL):
		r.Type = objabi.R_CALLARM
		if targ.Type == sym.SDYNIMPORT {
			addpltsym(ctxt, targ)
			r.Sym = ctxt.Syms.Lookup(".plt", 0)
			r.Add = int64(braddoff(int32(r.Add), targ.Plt()/4))
		}

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_REL32): // R_ARM_REL32
		r.Type = objabi.R_PCREL

		r.Add += 4
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_ABS32):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_ARM_ABS32 relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = objabi.R_ADDR
		return true

		// we can just ignore this, because we are targeting ARM V5+ anyway
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_V4BX):
		if r.Sym != nil {
			// R_ARM_V4BX is ABS relocation, so this symbol is a dummy symbol, ignore it
			r.Sym.Type = 0
		}

		r.Sym = nil
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_PC24),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_JUMP24):
		r.Type = objabi.R_CALLARM
		if targ.Type == sym.SDYNIMPORT {
			addpltsym(ctxt, targ)
			r.Sym = ctxt.Syms.Lookup(".plt", 0)
			r.Add = int64(braddoff(int32(r.Add), targ.Plt()/4))
		}

		return true
	}

	// Handle references to ELF symbols from our own object files.
	if targ.Type != sym.SDYNIMPORT {
		return true
	}

	switch r.Type {
	case objabi.R_CALLARM:
		if ctxt.LinkMode == ld.LinkExternal {
			// External linker will do this relocation.
			return true
		}
		addpltsym(ctxt, targ)
		r.Sym = ctxt.Syms.Lookup(".plt", 0)
		r.Add = int64(targ.Plt())
		return true

	case objabi.R_ADDR:
		if s.Type != sym.SDATA {
			break
		}
		if ctxt.IsELF {
			ld.Adddynsym(ctxt, targ)
			rel := ctxt.Syms.Lookup(".rel", 0)
			rel.AddAddrPlus(ctxt.Arch, s, int64(r.Off))
			rel.AddUint32(ctxt.Arch, ld.ELF32_R_INFO(uint32(targ.Dynid), uint32(elf.R_ARM_GLOB_DAT))) // we need a nil + A dynamic reloc
			r.Type = objabi.R_CONST                                                                   // write r->add during relocsym
			r.Sym = nil
			return true
		}
	}

	return false
}

func elfreloc1(ctxt *ld.Link, r *sym.Reloc, sectoff int64) bool {
	ctxt.Out.Write32(uint32(sectoff))

	elfsym := r.Xsym.ElfsymForReloc()
	switch r.Type {
	default:
		return false
	case objabi.R_ADDR:
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

func elfsetupplt(ctxt *ld.Link) {
	plt := ctxt.Syms.Lookup(".plt", 0)
	got := ctxt.Syms.Lookup(".got.plt", 0)
	if plt.Size == 0 {
		// str lr, [sp, #-4]!
		plt.AddUint32(ctxt.Arch, 0xe52de004)

		// ldr lr, [pc, #4]
		plt.AddUint32(ctxt.Arch, 0xe59fe004)

		// add lr, pc, lr
		plt.AddUint32(ctxt.Arch, 0xe08fe00e)

		// ldr pc, [lr, #8]!
		plt.AddUint32(ctxt.Arch, 0xe5bef008)

		// .word &GLOBAL_OFFSET_TABLE[0] - .
		plt.AddPCRelPlus(ctxt.Arch, got, 4)

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
func trampoline(ctxt *ld.Link, r *sym.Reloc, s *sym.Symbol) {
	switch r.Type {
	case objabi.R_CALLARM:
		// r.Add is the instruction
		// low 24-bit encodes the target address
		t := (ld.Symaddr(r.Sym) + int64(signext24(r.Add&0xffffff)*4) - (s.Value + int64(r.Off))) / 4
		if t > 0x7fffff || t < -0x800000 || (*ld.FlagDebugTramp > 1 && s.File != r.Sym.File) {
			// direct call too far, need to insert trampoline.
			// look up existing trampolines first. if we found one within the range
			// of direct call, we can reuse it. otherwise create a new one.
			offset := (signext24(r.Add&0xffffff) + 2) * 4
			var tramp *sym.Symbol
			for i := 0; ; i++ {
				oName := r.Sym.Name
				name := oName + fmt.Sprintf("%+d-tramp%d", offset, i)
				tramp = ctxt.Syms.Lookup(name, int(r.Sym.Version))
				if oName == "runtime.deferreturn" {
					tramp.Attr.Set(sym.AttrDeferReturnTramp, true)
				}
				if tramp.Type == sym.SDYNIMPORT {
					// don't reuse trampoline defined in other module
					continue
				}
				if tramp.Value == 0 {
					// either the trampoline does not exist -- we need to create one,
					// or found one the address which is not assigned -- this will be
					// laid down immediately after the current function. use this one.
					break
				}

				t = (ld.Symaddr(tramp) - 8 - (s.Value + int64(r.Off))) / 4
				if t >= -0x800000 && t < 0x7fffff {
					// found an existing trampoline that is not too far
					// we can just use it
					break
				}
			}
			if tramp.Type == 0 {
				// trampoline does not exist, create one
				ctxt.AddTramp(tramp)
				if ctxt.DynlinkingGo() {
					if immrot(uint32(offset)) == 0 {
						ld.Errorf(s, "odd offset in dynlink direct call: %v+%d", r.Sym, offset)
					}
					gentrampdyn(ctxt.Arch, tramp, r.Sym, int64(offset))
				} else if ctxt.BuildMode == ld.BuildModeCArchive || ctxt.BuildMode == ld.BuildModeCShared || ctxt.BuildMode == ld.BuildModePIE {
					gentramppic(ctxt.Arch, tramp, r.Sym, int64(offset))
				} else {
					gentramp(ctxt.Arch, ctxt.LinkMode, tramp, r.Sym, int64(offset))
				}
			}
			// modify reloc to point to tramp, which will be resolved later
			r.Sym = tramp
			r.Add = r.Add&0xff000000 | 0xfffffe // clear the offset embedded in the instruction
			r.Done = false
		}
	default:
		ld.Errorf(s, "trampoline called with non-jump reloc: %d (%s)", r.Type, sym.RelocName(ctxt.Arch, r.Type))
	}
}

// generate a trampoline to target+offset
func gentramp(arch *sys.Arch, linkmode ld.LinkMode, tramp, target *sym.Symbol, offset int64) {
	tramp.Size = 12 // 3 instructions
	tramp.P = make([]byte, tramp.Size)
	t := ld.Symaddr(target) + offset
	o1 := uint32(0xe5900000 | 11<<12 | 15<<16) // MOVW (R15), R11 // R15 is actual pc + 8
	o2 := uint32(0xe12fff10 | 11)              // JMP  (R11)
	o3 := uint32(t)                            // WORD $target
	arch.ByteOrder.PutUint32(tramp.P, o1)
	arch.ByteOrder.PutUint32(tramp.P[4:], o2)
	arch.ByteOrder.PutUint32(tramp.P[8:], o3)

	if linkmode == ld.LinkExternal {
		r := tramp.AddRel()
		r.Off = 8
		r.Type = objabi.R_ADDR
		r.Siz = 4
		r.Sym = target
		r.Add = offset
	}
}

// generate a trampoline to target+offset in position independent code
func gentramppic(arch *sys.Arch, tramp, target *sym.Symbol, offset int64) {
	tramp.Size = 16 // 4 instructions
	tramp.P = make([]byte, tramp.Size)
	o1 := uint32(0xe5900000 | 11<<12 | 15<<16 | 4)  // MOVW 4(R15), R11 // R15 is actual pc + 8
	o2 := uint32(0xe0800000 | 11<<12 | 15<<16 | 11) // ADD R15, R11, R11
	o3 := uint32(0xe12fff10 | 11)                   // JMP  (R11)
	o4 := uint32(0)                                 // WORD $(target-pc) // filled in with relocation
	arch.ByteOrder.PutUint32(tramp.P, o1)
	arch.ByteOrder.PutUint32(tramp.P[4:], o2)
	arch.ByteOrder.PutUint32(tramp.P[8:], o3)
	arch.ByteOrder.PutUint32(tramp.P[12:], o4)

	r := tramp.AddRel()
	r.Off = 12
	r.Type = objabi.R_PCREL
	r.Siz = 4
	r.Sym = target
	r.Add = offset + 4
}

// generate a trampoline to target+offset in dynlink mode (using GOT)
func gentrampdyn(arch *sys.Arch, tramp, target *sym.Symbol, offset int64) {
	tramp.Size = 20                                 // 5 instructions
	o1 := uint32(0xe5900000 | 11<<12 | 15<<16 | 8)  // MOVW 8(R15), R11 // R15 is actual pc + 8
	o2 := uint32(0xe0800000 | 11<<12 | 15<<16 | 11) // ADD R15, R11, R11
	o3 := uint32(0xe5900000 | 11<<12 | 11<<16)      // MOVW (R11), R11
	o4 := uint32(0xe12fff10 | 11)                   // JMP  (R11)
	o5 := uint32(0)                                 // WORD $target@GOT // filled in with relocation
	o6 := uint32(0)
	if offset != 0 {
		// insert an instruction to add offset
		tramp.Size = 24 // 6 instructions
		o6 = o5
		o5 = o4
		o4 = 0xe2800000 | 11<<12 | 11<<16 | immrot(uint32(offset)) // ADD $offset, R11, R11
		o1 = uint32(0xe5900000 | 11<<12 | 15<<16 | 12)             // MOVW 12(R15), R11
	}
	tramp.P = make([]byte, tramp.Size)
	arch.ByteOrder.PutUint32(tramp.P, o1)
	arch.ByteOrder.PutUint32(tramp.P[4:], o2)
	arch.ByteOrder.PutUint32(tramp.P[8:], o3)
	arch.ByteOrder.PutUint32(tramp.P[12:], o4)
	arch.ByteOrder.PutUint32(tramp.P[16:], o5)
	if offset != 0 {
		arch.ByteOrder.PutUint32(tramp.P[20:], o6)
	}

	r := tramp.AddRel()
	r.Off = 16
	r.Type = objabi.R_GOTPCREL
	r.Siz = 4
	r.Sym = target
	r.Add = 8
	if offset != 0 {
		// increase reloc offset by 4 as we inserted an ADD instruction
		r.Off = 20
		r.Add = 12
	}
}

func archreloc(ctxt *ld.Link, r *sym.Reloc, s *sym.Symbol, val int64) (int64, bool) {
	if ctxt.LinkMode == ld.LinkExternal {
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
		return ld.Symaddr(r.Sym) + r.Add - ld.Symaddr(ctxt.Syms.Lookup(".got", 0)), true

	// The following three arch specific relocations are only for generation of
	// Linux/ARM ELF's PLT entry (3 assembler instruction)
	case objabi.R_PLT0: // add ip, pc, #0xXX00000
		if ld.Symaddr(ctxt.Syms.Lookup(".got.plt", 0)) < ld.Symaddr(ctxt.Syms.Lookup(".plt", 0)) {
			ld.Errorf(s, ".got.plt should be placed after .plt section.")
		}
		return 0xe28fc600 + (0xff & (int64(uint32(ld.Symaddr(r.Sym)-(ld.Symaddr(ctxt.Syms.Lookup(".plt", 0))+int64(r.Off))+r.Add)) >> 20)), true
	case objabi.R_PLT1: // add ip, ip, #0xYY000
		return 0xe28cca00 + (0xff & (int64(uint32(ld.Symaddr(r.Sym)-(ld.Symaddr(ctxt.Syms.Lookup(".plt", 0))+int64(r.Off))+r.Add+4)) >> 12)), true
	case objabi.R_PLT2: // ldr pc, [ip, #0xZZZ]!
		return 0xe5bcf000 + (0xfff & int64(uint32(ld.Symaddr(r.Sym)-(ld.Symaddr(ctxt.Syms.Lookup(".plt", 0))+int64(r.Off))+r.Add+8))), true
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

func archrelocvariant(ctxt *ld.Link, r *sym.Reloc, s *sym.Symbol, t int64) int64 {
	log.Fatalf("unexpected relocation variant")
	return t
}

func addpltreloc(ctxt *ld.Link, plt *sym.Symbol, got *sym.Symbol, s *sym.Symbol, typ objabi.RelocType) {
	r := plt.AddRel()
	r.Sym = got
	r.Off = int32(plt.Size)
	r.Siz = 4
	r.Type = typ
	r.Add = int64(s.Got()) - 8

	plt.Attr |= sym.AttrReachable
	plt.Size += 4
	plt.Grow(plt.Size)
}

func addpltsym(ctxt *ld.Link, s *sym.Symbol) {
	if s.Plt() >= 0 {
		return
	}

	ld.Adddynsym(ctxt, s)

	if ctxt.IsELF {
		plt := ctxt.Syms.Lookup(".plt", 0)
		got := ctxt.Syms.Lookup(".got.plt", 0)
		rel := ctxt.Syms.Lookup(".rel.plt", 0)
		if plt.Size == 0 {
			elfsetupplt(ctxt)
		}

		// .got entry
		s.SetGot(int32(got.Size))

		// In theory, all GOT should point to the first PLT entry,
		// Linux/ARM's dynamic linker will do that for us, but FreeBSD/ARM's
		// dynamic linker won't, so we'd better do it ourselves.
		got.AddAddrPlus(ctxt.Arch, plt, 0)

		// .plt entry, this depends on the .got entry
		s.SetPlt(int32(plt.Size))

		addpltreloc(ctxt, plt, got, s, objabi.R_PLT0) // add lr, pc, #0xXX00000
		addpltreloc(ctxt, plt, got, s, objabi.R_PLT1) // add lr, lr, #0xYY000
		addpltreloc(ctxt, plt, got, s, objabi.R_PLT2) // ldr pc, [lr, #0xZZZ]!

		// rel
		rel.AddAddrPlus(ctxt.Arch, got, int64(s.Got()))

		rel.AddUint32(ctxt.Arch, ld.ELF32_R_INFO(uint32(s.Dynid), uint32(elf.R_ARM_JUMP_SLOT)))
	} else {
		ld.Errorf(s, "addpltsym: unsupported binary format")
	}
}

func addgotsyminternal(ctxt *ld.Link, s *sym.Symbol) {
	if s.Got() >= 0 {
		return
	}

	got := ctxt.Syms.Lookup(".got", 0)
	s.SetGot(int32(got.Size))

	got.AddAddrPlus(ctxt.Arch, s, 0)

	if ctxt.IsELF {
	} else {
		ld.Errorf(s, "addgotsyminternal: unsupported binary format")
	}
}

func addgotsym(ctxt *ld.Link, s *sym.Symbol) {
	if s.Got() >= 0 {
		return
	}

	ld.Adddynsym(ctxt, s)
	got := ctxt.Syms.Lookup(".got", 0)
	s.SetGot(int32(got.Size))
	got.AddUint32(ctxt.Arch, 0)

	if ctxt.IsELF {
		rel := ctxt.Syms.Lookup(".rel", 0)
		rel.AddAddrPlus(ctxt.Arch, got, int64(s.Got()))
		rel.AddUint32(ctxt.Arch, ld.ELF32_R_INFO(uint32(s.Dynid), uint32(elf.R_ARM_GLOB_DAT)))
	} else {
		ld.Errorf(s, "addgotsym: unsupported binary format")
	}
}

func asmb(ctxt *ld.Link) {
	if ctxt.IsELF {
		ld.Asmbelfsetup()
	}

	sect := ld.Segtext.Sections[0]
	ctxt.Out.SeekSet(int64(sect.Vaddr - ld.Segtext.Vaddr + ld.Segtext.Fileoff))
	ld.Codeblk(ctxt, int64(sect.Vaddr), int64(sect.Length))
	for _, sect = range ld.Segtext.Sections[1:] {
		ctxt.Out.SeekSet(int64(sect.Vaddr - ld.Segtext.Vaddr + ld.Segtext.Fileoff))
		ld.Datblk(ctxt, int64(sect.Vaddr), int64(sect.Length))
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

		case objabi.Hwindows:
			symo = uint32(ld.Segdwarf.Fileoff + ld.Segdwarf.Filelen)
			symo = uint32(ld.Rnd(int64(symo), ld.PEFILEALIGN))
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
