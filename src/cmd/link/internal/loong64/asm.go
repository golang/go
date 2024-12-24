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

func elfsetupplt(ctxt *ld.Link, ldr *loader.Loader, plt, gotplt *loader.SymbolBuilder, dynamic loader.Sym) {
	return
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
			return int64(val&0xffc003ff | (t << 10)), noExtReloc, isOk
		}
		return int64(val&0xfe00001f | (t << 5)), noExtReloc, isOk
	case objabi.R_LOONG64_TLS_LE_HI,
		objabi.R_LOONG64_TLS_LE_LO:
		t := ldr.SymAddr(rs) + r.Add()
		if r.Type() == objabi.R_LOONG64_TLS_LE_LO {
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
		// Host object relocations that will be turned into a PLT call.
		// The PLT may be too far. Insert a trampoline for them.
		fallthrough
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
