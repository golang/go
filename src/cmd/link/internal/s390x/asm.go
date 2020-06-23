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

package s390x

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/ld"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/elf"
	"fmt"
	"sync"
)

// gentext generates assembly to append the local moduledata to the global
// moduledata linked list at initialization time. This is only done if the runtime
// is in a different module.
//
// <go.link.addmoduledata>:
// 	larl  %r2, <local.moduledata>
// 	jg    <runtime.addmoduledata@plt>
//	undef
//
// The job of appending the moduledata is delegated to runtime.addmoduledata.
func gentext2(ctxt *ld.Link, ldr *loader.Loader) {
	initfunc, addmoduledata := ld.PrepareAddmoduledata(ctxt)
	if initfunc == nil {
		return
	}

	// larl %r2, <local.moduledata>
	initfunc.AddUint8(0xc0)
	initfunc.AddUint8(0x20)
	initfunc.AddSymRef(ctxt.Arch, ctxt.Moduledata2, 6, objabi.R_PCREL, 4)
	r1 := initfunc.Relocs()
	ldr.SetRelocVariant(initfunc.Sym(), r1.Count()-1, sym.RV_390_DBL)

	// jg <runtime.addmoduledata[@plt]>
	initfunc.AddUint8(0xc0)
	initfunc.AddUint8(0xf4)
	initfunc.AddSymRef(ctxt.Arch, addmoduledata, 6, objabi.R_CALL, 4)
	r2 := initfunc.Relocs()
	ldr.SetRelocVariant(initfunc.Sym(), r2.Count()-1, sym.RV_390_DBL)

	// undef (for debugging)
	initfunc.AddUint32(ctxt.Arch, 0)
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
			ldr.Errorf(s, "unexpected relocation type %d", r.Type())
			return false
		}

		// Handle relocations found in ELF object files.
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_12),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GOT12):
		ldr.Errorf(s, "s390x 12-bit relocations have not been implemented (relocation type %d)", r.Type()-objabi.ElfRelocOffset)
		return false

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_8),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_16),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_32),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_64):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected R_390_nn relocation for dynamic symbol %s", ldr.SymName(targ))
		}

		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ADDR)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PC16),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PC32),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PC64):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected R_390_PCnn relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		// TODO(mwhudson): the test of VisibilityHidden here probably doesn't make
		// sense and should be removed when someone has thought about it properly.
		if (targType == 0 || targType == sym.SXREF) && !ldr.AttrVisibilityHidden(targ) {
			ldr.Errorf(s, "unknown symbol %s in pcrel", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		su.SetRelocAdd(rIdx, r.Add()+int64(r.Siz()))
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GOT16),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GOT32),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GOT64):
		ldr.Errorf(s, "unimplemented S390x relocation: %v", r.Type()-objabi.ElfRelocOffset)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PLT16DBL),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PLT32DBL):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		ldr.SetRelocVariant(s, rIdx, sym.RV_390_DBL)
		su.SetRelocAdd(rIdx, r.Add()+int64(r.Siz()))
		if targType == sym.SDYNIMPORT {
			addpltsym2(target, ldr, syms, targ)
			r.SetSym(syms.PLT2)
			su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymPlt(targ)))
		}
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PLT32),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PLT64):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		su.SetRelocAdd(rIdx, r.Add()+int64(r.Siz()))
		if targType == sym.SDYNIMPORT {
			addpltsym2(target, ldr, syms, targ)
			r.SetSym(syms.PLT2)
			su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymPlt(targ)))
		}
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_COPY):
		ldr.Errorf(s, "unimplemented S390x relocation: %v", r.Type()-objabi.ElfRelocOffset)
		return false

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GLOB_DAT):
		ldr.Errorf(s, "unimplemented S390x relocation: %v", r.Type()-objabi.ElfRelocOffset)
		return false

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_JMP_SLOT):
		ldr.Errorf(s, "unimplemented S390x relocation: %v", r.Type()-objabi.ElfRelocOffset)
		return false

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_RELATIVE):
		ldr.Errorf(s, "unimplemented S390x relocation: %v", r.Type()-objabi.ElfRelocOffset)
		return false

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GOTOFF):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected R_390_GOTOFF relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_GOTOFF)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GOTPC):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		r.SetSym(syms.GOT2)
		su.SetRelocAdd(rIdx, r.Add()+int64(r.Siz()))
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PC16DBL),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PC32DBL):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		ldr.SetRelocVariant(s, rIdx, sym.RV_390_DBL)
		su.SetRelocAdd(rIdx, r.Add()+int64(r.Siz()))
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected R_390_PCnnDBL relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GOTPCDBL):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		ldr.SetRelocVariant(s, rIdx, sym.RV_390_DBL)
		r.SetSym(syms.GOT2)
		su.SetRelocAdd(rIdx, r.Add()+int64(r.Siz()))
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GOTENT):
		addgotsym2(target, ldr, syms, targ)
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		ldr.SetRelocVariant(s, rIdx, sym.RV_390_DBL)
		r.SetSym(syms.GOT2)
		su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymGot(targ))+int64(r.Siz()))
		return true
	}
	// Handle references to ELF symbols from our own object files.
	if targType != sym.SDYNIMPORT {
		return true
	}

	return false
}

func elfreloc1(ctxt *ld.Link, r *sym.Reloc, sectoff int64) bool {
	ctxt.Out.Write64(uint64(sectoff))

	elfsym := ld.ElfSymForReloc(ctxt, r.Xsym)
	switch r.Type {
	default:
		return false
	case objabi.R_TLS_LE:
		switch r.Siz {
		default:
			return false
		case 4:
			// WARNING - silently ignored by linker in ELF64
			ctxt.Out.Write64(uint64(elf.R_390_TLS_LE32) | uint64(elfsym)<<32)
		case 8:
			// WARNING - silently ignored by linker in ELF32
			ctxt.Out.Write64(uint64(elf.R_390_TLS_LE64) | uint64(elfsym)<<32)
		}
	case objabi.R_TLS_IE:
		switch r.Siz {
		default:
			return false
		case 4:
			ctxt.Out.Write64(uint64(elf.R_390_TLS_IEENT) | uint64(elfsym)<<32)
		}
	case objabi.R_ADDR, objabi.R_DWARFSECREF:
		switch r.Siz {
		default:
			return false
		case 4:
			ctxt.Out.Write64(uint64(elf.R_390_32) | uint64(elfsym)<<32)
		case 8:
			ctxt.Out.Write64(uint64(elf.R_390_64) | uint64(elfsym)<<32)
		}
	case objabi.R_GOTPCREL:
		if r.Siz == 4 {
			ctxt.Out.Write64(uint64(elf.R_390_GOTENT) | uint64(elfsym)<<32)
		} else {
			return false
		}
	case objabi.R_PCREL, objabi.R_PCRELDBL, objabi.R_CALL:
		elfrel := elf.R_390_NONE
		isdbl := r.Variant&sym.RV_TYPE_MASK == sym.RV_390_DBL
		// TODO(mundaym): all DBL style relocations should be
		// signalled using the variant - see issue 14218.
		switch r.Type {
		case objabi.R_PCRELDBL, objabi.R_CALL:
			isdbl = true
		}
		if r.Xsym.Type == sym.SDYNIMPORT && (r.Xsym.ElfType() == elf.STT_FUNC || r.Type == objabi.R_CALL) {
			if isdbl {
				switch r.Siz {
				case 2:
					elfrel = elf.R_390_PLT16DBL
				case 4:
					elfrel = elf.R_390_PLT32DBL
				}
			} else {
				switch r.Siz {
				case 4:
					elfrel = elf.R_390_PLT32
				case 8:
					elfrel = elf.R_390_PLT64
				}
			}
		} else {
			if isdbl {
				switch r.Siz {
				case 2:
					elfrel = elf.R_390_PC16DBL
				case 4:
					elfrel = elf.R_390_PC32DBL
				}
			} else {
				switch r.Siz {
				case 2:
					elfrel = elf.R_390_PC16
				case 4:
					elfrel = elf.R_390_PC32
				case 8:
					elfrel = elf.R_390_PC64
				}
			}
		}
		if elfrel == elf.R_390_NONE {
			return false // unsupported size/dbl combination
		}
		ctxt.Out.Write64(uint64(elfrel) | uint64(elfsym)<<32)
	}

	ctxt.Out.Write64(uint64(r.Xadd))
	return true
}

func elfsetupplt(ctxt *ld.Link, plt, got *loader.SymbolBuilder, dynamic loader.Sym) {
	if plt.Size() == 0 {
		// stg     %r1,56(%r15)
		plt.AddUint8(0xe3)
		plt.AddUint8(0x10)
		plt.AddUint8(0xf0)
		plt.AddUint8(0x38)
		plt.AddUint8(0x00)
		plt.AddUint8(0x24)
		// larl    %r1,_GLOBAL_OFFSET_TABLE_
		plt.AddUint8(0xc0)
		plt.AddUint8(0x10)
		plt.AddSymRef(ctxt.Arch, got.Sym(), 6, objabi.R_PCRELDBL, 4)
		// mvc     48(8,%r15),8(%r1)
		plt.AddUint8(0xd2)
		plt.AddUint8(0x07)
		plt.AddUint8(0xf0)
		plt.AddUint8(0x30)
		plt.AddUint8(0x10)
		plt.AddUint8(0x08)
		// lg      %r1,16(%r1)
		plt.AddUint8(0xe3)
		plt.AddUint8(0x10)
		plt.AddUint8(0x10)
		plt.AddUint8(0x10)
		plt.AddUint8(0x00)
		plt.AddUint8(0x04)
		// br      %r1
		plt.AddUint8(0x07)
		plt.AddUint8(0xf1)
		// nopr    %r0
		plt.AddUint8(0x07)
		plt.AddUint8(0x00)
		// nopr    %r0
		plt.AddUint8(0x07)
		plt.AddUint8(0x00)
		// nopr    %r0
		plt.AddUint8(0x07)
		plt.AddUint8(0x00)

		// assume got->size == 0 too
		got.AddAddrPlus(ctxt.Arch, dynamic, 0)

		got.AddUint64(ctxt.Arch, 0)
		got.AddUint64(ctxt.Arch, 0)
	}
}

func machoreloc1(arch *sys.Arch, out *ld.OutBuf, s *sym.Symbol, r *sym.Reloc, sectoff int64) bool {
	return false
}

func archreloc(target *ld.Target, syms *ld.ArchSyms, r *sym.Reloc, s *sym.Symbol, val int64) (int64, bool) {
	if target.IsExternal() {
		return val, false
	}

	switch r.Type {
	case objabi.R_CONST:
		return r.Add, true
	case objabi.R_GOTOFF:
		return ld.Symaddr(r.Sym) + r.Add - ld.Symaddr(syms.GOT), true
	}

	return val, false
}

func archrelocvariant(target *ld.Target, syms *ld.ArchSyms, r *sym.Reloc, s *sym.Symbol, t int64) int64 {
	switch r.Variant & sym.RV_TYPE_MASK {
	default:
		ld.Errorf(s, "unexpected relocation variant %d", r.Variant)
		return t

	case sym.RV_NONE:
		return t

	case sym.RV_390_DBL:
		if (t & 1) != 0 {
			ld.Errorf(s, "%s+%v is not 2-byte aligned", r.Sym.Name, r.Sym.Value)
		}
		return t >> 1
	}
}

func addpltsym2(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s loader.Sym) {
	if ldr.SymPlt(s) >= 0 {
		return
	}

	ld.Adddynsym2(ldr, target, syms, s)

	if target.IsElf() {
		plt := ldr.MakeSymbolUpdater(syms.PLT2)
		got := ldr.MakeSymbolUpdater(syms.GOT2)
		rela := ldr.MakeSymbolUpdater(syms.RelaPLT2)
		if plt.Size() == 0 {
			panic("plt is not set up")
		}
		// larl    %r1,_GLOBAL_OFFSET_TABLE_+index

		plt.AddUint8(0xc0)
		plt.AddUint8(0x10)
		plt.AddPCRelPlus(target.Arch, got.Sym(), got.Size()+6)
		pltrelocs := plt.Relocs()
		ldr.SetRelocVariant(plt.Sym(), pltrelocs.Count()-1, sym.RV_390_DBL)

		// add to got: pointer to current pos in plt
		got.AddAddrPlus(target.Arch, plt.Sym(), plt.Size()+8) // weird but correct
		// lg      %r1,0(%r1)
		plt.AddUint8(0xe3)
		plt.AddUint8(0x10)
		plt.AddUint8(0x10)
		plt.AddUint8(0x00)
		plt.AddUint8(0x00)
		plt.AddUint8(0x04)
		// br      %r1
		plt.AddUint8(0x07)
		plt.AddUint8(0xf1)
		// basr    %r1,%r0
		plt.AddUint8(0x0d)
		plt.AddUint8(0x10)
		// lgf     %r1,12(%r1)
		plt.AddUint8(0xe3)
		plt.AddUint8(0x10)
		plt.AddUint8(0x10)
		plt.AddUint8(0x0c)
		plt.AddUint8(0x00)
		plt.AddUint8(0x14)
		// jg .plt
		plt.AddUint8(0xc0)
		plt.AddUint8(0xf4)

		plt.AddUint32(target.Arch, uint32(-((plt.Size() - 2) >> 1))) // roll-your-own relocation
		//.plt index
		plt.AddUint32(target.Arch, uint32(rela.Size())) // rela size before current entry

		// rela
		rela.AddAddrPlus(target.Arch, got.Sym(), got.Size()-8)

		sDynid := ldr.SymDynid(s)
		rela.AddUint64(target.Arch, ld.ELF64_R_INFO(uint32(sDynid), uint32(elf.R_390_JMP_SLOT)))
		rela.AddUint64(target.Arch, 0)

		ldr.SetPlt(s, int32(plt.Size()-32))

	} else {
		ldr.Errorf(s, "addpltsym: unsupported binary format")
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
		rela := ldr.MakeSymbolUpdater(syms.Rela2)
		rela.AddAddrPlus(target.Arch, got.Sym(), int64(ldr.SymGot(s)))
		rela.AddUint64(target.Arch, ld.ELF64_R_INFO(uint32(ldr.SymDynid(s)), uint32(elf.R_390_GLOB_DAT)))
		rela.AddUint64(target.Arch, 0)
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
		if !ctxt.IsELF {
			ld.Errorf(nil, "unsupported executable format")
		}
		symo = uint32(ld.Segdwarf.Fileoff + ld.Segdwarf.Filelen)
		symo = uint32(ld.Rnd(int64(symo), int64(*ld.FlagRound)))

		ctxt.Out.SeekSet(int64(symo))
		ld.Asmelfsym(ctxt)
		ctxt.Out.Write(ld.Elfstrdat)

		if ctxt.LinkMode == ld.LinkExternal {
			ld.Elfemitreloc(ctxt)
		}
	}

	ctxt.Out.SeekSet(0)
	switch ctxt.HeadType {
	default:
		ld.Errorf(nil, "unsupported operating system")
	case objabi.Hlinux:
		ld.Asmbelf(ctxt, int64(symo))
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
