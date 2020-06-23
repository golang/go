// Inferno utils/8l/asm.c
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/8l/asm.c
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

package x86

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/ld"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/elf"
	"log"
	"sync"
)

func gentext2(ctxt *ld.Link, ldr *loader.Loader) {
	if ctxt.DynlinkingGo() {
		// We need get_pc_thunk.
	} else {
		switch ctxt.BuildMode {
		case ld.BuildModeCArchive:
			if !ctxt.IsELF {
				return
			}
		case ld.BuildModePIE, ld.BuildModeCShared, ld.BuildModePlugin:
			// We need get_pc_thunk.
		default:
			return
		}
	}

	// Generate little thunks that load the PC of the next instruction into a register.
	thunks := make([]loader.Sym, 0, 7+len(ctxt.Textp2))
	for _, r := range [...]struct {
		name string
		num  uint8
	}{
		{"ax", 0},
		{"cx", 1},
		{"dx", 2},
		{"bx", 3},
		// sp
		{"bp", 5},
		{"si", 6},
		{"di", 7},
	} {
		thunkfunc := ldr.CreateSymForUpdate("__x86.get_pc_thunk."+r.name, 0)
		thunkfunc.SetType(sym.STEXT)
		ldr.SetAttrLocal(thunkfunc.Sym(), true)
		o := func(op ...uint8) {
			for _, op1 := range op {
				thunkfunc.AddUint8(op1)
			}
		}
		// 8b 04 24	mov    (%esp),%eax
		// Destination register is in bits 3-5 of the middle byte, so add that in.
		o(0x8b, 0x04+r.num<<3, 0x24)
		// c3		ret
		o(0xc3)

		thunks = append(thunks, thunkfunc.Sym())
	}
	ctxt.Textp2 = append(thunks, ctxt.Textp2...) // keep Textp2 in dependency order

	initfunc, addmoduledata := ld.PrepareAddmoduledata(ctxt)
	if initfunc == nil {
		return
	}

	o := func(op ...uint8) {
		for _, op1 := range op {
			initfunc.AddUint8(op1)
		}
	}

	// go.link.addmoduledata:
	//      53                      push %ebx
	//      e8 00 00 00 00          call __x86.get_pc_thunk.cx + R_CALL __x86.get_pc_thunk.cx
	//      8d 81 00 00 00 00       lea 0x0(%ecx), %eax + R_PCREL ctxt.Moduledata
	//      8d 99 00 00 00 00       lea 0x0(%ecx), %ebx + R_GOTPC _GLOBAL_OFFSET_TABLE_
	//      e8 00 00 00 00          call runtime.addmoduledata@plt + R_CALL runtime.addmoduledata
	//      5b                      pop %ebx
	//      c3                      ret

	o(0x53)

	o(0xe8)
	initfunc.AddSymRef(ctxt.Arch, ldr.Lookup("__x86.get_pc_thunk.cx", 0), 0, objabi.R_CALL, 4)

	o(0x8d, 0x81)
	initfunc.AddPCRelPlus(ctxt.Arch, ctxt.Moduledata2, 6)

	o(0x8d, 0x99)
	gotsym := ldr.LookupOrCreateSym("_GLOBAL_OFFSET_TABLE_", 0)
	initfunc.AddSymRef(ctxt.Arch, gotsym, 12, objabi.R_PCREL, 4)
	o(0xe8)
	initfunc.AddSymRef(ctxt.Arch, addmoduledata, 0, objabi.R_CALL, 4)

	o(0x5b)

	o(0xc3)
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
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_386_PC32):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected R_386_PC32 relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		// TODO(mwhudson): the test of VisibilityHidden here probably doesn't make
		// sense and should be removed when someone has thought about it properly.
		if (targType == 0 || targType == sym.SXREF) && !ldr.AttrVisibilityHidden(targ) {
			ldr.Errorf(s, "unknown symbol %s in pcrel", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		su.SetRelocAdd(rIdx, r.Add()+4)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_386_PLT32):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		su.SetRelocAdd(rIdx, r.Add()+4)
		if targType == sym.SDYNIMPORT {
			addpltsym2(target, ldr, syms, targ)
			su.SetRelocSym(rIdx, syms.PLT2)
			su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymPlt(targ)))
		}

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_386_GOT32),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_386_GOT32X):
		su := ldr.MakeSymbolUpdater(s)
		if targType != sym.SDYNIMPORT {
			// have symbol
			sData := ldr.Data(s)

			if r.Off() >= 2 && sData[r.Off()-2] == 0x8b {
				su.MakeWritable()

				// turn MOVL of GOT entry into LEAL of symbol address, relative to GOT.
				writeableData := su.Data()
				writeableData[r.Off()-2] = 0x8d
				su.SetRelocType(rIdx, objabi.R_GOTOFF)
				return true
			}

			if r.Off() >= 2 && sData[r.Off()-2] == 0xff && sData[r.Off()-1] == 0xb3 {
				su.MakeWritable()
				// turn PUSHL of GOT entry into PUSHL of symbol itself.
				// use unnecessary SS prefix to keep instruction same length.
				writeableData := su.Data()
				writeableData[r.Off()-2] = 0x36
				writeableData[r.Off()-1] = 0x68
				su.SetRelocType(rIdx, objabi.R_ADDR)
				return true
			}

			ldr.Errorf(s, "unexpected GOT reloc for non-dynamic symbol %s", ldr.SymName(targ))
			return false
		}

		addgotsym2(target, ldr, syms, targ)
		su.SetRelocType(rIdx, objabi.R_CONST) // write r->add during relocsym
		su.SetRelocSym(rIdx, 0)
		su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymGot(targ)))
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_386_GOTOFF):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_GOTOFF)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_386_GOTPC):
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_PCREL)
		su.SetRelocSym(rIdx, syms.GOT2)
		su.SetRelocAdd(rIdx, r.Add()+4)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_386_32):
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected R_386_32 relocation for dynamic symbol %s", ldr.SymName(targ))
		}
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ADDR)
		return true

	case objabi.MachoRelocOffset + ld.MACHO_GENERIC_RELOC_VANILLA*2 + 0:
		su := ldr.MakeSymbolUpdater(s)
		su.SetRelocType(rIdx, objabi.R_ADDR)
		if targType == sym.SDYNIMPORT {
			ldr.Errorf(s, "unexpected reloc for dynamic symbol %s", ldr.SymName(targ))
		}
		return true

	case objabi.MachoRelocOffset + ld.MACHO_GENERIC_RELOC_VANILLA*2 + 1:
		su := ldr.MakeSymbolUpdater(s)
		if targType == sym.SDYNIMPORT {
			addpltsym2(target, ldr, syms, targ)
			su.SetRelocSym(rIdx, syms.PLT2)
			su.SetRelocAdd(rIdx, int64(ldr.SymPlt(targ)))
			su.SetRelocType(rIdx, objabi.R_PCREL)
			return true
		}

		su.SetRelocType(rIdx, objabi.R_PCREL)
		return true

	case objabi.MachoRelocOffset + ld.MACHO_FAKE_GOTPCREL:
		su := ldr.MakeSymbolUpdater(s)
		if targType != sym.SDYNIMPORT {
			// have symbol
			// turn MOVL of GOT entry into LEAL of symbol itself
			sData := ldr.Data(s)
			if r.Off() < 2 || sData[r.Off()-2] != 0x8b {
				ldr.Errorf(s, "unexpected GOT reloc for non-dynamic symbol %s", ldr.SymName(targ))
				return false
			}

			su.MakeWritable()
			writeableData := su.Data()
			writeableData[r.Off()-2] = 0x8d
			su.SetRelocType(rIdx, objabi.R_PCREL)
			return true
		}

		addgotsym2(target, ldr, syms, targ)
		su.SetRelocSym(rIdx, syms.GOT2)
		su.SetRelocAdd(rIdx, r.Add()+int64(ldr.SymGot(targ)))
		su.SetRelocType(rIdx, objabi.R_PCREL)
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
	case objabi.R_CALL,
		objabi.R_PCREL:
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
			rel.AddUint32(target.Arch, ld.ELF32_R_INFO(uint32(ldr.SymDynid(targ)), uint32(elf.R_386_32)))
			su := ldr.MakeSymbolUpdater(s)
			su.SetRelocType(rIdx, objabi.R_CONST) // write r->add during relocsym
			su.SetRelocSym(rIdx, 0)
			return true
		}

		if target.IsDarwin() && ldr.SymSize(s) == int64(target.Arch.PtrSize) && r.Off() == 0 {
			// Mach-O relocations are a royal pain to lay out.
			// They use a compact stateful bytecode representation
			// that is too much bother to deal with.
			// Instead, interpret the C declaration
			//	void *_Cvar_stderr = &stderr;
			// as making _Cvar_stderr the name of a GOT entry
			// for stderr. This is separate from the usual GOT entry,
			// just in case the C code assigns to the variable,
			// and of course it only works for single pointers,
			// but we only need to support cgo and that's all it needs.
			ld.Adddynsym2(ldr, target, syms, targ)

			got := ldr.MakeSymbolUpdater(syms.GOT2)
			su := ldr.MakeSymbolUpdater(s)
			su.SetType(got.Type())
			got.PrependSub(s)
			su.SetValue(got.Size())
			got.AddUint32(target.Arch, 0)
			leg := ldr.MakeSymbolUpdater(syms.LinkEditGOT2)
			leg.AddUint32(target.Arch, uint32(ldr.SymDynid(targ)))
			su.SetRelocType(rIdx, objabi.ElfRelocOffset) // ignore during relocsym
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
			ctxt.Out.Write32(uint32(elf.R_386_32) | uint32(elfsym)<<8)
		} else {
			return false
		}
	case objabi.R_GOTPCREL:
		if r.Siz == 4 {
			ctxt.Out.Write32(uint32(elf.R_386_GOTPC))
			if r.Xsym.Name != "_GLOBAL_OFFSET_TABLE_" {
				ctxt.Out.Write32(uint32(sectoff))
				ctxt.Out.Write32(uint32(elf.R_386_GOT32) | uint32(elfsym)<<8)
			}
		} else {
			return false
		}
	case objabi.R_CALL:
		if r.Siz == 4 {
			if r.Xsym.Type == sym.SDYNIMPORT {
				ctxt.Out.Write32(uint32(elf.R_386_PLT32) | uint32(elfsym)<<8)
			} else {
				ctxt.Out.Write32(uint32(elf.R_386_PC32) | uint32(elfsym)<<8)
			}
		} else {
			return false
		}
	case objabi.R_PCREL:
		if r.Siz == 4 {
			ctxt.Out.Write32(uint32(elf.R_386_PC32) | uint32(elfsym)<<8)
		} else {
			return false
		}
	case objabi.R_TLS_LE:
		if r.Siz == 4 {
			ctxt.Out.Write32(uint32(elf.R_386_TLS_LE) | uint32(elfsym)<<8)
		} else {
			return false
		}
	case objabi.R_TLS_IE:
		if r.Siz == 4 {
			ctxt.Out.Write32(uint32(elf.R_386_GOTPC))
			ctxt.Out.Write32(uint32(sectoff))
			ctxt.Out.Write32(uint32(elf.R_386_TLS_GOTIE) | uint32(elfsym)<<8)
		} else {
			return false
		}
	}

	return true
}

func machoreloc1(arch *sys.Arch, out *ld.OutBuf, s *sym.Symbol, r *sym.Reloc, sectoff int64) bool {
	return false
}

func pereloc1(arch *sys.Arch, out *ld.OutBuf, s *sym.Symbol, r *sym.Reloc, sectoff int64) bool {
	var v uint32

	rs := r.Xsym

	if rs.Dynid < 0 {
		ld.Errorf(s, "reloc %d (%s) to non-coff symbol %s type=%d (%s)", r.Type, sym.RelocName(arch, r.Type), rs.Name, rs.Type, rs.Type)
		return false
	}

	out.Write32(uint32(sectoff))
	out.Write32(uint32(rs.Dynid))

	switch r.Type {
	default:
		return false

	case objabi.R_DWARFSECREF:
		v = ld.IMAGE_REL_I386_SECREL

	case objabi.R_ADDR:
		v = ld.IMAGE_REL_I386_DIR32

	case objabi.R_CALL,
		objabi.R_PCREL:
		v = ld.IMAGE_REL_I386_REL32
	}

	out.Write16(uint16(v))

	return true
}

func archreloc2(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, r loader.Reloc2, rr *loader.ExtReloc, sym loader.Sym, val int64) (int64, bool, bool) {
	return val, false, false
}

func archrelocvariant(target *ld.Target, syms *ld.ArchSyms, r *sym.Reloc, s *sym.Symbol, t int64) int64 {
	log.Fatalf("unexpected relocation variant")
	return t
}

func elfsetupplt(ctxt *ld.Link, plt, got *loader.SymbolBuilder, dynamic loader.Sym) {
	if plt.Size() == 0 {
		// pushl got+4
		plt.AddUint8(0xff)

		plt.AddUint8(0x35)
		plt.AddAddrPlus(ctxt.Arch, got.Sym(), 4)

		// jmp *got+8
		plt.AddUint8(0xff)

		plt.AddUint8(0x25)
		plt.AddAddrPlus(ctxt.Arch, got.Sym(), 8)

		// zero pad
		plt.AddUint32(ctxt.Arch, 0)

		// assume got->size == 0 too
		got.AddAddrPlus(ctxt.Arch, dynamic, 0)

		got.AddUint32(ctxt.Arch, 0)
		got.AddUint32(ctxt.Arch, 0)
	}
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

		// jmpq *got+size
		plt.AddUint8(0xff)

		plt.AddUint8(0x25)
		plt.AddAddrPlus(target.Arch, got.Sym(), got.Size())

		// add to got: pointer to current pos in plt
		got.AddAddrPlus(target.Arch, plt.Sym(), plt.Size())

		// pushl $x
		plt.AddUint8(0x68)

		plt.AddUint32(target.Arch, uint32(rel.Size()))

		// jmp .plt
		plt.AddUint8(0xe9)

		plt.AddUint32(target.Arch, uint32(-(plt.Size() + 4)))

		// rel
		rel.AddAddrPlus(target.Arch, got.Sym(), got.Size()-4)

		sDynid := ldr.SymDynid(s)
		rel.AddUint32(target.Arch, ld.ELF32_R_INFO(uint32(sDynid), uint32(elf.R_386_JMP_SLOT)))

		ldr.SetPlt(s, int32(plt.Size()-16))
	} else if target.IsDarwin() {
		// Same laziness as in 6l.

		plt := ldr.MakeSymbolUpdater(syms.PLT2)

		addgotsym2(target, ldr, syms, s)

		sDynid := ldr.SymDynid(s)
		lep := ldr.MakeSymbolUpdater(syms.LinkEditPLT2)
		lep.AddUint32(target.Arch, uint32(sDynid))

		// jmpq *got+size(IP)
		ldr.SetPlt(s, int32(plt.Size()))

		plt.AddUint8(0xff)
		plt.AddUint8(0x25)
		plt.AddAddrPlus(target.Arch, syms.GOT2, int64(ldr.SymGot(s)))
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
	got.AddUint32(target.Arch, 0)

	if target.IsElf() {
		rel := ldr.MakeSymbolUpdater(syms.Rel2)
		rel.AddAddrPlus(target.Arch, got.Sym(), int64(ldr.SymGot(s)))
		rel.AddUint32(target.Arch, ld.ELF32_R_INFO(uint32(ldr.SymDynid(s)), uint32(elf.R_386_GLOB_DAT)))
	} else if target.IsDarwin() {
		leg := ldr.MakeSymbolUpdater(syms.LinkEditGOT2)
		leg.AddUint32(target.Arch, uint32(ldr.SymDynid(s)))
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
	f := func(ctxt *ld.Link, out *ld.OutBuf, start, length int64) {
		ld.CodeblkPad(ctxt, out, start, length, []byte{0xCC})
	}
	ld.WriteParallel(&wg, f, ctxt, offset, sect.Vaddr, sect.Length)

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
	machlink := uint32(0)
	if ctxt.HeadType == objabi.Hdarwin {
		machlink = uint32(ld.Domacholink(ctxt))
	}

	ld.Symsize = 0
	ld.Spsize = 0
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

		case objabi.Hdarwin:
			symo = uint32(ld.Segdwarf.Fileoff + uint64(ld.Rnd(int64(ld.Segdwarf.Filelen), int64(*ld.FlagRound))) + uint64(machlink))

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

		case objabi.Hdarwin:
			if ctxt.LinkMode == ld.LinkExternal {
				ld.Machoemitreloc(ctxt)
			}
		}
	}

	ctxt.Out.SeekSet(0)
	switch ctxt.HeadType {
	default:
	case objabi.Hplan9: /* plan9 */
		magic := int32(4*11*11 + 7)

		ctxt.Out.Write32b(uint32(magic))              /* magic */
		ctxt.Out.Write32b(uint32(ld.Segtext.Filelen)) /* sizes */
		ctxt.Out.Write32b(uint32(ld.Segdata.Filelen))
		ctxt.Out.Write32b(uint32(ld.Segdata.Length - ld.Segdata.Filelen))
		ctxt.Out.Write32b(uint32(ld.Symsize))          /* nsyms */
		ctxt.Out.Write32b(uint32(ld.Entryvalue(ctxt))) /* va of entry */
		ctxt.Out.Write32b(uint32(ld.Spsize))           /* sp offsets */
		ctxt.Out.Write32b(uint32(ld.Lcsize))           /* line offsets */

	case objabi.Hdarwin:
		ld.Asmbmacho(ctxt)

	case objabi.Hlinux,
		objabi.Hfreebsd,
		objabi.Hnetbsd,
		objabi.Hopenbsd:
		ld.Asmbelf(ctxt, int64(symo))

	case objabi.Hwindows:
		ld.Asmbpe(ctxt)
	}
}
