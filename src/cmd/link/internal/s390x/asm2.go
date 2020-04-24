// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package s390x

import (
	"cmd/internal/objabi"
	"cmd/link/internal/ld"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/elf"
)

func adddynrel(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s *sym.Symbol, r *sym.Reloc) bool {
	targ := r.Sym
	r.InitExt()

	switch r.Type {
	default:
		if r.Type >= objabi.ElfRelocOffset {
			ld.Errorf(s, "unexpected relocation type %d", r.Type)
			return false
		}

		// Handle relocations found in ELF object files.
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_12),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GOT12):
		ld.Errorf(s, "s390x 12-bit relocations have not been implemented (relocation type %d)", r.Type-objabi.ElfRelocOffset)
		return false

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_8),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_16),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_32),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_64):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_390_nn relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = objabi.R_ADDR
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PC16),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PC32),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PC64):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_390_PCnn relocation for dynamic symbol %s", targ.Name)
		}
		// TODO(mwhudson): the test of VisibilityHidden here probably doesn't make
		// sense and should be removed when someone has thought about it properly.
		if (targ.Type == 0 || targ.Type == sym.SXREF) && !targ.Attr.VisibilityHidden() {
			ld.Errorf(s, "unknown symbol %s in pcrel", targ.Name)
		}
		r.Type = objabi.R_PCREL
		r.Add += int64(r.Siz)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GOT16),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GOT32),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GOT64):
		ld.Errorf(s, "unimplemented S390x relocation: %v", r.Type-objabi.ElfRelocOffset)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PLT16DBL),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PLT32DBL):
		r.Type = objabi.R_PCREL
		r.Variant = sym.RV_390_DBL
		r.Add += int64(r.Siz)
		if targ.Type == sym.SDYNIMPORT {
			addpltsym(target, syms, targ)
			r.Sym = syms.PLT
			r.Add += int64(targ.Plt())
		}
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PLT32),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PLT64):
		r.Type = objabi.R_PCREL
		r.Add += int64(r.Siz)
		if targ.Type == sym.SDYNIMPORT {
			addpltsym(target, syms, targ)
			r.Sym = syms.PLT
			r.Add += int64(targ.Plt())
		}
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_COPY):
		ld.Errorf(s, "unimplemented S390x relocation: %v", r.Type-objabi.ElfRelocOffset)
		return false

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GLOB_DAT):
		ld.Errorf(s, "unimplemented S390x relocation: %v", r.Type-objabi.ElfRelocOffset)
		return false

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_JMP_SLOT):
		ld.Errorf(s, "unimplemented S390x relocation: %v", r.Type-objabi.ElfRelocOffset)
		return false

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_RELATIVE):
		ld.Errorf(s, "unimplemented S390x relocation: %v", r.Type-objabi.ElfRelocOffset)
		return false

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GOTOFF):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_390_GOTOFF relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = objabi.R_GOTOFF
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GOTPC):
		r.Type = objabi.R_PCREL
		r.Sym = syms.GOT
		r.Add += int64(r.Siz)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PC16DBL),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_PC32DBL):
		r.Type = objabi.R_PCREL
		r.Variant = sym.RV_390_DBL
		r.Add += int64(r.Siz)
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_390_PCnnDBL relocation for dynamic symbol %s", targ.Name)
		}
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GOTPCDBL):
		r.Type = objabi.R_PCREL
		r.Variant = sym.RV_390_DBL
		r.Sym = syms.GOT
		r.Add += int64(r.Siz)
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_390_GOTENT):
		addgotsym(target, syms, targ)

		r.Type = objabi.R_PCREL
		r.Variant = sym.RV_390_DBL
		r.Sym = syms.GOT
		r.Add += int64(targ.Got())
		r.Add += int64(r.Siz)
		return true
	}
	// Handle references to ELF symbols from our own object files.
	if targ.Type != sym.SDYNIMPORT {
		return true
	}

	return false
}

func addpltsym(target *ld.Target, syms *ld.ArchSyms, s *sym.Symbol) {
	if s.Plt() >= 0 {
		return
	}

	ld.Adddynsym(target, syms, s)

	if target.IsElf() {
		plt := syms.PLT
		got := syms.GOT
		rela := syms.RelaPLT
		if plt.Size == 0 {
			panic("plt is not set up")
		}
		// larl    %r1,_GLOBAL_OFFSET_TABLE_+index

		plt.AddUint8(0xc0)
		plt.AddUint8(0x10)
		plt.AddPCRelPlus(target.Arch, got, got.Size+6) // need variant?

		// add to got: pointer to current pos in plt
		got.AddAddrPlus(target.Arch, plt, plt.Size+8) // weird but correct
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

		plt.AddUint32(target.Arch, uint32(-((plt.Size - 2) >> 1))) // roll-your-own relocation
		//.plt index
		plt.AddUint32(target.Arch, uint32(rela.Size)) // rela size before current entry

		// rela
		rela.AddAddrPlus(target.Arch, got, got.Size-8)

		rela.AddUint64(target.Arch, ld.ELF64_R_INFO(uint32(s.Dynid), uint32(elf.R_390_JMP_SLOT)))
		rela.AddUint64(target.Arch, 0)

		s.SetPlt(int32(plt.Size - 32))

	} else {
		ld.Errorf(s, "addpltsym: unsupported binary format")
	}
}

func addgotsym(target *ld.Target, syms *ld.ArchSyms, s *sym.Symbol) {
	if s.Got() >= 0 {
		return
	}

	ld.Adddynsym(target, syms, s)
	got := syms.GOT
	s.SetGot(int32(got.Size))
	got.AddUint64(target.Arch, 0)

	if target.IsElf() {
		rela := syms.Rela
		rela.AddAddrPlus(target.Arch, got, int64(s.Got()))
		rela.AddUint64(target.Arch, ld.ELF64_R_INFO(uint32(s.Dynid), uint32(elf.R_390_GLOB_DAT)))
		rela.AddUint64(target.Arch, 0)
	} else {
		ld.Errorf(s, "addgotsym: unsupported binary format")
	}
}
