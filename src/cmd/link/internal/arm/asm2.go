// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm

import (
	"cmd/internal/objabi"
	"cmd/link/internal/ld"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/elf"
)

// Temporary dumping ground for sym.Symbol version of helper
// functions in asm.go, still being used for some oses.
// FIXME: get rid of this file when dodata() is completely
// converted.

func adddynrel(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s *sym.Symbol, r *sym.Reloc) bool {
	targ := r.Sym

	switch r.Type {
	default:
		if r.Type >= objabi.ElfRelocOffset {
			ld.Errorf(s, "unexpected relocation type %d (%s)", r.Type, sym.RelocName(target.Arch, r.Type))
			return false
		}

		// Handle relocations found in ELF object files.
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_PLT32):
		r.Type = objabi.R_CALLARM

		if targ.Type == sym.SDYNIMPORT {
			addpltsym(target, syms, targ)
			r.Sym = syms.PLT
			r.Add = int64(braddoff(int32(r.Add), targ.Plt()/4))
		}

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_THM_PC22): // R_ARM_THM_CALL
		ld.Exitf("R_ARM_THM_CALL, are you using -marm?")
		return false

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_GOT32): // R_ARM_GOT_BREL
		if targ.Type != sym.SDYNIMPORT {
			addgotsyminternal(target, syms, targ)
		} else {
			addgotsym(target, syms, targ)
		}

		r.Type = objabi.R_CONST // write r->add during relocsym
		r.Sym = nil
		r.Add += int64(targ.Got())
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_GOT_PREL): // GOT(nil) + A - nil
		if targ.Type != sym.SDYNIMPORT {
			addgotsyminternal(target, syms, targ)
		} else {
			addgotsym(target, syms, targ)
		}

		r.Type = objabi.R_PCREL
		r.Sym = syms.GOT
		r.Add += int64(targ.Got()) + 4
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_GOTOFF): // R_ARM_GOTOFF32
		r.Type = objabi.R_GOTOFF

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_GOTPC): // R_ARM_BASE_PREL
		r.Type = objabi.R_PCREL

		r.Sym = syms.GOT
		r.Add += 4
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_CALL):
		r.Type = objabi.R_CALLARM
		if targ.Type == sym.SDYNIMPORT {
			addpltsym(target, syms, targ)
			r.Sym = syms.PLT
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
			addpltsym(target, syms, targ)
			r.Sym = syms.PLT
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
		if target.IsExternal() {
			// External linker will do this relocation.
			return true
		}
		addpltsym(target, syms, targ)
		r.Sym = syms.PLT
		r.Add = int64(targ.Plt())
		return true

	case objabi.R_ADDR:
		if s.Type != sym.SDATA {
			break
		}
		if target.IsElf() {
			ld.Adddynsym(target, syms, targ)
			rel := syms.Rel
			rel.AddAddrPlus(target.Arch, s, int64(r.Off))
			rel.AddUint32(target.Arch, ld.ELF32_R_INFO(uint32(targ.Dynid), uint32(elf.R_ARM_GLOB_DAT))) // we need a nil + A dynamic reloc
			r.Type = objabi.R_CONST                                                                     // write r->add during relocsym
			r.Sym = nil
			return true
		}
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
		got := syms.GOTPLT
		rel := syms.RelPLT
		if plt.Size == 0 {
			panic("plt is not set up")
		}

		// .got entry
		s.SetGot(int32(got.Size))

		// In theory, all GOT should point to the first PLT entry,
		// Linux/ARM's dynamic linker will do that for us, but FreeBSD/ARM's
		// dynamic linker won't, so we'd better do it ourselves.
		got.AddAddrPlus(target.Arch, plt, 0)

		// .plt entry, this depends on the .got entry
		s.SetPlt(int32(plt.Size))

		addpltreloc(plt, got, s, objabi.R_PLT0) // add lr, pc, #0xXX00000
		addpltreloc(plt, got, s, objabi.R_PLT1) // add lr, lr, #0xYY000
		addpltreloc(plt, got, s, objabi.R_PLT2) // ldr pc, [lr, #0xZZZ]!

		// rel
		rel.AddAddrPlus(target.Arch, got, int64(s.Got()))

		rel.AddUint32(target.Arch, ld.ELF32_R_INFO(uint32(s.Dynid), uint32(elf.R_ARM_JUMP_SLOT)))
	} else {
		ld.Errorf(s, "addpltsym: unsupported binary format")
	}
}

func addpltreloc(plt *sym.Symbol, got *sym.Symbol, s *sym.Symbol, typ objabi.RelocType) {
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

func addgotsym(target *ld.Target, syms *ld.ArchSyms, s *sym.Symbol) {
	if s.Got() >= 0 {
		return
	}

	ld.Adddynsym(target, syms, s)
	got := syms.GOT
	s.SetGot(int32(got.Size))
	got.AddUint32(target.Arch, 0)

	if target.IsElf() {
		rel := syms.Rel
		rel.AddAddrPlus(target.Arch, got, int64(s.Got()))
		rel.AddUint32(target.Arch, ld.ELF32_R_INFO(uint32(s.Dynid), uint32(elf.R_ARM_GLOB_DAT)))
	} else {
		ld.Errorf(s, "addgotsym: unsupported binary format")
	}
}

func addgotsyminternal(target *ld.Target, syms *ld.ArchSyms, s *sym.Symbol) {
	if s.Got() >= 0 {
		return
	}

	got := syms.GOT
	s.SetGot(int32(got.Size))

	got.AddAddrPlus(target.Arch, s, 0)

	if target.IsElf() {
	} else {
		ld.Errorf(s, "addgotsyminternal: unsupported binary format")
	}
}
