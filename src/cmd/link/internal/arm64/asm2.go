// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

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
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_PREL32):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_AARCH64_PREL32 relocation for dynamic symbol %s", targ.Name)
		}
		// TODO(mwhudson): the test of VisibilityHidden here probably doesn't make
		// sense and should be removed when someone has thought about it properly.
		if (targ.Type == 0 || targ.Type == sym.SXREF) && !targ.Attr.VisibilityHidden() {
			ld.Errorf(s, "unknown symbol %s in pcrel", targ.Name)
		}
		r.Type = objabi.R_PCREL
		r.Add += 4
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_PREL64):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_AARCH64_PREL64 relocation for dynamic symbol %s", targ.Name)
		}
		if targ.Type == 0 || targ.Type == sym.SXREF {
			ld.Errorf(s, "unknown symbol %s in pcrel", targ.Name)
		}
		r.Type = objabi.R_PCREL
		r.Add += 8
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_CALL26),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_JUMP26):
		if targ.Type == sym.SDYNIMPORT {
			addpltsym(target, syms, targ)
			r.Sym = syms.PLT
			r.Add += int64(targ.Plt())
		}
		if (targ.Type == 0 || targ.Type == sym.SXREF) && !targ.Attr.VisibilityHidden() {
			ld.Errorf(s, "unknown symbol %s in callarm64", targ.Name)
		}
		r.Type = objabi.R_CALLARM64
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_ADR_GOT_PAGE),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_LD64_GOT_LO12_NC):
		if targ.Type != sym.SDYNIMPORT {
			// have symbol
			// TODO: turn LDR of GOT entry into ADR of symbol itself
		}

		// fall back to using GOT
		// TODO: just needs relocation, no need to put in .dynsym
		addgotsym(target, syms, targ)

		r.Type = objabi.R_ARM64_GOT
		r.Sym = syms.GOT
		r.Add += int64(targ.Got())
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_ADR_PREL_PG_HI21),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_ADD_ABS_LO12_NC):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected relocation for dynamic symbol %s", targ.Name)
		}
		if targ.Type == 0 || targ.Type == sym.SXREF {
			ld.Errorf(s, "unknown symbol %s", targ.Name)
		}
		r.Type = objabi.R_ARM64_PCREL
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_ABS64):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_AARCH64_ABS64 relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = objabi.R_ADDR
		if target.IsPIE() && target.IsInternal() {
			// For internal linking PIE, this R_ADDR relocation cannot
			// be resolved statically. We need to generate a dynamic
			// relocation. Let the code below handle it.
			break
		}
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_LDST8_ABS_LO12_NC):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = objabi.R_ARM64_LDST8
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_LDST32_ABS_LO12_NC):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = objabi.R_ARM64_LDST32
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_LDST64_ABS_LO12_NC):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = objabi.R_ARM64_LDST64
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_LDST128_ABS_LO12_NC):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = objabi.R_ARM64_LDST128
		return true
	}

	switch r.Type {
	case objabi.R_CALL,
		objabi.R_PCREL,
		objabi.R_CALLARM64:
		if targ.Type != sym.SDYNIMPORT {
			// nothing to do, the relocation will be laid out in reloc
			return true
		}
		if target.IsExternal() {
			// External linker will do this relocation.
			return true
		}

	case objabi.R_ADDR:
		if s.Type == sym.STEXT && target.IsElf() {
			// The code is asking for the address of an external
			// function. We provide it with the address of the
			// correspondent GOT symbol.
			addgotsym(target, syms, targ)

			r.Sym = syms.GOT
			r.Add += int64(targ.Got())
			return true
		}

		// Process dynamic relocations for the data sections.
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
			switch s.Name {
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
			if s.Type != sym.SDATA && s.Type != sym.SRODATA {
				break
			}
		}

		if target.IsElf() {
			// Generate R_AARCH64_RELATIVE relocations for best
			// efficiency in the dynamic linker.
			//
			// As noted above, symbol addresses have not been
			// assigned yet, so we can't generate the final reloc
			// entry yet. We ultimately want:
			//
			// r_offset = s + r.Off
			// r_info = R_AARCH64_RELATIVE
			// r_addend = targ + r.Add
			//
			// The dynamic linker will set *offset = base address +
			// addend.
			//
			// AddAddrPlus is used for r_offset and r_addend to
			// generate new R_ADDR relocations that will update
			// these fields in the 'reloc' phase.
			rela := syms.Rela
			rela.AddAddrPlus(target.Arch, s, int64(r.Off))
			if r.Siz == 8 {
				rela.AddUint64(target.Arch, ld.ELF64_R_INFO(0, uint32(elf.R_AARCH64_RELATIVE)))
			} else {
				ld.Errorf(s, "unexpected relocation for dynamic symbol %s", targ.Name)
			}
			rela.AddAddrPlus(target.Arch, targ, int64(r.Add))
			// Not mark r done here. So we still apply it statically,
			// so in the file content we'll also have the right offset
			// to the relocation target. So it can be examined statically
			// (e.g. go version).
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
		gotplt := syms.GOTPLT
		rela := syms.RelaPLT
		if plt.Size == 0 {
			panic("plt is not set up")
		}

		// adrp    x16, &got.plt[0]
		plt.AddAddrPlus4(gotplt, gotplt.Size)
		plt.SetUint32(target.Arch, plt.Size-4, 0x90000010)
		plt.R[len(plt.R)-1].Type = objabi.R_ARM64_GOT

		// <offset> is the offset value of &got.plt[n] to &got.plt[0]
		// ldr     x17, [x16, <offset>]
		plt.AddAddrPlus4(gotplt, gotplt.Size)
		plt.SetUint32(target.Arch, plt.Size-4, 0xf9400211)
		plt.R[len(plt.R)-1].Type = objabi.R_ARM64_GOT

		// add     x16, x16, <offset>
		plt.AddAddrPlus4(gotplt, gotplt.Size)
		plt.SetUint32(target.Arch, plt.Size-4, 0x91000210)
		plt.R[len(plt.R)-1].Type = objabi.R_ARM64_PCREL

		// br      x17
		plt.AddUint32(target.Arch, 0xd61f0220)

		// add to got.plt: pointer to plt[0]
		gotplt.AddAddrPlus(target.Arch, plt, 0)

		// rela
		rela.AddAddrPlus(target.Arch, gotplt, gotplt.Size-8)
		rela.AddUint64(target.Arch, ld.ELF64_R_INFO(uint32(s.Dynid), uint32(elf.R_AARCH64_JUMP_SLOT)))
		rela.AddUint64(target.Arch, 0)

		s.SetPlt(int32(plt.Size - 16))
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
		rela.AddUint64(target.Arch, ld.ELF64_R_INFO(uint32(s.Dynid), uint32(elf.R_AARCH64_GLOB_DAT)))
		rela.AddUint64(target.Arch, 0)
	} else {
		ld.Errorf(s, "addgotsym: unsupported binary format")
	}
}
