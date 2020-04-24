// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ppc64

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
	if target.IsElf() {
		return addelfdynrel(target, syms, s, r)
	} else if target.IsAIX() {
		return ld.Xcoffadddynrel(target, ldr, s, r)
	}
	return false
}

func addelfdynrel(target *ld.Target, syms *ld.ArchSyms, s *sym.Symbol, r *sym.Reloc) bool {
	targ := r.Sym
	r.InitExt()

	switch r.Type {
	default:
		if r.Type >= objabi.ElfRelocOffset {
			ld.Errorf(s, "unexpected relocation type %d (%s)", r.Type, sym.RelocName(target.Arch, r.Type))
			return false
		}

		// Handle relocations found in ELF object files.
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL24):
		r.Type = objabi.R_CALLPOWER

		// This is a local call, so the caller isn't setting
		// up r12 and r2 is the same for the caller and
		// callee. Hence, we need to go to the local entry
		// point.  (If we don't do this, the callee will try
		// to use r12 to compute r2.)
		r.Add += int64(r.Sym.Localentry()) * 4

		if targ.Type == sym.SDYNIMPORT {
			// Should have been handled in elfsetupplt
			ld.Errorf(s, "unexpected R_PPC64_REL24 for dyn import")
		}

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC_REL32):
		r.Type = objabi.R_PCREL
		r.Add += 4

		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_PPC_REL32 for dyn import")
		}

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_ADDR64):
		r.Type = objabi.R_ADDR
		if targ.Type == sym.SDYNIMPORT {
			// These happen in .toc sections
			ld.Adddynsym(target, syms, targ)

			rela := syms.Rela
			rela.AddAddrPlus(target.Arch, s, int64(r.Off))
			rela.AddUint64(target.Arch, ld.ELF64_R_INFO(uint32(targ.Dynid), uint32(elf.R_PPC64_ADDR64)))
			rela.AddUint64(target.Arch, uint64(r.Add))
			r.Type = objabi.ElfRelocOffset // ignore during relocsym
		}

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16):
		r.Type = objabi.R_POWER_TOC
		r.Variant = sym.RV_POWER_LO | sym.RV_CHECK_OVERFLOW
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16_LO):
		r.Type = objabi.R_POWER_TOC
		r.Variant = sym.RV_POWER_LO
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16_HA):
		r.Type = objabi.R_POWER_TOC
		r.Variant = sym.RV_POWER_HA | sym.RV_CHECK_OVERFLOW
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16_HI):
		r.Type = objabi.R_POWER_TOC
		r.Variant = sym.RV_POWER_HI | sym.RV_CHECK_OVERFLOW
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16_DS):
		r.Type = objabi.R_POWER_TOC
		r.Variant = sym.RV_POWER_DS | sym.RV_CHECK_OVERFLOW
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_TOC16_LO_DS):
		r.Type = objabi.R_POWER_TOC
		r.Variant = sym.RV_POWER_DS
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL16_LO):
		r.Type = objabi.R_PCREL
		r.Variant = sym.RV_POWER_LO
		r.Add += 2 // Compensate for relocation size of 2
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL16_HI):
		r.Type = objabi.R_PCREL
		r.Variant = sym.RV_POWER_HI | sym.RV_CHECK_OVERFLOW
		r.Add += 2
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_PPC64_REL16_HA):
		r.Type = objabi.R_PCREL
		r.Variant = sym.RV_POWER_HA | sym.RV_CHECK_OVERFLOW
		r.Add += 2
		return true
	}

	// Handle references to ELF symbols from our own object files.
	if targ.Type != sym.SDYNIMPORT {
		return true
	}

	// TODO(austin): Translate our relocations to ELF

	return false
}
