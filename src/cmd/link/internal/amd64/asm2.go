// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package amd64

import (
	"cmd/internal/objabi"
	"cmd/link/internal/ld"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/elf"
)

// Temporary dumping around for sym.Symbol version of helper
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
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_X86_64_PC32):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_X86_64_PC32 relocation for dynamic symbol %s", targ.Name)
		}
		// TODO(mwhudson): the test of VisibilityHidden here probably doesn't make
		// sense and should be removed when someone has thought about it properly.
		if (targ.Type == 0 || targ.Type == sym.SXREF) && !targ.Attr.VisibilityHidden() {
			ld.Errorf(s, "unknown symbol %s in pcrel", targ.Name)
		}
		r.Type = objabi.R_PCREL
		r.Add += 4
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_X86_64_PC64):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_X86_64_PC64 relocation for dynamic symbol %s", targ.Name)
		}
		if targ.Type == 0 || targ.Type == sym.SXREF {
			ld.Errorf(s, "unknown symbol %s in pcrel", targ.Name)
		}
		r.Type = objabi.R_PCREL
		r.Add += 8
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_X86_64_PLT32):
		r.Type = objabi.R_PCREL
		r.Add += 4
		if targ.Type == sym.SDYNIMPORT {
			addpltsym(target, syms, targ)
			r.Sym = syms.PLT
			r.Add += int64(targ.Plt())
		}

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_X86_64_GOTPCREL),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_X86_64_GOTPCRELX),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_X86_64_REX_GOTPCRELX):
		if targ.Type != sym.SDYNIMPORT {
			// have symbol
			if r.Off >= 2 && s.P[r.Off-2] == 0x8b {
				makeWritable(s)
				// turn MOVQ of GOT entry into LEAQ of symbol itself
				s.P[r.Off-2] = 0x8d

				r.Type = objabi.R_PCREL
				r.Add += 4
				return true
			}
		}

		// fall back to using GOT and hope for the best (CMOV*)
		// TODO: just needs relocation, no need to put in .dynsym
		addgotsym(target, syms, targ)

		r.Type = objabi.R_PCREL
		r.Sym = syms.GOT
		r.Add += 4
		r.Add += int64(targ.Got())
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_X86_64_64):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_X86_64_64 relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = objabi.R_ADDR
		if target.IsPIE() && target.IsInternal() {
			// For internal linking PIE, this R_ADDR relocation cannot
			// be resolved statically. We need to generate a dynamic
			// relocation. Let the code below handle it.
			break
		}
		return true

	// Handle relocations found in Mach-O object files.
	case objabi.MachoRelocOffset + ld.MACHO_X86_64_RELOC_UNSIGNED*2 + 0,
		objabi.MachoRelocOffset + ld.MACHO_X86_64_RELOC_SIGNED*2 + 0,
		objabi.MachoRelocOffset + ld.MACHO_X86_64_RELOC_BRANCH*2 + 0:
		// TODO: What is the difference between all these?
		r.Type = objabi.R_ADDR

		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected reloc for dynamic symbol %s", targ.Name)
		}
		return true

	case objabi.MachoRelocOffset + ld.MACHO_X86_64_RELOC_BRANCH*2 + 1:
		if targ.Type == sym.SDYNIMPORT {
			addpltsym(target, syms, targ)
			r.Sym = syms.PLT
			r.Add = int64(targ.Plt())
			r.Type = objabi.R_PCREL
			return true
		}
		fallthrough

	case objabi.MachoRelocOffset + ld.MACHO_X86_64_RELOC_UNSIGNED*2 + 1,
		objabi.MachoRelocOffset + ld.MACHO_X86_64_RELOC_SIGNED*2 + 1,
		objabi.MachoRelocOffset + ld.MACHO_X86_64_RELOC_SIGNED_1*2 + 1,
		objabi.MachoRelocOffset + ld.MACHO_X86_64_RELOC_SIGNED_2*2 + 1,
		objabi.MachoRelocOffset + ld.MACHO_X86_64_RELOC_SIGNED_4*2 + 1:
		r.Type = objabi.R_PCREL

		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected pc-relative reloc for dynamic symbol %s", targ.Name)
		}
		return true

	case objabi.MachoRelocOffset + ld.MACHO_X86_64_RELOC_GOT_LOAD*2 + 1:
		if targ.Type != sym.SDYNIMPORT {
			// have symbol
			// turn MOVQ of GOT entry into LEAQ of symbol itself
			if r.Off < 2 || s.P[r.Off-2] != 0x8b {
				ld.Errorf(s, "unexpected GOT_LOAD reloc for non-dynamic symbol %s", targ.Name)
				return false
			}

			makeWritable(s)
			s.P[r.Off-2] = 0x8d
			r.Type = objabi.R_PCREL
			return true
		}
		fallthrough

	case objabi.MachoRelocOffset + ld.MACHO_X86_64_RELOC_GOT*2 + 1:
		if targ.Type != sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected GOT reloc for non-dynamic symbol %s", targ.Name)
		}
		addgotsym(target, syms, targ)
		r.Type = objabi.R_PCREL
		r.Sym = syms.GOT
		r.Add += int64(targ.Got())
		return true
	}

	switch r.Type {
	case objabi.R_CALL,
		objabi.R_PCREL:
		if targ.Type != sym.SDYNIMPORT {
			// nothing to do, the relocation will be laid out in reloc
			return true
		}
		if target.IsExternal() {
			// External linker will do this relocation.
			return true
		}
		// Internal linking, for both ELF and Mach-O.
		// Build a PLT entry and change the relocation target to that entry.
		addpltsym(target, syms, targ)
		r.Sym = syms.PLT
		r.Add = int64(targ.Plt())
		return true

	case objabi.R_ADDR:
		if s.Type == sym.STEXT && target.IsElf() {
			if target.IsSolaris() {
				addpltsym(target, syms, targ)
				r.Sym = syms.PLT
				r.Add += int64(targ.Plt())
				return true
			}
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
			// Generate R_X86_64_RELATIVE relocations for best
			// efficiency in the dynamic linker.
			//
			// As noted above, symbol addresses have not been
			// assigned yet, so we can't generate the final reloc
			// entry yet. We ultimately want:
			//
			// r_offset = s + r.Off
			// r_info = R_X86_64_RELATIVE
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
				rela.AddUint64(target.Arch, ld.ELF64_R_INFO(0, uint32(elf.R_X86_64_RELATIVE)))
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

		if target.IsDarwin() && s.Size == int64(target.Arch.PtrSize) && r.Off == 0 {
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
			ld.Adddynsym(target, syms, targ)

			got := syms.GOT
			s.Type = got.Type
			s.Attr |= sym.AttrSubSymbol
			s.Outer = got
			s.Sub = got.Sub
			got.Sub = s
			s.Value = got.Size
			got.AddUint64(target.Arch, 0)
			syms.LinkEditGOT.AddUint32(target.Arch, uint32(targ.Dynid))
			r.Type = objabi.ElfRelocOffset // ignore during relocsym
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
		rela := syms.RelaPLT
		if plt.Size == 0 {
			panic("plt is not set up")
		}

		// jmpq *got+size(IP)
		plt.AddUint8(0xff)

		plt.AddUint8(0x25)
		plt.AddPCRelPlus(target.Arch, got, got.Size)

		// add to got: pointer to current pos in plt
		got.AddAddrPlus(target.Arch, plt, plt.Size)

		// pushq $x
		plt.AddUint8(0x68)

		plt.AddUint32(target.Arch, uint32((got.Size-24-8)/8))

		// jmpq .plt
		plt.AddUint8(0xe9)

		plt.AddUint32(target.Arch, uint32(-(plt.Size + 4)))

		// rela
		rela.AddAddrPlus(target.Arch, got, got.Size-8)

		rela.AddUint64(target.Arch, ld.ELF64_R_INFO(uint32(s.Dynid), uint32(elf.R_X86_64_JMP_SLOT)))
		rela.AddUint64(target.Arch, 0)

		s.SetPlt(int32(plt.Size - 16))
	} else if target.IsDarwin() {
		// To do lazy symbol lookup right, we're supposed
		// to tell the dynamic loader which library each
		// symbol comes from and format the link info
		// section just so. I'm too lazy (ha!) to do that
		// so for now we'll just use non-lazy pointers,
		// which don't need to be told which library to use.
		//
		// https://networkpx.blogspot.com/2009/09/about-lcdyldinfoonly-command.html
		// has details about what we're avoiding.

		addgotsym(target, syms, s)
		plt := syms.PLT

		syms.LinkEditPLT.AddUint32(target.Arch, uint32(s.Dynid))

		// jmpq *got+size(IP)
		s.SetPlt(int32(plt.Size))

		plt.AddUint8(0xff)
		plt.AddUint8(0x25)
		plt.AddPCRelPlus(target.Arch, syms.GOT, int64(s.Got()))
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
		rela.AddUint64(target.Arch, ld.ELF64_R_INFO(uint32(s.Dynid), uint32(elf.R_X86_64_GLOB_DAT)))
		rela.AddUint64(target.Arch, 0)
	} else if target.IsDarwin() {
		syms.LinkEditGOT.AddUint32(target.Arch, uint32(s.Dynid))
	} else {
		ld.Errorf(s, "addgotsym: unsupported binary format")
	}
}
