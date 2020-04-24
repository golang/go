// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86

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
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_386_PC32):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_386_PC32 relocation for dynamic symbol %s", targ.Name)
		}
		// TODO(mwhudson): the test of VisibilityHidden here probably doesn't make
		// sense and should be removed when someone has thought about it properly.
		if (targ.Type == 0 || targ.Type == sym.SXREF) && !targ.Attr.VisibilityHidden() {
			ld.Errorf(s, "unknown symbol %s in pcrel", targ.Name)
		}
		r.Type = objabi.R_PCREL
		r.Add += 4
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_386_PLT32):
		r.Type = objabi.R_PCREL
		r.Add += 4
		if targ.Type == sym.SDYNIMPORT {
			addpltsym(target, syms, targ)
			r.Sym = syms.PLT
			r.Add += int64(targ.Plt())
		}

		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_386_GOT32),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_386_GOT32X):
		if targ.Type != sym.SDYNIMPORT {
			// have symbol
			if r.Off >= 2 && s.P[r.Off-2] == 0x8b {
				// turn MOVL of GOT entry into LEAL of symbol address, relative to GOT.
				s.P[r.Off-2] = 0x8d

				r.Type = objabi.R_GOTOFF
				return true
			}

			if r.Off >= 2 && s.P[r.Off-2] == 0xff && s.P[r.Off-1] == 0xb3 {
				// turn PUSHL of GOT entry into PUSHL of symbol itself.
				// use unnecessary SS prefix to keep instruction same length.
				s.P[r.Off-2] = 0x36

				s.P[r.Off-1] = 0x68
				r.Type = objabi.R_ADDR
				return true
			}

			ld.Errorf(s, "unexpected GOT reloc for non-dynamic symbol %s", targ.Name)
			return false
		}

		addgotsym(target, syms, targ)
		r.Type = objabi.R_CONST // write r->add during relocsym
		r.Sym = nil
		r.Add += int64(targ.Got())
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_386_GOTOFF):
		r.Type = objabi.R_GOTOFF
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_386_GOTPC):
		r.Type = objabi.R_PCREL
		r.Sym = syms.GOT
		r.Add += 4
		return true

	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_386_32):
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected R_386_32 relocation for dynamic symbol %s", targ.Name)
		}
		r.Type = objabi.R_ADDR
		return true

	case objabi.MachoRelocOffset + ld.MACHO_GENERIC_RELOC_VANILLA*2 + 0:
		r.Type = objabi.R_ADDR
		if targ.Type == sym.SDYNIMPORT {
			ld.Errorf(s, "unexpected reloc for dynamic symbol %s", targ.Name)
		}
		return true

	case objabi.MachoRelocOffset + ld.MACHO_GENERIC_RELOC_VANILLA*2 + 1:
		if targ.Type == sym.SDYNIMPORT {
			addpltsym(target, syms, targ)
			r.Sym = syms.PLT
			r.Add = int64(targ.Plt())
			r.Type = objabi.R_PCREL
			return true
		}

		r.Type = objabi.R_PCREL
		return true

	case objabi.MachoRelocOffset + ld.MACHO_FAKE_GOTPCREL:
		if targ.Type != sym.SDYNIMPORT {
			// have symbol
			// turn MOVL of GOT entry into LEAL of symbol itself
			if r.Off < 2 || s.P[r.Off-2] != 0x8b {
				ld.Errorf(s, "unexpected GOT reloc for non-dynamic symbol %s", targ.Name)
				return false
			}

			s.P[r.Off-2] = 0x8d
			r.Type = objabi.R_PCREL
			return true
		}

		addgotsym(target, syms, targ)
		r.Sym = syms.GOT
		r.Add += int64(targ.Got())
		r.Type = objabi.R_PCREL
		return true
	}

	// Handle references to ELF symbols from our own object files.
	if targ.Type != sym.SDYNIMPORT {
		return true
	}
	switch r.Type {
	case objabi.R_CALL,
		objabi.R_PCREL:
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
			rel.AddUint32(target.Arch, ld.ELF32_R_INFO(uint32(targ.Dynid), uint32(elf.R_386_32)))
			r.Type = objabi.R_CONST // write r->add during relocsym
			r.Sym = nil
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
			got.AddUint32(target.Arch, 0)
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
		rel := syms.RelPLT
		if plt.Size == 0 {
			panic("plt is not set up")
		}

		// jmpq *got+size
		plt.AddUint8(0xff)

		plt.AddUint8(0x25)
		plt.AddAddrPlus(target.Arch, got, got.Size)

		// add to got: pointer to current pos in plt
		got.AddAddrPlus(target.Arch, plt, plt.Size)

		// pushl $x
		plt.AddUint8(0x68)

		plt.AddUint32(target.Arch, uint32(rel.Size))

		// jmp .plt
		plt.AddUint8(0xe9)

		plt.AddUint32(target.Arch, uint32(-(plt.Size + 4)))

		// rel
		rel.AddAddrPlus(target.Arch, got, got.Size-4)

		rel.AddUint32(target.Arch, ld.ELF32_R_INFO(uint32(s.Dynid), uint32(elf.R_386_JMP_SLOT)))

		s.SetPlt(int32(plt.Size - 16))
	} else if target.IsDarwin() {
		// Same laziness as in 6l.

		plt := syms.PLT

		addgotsym(target, syms, s)

		syms.LinkEditPLT.AddUint32(target.Arch, uint32(s.Dynid))

		// jmpq *got+size(IP)
		s.SetPlt(int32(plt.Size))

		plt.AddUint8(0xff)
		plt.AddUint8(0x25)
		plt.AddAddrPlus(target.Arch, syms.GOT, int64(s.Got()))
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
	got.AddUint32(target.Arch, 0)

	if target.IsElf() {
		rel := syms.Rel
		rel.AddAddrPlus(target.Arch, got, int64(s.Got()))
		rel.AddUint32(target.Arch, ld.ELF32_R_INFO(uint32(s.Dynid), uint32(elf.R_386_GLOB_DAT)))
	} else if target.IsDarwin() {
		syms.LinkEditGOT.AddUint32(target.Arch, uint32(s.Dynid))
	} else {
		ld.Errorf(s, "addgotsym: unsupported binary format")
	}
}
