// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package riscv64

import (
	"cmd/internal/obj/riscv"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/ld"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/elf"
	"fmt"
	"log"
	"sort"
	"sync"
)

// fakeLabelName matches the RISCV_FAKE_LABEL_NAME from binutils.
const fakeLabelName = ".L0 "

func gentext2(ctxt *ld.Link, ldr *loader.Loader) {
}

func adddynrela(target *ld.Target, syms *ld.ArchSyms, rel *sym.Symbol, s *sym.Symbol, r *sym.Reloc) {
	log.Fatalf("adddynrela not implemented")
}

func adddynrel(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, s *sym.Symbol, r *sym.Reloc) bool {
	log.Fatalf("adddynrel not implemented")
	return false
}

func findHI20Symbol(ctxt *ld.Link, val int64) *sym.Symbol {
	for idx := sort.Search(len(ctxt.Textp), func(i int) bool { return ctxt.Textp[i].Value >= val }); idx < len(ctxt.Textp); idx++ {
		s := ctxt.Textp[idx]
		if s.Value != val {
			return nil
		}
		if s.Type == sym.STEXT && s.Name == fakeLabelName {
			return s
		}
	}
	return nil
}

func elfreloc1(ctxt *ld.Link, s *sym.Symbol, r *sym.Reloc, sectoff int64) bool {
	elfsym := ld.ElfSymForReloc(ctxt, r.Xsym)
	switch r.Type {
	case objabi.R_ADDR:
		ctxt.Out.Write64(uint64(sectoff))
		switch r.Siz {
		case 4:
			ctxt.Out.Write64(uint64(elf.R_RISCV_32) | uint64(elfsym)<<32)
		case 8:
			ctxt.Out.Write64(uint64(elf.R_RISCV_64) | uint64(elfsym)<<32)
		default:
			ld.Errorf(nil, "unknown size %d for %v relocation", r.Siz, r.Type)
			return false
		}
		ctxt.Out.Write64(uint64(r.Xadd))

	case objabi.R_CALLRISCV:
		// Call relocations are currently handled via R_RISCV_PCREL_ITYPE.
		// TODO(jsing): Consider generating elf.R_RISCV_CALL instead of a
		// HI20/LO12_I pair.

	case objabi.R_RISCV_PCREL_ITYPE, objabi.R_RISCV_PCREL_STYPE, objabi.R_RISCV_TLS_IE_ITYPE, objabi.R_RISCV_TLS_IE_STYPE:
		// Find the text symbol for the AUIPC instruction targeted
		// by this relocation.
		hi20Sym := findHI20Symbol(ctxt, s.Value+int64(r.Off))
		if hi20Sym == nil || hi20Sym.Type != sym.STEXT {
			ld.Errorf(nil, "failed to find text symbol for HI20 relocation at %d (%x)", sectoff, s.Value+int64(r.Off))
			return false
		}

		// Emit two relocations - a R_RISCV_PCREL_HI20 relocation and a
		// corresponding R_RISCV_PCREL_LO12_I or R_RISCV_PCREL_LO12_S relocation.
		// Note that the LO12 relocation must refer to a text symbol that points
		// to the instruction that has the HI20 relocation given for a symbol.
		var hiRel, loRel elf.R_RISCV
		switch r.Type {
		case objabi.R_RISCV_PCREL_ITYPE:
			hiRel, loRel = elf.R_RISCV_PCREL_HI20, elf.R_RISCV_PCREL_LO12_I
		case objabi.R_RISCV_PCREL_STYPE:
			hiRel, loRel = elf.R_RISCV_PCREL_HI20, elf.R_RISCV_PCREL_LO12_S
		case objabi.R_RISCV_TLS_IE_ITYPE:
			hiRel, loRel = elf.R_RISCV_TLS_GOT_HI20, elf.R_RISCV_PCREL_LO12_I
		case objabi.R_RISCV_TLS_IE_STYPE:
			hiRel, loRel = elf.R_RISCV_TLS_GOT_HI20, elf.R_RISCV_PCREL_LO12_S
		}
		ctxt.Out.Write64(uint64(sectoff))
		ctxt.Out.Write64(uint64(hiRel) | uint64(elfsym)<<32)
		ctxt.Out.Write64(uint64(r.Xadd))
		ctxt.Out.Write64(uint64(sectoff + 4))
		ctxt.Out.Write64(uint64(loRel) | uint64(hi20Sym.Got())<<32)
		ctxt.Out.Write64(uint64(0))

	default:
		return false
	}

	return true
}

func elfsetupplt(ctxt *ld.Link, plt, gotplt *loader.SymbolBuilder, dynamic loader.Sym) {
	log.Fatalf("elfsetuplt")
}

func machoreloc1(arch *sys.Arch, out *ld.OutBuf, s *sym.Symbol, r *sym.Reloc, sectoff int64) bool {
	log.Fatalf("machoreloc1 not implemented")
	return false
}

func archreloc(target *ld.Target, syms *ld.ArchSyms, r *sym.Reloc, s *sym.Symbol, val int64) (int64, bool) {
	if target.IsExternal() {
		switch r.Type {
		case objabi.R_CALLRISCV:
			r.Done = false
			r.Xsym = r.Sym
			r.Xadd = r.Add
			return val, true

		case objabi.R_RISCV_PCREL_ITYPE, objabi.R_RISCV_PCREL_STYPE, objabi.R_RISCV_TLS_IE_ITYPE, objabi.R_RISCV_TLS_IE_STYPE:
			r.Done = false

			// Set up addend for eventual relocation via outer symbol.
			rs := r.Sym
			r.Xadd = r.Add
			for rs.Outer != nil {
				r.Xadd += ld.Symaddr(rs) - ld.Symaddr(rs.Outer)
				rs = rs.Outer
			}

			if rs.Type != sym.SHOSTOBJ && rs.Type != sym.SDYNIMPORT && rs.Sect == nil {
				ld.Errorf(s, "missing section for %s", rs.Name)
			}
			r.Xsym = rs

			return val, true
		}

		return val, false
	}

	switch r.Type {
	case objabi.R_CALLRISCV:
		// Nothing to do.
		return val, true

	case objabi.R_RISCV_TLS_IE_ITYPE, objabi.R_RISCV_TLS_IE_STYPE:
		// Nothing to do.
		return val, true

	case objabi.R_RISCV_PCREL_ITYPE, objabi.R_RISCV_PCREL_STYPE:
		pc := s.Value + int64(r.Off)
		off := ld.Symaddr(r.Sym) + r.Add - pc

		// Generate AUIPC and second instruction immediates.
		low, high, err := riscv.Split32BitImmediate(off)
		if err != nil {
			ld.Errorf(s, "R_RISCV_PCREL_ relocation does not fit in 32-bits: %d", off)
		}

		auipcImm, err := riscv.EncodeUImmediate(high)
		if err != nil {
			ld.Errorf(s, "cannot encode R_RISCV_PCREL_ AUIPC relocation offset for %s: %v", r.Sym.Name, err)
		}

		var secondImm, secondImmMask int64
		switch r.Type {
		case objabi.R_RISCV_PCREL_ITYPE:
			secondImmMask = riscv.ITypeImmMask
			secondImm, err = riscv.EncodeIImmediate(low)
			if err != nil {
				ld.Errorf(s, "cannot encode R_RISCV_PCREL_ITYPE I-type instruction relocation offset for %s: %v", r.Sym.Name, err)
			}
		case objabi.R_RISCV_PCREL_STYPE:
			secondImmMask = riscv.STypeImmMask
			secondImm, err = riscv.EncodeSImmediate(low)
			if err != nil {
				ld.Errorf(s, "cannot encode R_RISCV_PCREL_STYPE S-type instruction relocation offset for %s: %v", r.Sym.Name, err)
			}
		default:
			panic(fmt.Sprintf("Unknown relocation type: %v", r.Type))
		}

		auipc := int64(uint32(val))
		second := int64(uint32(val >> 32))

		auipc = (auipc &^ riscv.UTypeImmMask) | int64(uint32(auipcImm))
		second = (second &^ secondImmMask) | int64(uint32(secondImm))

		return second<<32 | auipc, true
	}

	return val, false
}

func archrelocvariant(target *ld.Target, syms *ld.ArchSyms, r *sym.Reloc, s *sym.Symbol, t int64) int64 {
	log.Fatalf("archrelocvariant")
	return -1
}

func genHi20TextSymbols(ctxt *ld.Link) {
	// Generate a local text symbol for each relocation target, as the
	// R_RISCV_PCREL_LO12_* relocations generated by elfreloc1 need it.
	var syms []*sym.Symbol
	for _, s := range ctxt.Textp {
		for _, r := range s.R {
			if r.Type != objabi.R_RISCV_PCREL_ITYPE && r.Type != objabi.R_RISCV_PCREL_STYPE &&
				r.Type != objabi.R_RISCV_TLS_IE_ITYPE && r.Type != objabi.R_RISCV_TLS_IE_STYPE {
				continue
			}
			sym := &sym.Symbol{
				Type:  sym.STEXT,
				Name:  fakeLabelName,
				Value: s.Value + int64(r.Off),
				Attr:  sym.AttrDuplicateOK | sym.AttrLocal | sym.AttrVisibilityHidden,
				Sect:  s.Sect,
			}
			syms = append(syms, sym)
		}
	}
	ctxt.Textp = append(ctxt.Textp, syms...)
	sort.SliceStable(ctxt.Textp, func(i, j int) bool { return ctxt.Textp[i].Value < ctxt.Textp[j].Value })
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

		genHi20TextSymbols(ctxt)

		ld.Asmelfsym(ctxt)
		ctxt.Out.Write(ld.Elfstrdat)

		if ctxt.LinkMode == ld.LinkExternal {
			ld.Elfemitreloc(ctxt)
		}
	}

	ctxt.Out.SeekSet(0)
	switch ctxt.HeadType {
	case objabi.Hlinux:
		ld.Asmbelf(ctxt, int64(symo))
	default:
		ld.Errorf(nil, "unsupported operating system")
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
