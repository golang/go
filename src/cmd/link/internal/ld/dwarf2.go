// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO/NICETOHAVE:
//   - eliminate DW_CLS_ if not used
//   - package info in compilation units
//   - assign types to their packages
//   - gdb uses c syntax, meaning clumsy quoting is needed for go identifiers. eg
//     ptype struct '[]uint8' and qualifiers need to be quoted away
//   - file:line info for variables
//   - make strings a typedef so prettyprinters can see the underlying string type

package ld

import (
	"cmd/internal/dwarf"
	"cmd/internal/objabi"
	"cmd/link/internal/sym"
	"fmt"
	"log"
)

func isDwarf64(ctxt *Link) bool {
	return ctxt.HeadType == objabi.Haix
}

var dwarfp []*sym.Symbol

// Every DIE manufactured by the linker has at least an AT_name
// attribute (but it will only be written out if it is listed in the abbrev).
// The compiler does create nameless DWARF DIEs (ex: concrete subprogram
// instance).
func newdie(ctxt *Link, parent *dwarf.DWDie, abbrev int, name string, version int) *dwarf.DWDie {
	die := new(dwarf.DWDie)
	die.Abbrev = abbrev
	die.Link = parent.Child
	parent.Child = die

	newattr(die, dwarf.DW_AT_name, dwarf.DW_CLS_STRING, int64(len(name)), name)

	if name != "" && (abbrev <= dwarf.DW_ABRV_VARIABLE || abbrev >= dwarf.DW_ABRV_NULLTYPE) {
		if abbrev != dwarf.DW_ABRV_VARIABLE || version == 0 {
			if abbrev == dwarf.DW_ABRV_COMPUNIT {
				// Avoid collisions with "real" symbol names.
				name = fmt.Sprintf(".pkg.%s.%d", name, len(ctxt.compUnits))
			}
			s := ctxt.Syms.Lookup(dwarf.InfoPrefix+name, version)
			s.Attr |= sym.AttrNotInSymbolTable
			s.Type = sym.SDWARFINFO
			die.Sym = s
		}
	}

	return die
}

/*
 *  Elf.
 */
func dwarfaddshstrings(ctxt *Link, shstrtab *sym.Symbol) {
	if *FlagW { // disable dwarf
		return
	}

	secs := []string{"abbrev", "frame", "info", "loc", "line", "pubnames", "pubtypes", "gdb_scripts", "ranges"}
	for _, sec := range secs {
		Addstring(shstrtab, ".debug_"+sec)
		if ctxt.LinkMode == LinkExternal {
			Addstring(shstrtab, elfRelType+".debug_"+sec)
		} else {
			Addstring(shstrtab, ".zdebug_"+sec)
		}
	}
}

// Add section symbols for DWARF debug info.  This is called before
// dwarfaddelfheaders.
func dwarfaddelfsectionsyms(ctxt *Link) {
	if *FlagW { // disable dwarf
		return
	}
	if ctxt.LinkMode != LinkExternal {
		return
	}

	s := ctxt.Syms.Lookup(".debug_info", 0)
	putelfsectionsym(ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	s = ctxt.Syms.Lookup(".debug_abbrev", 0)
	putelfsectionsym(ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	s = ctxt.Syms.Lookup(".debug_line", 0)
	putelfsectionsym(ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	s = ctxt.Syms.Lookup(".debug_frame", 0)
	putelfsectionsym(ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	s = ctxt.Syms.Lookup(".debug_loc", 0)
	if s.Sect != nil {
		putelfsectionsym(ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	}
	s = ctxt.Syms.Lookup(".debug_ranges", 0)
	if s.Sect != nil {
		putelfsectionsym(ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	}
}

// dwarfcompress compresses the DWARF sections. Relocations are applied
// on the fly. After this, dwarfp will contain a different (new) set of
// symbols, and sections may have been replaced.
func dwarfcompress(ctxt *Link) {
	supported := ctxt.IsELF || ctxt.HeadType == objabi.Hwindows || ctxt.HeadType == objabi.Hdarwin
	if !ctxt.compressDWARF || !supported || ctxt.LinkMode != LinkInternal {
		return
	}

	var start int
	var newDwarfp []*sym.Symbol
	Segdwarf.Sections = Segdwarf.Sections[:0]
	for i, s := range dwarfp {
		// Find the boundaries between sections and compress
		// the whole section once we've found the last of its
		// symbols.
		if i+1 >= len(dwarfp) || s.Sect != dwarfp[i+1].Sect {
			s1 := compressSyms(ctxt, dwarfp[start:i+1])
			if s1 == nil {
				// Compression didn't help.
				newDwarfp = append(newDwarfp, dwarfp[start:i+1]...)
				Segdwarf.Sections = append(Segdwarf.Sections, s.Sect)
			} else {
				compressedSegName := ".zdebug_" + s.Sect.Name[len(".debug_"):]
				sect := addsection(ctxt.Arch, &Segdwarf, compressedSegName, 04)
				sect.Length = uint64(len(s1))
				newSym := ctxt.Syms.Lookup(compressedSegName, 0)
				newSym.P = s1
				newSym.Size = int64(len(s1))
				newSym.Sect = sect
				newDwarfp = append(newDwarfp, newSym)
			}
			start = i + 1
		}
	}
	dwarfp = newDwarfp

	// Re-compute the locations of the compressed DWARF symbols
	// and sections, since the layout of these within the file is
	// based on Section.Vaddr and Symbol.Value.
	pos := Segdwarf.Vaddr
	var prevSect *sym.Section
	for _, s := range dwarfp {
		s.Value = int64(pos)
		if s.Sect != prevSect {
			s.Sect.Vaddr = uint64(s.Value)
			prevSect = s.Sect
		}
		if s.Sub != nil {
			log.Fatalf("%s: unexpected sub-symbols", s)
		}
		pos += uint64(s.Size)
		if ctxt.HeadType == objabi.Hwindows {
			pos = uint64(Rnd(int64(pos), PEFILEALIGN))
		}

	}
	Segdwarf.Length = pos - Segdwarf.Vaddr
}

type compilationUnitByStartPC []*sym.CompilationUnit

func (v compilationUnitByStartPC) Len() int      { return len(v) }
func (v compilationUnitByStartPC) Swap(i, j int) { v[i], v[j] = v[j], v[i] }

func (v compilationUnitByStartPC) Less(i, j int) bool {
	switch {
	case len(v[i].Textp2) == 0 && len(v[j].Textp2) == 0:
		return v[i].Lib.Pkg < v[j].Lib.Pkg
	case len(v[i].Textp2) != 0 && len(v[j].Textp2) == 0:
		return true
	case len(v[i].Textp2) == 0 && len(v[j].Textp2) != 0:
		return false
	default:
		return v[i].PCs[0].Start < v[j].PCs[0].Start
	}
}
