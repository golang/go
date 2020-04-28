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
	"cmd/internal/objabi"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"log"
)

func isDwarf64(ctxt *Link) bool {
	return ctxt.HeadType == objabi.Haix
}

// dwarfSecInfo2 is a replica of the dwarfSecInfo struct but with
// *sym.Symbol content instead of loader.Sym content.
type dwarfSecInfo2 struct {
	syms []*sym.Symbol
}

func (dsi *dwarfSecInfo2) secSym() *sym.Symbol {
	if len(dsi.syms) == 0 {
		return nil
	}
	return dsi.syms[0]
}

func (dsi *dwarfSecInfo2) subSyms() []*sym.Symbol {
	if len(dsi.syms) == 0 {
		return []*sym.Symbol{}
	}
	return dsi.syms[1:]
}

var dwarfp []dwarfSecInfo2

/*
 *  Elf.
 */
func dwarfaddshstrings(ctxt *Link, shstrtab *loader.SymbolBuilder) {
	if *FlagW { // disable dwarf
		return
	}

	secs := []string{"abbrev", "frame", "info", "loc", "line", "pubnames", "pubtypes", "gdb_scripts", "ranges"}
	for _, sec := range secs {
		shstrtab.Addstring(".debug_" + sec)
		if ctxt.IsExternal() {
			shstrtab.Addstring(elfRelType + ".debug_" + sec)
		} else {
			shstrtab.Addstring(".zdebug_" + sec)
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
	putelfsectionsym(ctxt, ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	s = ctxt.Syms.Lookup(".debug_abbrev", 0)
	putelfsectionsym(ctxt, ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	s = ctxt.Syms.Lookup(".debug_line", 0)
	putelfsectionsym(ctxt, ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	s = ctxt.Syms.Lookup(".debug_frame", 0)
	putelfsectionsym(ctxt, ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	s = ctxt.Syms.Lookup(".debug_loc", 0)
	if s.Sect != nil {
		putelfsectionsym(ctxt, ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	}
	s = ctxt.Syms.Lookup(".debug_ranges", 0)
	if s.Sect != nil {
		putelfsectionsym(ctxt, ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	}
}

// dwarfcompress compresses the DWARF sections. Relocations are applied
// on the fly. After this, dwarfp will contain a different (new) set of
// symbols, and sections may have been replaced.
func dwarfcompress(ctxt *Link) {
	// compressedSect is a helper type for parallelizing compression.
	type compressedSect struct {
		index      int
		compressed []byte
		syms       []loader.Sym
	}

	supported := ctxt.IsELF || ctxt.IsWindows() || ctxt.IsDarwin()
	if !ctxt.compressDWARF || !supported || ctxt.IsExternal() {
		return
	}

	var compressedCount int
	resChannel := make(chan compressedSect)
	for i := range dwarfp2 {
		go func(resIndex int, syms []loader.Sym) {
			resChannel <- compressedSect{resIndex, compressSyms(ctxt, syms), syms}
		}(compressedCount, dwarfp2[i].syms)
		compressedCount++
	}
	res := make([]compressedSect, compressedCount)
	for ; compressedCount > 0; compressedCount-- {
		r := <-resChannel
		res[r.index] = r
	}

	ldr := ctxt.loader
	var newDwarfp []dwarfSecInfo
	Segdwarf.Sections = Segdwarf.Sections[:0]
	for _, z := range res {
		s := z.syms[0]
		if z.compressed == nil {
			// Compression didn't help.
			ds := dwarfSecInfo{syms: z.syms}
			newDwarfp = append(newDwarfp, ds)
			Segdwarf.Sections = append(Segdwarf.Sections, ldr.SymSect(s))
		} else {
			compressedSegName := ".zdebug_" + ldr.SymSect(s).Name[len(".debug_"):]
			sect := addsection(ctxt.loader, ctxt.Arch, &Segdwarf, compressedSegName, 04)
			sect.Align = 1
			sect.Length = uint64(len(z.compressed))
			newSym := ldr.CreateSymForUpdate(compressedSegName, 0)
			newSym.SetReachable(true)
			newSym.SetData(z.compressed)
			newSym.SetSize(int64(len(z.compressed)))
			ldr.SetSymSect(newSym.Sym(), sect)
			ds := dwarfSecInfo{syms: []loader.Sym{newSym.Sym()}}
			newDwarfp = append(newDwarfp, ds)

			// compressed symbols are no longer needed.
			for _, s := range z.syms {
				ldr.SetAttrReachable(s, false)
				ldr.FreeSym(s)
			}
		}
	}
	dwarfp2 = newDwarfp

	// Re-compute the locations of the compressed DWARF symbols
	// and sections, since the layout of these within the file is
	// based on Section.Vaddr and Symbol.Value.
	pos := Segdwarf.Vaddr
	var prevSect *sym.Section
	for _, si := range dwarfp2 {
		for _, s := range si.syms {
			ldr.SetSymValue(s, int64(pos))
			sect := ldr.SymSect(s)
			if sect != prevSect {
				sect.Vaddr = uint64(pos)
				prevSect = sect
			}
			if ldr.SubSym(s) != 0 {
				log.Fatalf("%s: unexpected sub-symbols", ldr.SymName(s))
			}
			pos += uint64(ldr.SymSize(s))
			if ctxt.IsWindows() {
				pos = uint64(Rnd(int64(pos), PEFILEALIGN))
			}
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
