// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/gcprog"
	"cmd/internal/objabi"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"sync"
)

// Temporary dumping around for sym.Symbol version of helper
// functions in dodata(), still being used for some archs/oses.
// FIXME: get rid of this file when dodata() is completely
// converted.

func (ctxt *Link) dodata() {
	// Give zeros sized symbols space if necessary.
	fixZeroSizedSymbols(ctxt)

	// Collect data symbols by type into data.
	state := dodataState{ctxt: ctxt}
	for _, s := range ctxt.Syms.Allsym {
		if !s.Attr.Reachable() || s.Attr.Special() || s.Attr.SubSymbol() {
			continue
		}
		if s.Type <= sym.STEXT || s.Type >= sym.SXREF {
			continue
		}
		state.data[s.Type] = append(state.data[s.Type], s)
	}

	// Now that we have the data symbols, but before we start
	// to assign addresses, record all the necessary
	// dynamic relocations. These will grow the relocation
	// symbol, which is itself data.
	//
	// On darwin, we need the symbol table numbers for dynreloc.
	if ctxt.HeadType == objabi.Hdarwin {
		panic("not supported")
		//machosymorder(ctxt)
	}
	state.dynreloc(ctxt)

	// Move any RO data with relocations to a separate section.
	state.makeRelroForSharedLib(ctxt)

	// Temporary for debugging.
	symToIdx := make(map[*sym.Symbol]loader.Sym)
	for s := loader.Sym(1); s < loader.Sym(ctxt.loader.NSym()); s++ {
		sp := ctxt.loader.Syms[s]
		if sp != nil {
			symToIdx[sp] = s
		}
	}

	// Sort symbols.
	var wg sync.WaitGroup
	for symn := range state.data {
		symn := sym.SymKind(symn)
		wg.Add(1)
		go func() {
			state.data[symn], state.dataMaxAlign[symn] = dodataSect(ctxt, symn, state.data[symn], symToIdx)
			wg.Done()
		}()
	}
	wg.Wait()

	if ctxt.HeadType == objabi.Haix && ctxt.LinkMode == LinkExternal {
		// These symbols must have the same alignment as their section.
		// Otherwize, ld might change the layout of Go sections.
		ctxt.Syms.ROLookup("runtime.data", 0).Align = state.dataMaxAlign[sym.SDATA]
		ctxt.Syms.ROLookup("runtime.bss", 0).Align = state.dataMaxAlign[sym.SBSS]
	}

	// Create *sym.Section objects and assign symbols to sections for
	// data/rodata (and related) symbols.
	state.allocateDataSections(ctxt)

	// Create *sym.Section objects and assign symbols to sections for
	// DWARF symbols.
	state.allocateDwarfSections(ctxt)

	/* number the sections */
	n := int16(1)

	for _, sect := range Segtext.Sections {
		sect.Extnum = n
		n++
	}
	for _, sect := range Segrodata.Sections {
		sect.Extnum = n
		n++
	}
	for _, sect := range Segrelrodata.Sections {
		sect.Extnum = n
		n++
	}
	for _, sect := range Segdata.Sections {
		sect.Extnum = n
		n++
	}
	for _, sect := range Segdwarf.Sections {
		sect.Extnum = n
		n++
	}
}

// makeRelroForSharedLib creates a section of readonly data if necessary.
func (state *dodataState) makeRelroForSharedLib(target *Link) {
	if !target.UseRelro() {
		return
	}

	// "read only" data with relocations needs to go in its own section
	// when building a shared library. We do this by boosting objects of
	// type SXXX with relocations to type SXXXRELRO.
	for _, symnro := range sym.ReadOnly {
		symnrelro := sym.RelROMap[symnro]

		ro := []*sym.Symbol{}
		relro := state.data[symnrelro]

		for _, s := range state.data[symnro] {
			isRelro := len(s.R) > 0
			switch s.Type {
			case sym.STYPE, sym.STYPERELRO, sym.SGOFUNCRELRO:
				// Symbols are not sorted yet, so it is possible
				// that an Outer symbol has been changed to a
				// relro Type before it reaches here.
				isRelro = true
			case sym.SFUNCTAB:
				if target.IsAIX() && s.Name == "runtime.etypes" {
					// runtime.etypes must be at the end of
					// the relro datas.
					isRelro = true
				}
			}
			if isRelro {
				s.Type = symnrelro
				if s.Outer != nil {
					s.Outer.Type = s.Type
				}
				relro = append(relro, s)
			} else {
				ro = append(ro, s)
			}
		}

		// Check that we haven't made two symbols with the same .Outer into
		// different types (because references two symbols with non-nil Outer
		// become references to the outer symbol + offset it's vital that the
		// symbol and the outer end up in the same section).
		for _, s := range relro {
			if s.Outer != nil && s.Outer.Type != s.Type {
				Errorf(s, "inconsistent types for symbol and its Outer %s (%v != %v)",
					s.Outer.Name, s.Type, s.Outer.Type)
			}
		}

		state.data[symnro] = ro
		state.data[symnrelro] = relro
	}
}

func dynrelocsym(ctxt *Link, s *sym.Symbol) {
	target := &ctxt.Target
	ldr := ctxt.loader
	syms := &ctxt.ArchSyms
	for ri := range s.R {
		r := &s.R[ri]
		if ctxt.BuildMode == BuildModePIE && ctxt.LinkMode == LinkInternal {
			// It's expected that some relocations will be done
			// later by relocsym (R_TLS_LE, R_ADDROFF), so
			// don't worry if Adddynrel returns false.
			thearch.Adddynrel(target, ldr, syms, s, r)
			continue
		}

		if r.Sym != nil && r.Sym.Type == sym.SDYNIMPORT || r.Type >= objabi.ElfRelocOffset {
			if r.Sym != nil && !r.Sym.Attr.Reachable() {
				Errorf(s, "dynamic relocation to unreachable symbol %s", r.Sym.Name)
			}
			if !thearch.Adddynrel(target, ldr, syms, s, r) {
				Errorf(s, "unsupported dynamic relocation for symbol %s (type=%d (%s) stype=%d (%s))", r.Sym.Name, r.Type, sym.RelocName(ctxt.Arch, r.Type), r.Sym.Type, r.Sym.Type)
			}
		}
	}
}

func (state *dodataState) dynreloc(ctxt *Link) {
	if ctxt.HeadType == objabi.Hwindows {
		return
	}
	// -d suppresses dynamic loader format, so we may as well not
	// compute these sections or mark their symbols as reachable.
	if *FlagD {
		return
	}

	for _, s := range ctxt.Textp {
		dynrelocsym(ctxt, s)
	}
	for _, syms := range state.data {
		for _, s := range syms {
			dynrelocsym(ctxt, s)
		}
	}
	if ctxt.IsELF {
		elfdynhash(ctxt)
	}
}

func Addstring(s *sym.Symbol, str string) int64 {
	if s.Type == 0 {
		s.Type = sym.SNOPTRDATA
	}
	s.Attr |= sym.AttrReachable
	r := s.Size
	if s.Name == ".shstrtab" {
		elfsetstring(s, str, int(r))
	}
	s.P = append(s.P, str...)
	s.P = append(s.P, 0)
	s.Size = int64(len(s.P))
	return r
}

// symalign returns the required alignment for the given symbol s.
func symalign(s *sym.Symbol) int32 {
	min := int32(thearch.Minalign)
	if s.Align >= min {
		return s.Align
	} else if s.Align != 0 {
		return min
	}
	if strings.HasPrefix(s.Name, "go.string.") || strings.HasPrefix(s.Name, "type..namedata.") {
		// String data is just bytes.
		// If we align it, we waste a lot of space to padding.
		return min
	}
	align := int32(thearch.Maxalign)
	for int64(align) > s.Size && align > min {
		align >>= 1
	}
	s.Align = align
	return align
}

func aligndatsize(datsize int64, s *sym.Symbol) int64 {
	return Rnd(datsize, int64(symalign(s)))
}

type GCProg struct {
	ctxt *Link
	sym  *sym.Symbol
	w    gcprog.Writer
}

func (p *GCProg) Init(ctxt *Link, name string) {
	p.ctxt = ctxt
	p.sym = ctxt.Syms.Lookup(name, 0)
	p.w.Init(p.writeByte(ctxt))
	if debugGCProg {
		fmt.Fprintf(os.Stderr, "ld: start GCProg %s\n", name)
		p.w.Debug(os.Stderr)
	}
}

func (p *GCProg) writeByte(ctxt *Link) func(x byte) {
	return func(x byte) {
		p.sym.AddUint8(x)
	}
}

func (p *GCProg) End(size int64) {
	p.w.ZeroUntil(size / int64(p.ctxt.Arch.PtrSize))
	p.w.End()
	if debugGCProg {
		fmt.Fprintf(os.Stderr, "ld: end GCProg\n")
	}
}

func (p *GCProg) AddSym(s *sym.Symbol) {
	typ := s.Gotype
	// Things without pointers should be in sym.SNOPTRDATA or sym.SNOPTRBSS;
	// everything we see should have pointers and should therefore have a type.
	if typ == nil {
		switch s.Name {
		case "runtime.data", "runtime.edata", "runtime.bss", "runtime.ebss":
			// Ignore special symbols that are sometimes laid out
			// as real symbols. See comment about dyld on darwin in
			// the address function.
			return
		}
		Errorf(s, "missing Go type information for global symbol: size %d", s.Size)
		return
	}

	ptrsize := int64(p.ctxt.Arch.PtrSize)
	nptr := decodetypePtrdata(p.ctxt.Arch, typ.P) / ptrsize

	if debugGCProg {
		fmt.Fprintf(os.Stderr, "gcprog sym: %s at %d (ptr=%d+%d)\n", s.Name, s.Value, s.Value/ptrsize, nptr)
	}

	if decodetypeUsegcprog(p.ctxt.Arch, typ.P) == 0 {
		// Copy pointers from mask into program.
		mask := decodetypeGcmask(p.ctxt, typ)
		for i := int64(0); i < nptr; i++ {
			if (mask[i/8]>>uint(i%8))&1 != 0 {
				p.w.Ptr(s.Value/ptrsize + i)
			}
		}
		return
	}

	// Copy program.
	prog := decodetypeGcprog(p.ctxt, typ)
	p.w.ZeroUntil(s.Value / ptrsize)
	p.w.Append(prog[4:], nptr)
}

// dataSortKey is used to sort a slice of data symbol *sym.Symbol pointers.
// The sort keys are kept inline to improve cache behavior while sorting.
type dataSortKey struct {
	size   int64
	name   string
	sym    *sym.Symbol
	symIdx loader.Sym
}

type bySizeAndName []dataSortKey

func (d bySizeAndName) Len() int      { return len(d) }
func (d bySizeAndName) Swap(i, j int) { d[i], d[j] = d[j], d[i] }
func (d bySizeAndName) Less(i, j int) bool {
	s1, s2 := d[i], d[j]
	if s1.size != s2.size {
		return s1.size < s2.size
	}
	if s1.name != s2.name {
		return s1.name < s2.name
	}
	return s1.symIdx < s2.symIdx
}

// fixZeroSizedSymbols gives a few special symbols with zero size some space.
func fixZeroSizedSymbols(ctxt *Link) {
	// The values in moduledata are filled out by relocations
	// pointing to the addresses of these special symbols.
	// Typically these symbols have no size and are not laid
	// out with their matching section.
	//
	// However on darwin, dyld will find the special symbol
	// in the first loaded module, even though it is local.
	//
	// (An hypothesis, formed without looking in the dyld sources:
	// these special symbols have no size, so their address
	// matches a real symbol. The dynamic linker assumes we
	// want the normal symbol with the same address and finds
	// it in the other module.)
	//
	// To work around this we lay out the symbls whose
	// addresses are vital for multi-module programs to work
	// as normal symbols, and give them a little size.
	//
	// On AIX, as all DATA sections are merged together, ld might not put
	// these symbols at the beginning of their respective section if there
	// aren't real symbols, their alignment might not match the
	// first symbol alignment. Therefore, there are explicitly put at the
	// beginning of their section with the same alignment.
	if !(ctxt.DynlinkingGo() && ctxt.HeadType == objabi.Hdarwin) && !(ctxt.HeadType == objabi.Haix && ctxt.LinkMode == LinkExternal) {
		return
	}

	bss := ctxt.Syms.Lookup("runtime.bss", 0)
	bss.Size = 8
	bss.Attr.Set(sym.AttrSpecial, false)

	ctxt.Syms.Lookup("runtime.ebss", 0).Attr.Set(sym.AttrSpecial, false)

	data := ctxt.Syms.Lookup("runtime.data", 0)
	data.Size = 8
	data.Attr.Set(sym.AttrSpecial, false)

	edata := ctxt.Syms.Lookup("runtime.edata", 0)
	edata.Attr.Set(sym.AttrSpecial, false)
	if ctxt.HeadType == objabi.Haix {
		// XCOFFTOC symbols are part of .data section.
		edata.Type = sym.SXCOFFTOC
	}

	types := ctxt.Syms.Lookup("runtime.types", 0)
	types.Type = sym.STYPE
	types.Size = 8
	types.Attr.Set(sym.AttrSpecial, false)

	etypes := ctxt.Syms.Lookup("runtime.etypes", 0)
	etypes.Type = sym.SFUNCTAB
	etypes.Attr.Set(sym.AttrSpecial, false)

	if ctxt.HeadType == objabi.Haix {
		rodata := ctxt.Syms.Lookup("runtime.rodata", 0)
		rodata.Type = sym.SSTRING
		rodata.Size = 8
		rodata.Attr.Set(sym.AttrSpecial, false)

		ctxt.Syms.Lookup("runtime.erodata", 0).Attr.Set(sym.AttrSpecial, false)
	}
}

// allocateDataSectionForSym creates a new sym.Section into which a a
// single symbol will be placed. Here "seg" is the segment into which
// the section will go, "s" is the symbol to be placed into the new
// section, and "rwx" contains permissions for the section.
func (state *dodataState) allocateDataSectionForSym(seg *sym.Segment, s *sym.Symbol, rwx int) *sym.Section {
	sect := addsection(state.ctxt.loader, state.ctxt.Arch, seg, s.Name, rwx)
	sect.Align = symalign(s)
	state.datsize = Rnd(state.datsize, int64(sect.Align))
	sect.Vaddr = uint64(state.datsize)
	return sect
}

// assignDsymsToSection assigns a collection of data symbols to a
// newly created section. "sect" is the section into which to place
// the symbols, "syms" holds the list of symbols to assign,
// "forceType" (if non-zero) contains a new sym type to apply to each
// sym during the assignment, and "aligner" is a hook to call to
// handle alignment during the assignment process.
func (state *dodataState) assignDsymsToSection(sect *sym.Section, syms []*sym.Symbol, forceType sym.SymKind, aligner func(datsize int64, s *sym.Symbol) int64) {
	for _, s := range syms {
		state.datsize = aligner(state.datsize, s)
		s.Sect = sect
		if forceType != sym.Sxxx {
			s.Type = forceType
		}
		s.Value = int64(uint64(state.datsize) - sect.Vaddr)
		state.datsize += s.Size
	}
	sect.Length = uint64(state.datsize) - sect.Vaddr
}

func (state *dodataState) assignToSection(sect *sym.Section, symn sym.SymKind, forceType sym.SymKind) {
	state.assignDsymsToSection(sect, state.data[symn], forceType, aligndatsize)
	state.checkdatsize(symn)
}

// allocateSingleSymSections walks through the bucketed data symbols
// with type 'symn', creates a new section for each sym, and assigns
// the sym to a newly created section. Section name is set from the
// symbol name. "Seg" is the segment into which to place the new
// section, "forceType" is the new sym.SymKind to assign to the symbol
// within the section, and "rwx" holds section permissions.
func (state *dodataState) allocateSingleSymSections(seg *sym.Segment, symn sym.SymKind, forceType sym.SymKind, rwx int) {
	for _, s := range state.data[symn] {
		sect := state.allocateDataSectionForSym(seg, s, rwx)
		s.Sect = sect
		s.Type = forceType
		s.Value = int64(uint64(state.datsize) - sect.Vaddr)
		state.datsize += s.Size
		sect.Length = uint64(state.datsize) - sect.Vaddr
	}
	state.checkdatsize(symn)
}

// allocateNamedSectionAndAssignSyms creates a new section with the
// specified name, then walks through the bucketed data symbols with
// type 'symn' and assigns each of them to this new section. "Seg" is
// the segment into which to place the new section, "secName" is the
// name to give to the new section, "forceType" (if non-zero) contains
// a new sym type to apply to each sym during the assignment, and
// "rwx" holds section permissions.
func (state *dodataState) allocateNamedSectionAndAssignSyms(seg *sym.Segment, secName string, symn sym.SymKind, forceType sym.SymKind, rwx int) *sym.Section {

	sect := state.allocateNamedDataSection(seg, secName, []sym.SymKind{symn}, rwx)
	state.assignDsymsToSection(sect, state.data[symn], forceType, aligndatsize)
	return sect
}

// allocateDataSections allocates sym.Section objects for data/rodata
// (and related) symbols, and then assigns symbols to those sections.
func (state *dodataState) allocateDataSections(ctxt *Link) {
	// Allocate sections.
	// Data is processed before segtext, because we need
	// to see all symbols in the .data and .bss sections in order
	// to generate garbage collection information.

	// Writable data sections that do not need any specialized handling.
	writable := []sym.SymKind{
		sym.SBUILDINFO,
		sym.SELFSECT,
		sym.SMACHO,
		sym.SMACHOGOT,
		sym.SWINDOWS,
	}
	for _, symn := range writable {
		state.allocateSingleSymSections(&Segdata, symn, sym.SDATA, 06)
	}

	// .got (and .toc on ppc64)
	if len(state.data[sym.SELFGOT]) > 0 {
		sect := state.allocateNamedSectionAndAssignSyms(&Segdata, ".got", sym.SELFGOT, sym.SDATA, 06)
		if ctxt.IsPPC64() {
			for _, s := range state.data[sym.SELFGOT] {
				// Resolve .TOC. symbol for this object file (ppc64)
				toc := ctxt.Syms.ROLookup(".TOC.", int(s.Version))
				if toc != nil {
					toc.Sect = sect
					toc.Outer = s
					toc.Sub = s.Sub
					s.Sub = toc

					toc.Value = 0x8000
				}
			}
		}
	}

	/* pointer-free data */
	sect := state.allocateNamedSectionAndAssignSyms(&Segdata, ".noptrdata", sym.SNOPTRDATA, sym.SDATA, 06)
	ctxt.Syms.Lookup("runtime.noptrdata", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.enoptrdata", 0).Sect = sect

	hasinitarr := ctxt.linkShared

	/* shared library initializer */
	switch ctxt.BuildMode {
	case BuildModeCArchive, BuildModeCShared, BuildModeShared, BuildModePlugin:
		hasinitarr = true
	}

	if ctxt.HeadType == objabi.Haix {
		if len(state.data[sym.SINITARR]) > 0 {
			Errorf(nil, "XCOFF format doesn't allow .init_array section")
		}
	}

	if hasinitarr && len(state.data[sym.SINITARR]) > 0 {
		state.allocateNamedSectionAndAssignSyms(&Segdata, ".init_array", sym.SINITARR, sym.Sxxx, 06)
	}

	/* data */
	sect = state.allocateNamedSectionAndAssignSyms(&Segdata, ".data", sym.SDATA, sym.SDATA, 06)
	ctxt.Syms.Lookup("runtime.data", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.edata", 0).Sect = sect
	dataGcEnd := state.datsize - int64(sect.Vaddr)

	// On AIX, TOC entries must be the last of .data
	// These aren't part of gc as they won't change during the runtime.
	state.assignToSection(sect, sym.SXCOFFTOC, sym.SDATA)
	state.checkdatsize(sym.SDATA)
	sect.Length = uint64(state.datsize) - sect.Vaddr

	/* bss */
	sect = state.allocateNamedSectionAndAssignSyms(&Segdata, ".bss", sym.SBSS, sym.Sxxx, 06)
	ctxt.Syms.Lookup("runtime.bss", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.ebss", 0).Sect = sect
	bssGcEnd := state.datsize - int64(sect.Vaddr)

	// Emit gcdata for bcc symbols now that symbol values have been assigned.
	gcsToEmit := []struct {
		symName string
		symKind sym.SymKind
		gcEnd   int64
	}{
		{"runtime.gcdata", sym.SDATA, dataGcEnd},
		{"runtime.gcbss", sym.SBSS, bssGcEnd},
	}
	for _, g := range gcsToEmit {
		var gc GCProg
		gc.Init(ctxt, g.symName)
		for _, s := range state.data[g.symKind] {
			gc.AddSym(s)
		}
		gc.End(g.gcEnd)
	}

	/* pointer-free bss */
	sect = state.allocateNamedSectionAndAssignSyms(&Segdata, ".noptrbss", sym.SNOPTRBSS, sym.Sxxx, 06)
	ctxt.Syms.Lookup("runtime.noptrbss", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.enoptrbss", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.end", 0).Sect = sect

	// Coverage instrumentation counters for libfuzzer.
	if len(state.data[sym.SLIBFUZZER_EXTRA_COUNTER]) > 0 {
		state.allocateNamedSectionAndAssignSyms(&Segdata, "__libfuzzer_extra_counters", sym.SLIBFUZZER_EXTRA_COUNTER, sym.Sxxx, 06)
	}

	if len(state.data[sym.STLSBSS]) > 0 {
		var sect *sym.Section
		// FIXME: not clear why it is sometimes necessary to suppress .tbss section creation.
		if (ctxt.IsELF || ctxt.HeadType == objabi.Haix) && (ctxt.LinkMode == LinkExternal || !*FlagD) {
			sect = addsection(ctxt.loader, ctxt.Arch, &Segdata, ".tbss", 06)
			sect.Align = int32(ctxt.Arch.PtrSize)
			// FIXME: why does this need to be set to zero?
			sect.Vaddr = 0
		}
		state.datsize = 0

		for _, s := range state.data[sym.STLSBSS] {
			state.datsize = aligndatsize(state.datsize, s)
			s.Sect = sect
			s.Value = state.datsize
			state.datsize += s.Size
		}
		state.checkdatsize(sym.STLSBSS)

		if sect != nil {
			sect.Length = uint64(state.datsize)
		}
	}

	/*
	 * We finished data, begin read-only data.
	 * Not all systems support a separate read-only non-executable data section.
	 * ELF and Windows PE systems do.
	 * OS X and Plan 9 do not.
	 * And if we're using external linking mode, the point is moot,
	 * since it's not our decision; that code expects the sections in
	 * segtext.
	 */
	var segro *sym.Segment
	if ctxt.IsELF && ctxt.LinkMode == LinkInternal {
		segro = &Segrodata
	} else if ctxt.HeadType == objabi.Hwindows {
		segro = &Segrodata
	} else {
		segro = &Segtext
	}

	state.datsize = 0

	/* read-only executable ELF, Mach-O sections */
	if len(state.data[sym.STEXT]) != 0 {
		Errorf(nil, "dodata found an sym.STEXT symbol: %s", state.data[sym.STEXT][0].Name)
	}
	state.allocateSingleSymSections(&Segtext, sym.SELFRXSECT, sym.SRODATA, 04)

	/* read-only data */
	sect = state.allocateNamedDataSection(segro, ".rodata", sym.ReadOnly, 04)
	ctxt.Syms.Lookup("runtime.rodata", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.erodata", 0).Sect = sect
	if !ctxt.UseRelro() {
		ctxt.Syms.Lookup("runtime.types", 0).Sect = sect
		ctxt.Syms.Lookup("runtime.etypes", 0).Sect = sect
	}
	for _, symn := range sym.ReadOnly {
		symnStartValue := state.datsize
		state.assignToSection(sect, symn, sym.SRODATA)
		if ctxt.HeadType == objabi.Haix {
			// Read-only symbols might be wrapped inside their outer
			// symbol.
			// XCOFF symbol table needs to know the size of
			// these outer symbols.
			xcoffUpdateOuterSize(ctxt, state.datsize-symnStartValue, symn)
		}
	}

	/* read-only ELF, Mach-O sections */
	state.allocateSingleSymSections(segro, sym.SELFROSECT, sym.SRODATA, 04)
	state.allocateSingleSymSections(segro, sym.SMACHOPLT, sym.SRODATA, 04)

	// There is some data that are conceptually read-only but are written to by
	// relocations. On GNU systems, we can arrange for the dynamic linker to
	// mprotect sections after relocations are applied by giving them write
	// permissions in the object file and calling them ".data.rel.ro.FOO". We
	// divide the .rodata section between actual .rodata and .data.rel.ro.rodata,
	// but for the other sections that this applies to, we just write a read-only
	// .FOO section or a read-write .data.rel.ro.FOO section depending on the
	// situation.
	// TODO(mwhudson): It would make sense to do this more widely, but it makes
	// the system linker segfault on darwin.
	const relroPerm = 06
	const fallbackPerm = 04
	relroSecPerm := fallbackPerm
	genrelrosecname := func(suffix string) string {
		return suffix
	}
	seg := segro

	if ctxt.UseRelro() {
		segrelro := &Segrelrodata
		if ctxt.LinkMode == LinkExternal && ctxt.HeadType != objabi.Haix {
			// Using a separate segment with an external
			// linker results in some programs moving
			// their data sections unexpectedly, which
			// corrupts the moduledata. So we use the
			// rodata segment and let the external linker
			// sort out a rel.ro segment.
			segrelro = segro
		} else {
			// Reset datsize for new segment.
			state.datsize = 0
		}

		genrelrosecname = func(suffix string) string {
			return ".data.rel.ro" + suffix
		}
		relroReadOnly := []sym.SymKind{}
		for _, symnro := range sym.ReadOnly {
			symn := sym.RelROMap[symnro]
			relroReadOnly = append(relroReadOnly, symn)
		}
		seg = segrelro
		relroSecPerm = relroPerm

		/* data only written by relocations */
		sect = state.allocateNamedDataSection(segrelro, genrelrosecname(""), relroReadOnly, relroSecPerm)

		ctxt.Syms.Lookup("runtime.types", 0).Sect = sect
		ctxt.Syms.Lookup("runtime.etypes", 0).Sect = sect

		for i, symnro := range sym.ReadOnly {
			if i == 0 && symnro == sym.STYPE && ctxt.HeadType != objabi.Haix {
				// Skip forward so that no type
				// reference uses a zero offset.
				// This is unlikely but possible in small
				// programs with no other read-only data.
				state.datsize++
			}

			symn := sym.RelROMap[symnro]
			symnStartValue := state.datsize

			for _, s := range state.data[symn] {
				if s.Outer != nil && s.Outer.Sect != nil && s.Outer.Sect != sect {
					Errorf(s, "s.Outer (%s) in different section from s, %s != %s", s.Outer.Name, s.Outer.Sect.Name, sect.Name)
				}
			}
			state.assignToSection(sect, symn, sym.SRODATA)
			if ctxt.HeadType == objabi.Haix {
				// Read-only symbols might be wrapped inside their outer
				// symbol.
				// XCOFF symbol table needs to know the size of
				// these outer symbols.
				xcoffUpdateOuterSize(ctxt, state.datsize-symnStartValue, symn)
			}
		}

		sect.Length = uint64(state.datsize) - sect.Vaddr
	}

	/* typelink */
	sect = state.allocateNamedDataSection(seg, genrelrosecname(".typelink"), []sym.SymKind{sym.STYPELINK}, relroSecPerm)
	typelink := ctxt.Syms.Lookup("runtime.typelink", 0)
	typelink.Sect = sect
	typelink.Type = sym.SRODATA
	state.datsize += typelink.Size
	state.checkdatsize(sym.STYPELINK)
	sect.Length = uint64(state.datsize) - sect.Vaddr

	/* itablink */
	sect = state.allocateNamedSectionAndAssignSyms(seg, genrelrosecname(".itablink"), sym.SITABLINK, sym.Sxxx, relroSecPerm)
	ctxt.Syms.Lookup("runtime.itablink", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.eitablink", 0).Sect = sect
	if ctxt.HeadType == objabi.Haix {
		// Store .itablink size because its symbols are wrapped
		// under an outer symbol: runtime.itablink.
		xcoffUpdateOuterSize(ctxt, int64(sect.Length), sym.SITABLINK)
	}

	/* gosymtab */
	sect = state.allocateNamedSectionAndAssignSyms(seg, genrelrosecname(".gosymtab"), sym.SSYMTAB, sym.SRODATA, relroSecPerm)
	ctxt.Syms.Lookup("runtime.symtab", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.esymtab", 0).Sect = sect

	/* gopclntab */
	sect = state.allocateNamedSectionAndAssignSyms(seg, genrelrosecname(".gopclntab"), sym.SPCLNTAB, sym.SRODATA, relroSecPerm)
	ctxt.Syms.Lookup("runtime.pclntab", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.epclntab", 0).Sect = sect

	// 6g uses 4-byte relocation offsets, so the entire segment must fit in 32 bits.
	if state.datsize != int64(uint32(state.datsize)) {
		Errorf(nil, "read-only data segment too large: %d", state.datsize)
	}

	for symn := sym.SELFRXSECT; symn < sym.SXREF; symn++ {
		ctxt.datap = append(ctxt.datap, state.data[symn]...)
	}
}

// allocateDwarfSections allocates sym.Section objects for DWARF
// symbols, and assigns symbols to sections.
func (state *dodataState) allocateDwarfSections(ctxt *Link) {

	alignOne := func(datsize int64, s *sym.Symbol) int64 { return datsize }

	for i := 0; i < len(dwarfp); i++ {
		// First the section symbol.
		s := dwarfp[i].secSym()
		sect := state.allocateNamedDataSection(&Segdwarf, s.Name, []sym.SymKind{}, 04)
		sect.Sym = s
		s.Sect = sect
		curType := s.Type
		s.Type = sym.SRODATA
		s.Value = int64(uint64(state.datsize) - sect.Vaddr)
		state.datsize += s.Size

		// Then any sub-symbols for the section symbol.
		subSyms := dwarfp[i].subSyms()
		state.assignDsymsToSection(sect, subSyms, sym.SRODATA, alignOne)

		for j := 0; j < len(subSyms); j++ {
			s := subSyms[j]
			if ctxt.HeadType == objabi.Haix && curType == sym.SDWARFLOC {
				// Update the size of .debug_loc for this symbol's
				// package.
				addDwsectCUSize(".debug_loc", s.File, uint64(s.Size))
			}
		}
		sect.Length = uint64(state.datsize) - sect.Vaddr
		state.checkdatsize(curType)
	}
}

func dodataSect(ctxt *Link, symn sym.SymKind, syms []*sym.Symbol, symToIdx map[*sym.Symbol]loader.Sym) (result []*sym.Symbol, maxAlign int32) {
	if ctxt.HeadType == objabi.Hdarwin {
		// Some symbols may no longer belong in syms
		// due to movement in machosymorder.
		newSyms := make([]*sym.Symbol, 0, len(syms))
		for _, s := range syms {
			if s.Type == symn {
				newSyms = append(newSyms, s)
			}
		}
		syms = newSyms
	}

	var head, tail *sym.Symbol
	symsSort := make([]dataSortKey, 0, len(syms))
	for _, s := range syms {
		if s.Attr.OnList() {
			log.Fatalf("symbol %s listed multiple times", s.Name)
		}
		s.Attr |= sym.AttrOnList
		switch {
		case s.Size < int64(len(s.P)):
			Errorf(s, "initialize bounds (%d < %d)", s.Size, len(s.P))
		case s.Size < 0:
			Errorf(s, "negative size (%d bytes)", s.Size)
		case s.Size > cutoff:
			Errorf(s, "symbol too large (%d bytes)", s.Size)
		}

		// If the usually-special section-marker symbols are being laid
		// out as regular symbols, put them either at the beginning or
		// end of their section.
		if (ctxt.DynlinkingGo() && ctxt.HeadType == objabi.Hdarwin) || (ctxt.HeadType == objabi.Haix && ctxt.LinkMode == LinkExternal) {
			switch s.Name {
			case "runtime.text", "runtime.bss", "runtime.data", "runtime.types", "runtime.rodata":
				head = s
				continue
			case "runtime.etext", "runtime.ebss", "runtime.edata", "runtime.etypes", "runtime.erodata":
				tail = s
				continue
			}
		}

		key := dataSortKey{
			size:   s.Size,
			name:   s.Name,
			sym:    s,
			symIdx: symToIdx[s],
		}

		switch s.Type {
		case sym.SELFGOT:
			// For ppc64, we want to interleave the .got and .toc sections
			// from input files. Both are type sym.SELFGOT, so in that case
			// we skip size comparison and fall through to the name
			// comparison (conveniently, .got sorts before .toc).
			key.size = 0
		}

		symsSort = append(symsSort, key)
	}

	sort.Sort(bySizeAndName(symsSort))

	off := 0
	if head != nil {
		syms[0] = head
		off++
	}
	for i, symSort := range symsSort {
		syms[i+off] = symSort.sym
		align := symalign(symSort.sym)
		if maxAlign < align {
			maxAlign = align
		}
	}
	if tail != nil {
		syms[len(syms)-1] = tail
	}

	if ctxt.IsELF && symn == sym.SELFROSECT {
		// Make .rela and .rela.plt contiguous, the ELF ABI requires this
		// and Solaris actually cares.
		reli, plti := -1, -1
		for i, s := range syms {
			switch s.Name {
			case ".rel.plt", ".rela.plt":
				plti = i
			case ".rel", ".rela":
				reli = i
			}
		}
		if reli >= 0 && plti >= 0 && plti != reli+1 {
			var first, second int
			if plti > reli {
				first, second = reli, plti
			} else {
				first, second = plti, reli
			}
			rel, plt := syms[reli], syms[plti]
			copy(syms[first+2:], syms[first+1:second])
			syms[first+0] = rel
			syms[first+1] = plt

			// Make sure alignment doesn't introduce a gap.
			// Setting the alignment explicitly prevents
			// symalign from basing it on the size and
			// getting it wrong.
			rel.Align = int32(ctxt.Arch.RegSize)
			plt.Align = int32(ctxt.Arch.RegSize)
		}
	}

	return syms, maxAlign
}
