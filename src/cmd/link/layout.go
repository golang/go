// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Executable image layout - address assignment.

package main

import (
	"debug/goobj"
)

// A layoutSection describes a single section to add to the
// final executable. Go binaries only have a fixed set of possible
// sections, and the symbol kind determines the section.
type layoutSection struct {
	Segment string
	Section string
	Kind    goobj.SymKind
	Index   int
}

// layout defines the layout of the generated Go executable.
// The order of entries here is the order in the executable.
// Entries with the same Segment name must be contiguous.
var layout = []layoutSection{
	{Segment: "text", Section: "text", Kind: goobj.STEXT},
	{Segment: "rodata", Section: "rodata", Kind: goobj.SRODATA},
	{Segment: "rodata", Section: "functab", Kind: goobj.SPCLNTAB},
	{Segment: "rodata", Section: "typelink", Kind: goobj.STYPELINK},
	{Segment: "data", Section: "noptrdata", Kind: goobj.SNOPTRDATA},
	{Segment: "data", Section: "data", Kind: goobj.SDATA},
	{Segment: "data", Section: "bss", Kind: goobj.SBSS},
	{Segment: "data", Section: "noptrbss", Kind: goobj.SNOPTRBSS},

	// Later:
	//	{"rodata", "type", goobj.STYPE},
	//	{"rodata", "string", goobj.SSTRING},
	//	{"rodata", "gostring", goobj.SGOSTRING},
	//	{"rodata", "gofunc", goobj.SGOFUNC},
}

// layoutByKind maps from SymKind to an entry in layout.
var layoutByKind []*layoutSection

func init() {
	// Build index from symbol type to layout entry.
	max := 0
	for _, sect := range layout {
		if max <= int(sect.Kind) {
			max = int(sect.Kind) + 1
		}
	}
	layoutByKind = make([]*layoutSection, max)
	for i := range layout {
		sect := &layout[i]
		layoutByKind[sect.Kind] = sect
		sect.Index = i
	}
}

// layout arranges symbols into sections and sections into segments,
// and then it assigns addresses to segments, sections, and symbols.
func (p *Prog) layout() {
	sections := make([]*Section, len(layout))

	// Assign symbols to sections using index, creating sections as needed.
	// Could keep sections separated by type during input instead.
	for _, sym := range p.SymOrder {
		kind := sym.Kind
		if kind < 0 || int(kind) >= len(layoutByKind) || layoutByKind[kind] == nil {
			p.errorf("%s: unexpected symbol kind %v", sym.SymID, kind)
			continue
		}
		lsect := layoutByKind[kind]
		sect := sections[lsect.Index]
		if sect == nil {
			sect = &Section{
				Name:  lsect.Section,
				Align: 1,
			}
			sections[lsect.Index] = sect
		}
		if sym.Data.Size > 0 || len(sym.Bytes) > 0 {
			sect.InFile = true
		}
		sym.Section = sect
		sect.Syms = append(sect.Syms, sym)

		// TODO(rsc): Incorporate alignment information.
		// First that information needs to be added to the object files.
		//
		// if sect.Align < Addr(sym.Align) {
		//	sect.Align = Addr(sym.Align)
		// }
	}

	// Assign sections to segments, creating segments as needed.
	var seg *Segment
	for i, sect := range sections {
		if sect == nil {
			continue
		}
		segName := layout[i].Segment

		// Special case: Mach-O does not support "rodata" segment,
		// so store read-only data in text segment.
		if p.GOOS == "darwin" && segName == "rodata" {
			segName = "text"
		}

		if seg == nil || seg.Name != segName {
			seg = &Segment{
				Name: segName,
			}
			p.Segments = append(p.Segments, seg)
		}
		sect.Segment = seg
		seg.Sections = append(seg.Sections, sect)
	}

	// Assign addresses.

	// TODO(rsc): This choice needs to be informed by both
	// the formatter and the target architecture.
	// And maybe eventually a command line flag (sigh).
	const segAlign = 4096

	// TODO(rsc): Use a larger amount on most systems, which will let the
	// compiler eliminate more nil checks.
	if p.UnmappedSize == 0 {
		p.UnmappedSize = segAlign
	}

	// TODO(rsc): addr := Addr(0) when generating a shared library or PIE.
	addr := p.UnmappedSize

	// Account for initial file header.
	hdrVirt, hdrFile := p.formatter.headerSize(p)
	addr += hdrVirt

	// Assign addresses to segments, sections, symbols.
	// Assign sizes to segments, sections.
	startVirt := addr
	startFile := hdrFile
	for _, seg := range p.Segments {
		addr = round(addr, segAlign)
		seg.VirtAddr = addr
		seg.FileOffset = startFile + seg.VirtAddr - startVirt
		for _, sect := range seg.Sections {
			addr = round(addr, sect.Align)
			sect.VirtAddr = addr
			for _, sym := range sect.Syms {
				// TODO(rsc): Respect alignment once we have that information.
				sym.Addr = addr
				addr += Addr(sym.Size)
			}
			sect.Size = addr - sect.VirtAddr
			if sect.InFile {
				seg.FileSize = addr - seg.VirtAddr
			}
		}
		seg.VirtSize = addr - seg.VirtAddr
	}

	// Define symbols for section names.
	var progEnd Addr
	for i, sect := range sections {
		name := layout[i].Section
		var start, end Addr
		if sect != nil {
			start = sect.VirtAddr
			end = sect.VirtAddr + sect.Size
		}
		p.defineConst("runtime."+name, start)
		p.defineConst("runtime.e"+name, end)
		progEnd = end
	}
	p.defineConst("runtime.end", progEnd)
}
