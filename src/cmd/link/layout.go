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
	{Segment: "data", Section: "data", Kind: goobj.SDATA},

	// Later:
	//	{"rodata", "type", goobj.STYPE},
	//	{"rodata", "string", goobj.SSTRING},
	//	{"rodata", "gostring", goobj.SGOSTRING},
	//	{"rodata", "gofunc", goobj.SGOFUNC},
	//	{"rodata", "rodata", goobj.SRODATA},
	//	{"rodata", "functab", goobj.SFUNCTAB},
	//	{"rodata", "typelink", goobj.STYPELINK},
	//	{"rodata", "symtab", goobj.SSYMTAB},
	//	{"rodata", "pclntab", goobj.SPCLNTAB},
	//	{"data", "noptrdata", goobj.SNOPTRDATA},
	//	{"data", "bss", goobj.SBSS},
	//	{"data", "noptrbss", goobj.SNOPTRBSS},
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
	for i, sect := range layout {
		layoutByKind[sect.Kind] = &layout[i]
		sect.Index = i
	}
}

// layout arranges symbols into sections and sections into segments,
// and then it assigns addresses to segments, sections, and symbols.
func (p *Prog) layout() {
	sections := make([]*Section, len(layout))

	// Assign symbols to sections using index, creating sections as needed.
	// Could keep sections separated by type during input instead.
	for _, sym := range p.Syms {
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
		if sym.Data.Size > 0 {
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
		if seg == nil || seg.Name != layout[i].Segment {
			seg = &Segment{
				Name: layout[i].Segment,
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
		seg.VirtSize = addr
	}
}
