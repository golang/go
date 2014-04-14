// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Loading of code and data fragments from package files into final image.

package main

import "os"

// load allocates segment images, populates them with data
// read from package files, and applies relocations to the data.
func (p *Prog) load() {
	// TODO(rsc): mmap the output file and store the data directly.
	// That will make writing the output file more efficient.
	for _, seg := range p.Segments {
		seg.Data = make([]byte, seg.FileSize)
	}
	for _, pkg := range p.Packages {
		p.loadPackage(pkg)
	}
}

// loadPackage loads and relocates data for all the
// symbols needed in the given package.
func (p *Prog) loadPackage(pkg *Package) {
	if pkg.File == "" {
		// This "package" contains internally generated symbols only.
		// All such symbols have a sym.Bytes field holding the actual data
		// (if any), plus relocations.
		for _, sym := range pkg.Syms {
			if sym.Bytes == nil {
				continue
			}
			seg := sym.Section.Segment
			off := sym.Addr - seg.VirtAddr
			data := seg.Data[off : off+Addr(sym.Size)]
			copy(data, sym.Bytes)
			p.relocateSym(sym, data)
		}
		return
	}

	// Package stored in file.
	f, err := os.Open(pkg.File)
	if err != nil {
		p.errorf("%v", err)
		return
	}
	defer f.Close()

	// TODO(rsc): Mmap file into memory.

	for _, sym := range pkg.Syms {
		if sym.Data.Size == 0 {
			continue
		}
		// TODO(rsc): If not using mmap, at least coalesce nearby reads.
		if sym.Section == nil {
			p.errorf("internal error: missing section for %s", sym.Name)
		}
		seg := sym.Section.Segment
		off := sym.Addr - seg.VirtAddr
		if off >= Addr(len(seg.Data)) || off+Addr(sym.Data.Size) > Addr(len(seg.Data)) {
			p.errorf("internal error: allocated space for %s too small: %d bytes for %d+%d (%d)", sym, len(seg.Data), off, sym.Data.Size, sym.Size)
		}
		data := seg.Data[off : off+Addr(sym.Data.Size)]
		_, err := f.ReadAt(data, sym.Data.Offset)
		if err != nil {
			p.errorf("reading %v: %v", sym.SymID, err)
		}
		p.relocateSym(sym, data)
	}
}

// TODO(rsc): Define full enumeration for relocation types.
const (
	R_ADDR  = 1
	R_SIZE  = 2
	R_PCREL = 5
)

// relocateSym applies relocations to sym's data.
func (p *Prog) relocateSym(sym *Sym, data []byte) {
	for i := range sym.Reloc {
		r := &sym.Reloc[i]
		targ := p.Syms[r.Sym]
		if targ == nil {
			p.errorf("%v: reference to undefined symbol %v", sym, r.Sym)
			continue
		}
		val := targ.Addr + Addr(r.Add)
		switch r.Type {
		default:
			p.errorf("%v: unknown relocation type %d", sym, r.Type)
		case R_ADDR:
			// ok
		case R_PCREL:
			val -= sym.Addr + Addr(r.Offset+r.Size)
		}
		frag := data[r.Offset : r.Offset+r.Size]
		switch r.Size {
		default:
			p.errorf("%v: unknown relocation size %d", sym, r.Size)
		case 4:
			// TODO(rsc): Check for overflow?
			p.byteorder.PutUint32(frag, uint32(val))
		case 8:
			p.byteorder.PutUint64(frag, uint64(val))
		}
	}
}
