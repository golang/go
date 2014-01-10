// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Loading of code and data fragments from package files into final image.

package main

import (
	"encoding/binary"
	"os"
)

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
		seg := sym.Section.Segment
		off := sym.Addr - seg.VirtAddr
		data := seg.Data[off : off+Addr(sym.Data.Size)]
		_, err := f.ReadAt(data, sym.Data.Offset)
		if err != nil {
			p.errorf("reading %v: %v", sym.SymID, err)
		}
		p.relocateSym(sym, data)
	}
}

// TODO(rsc): These are the relocation types and should be
// loaded from debug/goobj. They are not in debug/goobj
// because they are different for each architecture.
// The symbol file format needs to be revised to use an
// architecture-independent set of numbers, and then
// those should be fetched from debug/goobj instead of
// defined here. These are the amd64 numbers.
const (
	D_ADDR  = 120
	D_SIZE  = 246
	D_PCREL = 247
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
		case D_ADDR:
			// ok
		case D_PCREL:
			val -= sym.Addr + Addr(r.Offset+r.Size)
		}
		frag := data[r.Offset : r.Offset+r.Size]
		switch r.Size {
		default:
			p.errorf("%v: unknown relocation size %d", sym, r.Size)
		case 4:
			// TODO(rsc): Check for overflow?
			// TODO(rsc): Handle big-endian systems.
			binary.LittleEndian.PutUint32(frag, uint32(val))
		case 8:
			binary.LittleEndian.PutUint64(frag, uint64(val))
		}
	}
}
