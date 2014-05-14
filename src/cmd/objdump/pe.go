// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parsing of PE executables (Microsoft Windows).

package main

import (
	"debug/pe"
	"os"
	"sort"
)

func peSymbols(f *os.File) (syms []Sym, goarch string) {
	p, err := pe.NewFile(f)
	if err != nil {
		errorf("parsing %s: %v", f.Name(), err)
		return
	}

	// Build sorted list of addresses of all symbols.
	// We infer the size of a symbol by looking at where the next symbol begins.
	var addrs []uint64

	var imageBase uint64
	switch oh := p.OptionalHeader.(type) {
	case *pe.OptionalHeader32:
		imageBase = uint64(oh.ImageBase)
		goarch = "386"
	case *pe.OptionalHeader64:
		imageBase = oh.ImageBase
		goarch = "amd64"
	default:
		errorf("parsing %s: file format not recognized", f.Name())
		return
	}

	for _, s := range p.Symbols {
		const (
			N_UNDEF = 0  // An undefined (extern) symbol
			N_ABS   = -1 // An absolute symbol (e_value is a constant, not an address)
			N_DEBUG = -2 // A debugging symbol
		)
		sym := Sym{Name: s.Name, Addr: uint64(s.Value), Code: '?'}
		switch s.SectionNumber {
		case N_UNDEF:
			sym.Code = 'U'
		case N_ABS:
			sym.Code = 'C'
		case N_DEBUG:
			sym.Code = '?'
		default:
			if s.SectionNumber < 0 {
				errorf("parsing %s: invalid section number %d", f.Name(), s.SectionNumber)
				return
			}
			if len(p.Sections) < int(s.SectionNumber) {
				errorf("parsing %s: section number %d is large then max %d", f.Name(), s.SectionNumber, len(p.Sections))
				return
			}
			sect := p.Sections[s.SectionNumber-1]
			const (
				text  = 0x20
				data  = 0x40
				bss   = 0x80
				permX = 0x20000000
				permR = 0x40000000
				permW = 0x80000000
			)
			ch := sect.Characteristics
			switch {
			case ch&text != 0:
				sym.Code = 'T'
			case ch&data != 0:
				if ch&permW == 0 {
					sym.Code = 'R'
				} else {
					sym.Code = 'D'
				}
			case ch&bss != 0:
				sym.Code = 'B'
			}
			sym.Addr += imageBase + uint64(sect.VirtualAddress)
		}
		syms = append(syms, sym)
		addrs = append(addrs, sym.Addr)
	}

	sort.Sort(uint64s(addrs))
	for i := range syms {
		j := sort.Search(len(addrs), func(x int) bool { return addrs[x] > syms[i].Addr })
		if j < len(addrs) {
			syms[i].Size = int64(addrs[j] - syms[i].Addr)
		}
	}

	return
}
