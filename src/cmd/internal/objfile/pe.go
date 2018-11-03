// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parsing of PE executables (Microsoft Windows).

package objfile

import (
	"debug/dwarf"
	"debug/pe"
	"fmt"
	"io"
	"sort"
)

type peFile struct {
	pe *pe.File
}

func openPE(r io.ReaderAt) (rawFile, error) {
	f, err := pe.NewFile(r)
	if err != nil {
		return nil, err
	}
	return &peFile{f}, nil
}

func (f *peFile) symbols() ([]Sym, error) {
	// Build sorted list of addresses of all symbols.
	// We infer the size of a symbol by looking at where the next symbol begins.
	var addrs []uint64

	var imageBase uint64
	switch oh := f.pe.OptionalHeader.(type) {
	case *pe.OptionalHeader32:
		imageBase = uint64(oh.ImageBase)
	case *pe.OptionalHeader64:
		imageBase = oh.ImageBase
	}

	var syms []Sym
	for _, s := range f.pe.Symbols {
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
			if s.SectionNumber < 0 || len(f.pe.Sections) < int(s.SectionNumber) {
				return nil, fmt.Errorf("invalid section number in symbol table")
			}
			sect := f.pe.Sections[s.SectionNumber-1]
			const (
				text  = 0x20
				data  = 0x40
				bss   = 0x80
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

	return syms, nil
}

func (f *peFile) pcln() (textStart uint64, symtab, pclntab []byte, err error) {
	var imageBase uint64
	switch oh := f.pe.OptionalHeader.(type) {
	case *pe.OptionalHeader32:
		imageBase = uint64(oh.ImageBase)
	case *pe.OptionalHeader64:
		imageBase = oh.ImageBase
	default:
		return 0, nil, nil, fmt.Errorf("pe file format not recognized")
	}
	if sect := f.pe.Section(".text"); sect != nil {
		textStart = imageBase + uint64(sect.VirtualAddress)
	}
	if pclntab, err = loadPETable(f.pe, "runtime.pclntab", "runtime.epclntab"); err != nil {
		// We didn't find the symbols, so look for the names used in 1.3 and earlier.
		// TODO: Remove code looking for the old symbols when we no longer care about 1.3.
		var err2 error
		if pclntab, err2 = loadPETable(f.pe, "pclntab", "epclntab"); err2 != nil {
			return 0, nil, nil, err
		}
	}
	if symtab, err = loadPETable(f.pe, "runtime.symtab", "runtime.esymtab"); err != nil {
		// Same as above.
		var err2 error
		if symtab, err2 = loadPETable(f.pe, "symtab", "esymtab"); err2 != nil {
			return 0, nil, nil, err
		}
	}
	return textStart, symtab, pclntab, nil
}

func (f *peFile) text() (textStart uint64, text []byte, err error) {
	var imageBase uint64
	switch oh := f.pe.OptionalHeader.(type) {
	case *pe.OptionalHeader32:
		imageBase = uint64(oh.ImageBase)
	case *pe.OptionalHeader64:
		imageBase = oh.ImageBase
	default:
		return 0, nil, fmt.Errorf("pe file format not recognized")
	}
	sect := f.pe.Section(".text")
	if sect == nil {
		return 0, nil, fmt.Errorf("text section not found")
	}
	textStart = imageBase + uint64(sect.VirtualAddress)
	text, err = sect.Data()
	return
}

func findPESymbol(f *pe.File, name string) (*pe.Symbol, error) {
	for _, s := range f.Symbols {
		if s.Name != name {
			continue
		}
		if s.SectionNumber <= 0 {
			return nil, fmt.Errorf("symbol %s: invalid section number %d", name, s.SectionNumber)
		}
		if len(f.Sections) < int(s.SectionNumber) {
			return nil, fmt.Errorf("symbol %s: section number %d is larger than max %d", name, s.SectionNumber, len(f.Sections))
		}
		return s, nil
	}
	return nil, fmt.Errorf("no %s symbol found", name)
}

func loadPETable(f *pe.File, sname, ename string) ([]byte, error) {
	ssym, err := findPESymbol(f, sname)
	if err != nil {
		return nil, err
	}
	esym, err := findPESymbol(f, ename)
	if err != nil {
		return nil, err
	}
	if ssym.SectionNumber != esym.SectionNumber {
		return nil, fmt.Errorf("%s and %s symbols must be in the same section", sname, ename)
	}
	sect := f.Sections[ssym.SectionNumber-1]
	data, err := sect.Data()
	if err != nil {
		return nil, err
	}
	return data[ssym.Value:esym.Value], nil
}

func (f *peFile) goarch() string {
	// Not sure how to get the info we want from PE header.
	// Look in symbol table for telltale rt0 symbol.
	if _, err := findPESymbol(f.pe, "_rt0_386_windows"); err == nil {
		return "386"
	}
	if _, err := findPESymbol(f.pe, "_rt0_amd64_windows"); err == nil {
		return "amd64"
	}
	if _, err := findPESymbol(f.pe, "_rt0_arm_windows"); err == nil {
		return "arm"
	}
	return ""
}

func (f *peFile) loadAddress() (uint64, error) {
	return 0, fmt.Errorf("unknown load address")
}

func (f *peFile) dwarf() (*dwarf.Data, error) {
	return f.pe.DWARF()
}
