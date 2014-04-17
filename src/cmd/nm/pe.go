// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parsing of PE executables (Microsoft Windows).

package main

import (
	"debug/pe"
	"os"
)

func peSymbols(f *os.File) []Sym {
	p, err := pe.NewFile(f)
	if err != nil {
		errorf("parsing %s: %v", f.Name(), err)
		return nil
	}

	var imageBase uint64
	switch oh := p.OptionalHeader.(type) {
	case *pe.OptionalHeader32:
		imageBase = uint64(oh.ImageBase)
	case *pe.OptionalHeader64:
		imageBase = oh.ImageBase
	default:
		errorf("parsing %s: file format not recognized", f.Name())
		return nil
	}

	var syms []Sym
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
				return nil
			}
			if len(p.Sections) < int(s.SectionNumber) {
				errorf("parsing %s: section number %d is large then max %d", f.Name(), s.SectionNumber, len(p.Sections))
				return nil
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
	}

	return syms
}
