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

	var syms []Sym
	for _, s := range p.Symbols {
		sym := Sym{Name: s.Name, Addr: uint64(s.Value), Code: '?'}
		if s.SectionNumber == 0 {
			sym.Code = 'U'
		} else if int(s.SectionNumber) <= len(p.Sections) {
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
		}
		syms = append(syms, sym)
	}

	return syms
}
