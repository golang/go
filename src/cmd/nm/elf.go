// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parsing of ELF executables (Linux, FreeBSD, and so on).

package main

import (
	"debug/elf"
	"os"
)

func elfSymbols(f *os.File) []Sym {
	p, err := elf.NewFile(f)
	if err != nil {
		errorf("parsing %s: %v", f.Name(), err)
		return nil
	}

	elfSyms, err := p.Symbols()
	if err != nil {
		errorf("parsing %s: %v", f.Name(), err)
		return nil
	}

	var syms []Sym
	for _, s := range elfSyms {
		sym := Sym{Addr: s.Value, Name: s.Name, Size: int64(s.Size), Code: '?'}
		switch s.Section {
		case elf.SHN_UNDEF:
			sym.Code = 'U'
		case elf.SHN_COMMON:
			sym.Code = 'B'
		default:
			i := int(s.Section)
			if i <= 0 || i > len(p.Sections) {
				break
			}
			sect := p.Sections[i-1]
			switch sect.Flags & (elf.SHF_WRITE | elf.SHF_ALLOC | elf.SHF_EXECINSTR) {
			case elf.SHF_ALLOC | elf.SHF_EXECINSTR:
				sym.Code = 'T'
			case elf.SHF_ALLOC:
				sym.Code = 'R'
			case elf.SHF_ALLOC | elf.SHF_WRITE:
				sym.Code = 'D'
			}
		}
		if elf.ST_BIND(s.Info) == elf.STB_LOCAL {
			sym.Code += 'a' - 'A'
		}
		syms = append(syms, sym)
	}

	return syms
}
