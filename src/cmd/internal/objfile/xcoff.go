// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parsing of XCOFF executable (AIX)

package objfile

import (
	"cmd/internal/xcoff"
	"debug/dwarf"
	"fmt"
	"io"
	"unicode"
)

type xcoffFile struct {
	xcoff *xcoff.File
}

func openXcoff(r io.ReaderAt) (rawFile, error) {
	f, err := xcoff.NewFile(r)
	if err != nil {
		return nil, err
	}
	return &xcoffFile{f}, nil
}

func (f *xcoffFile) symbols() ([]Sym, error) {
	var syms []Sym
	for _, s := range f.xcoff.Symbols {
		const (
			N_UNDEF = 0  // An undefined (extern) symbol
			N_ABS   = -1 // An absolute symbol (e_value is a constant, not an address)
			N_DEBUG = -2 // A debugging symbol
		)
		sym := Sym{Name: s.Name, Addr: s.Value, Code: '?'}

		switch s.SectionNumber {
		case N_UNDEF:
			sym.Code = 'U'
		case N_ABS:
			sym.Code = 'C'
		case N_DEBUG:
			sym.Code = '?'
		default:
			if s.SectionNumber < 0 || len(f.xcoff.Sections) < int(s.SectionNumber) {
				return nil, fmt.Errorf("invalid section number in symbol table")
			}
			sect := f.xcoff.Sections[s.SectionNumber-1]

			// debug/xcoff returns an offset in the section not the actual address
			sym.Addr += sect.VirtualAddress

			if s.AuxCSect.SymbolType&0x3 == xcoff.XTY_LD {
				// The size of a function is contained in the
				// AUX_FCN entry
				sym.Size = s.AuxFcn.Size
			} else {
				sym.Size = s.AuxCSect.Length
			}

			sym.Size = s.AuxCSect.Length

			switch sect.Type {
			case xcoff.STYP_TEXT:
				if s.AuxCSect.StorageMappingClass == xcoff.XMC_RO {
					sym.Code = 'R'
				} else {
					sym.Code = 'T'
				}
			case xcoff.STYP_DATA:
				sym.Code = 'D'
			case xcoff.STYP_BSS:
				sym.Code = 'B'
			}

			if s.StorageClass == xcoff.C_HIDEXT {
				// Local symbol
				sym.Code = unicode.ToLower(sym.Code)
			}

		}
		syms = append(syms, sym)
	}

	return syms, nil
}

func (f *xcoffFile) pcln() (textStart uint64, symtab, pclntab []byte, err error) {
	if sect := f.xcoff.Section(".text"); sect != nil {
		textStart = sect.VirtualAddress
	}
	if pclntab, err = loadXCOFFTable(f.xcoff, "runtime.pclntab", "runtime.epclntab"); err != nil {
		return 0, nil, nil, err
	}
	if symtab, err = loadXCOFFTable(f.xcoff, "runtime.symtab", "runtime.esymtab"); err != nil {
		return 0, nil, nil, err
	}
	return textStart, symtab, pclntab, nil
}

func (f *xcoffFile) text() (textStart uint64, text []byte, err error) {
	sect := f.xcoff.Section(".text")
	if sect == nil {
		return 0, nil, fmt.Errorf("text section not found")
	}
	textStart = sect.VirtualAddress
	text, err = sect.Data()
	return
}

func findXCOFFSymbol(f *xcoff.File, name string) (*xcoff.Symbol, error) {
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

func loadXCOFFTable(f *xcoff.File, sname, ename string) ([]byte, error) {
	ssym, err := findXCOFFSymbol(f, sname)
	if err != nil {
		return nil, err
	}
	esym, err := findXCOFFSymbol(f, ename)
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

func (f *xcoffFile) goarch() string {
	switch f.xcoff.TargetMachine {
	case xcoff.U802TOCMAGIC:
		return "ppc"
	case xcoff.U64_TOCMAGIC:
		return "ppc64"
	}
	return ""
}

func (f *xcoffFile) loadAddress() (uint64, error) {
	return 0, fmt.Errorf("unknown load address")
}

func (f *xcoffFile) dwarf() (*dwarf.Data, error) {
	return f.xcoff.DWARF()
}
