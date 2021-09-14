// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parsing of ELF executables (Linux, FreeBSD, and so on).

package objfile

import (
	"debug/dwarf"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"io"
)

type elfFile struct {
	elf *elf.File
}

func openElf(r io.ReaderAt) (rawFile, error) {
	f, err := elf.NewFile(r)
	if err != nil {
		return nil, err
	}
	return &elfFile{f}, nil
}

// get Symbol Type from a s.Info, which also can derive the binding
// https://refspecs.linuxfoundation.org/elf/elf.pdf section 1-18
func getSymbolType(s elf.Symbol) string {
	symType := int(s.Info) & 0xf

	// Type expects a string. In the future this could be an enum.
	switch symType {
	case 0:
		return "STT_NOTYPE"
	case 1:
		return "STT_OBJECT"
	case 2:
		return "STT_FUNC"
	case 4:
		return "STT_FILE"
	case 13:
		return "STT_LOPROC"
	case 15:
		return "STT_HIPROC"
	}
	// We should not get here!
	return "UNKNOWN"
}

// get Symbol Binding from s.Info
func getSymbolBinding(s elf.Symbol) string {
	binding := s.Info >> 4
	switch binding {
	case 0:
		return "STB_LOCAL"
	case 1:
		return "STB_GLOBAL"
	case 2:
		return "STB_WEAK"
	case 13:
		return "STB_LOPROC"
	case 15:
		return "STB_HIPROC"
	}
	// We should not get here!
	return "UNKNOWN"
}

func (f *elfFile) symbols() ([]Sym, error) {
	elfSyms, err := f.elf.Symbols()
	if err != nil {
		return nil, err
	}

	var syms []Sym
	for _, s := range elfSyms {

		// Convert the s.Info (we can use to calculate binding and type) to unsigned int, then string
		symType := getSymbolType(s)
		binding := getSymbolBinding(s)
		sym := Sym{Addr: s.Value, Name: s.Name, Type: symType, Binding: binding, Size: int64(s.Size), Code: '?'}
		switch s.Section {
		case elf.SHN_UNDEF:
			sym.Code = 'U'
		case elf.SHN_COMMON:
			sym.Code = 'B'
		default:
			i := int(s.Section)
			if i < 0 || i >= len(f.elf.Sections) {
				break
			}
			sect := f.elf.Sections[i]
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
	return syms, nil
}

func (f *elfFile) pcln() (textStart uint64, symtab, pclntab []byte, err error) {
	if sect := f.elf.Section(".text"); sect != nil {
		textStart = sect.Addr
	}
	if sect := f.elf.Section(".gosymtab"); sect != nil {
		if symtab, err = sect.Data(); err != nil {
			return 0, nil, nil, err
		}
	}
	if sect := f.elf.Section(".gopclntab"); sect != nil {
		if pclntab, err = sect.Data(); err != nil {
			return 0, nil, nil, err
		}
	}
	return textStart, symtab, pclntab, nil
}

func (f *elfFile) text() (textStart uint64, text []byte, err error) {
	sect := f.elf.Section(".text")
	if sect == nil {
		return 0, nil, fmt.Errorf("text section not found")
	}
	textStart = sect.Addr
	text, err = sect.Data()
	return
}

func (f *elfFile) goarch() string {
	switch f.elf.Machine {
	case elf.EM_386:
		return "386"
	case elf.EM_X86_64:
		return "amd64"
	case elf.EM_ARM:
		return "arm"
	case elf.EM_AARCH64:
		return "arm64"
	case elf.EM_PPC64:
		if f.elf.ByteOrder == binary.LittleEndian {
			return "ppc64le"
		}
		return "ppc64"
	case elf.EM_S390:
		return "s390x"
	}
	return ""
}

func (f *elfFile) loadAddress() (uint64, error) {
	for _, p := range f.elf.Progs {
		if p.Type == elf.PT_LOAD && p.Flags&elf.PF_X != 0 {
			return p.Vaddr, nil
		}
	}
	return 0, fmt.Errorf("unknown load address")
}

func (f *elfFile) dwarf() (*dwarf.Data, error) {
	return f.elf.DWARF()
}
