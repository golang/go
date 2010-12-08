// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package elf implements access to ELF object files.
package elf

import (
	"bytes"
	"debug/dwarf"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// TODO: error reporting detail

/*
 * Internal ELF representation
 */

// A FileHeader represents an ELF file header.
type FileHeader struct {
	Class      Class
	Data       Data
	Version    Version
	OSABI      OSABI
	ABIVersion uint8
	ByteOrder  binary.ByteOrder
	Type       Type
	Machine    Machine
}

// A File represents an open ELF file.
type File struct {
	FileHeader
	Sections []*Section
	Progs    []*Prog
	closer   io.Closer
}

// A SectionHeader represents a single ELF section header.
type SectionHeader struct {
	Name      string
	Type      SectionType
	Flags     SectionFlag
	Addr      uint64
	Offset    uint64
	Size      uint64
	Link      uint32
	Info      uint32
	Addralign uint64
	Entsize   uint64
}

// A Section represents a single section in an ELF file.
type Section struct {
	SectionHeader

	// Embed ReaderAt for ReadAt method.
	// Do not embed SectionReader directly
	// to avoid having Read and Seek.
	// If a client wants Read and Seek it must use
	// Open() to avoid fighting over the seek offset
	// with other clients.
	io.ReaderAt
	sr *io.SectionReader
}

// Data reads and returns the contents of the ELF section.
func (s *Section) Data() ([]byte, os.Error) {
	dat := make([]byte, s.sr.Size())
	n, err := s.sr.ReadAt(dat, 0)
	return dat[0:n], err
}

// stringTable reads and returns the string table given by the
// specified link value.
func (f *File) stringTable(link uint32) ([]byte, os.Error) {
	if link <= 0 || link >= uint32(len(f.Sections)) {
		return nil, os.ErrorString("section has invalid string table link")
	}
	return f.Sections[link].Data()
}

// Open returns a new ReadSeeker reading the ELF section.
func (s *Section) Open() io.ReadSeeker { return io.NewSectionReader(s.sr, 0, 1<<63-1) }

// A ProgHeader represents a single ELF program header.
type ProgHeader struct {
	Type   ProgType
	Flags  ProgFlag
	Vaddr  uint64
	Paddr  uint64
	Filesz uint64
	Memsz  uint64
	Align  uint64
}

// A Prog represents a single ELF program header in an ELF binary.
type Prog struct {
	ProgHeader

	// Embed ReaderAt for ReadAt method.
	// Do not embed SectionReader directly
	// to avoid having Read and Seek.
	// If a client wants Read and Seek it must use
	// Open() to avoid fighting over the seek offset
	// with other clients.
	io.ReaderAt
	sr *io.SectionReader
}

// Open returns a new ReadSeeker reading the ELF program body.
func (p *Prog) Open() io.ReadSeeker { return io.NewSectionReader(p.sr, 0, 1<<63-1) }

// A Symbol represents an entry in an ELF symbol table section.
type Symbol struct {
	Name        string
	Info, Other byte
	Section     SectionIndex
	Value, Size uint64
}

/*
 * ELF reader
 */

type FormatError struct {
	off int64
	msg string
	val interface{}
}

func (e *FormatError) String() string {
	msg := e.msg
	if e.val != nil {
		msg += fmt.Sprintf(" '%v' ", e.val)
	}
	msg += fmt.Sprintf("in record at byte %#x", e.off)
	return msg
}

// Open opens the named file using os.Open and prepares it for use as an ELF binary.
func Open(name string) (*File, os.Error) {
	f, err := os.Open(name, os.O_RDONLY, 0)
	if err != nil {
		return nil, err
	}
	ff, err := NewFile(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	ff.closer = f
	return ff, nil
}

// Close closes the File.
// If the File was created using NewFile directly instead of Open,
// Close has no effect.
func (f *File) Close() os.Error {
	var err os.Error
	if f.closer != nil {
		err = f.closer.Close()
		f.closer = nil
	}
	return err
}

// SectionByType returns the first section in f with the
// given type, or nil if there is no such section.
func (f *File) SectionByType(typ SectionType) *Section {
	for _, s := range f.Sections {
		if s.Type == typ {
			return s
		}
	}
	return nil
}

// NewFile creates a new File for accessing an ELF binary in an underlying reader.
// The ELF binary is expected to start at position 0 in the ReaderAt.
func NewFile(r io.ReaderAt) (*File, os.Error) {
	sr := io.NewSectionReader(r, 0, 1<<63-1)
	// Read and decode ELF identifier
	var ident [16]uint8
	if _, err := r.ReadAt(ident[0:], 0); err != nil {
		return nil, err
	}
	if ident[0] != '\x7f' || ident[1] != 'E' || ident[2] != 'L' || ident[3] != 'F' {
		return nil, &FormatError{0, "bad magic number", ident[0:4]}
	}

	f := new(File)
	f.Class = Class(ident[EI_CLASS])
	switch f.Class {
	case ELFCLASS32:
	case ELFCLASS64:
		// ok
	default:
		return nil, &FormatError{0, "unknown ELF class", f.Class}
	}

	f.Data = Data(ident[EI_DATA])
	switch f.Data {
	case ELFDATA2LSB:
		f.ByteOrder = binary.LittleEndian
	case ELFDATA2MSB:
		f.ByteOrder = binary.BigEndian
	default:
		return nil, &FormatError{0, "unknown ELF data encoding", f.Data}
	}

	f.Version = Version(ident[EI_VERSION])
	if f.Version != EV_CURRENT {
		return nil, &FormatError{0, "unknown ELF version", f.Version}
	}

	f.OSABI = OSABI(ident[EI_OSABI])
	f.ABIVersion = ident[EI_ABIVERSION]

	// Read ELF file header
	var shoff int64
	var shentsize, shnum, shstrndx int
	shstrndx = -1
	switch f.Class {
	case ELFCLASS32:
		hdr := new(Header32)
		sr.Seek(0, 0)
		if err := binary.Read(sr, f.ByteOrder, hdr); err != nil {
			return nil, err
		}
		f.Type = Type(hdr.Type)
		f.Machine = Machine(hdr.Machine)
		if v := Version(hdr.Version); v != f.Version {
			return nil, &FormatError{0, "mismatched ELF version", v}
		}
		shoff = int64(hdr.Shoff)
		shentsize = int(hdr.Shentsize)
		shnum = int(hdr.Shnum)
		shstrndx = int(hdr.Shstrndx)
	case ELFCLASS64:
		hdr := new(Header64)
		sr.Seek(0, 0)
		if err := binary.Read(sr, f.ByteOrder, hdr); err != nil {
			return nil, err
		}
		f.Type = Type(hdr.Type)
		f.Machine = Machine(hdr.Machine)
		if v := Version(hdr.Version); v != f.Version {
			return nil, &FormatError{0, "mismatched ELF version", v}
		}
		shoff = int64(hdr.Shoff)
		shentsize = int(hdr.Shentsize)
		shnum = int(hdr.Shnum)
		shstrndx = int(hdr.Shstrndx)
	}
	if shstrndx < 0 || shstrndx >= shnum {
		return nil, &FormatError{0, "invalid ELF shstrndx", shstrndx}
	}

	// Read program headers
	// TODO

	// Read section headers
	f.Sections = make([]*Section, shnum)
	names := make([]uint32, shnum)
	for i := 0; i < shnum; i++ {
		off := shoff + int64(i)*int64(shentsize)
		sr.Seek(off, 0)
		s := new(Section)
		switch f.Class {
		case ELFCLASS32:
			sh := new(Section32)
			if err := binary.Read(sr, f.ByteOrder, sh); err != nil {
				return nil, err
			}
			names[i] = sh.Name
			s.SectionHeader = SectionHeader{
				Type:      SectionType(sh.Type),
				Flags:     SectionFlag(sh.Flags),
				Addr:      uint64(sh.Addr),
				Offset:    uint64(sh.Off),
				Size:      uint64(sh.Size),
				Link:      uint32(sh.Link),
				Info:      uint32(sh.Info),
				Addralign: uint64(sh.Addralign),
				Entsize:   uint64(sh.Entsize),
			}
		case ELFCLASS64:
			sh := new(Section64)
			if err := binary.Read(sr, f.ByteOrder, sh); err != nil {
				return nil, err
			}
			names[i] = sh.Name
			s.SectionHeader = SectionHeader{
				Type:      SectionType(sh.Type),
				Flags:     SectionFlag(sh.Flags),
				Offset:    uint64(sh.Off),
				Size:      uint64(sh.Size),
				Addr:      uint64(sh.Addr),
				Link:      uint32(sh.Link),
				Info:      uint32(sh.Info),
				Addralign: uint64(sh.Addralign),
				Entsize:   uint64(sh.Entsize),
			}
		}
		s.sr = io.NewSectionReader(r, int64(s.Offset), int64(s.Size))
		s.ReaderAt = s.sr
		f.Sections[i] = s
	}

	// Load section header string table.
	shstrtab, err := f.Sections[shstrndx].Data()
	if err != nil {
		return nil, err
	}
	for i, s := range f.Sections {
		var ok bool
		s.Name, ok = getString(shstrtab, int(names[i]))
		if !ok {
			return nil, &FormatError{shoff + int64(i*shentsize), "bad section name index", names[i]}
		}
	}

	return f, nil
}

// getSymbols returns a slice of Symbols from parsing the symbol table
// with the given type.
func (f *File) getSymbols(typ SectionType) ([]Symbol, os.Error) {
	switch f.Class {
	case ELFCLASS64:
		return f.getSymbols64(typ)

	case ELFCLASS32:
		return f.getSymbols32(typ)
	}

	return nil, os.ErrorString("not implemented")
}

func (f *File) getSymbols32(typ SectionType) ([]Symbol, os.Error) {
	symtabSection := f.SectionByType(typ)
	if symtabSection == nil {
		return nil, os.ErrorString("no symbol section")
	}

	data, err := symtabSection.Data()
	if err != nil {
		return nil, os.ErrorString("cannot load symbol section")
	}
	symtab := bytes.NewBuffer(data)
	if symtab.Len()%Sym32Size != 0 {
		return nil, os.ErrorString("length of symbol section is not a multiple of SymSize")
	}

	strdata, err := f.stringTable(symtabSection.Link)
	if err != nil {
		return nil, os.ErrorString("cannot load string table section")
	}

	// The first entry is all zeros.
	var skip [Sym32Size]byte
	symtab.Read(skip[0:])

	symbols := make([]Symbol, symtab.Len()/Sym32Size)

	i := 0
	var sym Sym32
	for symtab.Len() > 0 {
		binary.Read(symtab, f.ByteOrder, &sym)
		str, _ := getString(strdata, int(sym.Name))
		symbols[i].Name = str
		symbols[i].Info = sym.Info
		symbols[i].Other = sym.Other
		symbols[i].Section = SectionIndex(sym.Shndx)
		symbols[i].Value = uint64(sym.Value)
		symbols[i].Size = uint64(sym.Size)
		i++
	}

	return symbols, nil
}

func (f *File) getSymbols64(typ SectionType) ([]Symbol, os.Error) {
	symtabSection := f.SectionByType(typ)
	if symtabSection == nil {
		return nil, os.ErrorString("no symbol section")
	}

	data, err := symtabSection.Data()
	if err != nil {
		return nil, os.ErrorString("cannot load symbol section")
	}
	symtab := bytes.NewBuffer(data)
	if symtab.Len()%Sym64Size != 0 {
		return nil, os.ErrorString("length of symbol section is not a multiple of Sym64Size")
	}

	strdata, err := f.stringTable(symtabSection.Link)
	if err != nil {
		return nil, os.ErrorString("cannot load string table section")
	}

	// The first entry is all zeros.
	var skip [Sym64Size]byte
	symtab.Read(skip[0:])

	symbols := make([]Symbol, symtab.Len()/Sym64Size)

	i := 0
	var sym Sym64
	for symtab.Len() > 0 {
		binary.Read(symtab, f.ByteOrder, &sym)
		str, _ := getString(strdata, int(sym.Name))
		symbols[i].Name = str
		symbols[i].Info = sym.Info
		symbols[i].Other = sym.Other
		symbols[i].Section = SectionIndex(sym.Shndx)
		symbols[i].Value = sym.Value
		symbols[i].Size = sym.Size
		i++
	}

	return symbols, nil
}

// getString extracts a string from an ELF string table.
func getString(section []byte, start int) (string, bool) {
	if start < 0 || start >= len(section) {
		return "", false
	}

	for end := start; end < len(section); end++ {
		if section[end] == 0 {
			return string(section[start:end]), true
		}
	}
	return "", false
}

// Section returns a section with the given name, or nil if no such
// section exists.
func (f *File) Section(name string) *Section {
	for _, s := range f.Sections {
		if s.Name == name {
			return s
		}
	}
	return nil
}

// applyRelocations applies relocations to dst. rels is a relocations section
// in RELA format.
func (f *File) applyRelocations(dst []byte, rels []byte) os.Error {
	if f.Class == ELFCLASS64 && f.Machine == EM_X86_64 {
		return f.applyRelocationsAMD64(dst, rels)
	}

	return os.ErrorString("not implemented")
}

func (f *File) applyRelocationsAMD64(dst []byte, rels []byte) os.Error {
	if len(rels)%Sym64Size != 0 {
		return os.ErrorString("length of relocation section is not a multiple of Sym64Size")
	}

	symbols, err := f.getSymbols(SHT_SYMTAB)
	if err != nil {
		return err
	}

	b := bytes.NewBuffer(rels)
	var rela Rela64

	for b.Len() > 0 {
		binary.Read(b, f.ByteOrder, &rela)
		symNo := rela.Info >> 32
		t := R_X86_64(rela.Info & 0xffff)

		if symNo >= uint64(len(symbols)) {
			continue
		}
		sym := &symbols[symNo]
		if SymType(sym.Info&0xf) != STT_SECTION {
			// We don't handle non-section relocations for now.
			continue
		}

		switch t {
		case R_X86_64_64:
			if rela.Off+8 >= uint64(len(dst)) || rela.Addend < 0 {
				continue
			}
			f.ByteOrder.PutUint64(dst[rela.Off:rela.Off+8], uint64(rela.Addend))
		case R_X86_64_32:
			if rela.Off+4 >= uint64(len(dst)) || rela.Addend < 0 {
				continue
			}
			f.ByteOrder.PutUint32(dst[rela.Off:rela.Off+4], uint32(rela.Addend))
		}
	}

	return nil
}

func (f *File) DWARF() (*dwarf.Data, os.Error) {
	// There are many other DWARF sections, but these
	// are the required ones, and the debug/dwarf package
	// does not use the others, so don't bother loading them.
	var names = [...]string{"abbrev", "info", "str"}
	var dat [len(names)][]byte
	for i, name := range names {
		name = ".debug_" + name
		s := f.Section(name)
		if s == nil {
			continue
		}
		b, err := s.Data()
		if err != nil && uint64(len(b)) < s.Size {
			return nil, err
		}
		dat[i] = b
	}

	// If there's a relocation table for .debug_info, we have to process it
	// now otherwise the data in .debug_info is invalid for x86-64 objects.
	rela := f.Section(".rela.debug_info")
	if rela != nil && rela.Type == SHT_RELA && f.Machine == EM_X86_64 {
		data, err := rela.Data()
		if err != nil {
			return nil, err
		}
		err = f.applyRelocations(dat[1], data)
		if err != nil {
			return nil, err
		}
	}

	abbrev, info, str := dat[0], dat[1], dat[2]
	return dwarf.New(abbrev, nil, nil, info, nil, nil, nil, str)
}

// ImportedSymbols returns the names of all symbols
// referred to by the binary f that are expected to be
// satisfied by other libraries at dynamic load time.
// It does not return weak symbols.
func (f *File) ImportedSymbols() ([]string, os.Error) {
	sym, err := f.getSymbols(SHT_DYNSYM)
	if err != nil {
		return nil, err
	}
	var all []string
	for _, s := range sym {
		if ST_BIND(s.Info) == STB_GLOBAL && s.Section == SHN_UNDEF {
			all = append(all, s.Name)
		}
	}
	return all, nil
}

// ImportedLibraries returns the names of all libraries
// referred to by the binary f that are expected to be
// linked with the binary at dynamic link time.
func (f *File) ImportedLibraries() ([]string, os.Error) {
	ds := f.SectionByType(SHT_DYNAMIC)
	if ds == nil {
		// not dynamic, so no libraries
		return nil, nil
	}
	d, err := ds.Data()
	if err != nil {
		return nil, err
	}
	str, err := f.stringTable(ds.Link)
	if err != nil {
		return nil, err
	}
	var all []string
	for len(d) > 0 {
		var tag DynTag
		var value uint64
		switch f.Class {
		case ELFCLASS32:
			tag = DynTag(f.ByteOrder.Uint32(d[0:4]))
			value = uint64(f.ByteOrder.Uint32(d[4:8]))
			d = d[8:]
		case ELFCLASS64:
			tag = DynTag(f.ByteOrder.Uint64(d[0:8]))
			value = f.ByteOrder.Uint64(d[8:16])
			d = d[16:]
		}
		if tag == DT_NEEDED {
			s, ok := getString(str, int(value))
			if ok {
				all = append(all, s)
			}
		}
	}

	return all, nil
}
