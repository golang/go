// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package pe implements access to PE (Microsoft Windows Portable Executable) files.
package pe

import (
	"bytes"
	"compress/zlib"
	"debug/dwarf"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
)

// Avoid use of post-Go 1.4 io features, to make safe for toolchain bootstrap.
const seekStart = 0

// A File represents an open PE file.
type File struct {
	FileHeader
	OptionalHeader interface{} // of type *OptionalHeader32 or *OptionalHeader64
	Sections       []*Section
	Symbols        []*Symbol    // COFF symbols with auxiliary symbol records removed
	COFFSymbols    []COFFSymbol // all COFF symbols (including auxiliary symbol records)
	StringTable    StringTable

	closer io.Closer
}

// Open opens the named file using os.Open and prepares it for use as a PE binary.
func Open(name string) (*File, error) {
	f, err := os.Open(name)
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
func (f *File) Close() error {
	var err error
	if f.closer != nil {
		err = f.closer.Close()
		f.closer = nil
	}
	return err
}

var (
	sizeofOptionalHeader32 = uint16(binary.Size(OptionalHeader32{}))
	sizeofOptionalHeader64 = uint16(binary.Size(OptionalHeader64{}))
)

// TODO(brainman): add Load function, as a replacement for NewFile, that does not call removeAuxSymbols (for performance)

// NewFile creates a new File for accessing a PE binary in an underlying reader.
func NewFile(r io.ReaderAt) (*File, error) {
	f := new(File)
	sr := io.NewSectionReader(r, 0, 1<<63-1)

	var dosheader [96]byte
	if _, err := r.ReadAt(dosheader[0:], 0); err != nil {
		return nil, err
	}
	var base int64
	if dosheader[0] == 'M' && dosheader[1] == 'Z' {
		signoff := int64(binary.LittleEndian.Uint32(dosheader[0x3c:]))
		var sign [4]byte
		r.ReadAt(sign[:], signoff)
		if !(sign[0] == 'P' && sign[1] == 'E' && sign[2] == 0 && sign[3] == 0) {
			return nil, fmt.Errorf("Invalid PE COFF file signature of %v.", sign)
		}
		base = signoff + 4
	} else {
		base = int64(0)
	}
	sr.Seek(base, seekStart)
	if err := binary.Read(sr, binary.LittleEndian, &f.FileHeader); err != nil {
		return nil, err
	}
	switch f.FileHeader.Machine {
	case IMAGE_FILE_MACHINE_UNKNOWN, IMAGE_FILE_MACHINE_ARMNT, IMAGE_FILE_MACHINE_AMD64, IMAGE_FILE_MACHINE_I386:
	default:
		return nil, fmt.Errorf("Unrecognised COFF file header machine value of 0x%x.", f.FileHeader.Machine)
	}

	var err error

	// Read string table.
	f.StringTable, err = readStringTable(&f.FileHeader, sr)
	if err != nil {
		return nil, err
	}

	// Read symbol table.
	f.COFFSymbols, err = readCOFFSymbols(&f.FileHeader, sr)
	if err != nil {
		return nil, err
	}
	f.Symbols, err = removeAuxSymbols(f.COFFSymbols, f.StringTable)
	if err != nil {
		return nil, err
	}

	// Read optional header.
	sr.Seek(base, seekStart)
	if err := binary.Read(sr, binary.LittleEndian, &f.FileHeader); err != nil {
		return nil, err
	}
	var oh32 OptionalHeader32
	var oh64 OptionalHeader64
	switch f.FileHeader.SizeOfOptionalHeader {
	case sizeofOptionalHeader32:
		if err := binary.Read(sr, binary.LittleEndian, &oh32); err != nil {
			return nil, err
		}
		if oh32.Magic != 0x10b { // PE32
			return nil, fmt.Errorf("pe32 optional header has unexpected Magic of 0x%x", oh32.Magic)
		}
		f.OptionalHeader = &oh32
	case sizeofOptionalHeader64:
		if err := binary.Read(sr, binary.LittleEndian, &oh64); err != nil {
			return nil, err
		}
		if oh64.Magic != 0x20b { // PE32+
			return nil, fmt.Errorf("pe32+ optional header has unexpected Magic of 0x%x", oh64.Magic)
		}
		f.OptionalHeader = &oh64
	}

	// Process sections.
	f.Sections = make([]*Section, f.FileHeader.NumberOfSections)
	for i := 0; i < int(f.FileHeader.NumberOfSections); i++ {
		sh := new(SectionHeader32)
		if err := binary.Read(sr, binary.LittleEndian, sh); err != nil {
			return nil, err
		}
		name, err := sh.fullName(f.StringTable)
		if err != nil {
			return nil, err
		}
		s := new(Section)
		s.SectionHeader = SectionHeader{
			Name:                 name,
			VirtualSize:          sh.VirtualSize,
			VirtualAddress:       sh.VirtualAddress,
			Size:                 sh.SizeOfRawData,
			Offset:               sh.PointerToRawData,
			PointerToRelocations: sh.PointerToRelocations,
			PointerToLineNumbers: sh.PointerToLineNumbers,
			NumberOfRelocations:  sh.NumberOfRelocations,
			NumberOfLineNumbers:  sh.NumberOfLineNumbers,
			Characteristics:      sh.Characteristics,
		}
		r2 := r
		if sh.PointerToRawData == 0 { // .bss must have all 0s
			r2 = zeroReaderAt{}
		}
		s.sr = io.NewSectionReader(r2, int64(s.SectionHeader.Offset), int64(s.SectionHeader.Size))
		s.ReaderAt = s.sr
		f.Sections[i] = s
	}
	for i := range f.Sections {
		var err error
		f.Sections[i].Relocs, err = readRelocs(&f.Sections[i].SectionHeader, sr)
		if err != nil {
			return nil, err
		}
	}

	return f, nil
}

// zeroReaderAt is ReaderAt that reads 0s.
type zeroReaderAt struct{}

// ReadAt writes len(p) 0s into p.
func (w zeroReaderAt) ReadAt(p []byte, off int64) (n int, err error) {
	for i := range p {
		p[i] = 0
	}
	return len(p), nil
}

// getString extracts a string from symbol string table.
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

// Section returns the first section with the given name, or nil if no such
// section exists.
func (f *File) Section(name string) *Section {
	for _, s := range f.Sections {
		if s.Name == name {
			return s
		}
	}
	return nil
}

func (f *File) DWARF() (*dwarf.Data, error) {
	dwarfSuffix := func(s *Section) string {
		switch {
		case strings.HasPrefix(s.Name, ".debug_"):
			return s.Name[7:]
		case strings.HasPrefix(s.Name, ".zdebug_"):
			return s.Name[8:]
		default:
			return ""
		}

	}

	// sectionData gets the data for s and checks its size.
	sectionData := func(s *Section) ([]byte, error) {
		b, err := s.Data()
		if err != nil && uint32(len(b)) < s.Size {
			return nil, err
		}

		if 0 < s.VirtualSize && s.VirtualSize < s.Size {
			b = b[:s.VirtualSize]
		}

		if len(b) >= 12 && string(b[:4]) == "ZLIB" {
			dlen := binary.BigEndian.Uint64(b[4:12])
			dbuf := make([]byte, dlen)
			r, err := zlib.NewReader(bytes.NewBuffer(b[12:]))
			if err != nil {
				return nil, err
			}
			if _, err := io.ReadFull(r, dbuf); err != nil {
				return nil, err
			}
			if err := r.Close(); err != nil {
				return nil, err
			}
			b = dbuf
		}
		return b, nil
	}

	// There are many other DWARF sections, but these
	// are the ones the debug/dwarf package uses.
	// Don't bother loading others.
	var dat = map[string][]byte{"abbrev": nil, "info": nil, "str": nil, "line": nil, "ranges": nil}
	for _, s := range f.Sections {
		suffix := dwarfSuffix(s)
		if suffix == "" {
			continue
		}
		if _, ok := dat[suffix]; !ok {
			continue
		}

		b, err := sectionData(s)
		if err != nil {
			return nil, err
		}
		dat[suffix] = b
	}

	d, err := dwarf.New(dat["abbrev"], nil, nil, dat["info"], dat["line"], nil, dat["ranges"], dat["str"])
	if err != nil {
		return nil, err
	}

	// Look for DWARF4 .debug_types sections.
	for i, s := range f.Sections {
		suffix := dwarfSuffix(s)
		if suffix != "types" {
			continue
		}

		b, err := sectionData(s)
		if err != nil {
			return nil, err
		}

		err = d.AddTypes(fmt.Sprintf("types-%d", i), b)
		if err != nil {
			return nil, err
		}
	}

	return d, nil
}

// TODO(brainman): document ImportDirectory once we decide what to do with it.

type ImportDirectory struct {
	OriginalFirstThunk uint32
	TimeDateStamp      uint32
	ForwarderChain     uint32
	Name               uint32
	FirstThunk         uint32

	dll string
}

// ImportedSymbols returns the names of all symbols
// referred to by the binary f that are expected to be
// satisfied by other libraries at dynamic load time.
// It does not return weak symbols.
func (f *File) ImportedSymbols() ([]string, error) {
	pe64 := f.Machine == IMAGE_FILE_MACHINE_AMD64

	// grab the number of data directory entries
	var dd_length uint32
	if pe64 {
		dd_length = f.OptionalHeader.(*OptionalHeader64).NumberOfRvaAndSizes
	} else {
		dd_length = f.OptionalHeader.(*OptionalHeader32).NumberOfRvaAndSizes
	}

	// check that the length of data directory entries is large
	// enough to include the imports directory.
	if dd_length < IMAGE_DIRECTORY_ENTRY_IMPORT+1 {
		return nil, nil
	}

	// grab the import data directory entry
	var idd DataDirectory
	if pe64 {
		idd = f.OptionalHeader.(*OptionalHeader64).DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT]
	} else {
		idd = f.OptionalHeader.(*OptionalHeader32).DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT]
	}

	// figure out which section contains the import directory table
	var ds *Section
	ds = nil
	for _, s := range f.Sections {
		if s.VirtualAddress <= idd.VirtualAddress && idd.VirtualAddress < s.VirtualAddress+s.VirtualSize {
			ds = s
			break
		}
	}

	// didn't find a section, so no import libraries were found
	if ds == nil {
		return nil, nil
	}

	d, err := ds.Data()
	if err != nil {
		return nil, err
	}

	// seek to the virtual address specified in the import data directory
	d = d[idd.VirtualAddress-ds.VirtualAddress:]

	// start decoding the import directory
	var ida []ImportDirectory
	for len(d) > 0 {
		var dt ImportDirectory
		dt.OriginalFirstThunk = binary.LittleEndian.Uint32(d[0:4])
		dt.TimeDateStamp = binary.LittleEndian.Uint32(d[4:8])
		dt.ForwarderChain = binary.LittleEndian.Uint32(d[8:12])
		dt.Name = binary.LittleEndian.Uint32(d[12:16])
		dt.FirstThunk = binary.LittleEndian.Uint32(d[16:20])
		d = d[20:]
		if dt.OriginalFirstThunk == 0 {
			break
		}
		ida = append(ida, dt)
	}
	// TODO(brainman): this needs to be rewritten
	//  ds.Data() returns contents of section containing import table. Why store in variable called "names"?
	//  Why we are retrieving it second time? We already have it in "d", and it is not modified anywhere.
	//  getString does not extracts a string from symbol string table (as getString doco says).
	//  Why ds.Data() called again and again in the loop?
	//  Needs test before rewrite.
	names, _ := ds.Data()
	var all []string
	for _, dt := range ida {
		dt.dll, _ = getString(names, int(dt.Name-ds.VirtualAddress))
		d, _ = ds.Data()
		// seek to OriginalFirstThunk
		d = d[dt.OriginalFirstThunk-ds.VirtualAddress:]
		for len(d) > 0 {
			if pe64 { // 64bit
				va := binary.LittleEndian.Uint64(d[0:8])
				d = d[8:]
				if va == 0 {
					break
				}
				if va&0x8000000000000000 > 0 { // is Ordinal
					// TODO add dynimport ordinal support.
				} else {
					fn, _ := getString(names, int(uint32(va)-ds.VirtualAddress+2))
					all = append(all, fn+":"+dt.dll)
				}
			} else { // 32bit
				va := binary.LittleEndian.Uint32(d[0:4])
				d = d[4:]
				if va == 0 {
					break
				}
				if va&0x80000000 > 0 { // is Ordinal
					// TODO add dynimport ordinal support.
					//ord := va&0x0000FFFF
				} else {
					fn, _ := getString(names, int(va-ds.VirtualAddress+2))
					all = append(all, fn+":"+dt.dll)
				}
			}
		}
	}

	return all, nil
}

// ImportedLibraries returns the names of all libraries
// referred to by the binary f that are expected to be
// linked with the binary at dynamic link time.
func (f *File) ImportedLibraries() ([]string, error) {
	// TODO
	// cgo -dynimport don't use this for windows PE, so just return.
	return nil, nil
}

// FormatError is unused.
// The type is retained for compatibility.
type FormatError struct {
}

func (e *FormatError) Error() string {
	return "unknown error"
}
