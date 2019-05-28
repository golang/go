// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package xcoff implements access to XCOFF (Extended Common Object File Format) files.
package xcoff

import (
	"debug/dwarf"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
)

// SectionHeader holds information about an XCOFF section header.
type SectionHeader struct {
	Name           string
	VirtualAddress uint64
	Size           uint64
	Type           uint32
	Relptr         uint64
	Nreloc         uint32
}

type Section struct {
	SectionHeader
	Relocs []Reloc
	io.ReaderAt
	sr *io.SectionReader
}

// AuxiliaryCSect holds information about an XCOFF symbol in an AUX_CSECT entry.
type AuxiliaryCSect struct {
	Length              int64
	StorageMappingClass int
	SymbolType          int
}

// AuxiliaryFcn holds information about an XCOFF symbol in an AUX_FCN entry.
type AuxiliaryFcn struct {
	Size int64
}

type Symbol struct {
	Name          string
	Value         uint64
	SectionNumber int
	StorageClass  int
	AuxFcn        AuxiliaryFcn
	AuxCSect      AuxiliaryCSect
}

type Reloc struct {
	VirtualAddress   uint64
	Symbol           *Symbol
	Signed           bool
	InstructionFixed bool
	Length           uint8
	Type             uint8
}

// ImportedSymbol holds information about an imported XCOFF symbol.
type ImportedSymbol struct {
	Name    string
	Library string
}

// FileHeader holds information about an XCOFF file header.
type FileHeader struct {
	TargetMachine uint16
}

// A File represents an open XCOFF file.
type File struct {
	FileHeader
	Sections     []*Section
	Symbols      []*Symbol
	StringTable  []byte
	LibraryPaths []string

	closer io.Closer
}

// Open opens the named file using os.Open and prepares it for use as an XCOFF binary.
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

// Section returns the first section with the given name, or nil if no such
// section exists.
// Xcoff have section's name limited to 8 bytes. Some sections like .gosymtab
// can be trunked but this method will still find them.
func (f *File) Section(name string) *Section {
	for _, s := range f.Sections {
		if s.Name == name || (len(name) > 8 && s.Name == name[:8]) {
			return s
		}
	}
	return nil
}

// SectionByType returns the first section in f with the
// given type, or nil if there is no such section.
func (f *File) SectionByType(typ uint32) *Section {
	for _, s := range f.Sections {
		if s.Type == typ {
			return s
		}
	}
	return nil
}

// cstring converts ASCII byte sequence b to string.
// It stops once it finds 0 or reaches end of b.
func cstring(b []byte) string {
	var i int
	for i = 0; i < len(b) && b[i] != 0; i++ {
	}
	return string(b[:i])
}

// getString extracts a string from an XCOFF string table.
func getString(st []byte, offset uint32) (string, bool) {
	if offset < 4 || int(offset) >= len(st) {
		return "", false
	}
	return cstring(st[offset:]), true
}

// NewFile creates a new File for accessing an XCOFF binary in an underlying reader.
func NewFile(r io.ReaderAt) (*File, error) {
	sr := io.NewSectionReader(r, 0, 1<<63-1)
	// Read XCOFF target machine
	var magic uint16
	if err := binary.Read(sr, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != U802TOCMAGIC && magic != U64_TOCMAGIC {
		return nil, fmt.Errorf("unrecognised XCOFF magic: 0x%x", magic)
	}

	f := new(File)
	f.TargetMachine = magic

	// Read XCOFF file header
	if _, err := sr.Seek(0, os.SEEK_SET); err != nil {
		return nil, err
	}
	var nscns uint16
	var symptr uint64
	var nsyms int32
	var opthdr uint16
	var hdrsz int
	switch f.TargetMachine {
	case U802TOCMAGIC:
		fhdr := new(FileHeader32)
		if err := binary.Read(sr, binary.BigEndian, fhdr); err != nil {
			return nil, err
		}
		nscns = fhdr.Fnscns
		symptr = uint64(fhdr.Fsymptr)
		nsyms = fhdr.Fnsyms
		opthdr = fhdr.Fopthdr
		hdrsz = FILHSZ_32
	case U64_TOCMAGIC:
		fhdr := new(FileHeader64)
		if err := binary.Read(sr, binary.BigEndian, fhdr); err != nil {
			return nil, err
		}
		nscns = fhdr.Fnscns
		symptr = fhdr.Fsymptr
		nsyms = fhdr.Fnsyms
		opthdr = fhdr.Fopthdr
		hdrsz = FILHSZ_64
	}

	if symptr == 0 || nsyms <= 0 {
		return nil, fmt.Errorf("no symbol table")
	}

	// Read string table (located right after symbol table).
	offset := symptr + uint64(nsyms)*SYMESZ
	if _, err := sr.Seek(int64(offset), os.SEEK_SET); err != nil {
		return nil, err
	}
	// The first 4 bytes contain the length (in bytes).
	var l uint32
	if err := binary.Read(sr, binary.BigEndian, &l); err != nil {
		return nil, err
	}
	if l > 4 {
		if _, err := sr.Seek(int64(offset), os.SEEK_SET); err != nil {
			return nil, err
		}
		f.StringTable = make([]byte, l)
		if _, err := io.ReadFull(sr, f.StringTable); err != nil {
			return nil, err
		}
	}

	// Read section headers
	if _, err := sr.Seek(int64(hdrsz)+int64(opthdr), os.SEEK_SET); err != nil {
		return nil, err
	}
	f.Sections = make([]*Section, nscns)
	for i := 0; i < int(nscns); i++ {
		var scnptr uint64
		s := new(Section)
		switch f.TargetMachine {
		case U802TOCMAGIC:
			shdr := new(SectionHeader32)
			if err := binary.Read(sr, binary.BigEndian, shdr); err != nil {
				return nil, err
			}
			s.Name = cstring(shdr.Sname[:])
			s.VirtualAddress = uint64(shdr.Svaddr)
			s.Size = uint64(shdr.Ssize)
			scnptr = uint64(shdr.Sscnptr)
			s.Type = shdr.Sflags
			s.Relptr = uint64(shdr.Srelptr)
			s.Nreloc = uint32(shdr.Snreloc)
		case U64_TOCMAGIC:
			shdr := new(SectionHeader64)
			if err := binary.Read(sr, binary.BigEndian, shdr); err != nil {
				return nil, err
			}
			s.Name = cstring(shdr.Sname[:])
			s.VirtualAddress = shdr.Svaddr
			s.Size = shdr.Ssize
			scnptr = shdr.Sscnptr
			s.Type = shdr.Sflags
			s.Relptr = shdr.Srelptr
			s.Nreloc = shdr.Snreloc
		}
		r2 := r
		if scnptr == 0 { // .bss must have all 0s
			r2 = zeroReaderAt{}
		}
		s.sr = io.NewSectionReader(r2, int64(scnptr), int64(s.Size))
		s.ReaderAt = s.sr
		f.Sections[i] = s
	}

	// Symbol map needed by relocation
	var idxToSym = make(map[int]*Symbol)

	// Read symbol table
	if _, err := sr.Seek(int64(symptr), os.SEEK_SET); err != nil {
		return nil, err
	}
	f.Symbols = make([]*Symbol, 0)
	for i := 0; i < int(nsyms); i++ {
		var numaux int
		var ok, needAuxFcn bool
		sym := new(Symbol)
		switch f.TargetMachine {
		case U802TOCMAGIC:
			se := new(SymEnt32)
			if err := binary.Read(sr, binary.BigEndian, se); err != nil {
				return nil, err
			}
			numaux = int(se.Nnumaux)
			sym.SectionNumber = int(se.Nscnum)
			sym.StorageClass = int(se.Nsclass)
			sym.Value = uint64(se.Nvalue)
			needAuxFcn = se.Ntype&SYM_TYPE_FUNC != 0 && numaux > 1
			zeroes := binary.BigEndian.Uint32(se.Nname[:4])
			if zeroes != 0 {
				sym.Name = cstring(se.Nname[:])
			} else {
				offset := binary.BigEndian.Uint32(se.Nname[4:])
				sym.Name, ok = getString(f.StringTable, offset)
				if !ok {
					goto skip
				}
			}
		case U64_TOCMAGIC:
			se := new(SymEnt64)
			if err := binary.Read(sr, binary.BigEndian, se); err != nil {
				return nil, err
			}
			numaux = int(se.Nnumaux)
			sym.SectionNumber = int(se.Nscnum)
			sym.StorageClass = int(se.Nsclass)
			sym.Value = se.Nvalue
			needAuxFcn = se.Ntype&SYM_TYPE_FUNC != 0 && numaux > 1
			sym.Name, ok = getString(f.StringTable, se.Noffset)
			if !ok {
				goto skip
			}
		}
		if sym.StorageClass != C_EXT && sym.StorageClass != C_WEAKEXT && sym.StorageClass != C_HIDEXT {
			goto skip
		}
		// Must have at least one csect auxiliary entry.
		if numaux < 1 || i+numaux >= int(nsyms) {
			goto skip
		}

		if sym.SectionNumber > int(nscns) {
			goto skip
		}
		if sym.SectionNumber == 0 {
			sym.Value = 0
		} else {
			sym.Value -= f.Sections[sym.SectionNumber-1].VirtualAddress
		}

		idxToSym[i] = sym

		// If this symbol is a function, it must retrieve its size from
		// its AUX_FCN entry.
		// It can happen that a function symbol doesn't have any AUX_FCN.
		// In this case, needAuxFcn is false and their size will be set to 0.
		if needAuxFcn {
			switch f.TargetMachine {
			case U802TOCMAGIC:
				aux := new(AuxFcn32)
				if err := binary.Read(sr, binary.BigEndian, aux); err != nil {
					return nil, err
				}
				sym.AuxFcn.Size = int64(aux.Xfsize)
			case U64_TOCMAGIC:
				aux := new(AuxFcn64)
				if err := binary.Read(sr, binary.BigEndian, aux); err != nil {
					return nil, err
				}
				sym.AuxFcn.Size = int64(aux.Xfsize)
			}
		}

		// Read csect auxiliary entry (by convention, it is the last).
		if !needAuxFcn {
			if _, err := sr.Seek(int64(numaux-1)*SYMESZ, os.SEEK_CUR); err != nil {
				return nil, err
			}
		}
		i += numaux
		numaux = 0
		switch f.TargetMachine {
		case U802TOCMAGIC:
			aux := new(AuxCSect32)
			if err := binary.Read(sr, binary.BigEndian, aux); err != nil {
				return nil, err
			}
			sym.AuxCSect.SymbolType = int(aux.Xsmtyp & 0x7)
			sym.AuxCSect.StorageMappingClass = int(aux.Xsmclas)
			sym.AuxCSect.Length = int64(aux.Xscnlen)
		case U64_TOCMAGIC:
			aux := new(AuxCSect64)
			if err := binary.Read(sr, binary.BigEndian, aux); err != nil {
				return nil, err
			}
			sym.AuxCSect.SymbolType = int(aux.Xsmtyp & 0x7)
			sym.AuxCSect.StorageMappingClass = int(aux.Xsmclas)
			sym.AuxCSect.Length = int64(aux.Xscnlenhi)<<32 | int64(aux.Xscnlenlo)
		}
		f.Symbols = append(f.Symbols, sym)
	skip:
		i += numaux // Skip auxiliary entries
		if _, err := sr.Seek(int64(numaux)*SYMESZ, os.SEEK_CUR); err != nil {
			return nil, err
		}
	}

	// Read relocations
	// Only for .data or .text section
	for _, sect := range f.Sections {
		if sect.Type != STYP_TEXT && sect.Type != STYP_DATA {
			continue
		}
		sect.Relocs = make([]Reloc, sect.Nreloc)
		if sect.Relptr == 0 {
			continue
		}
		if _, err := sr.Seek(int64(sect.Relptr), os.SEEK_SET); err != nil {
			return nil, err
		}
		for i := uint32(0); i < sect.Nreloc; i++ {
			switch f.TargetMachine {
			case U802TOCMAGIC:
				rel := new(Reloc32)
				if err := binary.Read(sr, binary.BigEndian, rel); err != nil {
					return nil, err
				}
				sect.Relocs[i].VirtualAddress = uint64(rel.Rvaddr)
				sect.Relocs[i].Symbol = idxToSym[int(rel.Rsymndx)]
				sect.Relocs[i].Type = rel.Rtype
				sect.Relocs[i].Length = rel.Rsize&0x3F + 1

				if rel.Rsize&0x80 == 1 {
					sect.Relocs[i].Signed = true
				}
				if rel.Rsize&0x40 == 1 {
					sect.Relocs[i].InstructionFixed = true
				}

			case U64_TOCMAGIC:
				rel := new(Reloc64)
				if err := binary.Read(sr, binary.BigEndian, rel); err != nil {
					return nil, err
				}
				sect.Relocs[i].VirtualAddress = rel.Rvaddr
				sect.Relocs[i].Symbol = idxToSym[int(rel.Rsymndx)]
				sect.Relocs[i].Type = rel.Rtype
				sect.Relocs[i].Length = rel.Rsize&0x3F + 1
				if rel.Rsize&0x80 == 1 {
					sect.Relocs[i].Signed = true
				}
				if rel.Rsize&0x40 == 1 {
					sect.Relocs[i].InstructionFixed = true
				}
			}
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

// Data reads and returns the contents of the XCOFF section s.
func (s *Section) Data() ([]byte, error) {
	dat := make([]byte, s.sr.Size())
	n, err := s.sr.ReadAt(dat, 0)
	if n == len(dat) {
		err = nil
	}
	return dat[:n], err
}

// CSect reads and returns the contents of a csect.
func (f *File) CSect(name string) []byte {
	for _, sym := range f.Symbols {
		if sym.Name == name && sym.AuxCSect.SymbolType == XTY_SD {
			if i := sym.SectionNumber - 1; 0 <= i && i < len(f.Sections) {
				s := f.Sections[i]
				if sym.Value+uint64(sym.AuxCSect.Length) <= s.Size {
					dat := make([]byte, sym.AuxCSect.Length)
					_, err := s.sr.ReadAt(dat, int64(sym.Value))
					if err != nil {
						return nil
					}
					return dat
				}
			}
			break
		}
	}
	return nil
}

func (f *File) DWARF() (*dwarf.Data, error) {
	// There are many other DWARF sections, but these
	// are the ones the debug/dwarf package uses.
	// Don't bother loading others.
	var subtypes = [...]uint32{SSUBTYP_DWABREV, SSUBTYP_DWINFO, SSUBTYP_DWLINE, SSUBTYP_DWRNGES, SSUBTYP_DWSTR}
	var dat [len(subtypes)][]byte
	for i, subtype := range subtypes {
		s := f.SectionByType(STYP_DWARF | subtype)
		if s != nil {
			b, err := s.Data()
			if err != nil && uint64(len(b)) < s.Size {
				return nil, err
			}
			dat[i] = b
		}
	}

	abbrev, info, line, ranges, str := dat[0], dat[1], dat[2], dat[3], dat[4]
	return dwarf.New(abbrev, nil, nil, info, line, nil, ranges, str)
}

// readImportID returns the import file IDs stored inside the .loader section.
// Library name pattern is either path/base/member or base/member
func (f *File) readImportIDs(s *Section) ([]string, error) {
	// Read loader header
	if _, err := s.sr.Seek(0, os.SEEK_SET); err != nil {
		return nil, err
	}
	var istlen uint32
	var nimpid int32
	var impoff uint64
	switch f.TargetMachine {
	case U802TOCMAGIC:
		lhdr := new(LoaderHeader32)
		if err := binary.Read(s.sr, binary.BigEndian, lhdr); err != nil {
			return nil, err
		}
		istlen = lhdr.Listlen
		nimpid = lhdr.Lnimpid
		impoff = uint64(lhdr.Limpoff)
	case U64_TOCMAGIC:
		lhdr := new(LoaderHeader64)
		if err := binary.Read(s.sr, binary.BigEndian, lhdr); err != nil {
			return nil, err
		}
		istlen = lhdr.Listlen
		nimpid = lhdr.Lnimpid
		impoff = lhdr.Limpoff
	}

	// Read loader import file ID table
	if _, err := s.sr.Seek(int64(impoff), os.SEEK_SET); err != nil {
		return nil, err
	}
	table := make([]byte, istlen)
	if _, err := io.ReadFull(s.sr, table); err != nil {
		return nil, err
	}

	offset := 0
	// First import file ID is the default LIBPATH value
	libpath := cstring(table[offset:])
	f.LibraryPaths = strings.Split(libpath, ":")
	offset += len(libpath) + 3 // 3 null bytes
	all := make([]string, 0)
	for i := 1; i < int(nimpid); i++ {
		impidpath := cstring(table[offset:])
		offset += len(impidpath) + 1
		impidbase := cstring(table[offset:])
		offset += len(impidbase) + 1
		impidmem := cstring(table[offset:])
		offset += len(impidmem) + 1
		var path string
		if len(impidpath) > 0 {
			path = impidpath + "/" + impidbase + "/" + impidmem
		} else {
			path = impidbase + "/" + impidmem
		}
		all = append(all, path)
	}

	return all, nil
}

// ImportedSymbols returns the names of all symbols
// referred to by the binary f that are expected to be
// satisfied by other libraries at dynamic load time.
// It does not return weak symbols.
func (f *File) ImportedSymbols() ([]ImportedSymbol, error) {
	s := f.SectionByType(STYP_LOADER)
	if s == nil {
		return nil, nil
	}
	// Read loader header
	if _, err := s.sr.Seek(0, os.SEEK_SET); err != nil {
		return nil, err
	}
	var stlen uint32
	var stoff uint64
	var nsyms int32
	var symoff uint64
	switch f.TargetMachine {
	case U802TOCMAGIC:
		lhdr := new(LoaderHeader32)
		if err := binary.Read(s.sr, binary.BigEndian, lhdr); err != nil {
			return nil, err
		}
		stlen = lhdr.Lstlen
		stoff = uint64(lhdr.Lstoff)
		nsyms = lhdr.Lnsyms
		symoff = LDHDRSZ_32
	case U64_TOCMAGIC:
		lhdr := new(LoaderHeader64)
		if err := binary.Read(s.sr, binary.BigEndian, lhdr); err != nil {
			return nil, err
		}
		stlen = lhdr.Lstlen
		stoff = lhdr.Lstoff
		nsyms = lhdr.Lnsyms
		symoff = lhdr.Lsymoff
	}

	// Read loader section string table
	if _, err := s.sr.Seek(int64(stoff), os.SEEK_SET); err != nil {
		return nil, err
	}
	st := make([]byte, stlen)
	if _, err := io.ReadFull(s.sr, st); err != nil {
		return nil, err
	}

	// Read imported libraries
	libs, err := f.readImportIDs(s)
	if err != nil {
		return nil, err
	}

	// Read loader symbol table
	if _, err := s.sr.Seek(int64(symoff), os.SEEK_SET); err != nil {
		return nil, err
	}
	all := make([]ImportedSymbol, 0)
	for i := 0; i < int(nsyms); i++ {
		var name string
		var ifile int32
		var ok bool
		switch f.TargetMachine {
		case U802TOCMAGIC:
			ldsym := new(LoaderSymbol32)
			if err := binary.Read(s.sr, binary.BigEndian, ldsym); err != nil {
				return nil, err
			}
			if ldsym.Lsmtype&0x40 == 0 {
				continue // Imported symbols only
			}
			zeroes := binary.BigEndian.Uint32(ldsym.Lname[:4])
			if zeroes != 0 {
				name = cstring(ldsym.Lname[:])
			} else {
				offset := binary.BigEndian.Uint32(ldsym.Lname[4:])
				name, ok = getString(st, offset)
				if !ok {
					continue
				}
			}
			ifile = ldsym.Lifile
		case U64_TOCMAGIC:
			ldsym := new(LoaderSymbol64)
			if err := binary.Read(s.sr, binary.BigEndian, ldsym); err != nil {
				return nil, err
			}
			if ldsym.Lsmtype&0x40 == 0 {
				continue // Imported symbols only
			}
			name, ok = getString(st, ldsym.Loffset)
			if !ok {
				continue
			}
			ifile = ldsym.Lifile
		}
		var sym ImportedSymbol
		sym.Name = name
		if ifile >= 1 && int(ifile) <= len(libs) {
			sym.Library = libs[ifile-1]
		}
		all = append(all, sym)
	}

	return all, nil
}

// ImportedLibraries returns the names of all libraries
// referred to by the binary f that are expected to be
// linked with the binary at dynamic link time.
func (f *File) ImportedLibraries() ([]string, error) {
	s := f.SectionByType(STYP_LOADER)
	if s == nil {
		return nil, nil
	}
	all, err := f.readImportIDs(s)
	return all, err
}
