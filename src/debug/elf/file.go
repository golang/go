// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package elf implements access to ELF object files.

# Security

This package is not designed to be hardened against adversarial inputs, and is
outside the scope of https://go.dev/security/policy. In particular, only basic
validation is done when parsing object files. As such, care should be taken when
parsing untrusted inputs, as parsing malformed files may consume significant
resources, or cause panics.
*/
package elf

import (
	"bytes"
	"compress/zlib"
	"debug/dwarf"
	"encoding/binary"
	"errors"
	"fmt"
	"internal/saferio"
	"internal/zstd"
	"io"
	"math"
	"os"
	"strings"
	"unsafe"
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
	Entry      uint64
}

// A File represents an open ELF file.
type File struct {
	FileHeader
	Sections    []*Section
	Progs       []*Prog
	closer      io.Closer
	dynVers     []DynamicVersion
	dynVerNeeds []DynamicVersionNeed
	gnuVersym   []byte
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

	// FileSize is the size of this section in the file in bytes.
	// If a section is compressed, FileSize is the size of the
	// compressed data, while Size (above) is the size of the
	// uncompressed data.
	FileSize uint64
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
	//
	// ReaderAt may be nil if the section is not easily available
	// in a random-access form. For example, a compressed section
	// may have a nil ReaderAt.
	io.ReaderAt
	sr *io.SectionReader

	compressionType   CompressionType
	compressionOffset int64
}

// Data reads and returns the contents of the ELF section.
// Even if the section is stored compressed in the ELF file,
// Data returns uncompressed data.
//
// For an [SHT_NOBITS] section, Data always returns a non-nil error.
func (s *Section) Data() ([]byte, error) {
	return saferio.ReadData(s.Open(), s.Size)
}

// stringTable reads and returns the string table given by the
// specified link value.
func (f *File) stringTable(link uint32) ([]byte, error) {
	if link <= 0 || link >= uint32(len(f.Sections)) {
		return nil, errors.New("section has invalid string table link")
	}
	return f.Sections[link].Data()
}

// Open returns a new ReadSeeker reading the ELF section.
// Even if the section is stored compressed in the ELF file,
// the ReadSeeker reads uncompressed data.
//
// For an [SHT_NOBITS] section, all calls to the opened reader
// will return a non-nil error.
func (s *Section) Open() io.ReadSeeker {
	if s.Type == SHT_NOBITS {
		return io.NewSectionReader(&nobitsSectionReader{}, 0, int64(s.Size))
	}

	var zrd func(io.Reader) (io.ReadCloser, error)
	if s.Flags&SHF_COMPRESSED == 0 {

		if !strings.HasPrefix(s.Name, ".zdebug") {
			return io.NewSectionReader(s.sr, 0, 1<<63-1)
		}

		b := make([]byte, 12)
		n, _ := s.sr.ReadAt(b, 0)
		if n != 12 || string(b[:4]) != "ZLIB" {
			return io.NewSectionReader(s.sr, 0, 1<<63-1)
		}

		s.compressionOffset = 12
		s.compressionType = COMPRESS_ZLIB
		s.Size = binary.BigEndian.Uint64(b[4:12])
		zrd = zlib.NewReader

	} else if s.Flags&SHF_ALLOC != 0 {
		return errorReader{&FormatError{int64(s.Offset),
			"SHF_COMPRESSED applies only to non-allocable sections", s.compressionType}}
	}

	switch s.compressionType {
	case COMPRESS_ZLIB:
		zrd = zlib.NewReader
	case COMPRESS_ZSTD:
		zrd = func(r io.Reader) (io.ReadCloser, error) {
			return io.NopCloser(zstd.NewReader(r)), nil
		}
	}

	if zrd == nil {
		return errorReader{&FormatError{int64(s.Offset), "unknown compression type", s.compressionType}}
	}

	return &readSeekerFromReader{
		reset: func() (io.Reader, error) {
			fr := io.NewSectionReader(s.sr, s.compressionOffset, int64(s.FileSize)-s.compressionOffset)
			return zrd(fr)
		},
		size: int64(s.Size),
	}
}

// A ProgHeader represents a single ELF program header.
type ProgHeader struct {
	Type   ProgType
	Flags  ProgFlag
	Off    uint64
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

	// HasVersion reports whether the symbol has any version information.
	// This will only be true for the dynamic symbol table.
	HasVersion bool
	// VersionIndex is the symbol's version index.
	// Use the methods of the [VersionIndex] type to access it.
	// This field is only meaningful if HasVersion is true.
	VersionIndex VersionIndex

	Section     SectionIndex
	Value, Size uint64

	// These fields are present only for the dynamic symbol table.
	Version string
	Library string
}

/*
 * ELF reader
 */

type FormatError struct {
	off int64
	msg string
	val any
}

func (e *FormatError) Error() string {
	msg := e.msg
	if e.val != nil {
		msg += fmt.Sprintf(" '%v' ", e.val)
	}
	msg += fmt.Sprintf("in record at byte %#x", e.off)
	return msg
}

// Open opens the named file using [os.Open] and prepares it for use as an ELF binary.
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

// Close closes the [File].
// If the [File] was created using [NewFile] directly instead of [Open],
// Close has no effect.
func (f *File) Close() error {
	var err error
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

// NewFile creates a new [File] for accessing an ELF binary in an underlying reader.
// The ELF binary is expected to start at position 0 in the ReaderAt.
func NewFile(r io.ReaderAt) (*File, error) {
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
	var bo binary.ByteOrder
	switch f.Data {
	case ELFDATA2LSB:
		bo = binary.LittleEndian
	case ELFDATA2MSB:
		bo = binary.BigEndian
	default:
		return nil, &FormatError{0, "unknown ELF data encoding", f.Data}
	}
	f.ByteOrder = bo

	f.Version = Version(ident[EI_VERSION])
	if f.Version != EV_CURRENT {
		return nil, &FormatError{0, "unknown ELF version", f.Version}
	}

	f.OSABI = OSABI(ident[EI_OSABI])
	f.ABIVersion = ident[EI_ABIVERSION]

	// Read ELF file header
	var phoff int64
	var phentsize, phnum int
	var shoff int64
	var shentsize, shnum, shstrndx int
	switch f.Class {
	case ELFCLASS32:
		var hdr Header32
		data := make([]byte, unsafe.Sizeof(hdr))
		if _, err := sr.ReadAt(data, 0); err != nil {
			return nil, err
		}
		f.Type = Type(bo.Uint16(data[unsafe.Offsetof(hdr.Type):]))
		f.Machine = Machine(bo.Uint16(data[unsafe.Offsetof(hdr.Machine):]))
		f.Entry = uint64(bo.Uint32(data[unsafe.Offsetof(hdr.Entry):]))
		if v := Version(bo.Uint32(data[unsafe.Offsetof(hdr.Version):])); v != f.Version {
			return nil, &FormatError{0, "mismatched ELF version", v}
		}
		phoff = int64(bo.Uint32(data[unsafe.Offsetof(hdr.Phoff):]))
		phentsize = int(bo.Uint16(data[unsafe.Offsetof(hdr.Phentsize):]))
		phnum = int(bo.Uint16(data[unsafe.Offsetof(hdr.Phnum):]))
		shoff = int64(bo.Uint32(data[unsafe.Offsetof(hdr.Shoff):]))
		shentsize = int(bo.Uint16(data[unsafe.Offsetof(hdr.Shentsize):]))
		shnum = int(bo.Uint16(data[unsafe.Offsetof(hdr.Shnum):]))
		shstrndx = int(bo.Uint16(data[unsafe.Offsetof(hdr.Shstrndx):]))
	case ELFCLASS64:
		var hdr Header64
		data := make([]byte, unsafe.Sizeof(hdr))
		if _, err := sr.ReadAt(data, 0); err != nil {
			return nil, err
		}
		f.Type = Type(bo.Uint16(data[unsafe.Offsetof(hdr.Type):]))
		f.Machine = Machine(bo.Uint16(data[unsafe.Offsetof(hdr.Machine):]))
		f.Entry = bo.Uint64(data[unsafe.Offsetof(hdr.Entry):])
		if v := Version(bo.Uint32(data[unsafe.Offsetof(hdr.Version):])); v != f.Version {
			return nil, &FormatError{0, "mismatched ELF version", v}
		}
		phoff = int64(bo.Uint64(data[unsafe.Offsetof(hdr.Phoff):]))
		phentsize = int(bo.Uint16(data[unsafe.Offsetof(hdr.Phentsize):]))
		phnum = int(bo.Uint16(data[unsafe.Offsetof(hdr.Phnum):]))
		shoff = int64(bo.Uint64(data[unsafe.Offsetof(hdr.Shoff):]))
		shentsize = int(bo.Uint16(data[unsafe.Offsetof(hdr.Shentsize):]))
		shnum = int(bo.Uint16(data[unsafe.Offsetof(hdr.Shnum):]))
		shstrndx = int(bo.Uint16(data[unsafe.Offsetof(hdr.Shstrndx):]))
	}

	if shoff < 0 {
		return nil, &FormatError{0, "invalid shoff", shoff}
	}
	if phoff < 0 {
		return nil, &FormatError{0, "invalid phoff", phoff}
	}

	if shoff == 0 && shnum != 0 {
		return nil, &FormatError{0, "invalid ELF shnum for shoff=0", shnum}
	}

	if shnum > 0 && shstrndx >= shnum {
		return nil, &FormatError{0, "invalid ELF shstrndx", shstrndx}
	}

	var wantPhentsize, wantShentsize int
	switch f.Class {
	case ELFCLASS32:
		wantPhentsize = 8 * 4
		wantShentsize = 10 * 4
	case ELFCLASS64:
		wantPhentsize = 2*4 + 6*8
		wantShentsize = 4*4 + 6*8
	}
	if phnum > 0 && phentsize < wantPhentsize {
		return nil, &FormatError{0, "invalid ELF phentsize", phentsize}
	}

	// Read program headers
	f.Progs = make([]*Prog, phnum)
	phdata, err := saferio.ReadDataAt(sr, uint64(phnum)*uint64(phentsize), phoff)
	if err != nil {
		return nil, err
	}
	for i := 0; i < phnum; i++ {
		off := uintptr(i) * uintptr(phentsize)
		p := new(Prog)
		switch f.Class {
		case ELFCLASS32:
			var ph Prog32
			p.ProgHeader = ProgHeader{
				Type:   ProgType(bo.Uint32(phdata[off+unsafe.Offsetof(ph.Type):])),
				Flags:  ProgFlag(bo.Uint32(phdata[off+unsafe.Offsetof(ph.Flags):])),
				Off:    uint64(bo.Uint32(phdata[off+unsafe.Offsetof(ph.Off):])),
				Vaddr:  uint64(bo.Uint32(phdata[off+unsafe.Offsetof(ph.Vaddr):])),
				Paddr:  uint64(bo.Uint32(phdata[off+unsafe.Offsetof(ph.Paddr):])),
				Filesz: uint64(bo.Uint32(phdata[off+unsafe.Offsetof(ph.Filesz):])),
				Memsz:  uint64(bo.Uint32(phdata[off+unsafe.Offsetof(ph.Memsz):])),
				Align:  uint64(bo.Uint32(phdata[off+unsafe.Offsetof(ph.Align):])),
			}
		case ELFCLASS64:
			var ph Prog64
			p.ProgHeader = ProgHeader{
				Type:   ProgType(bo.Uint32(phdata[off+unsafe.Offsetof(ph.Type):])),
				Flags:  ProgFlag(bo.Uint32(phdata[off+unsafe.Offsetof(ph.Flags):])),
				Off:    bo.Uint64(phdata[off+unsafe.Offsetof(ph.Off):]),
				Vaddr:  bo.Uint64(phdata[off+unsafe.Offsetof(ph.Vaddr):]),
				Paddr:  bo.Uint64(phdata[off+unsafe.Offsetof(ph.Paddr):]),
				Filesz: bo.Uint64(phdata[off+unsafe.Offsetof(ph.Filesz):]),
				Memsz:  bo.Uint64(phdata[off+unsafe.Offsetof(ph.Memsz):]),
				Align:  bo.Uint64(phdata[off+unsafe.Offsetof(ph.Align):]),
			}
		}
		if int64(p.Off) < 0 {
			return nil, &FormatError{phoff + int64(off), "invalid program header offset", p.Off}
		}
		if int64(p.Filesz) < 0 {
			return nil, &FormatError{phoff + int64(off), "invalid program header file size", p.Filesz}
		}
		p.sr = io.NewSectionReader(r, int64(p.Off), int64(p.Filesz))
		p.ReaderAt = p.sr
		f.Progs[i] = p
	}

	// If the number of sections is greater than or equal to SHN_LORESERVE
	// (0xff00), shnum has the value zero and the actual number of section
	// header table entries is contained in the sh_size field of the section
	// header at index 0.
	if shoff > 0 && shnum == 0 {
		var typ, link uint32
		sr.Seek(shoff, io.SeekStart)
		switch f.Class {
		case ELFCLASS32:
			sh := new(Section32)
			if err := binary.Read(sr, bo, sh); err != nil {
				return nil, err
			}
			shnum = int(sh.Size)
			typ = sh.Type
			link = sh.Link
		case ELFCLASS64:
			sh := new(Section64)
			if err := binary.Read(sr, bo, sh); err != nil {
				return nil, err
			}
			shnum = int(sh.Size)
			typ = sh.Type
			link = sh.Link
		}
		if SectionType(typ) != SHT_NULL {
			return nil, &FormatError{shoff, "invalid type of the initial section", SectionType(typ)}
		}

		if shnum < int(SHN_LORESERVE) {
			return nil, &FormatError{shoff, "invalid ELF shnum contained in sh_size", shnum}
		}

		// If the section name string table section index is greater than or
		// equal to SHN_LORESERVE (0xff00), this member has the value
		// SHN_XINDEX (0xffff) and the actual index of the section name
		// string table section is contained in the sh_link field of the
		// section header at index 0.
		if shstrndx == int(SHN_XINDEX) {
			shstrndx = int(link)
			if shstrndx < int(SHN_LORESERVE) {
				return nil, &FormatError{shoff, "invalid ELF shstrndx contained in sh_link", shstrndx}
			}
		}
	}

	if shnum > 0 && shentsize < wantShentsize {
		return nil, &FormatError{0, "invalid ELF shentsize", shentsize}
	}

	// Read section headers
	c := saferio.SliceCap[Section](uint64(shnum))
	if c < 0 {
		return nil, &FormatError{0, "too many sections", shnum}
	}
	if shnum > 0 && ((1<<64)-1)/uint64(shnum) < uint64(shentsize) {
		return nil, &FormatError{0, "section header overflow", shnum}
	}
	f.Sections = make([]*Section, 0, c)
	names := make([]uint32, 0, c)
	shdata, err := saferio.ReadDataAt(sr, uint64(shnum)*uint64(shentsize), shoff)
	if err != nil {
		return nil, err
	}
	for i := 0; i < shnum; i++ {
		off := uintptr(i) * uintptr(shentsize)
		s := new(Section)
		switch f.Class {
		case ELFCLASS32:
			var sh Section32
			names = append(names, bo.Uint32(shdata[off+unsafe.Offsetof(sh.Name):]))
			s.SectionHeader = SectionHeader{
				Type:      SectionType(bo.Uint32(shdata[off+unsafe.Offsetof(sh.Type):])),
				Flags:     SectionFlag(bo.Uint32(shdata[off+unsafe.Offsetof(sh.Flags):])),
				Addr:      uint64(bo.Uint32(shdata[off+unsafe.Offsetof(sh.Addr):])),
				Offset:    uint64(bo.Uint32(shdata[off+unsafe.Offsetof(sh.Off):])),
				FileSize:  uint64(bo.Uint32(shdata[off+unsafe.Offsetof(sh.Size):])),
				Link:      bo.Uint32(shdata[off+unsafe.Offsetof(sh.Link):]),
				Info:      bo.Uint32(shdata[off+unsafe.Offsetof(sh.Info):]),
				Addralign: uint64(bo.Uint32(shdata[off+unsafe.Offsetof(sh.Addralign):])),
				Entsize:   uint64(bo.Uint32(shdata[off+unsafe.Offsetof(sh.Entsize):])),
			}
		case ELFCLASS64:
			var sh Section64
			names = append(names, bo.Uint32(shdata[off+unsafe.Offsetof(sh.Name):]))
			s.SectionHeader = SectionHeader{
				Type:      SectionType(bo.Uint32(shdata[off+unsafe.Offsetof(sh.Type):])),
				Flags:     SectionFlag(bo.Uint64(shdata[off+unsafe.Offsetof(sh.Flags):])),
				Offset:    bo.Uint64(shdata[off+unsafe.Offsetof(sh.Off):]),
				FileSize:  bo.Uint64(shdata[off+unsafe.Offsetof(sh.Size):]),
				Addr:      bo.Uint64(shdata[off+unsafe.Offsetof(sh.Addr):]),
				Link:      bo.Uint32(shdata[off+unsafe.Offsetof(sh.Link):]),
				Info:      bo.Uint32(shdata[off+unsafe.Offsetof(sh.Info):]),
				Addralign: bo.Uint64(shdata[off+unsafe.Offsetof(sh.Addralign):]),
				Entsize:   bo.Uint64(shdata[off+unsafe.Offsetof(sh.Entsize):]),
			}
		}
		if int64(s.Offset) < 0 {
			return nil, &FormatError{shoff + int64(off), "invalid section offset", int64(s.Offset)}
		}
		if int64(s.FileSize) < 0 {
			return nil, &FormatError{shoff + int64(off), "invalid section size", int64(s.FileSize)}
		}
		s.sr = io.NewSectionReader(r, int64(s.Offset), int64(s.FileSize))

		if s.Flags&SHF_COMPRESSED == 0 {
			s.ReaderAt = s.sr
			s.Size = s.FileSize
		} else {
			// Read the compression header.
			switch f.Class {
			case ELFCLASS32:
				var ch Chdr32
				chdata := make([]byte, unsafe.Sizeof(ch))
				if _, err := s.sr.ReadAt(chdata, 0); err != nil {
					return nil, err
				}
				s.compressionType = CompressionType(bo.Uint32(chdata[unsafe.Offsetof(ch.Type):]))
				s.Size = uint64(bo.Uint32(chdata[unsafe.Offsetof(ch.Size):]))
				s.Addralign = uint64(bo.Uint32(chdata[unsafe.Offsetof(ch.Addralign):]))
				s.compressionOffset = int64(unsafe.Sizeof(ch))
			case ELFCLASS64:
				var ch Chdr64
				chdata := make([]byte, unsafe.Sizeof(ch))
				if _, err := s.sr.ReadAt(chdata, 0); err != nil {
					return nil, err
				}
				s.compressionType = CompressionType(bo.Uint32(chdata[unsafe.Offsetof(ch.Type):]))
				s.Size = bo.Uint64(chdata[unsafe.Offsetof(ch.Size):])
				s.Addralign = bo.Uint64(chdata[unsafe.Offsetof(ch.Addralign):])
				s.compressionOffset = int64(unsafe.Sizeof(ch))
			}
		}

		f.Sections = append(f.Sections, s)
	}

	if len(f.Sections) == 0 {
		return f, nil
	}

	// Load section header string table.
	if shstrndx == 0 {
		// If the file has no section name string table,
		// shstrndx holds the value SHN_UNDEF (0).
		return f, nil
	}
	shstr := f.Sections[shstrndx]
	if shstr.Type != SHT_STRTAB {
		return nil, &FormatError{shoff + int64(shstrndx*shentsize), "invalid ELF section name string table type", shstr.Type}
	}
	shstrtab, err := shstr.Data()
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
// with the given type, along with the associated string table.
func (f *File) getSymbols(typ SectionType) ([]Symbol, []byte, error) {
	switch f.Class {
	case ELFCLASS64:
		return f.getSymbols64(typ)

	case ELFCLASS32:
		return f.getSymbols32(typ)
	}

	return nil, nil, errors.New("not implemented")
}

// ErrNoSymbols is returned by [File.Symbols] and [File.DynamicSymbols]
// if there is no such section in the File.
var ErrNoSymbols = errors.New("no symbol section")

func (f *File) getSymbols32(typ SectionType) ([]Symbol, []byte, error) {
	symtabSection := f.SectionByType(typ)
	if symtabSection == nil {
		return nil, nil, ErrNoSymbols
	}

	data, err := symtabSection.Data()
	if err != nil {
		return nil, nil, fmt.Errorf("cannot load symbol section: %w", err)
	}
	if len(data) == 0 {
		return nil, nil, ErrNoSymbols
	}
	if len(data)%Sym32Size != 0 {
		return nil, nil, errors.New("length of symbol section is not a multiple of SymSize")
	}

	strdata, err := f.stringTable(symtabSection.Link)
	if err != nil {
		return nil, nil, fmt.Errorf("cannot load string table section: %w", err)
	}

	// The first entry is all zeros.
	data = data[Sym32Size:]

	symbols := make([]Symbol, len(data)/Sym32Size)

	i := 0
	var sym Sym32
	for len(data) > 0 {
		sym.Name = f.ByteOrder.Uint32(data[0:4])
		sym.Value = f.ByteOrder.Uint32(data[4:8])
		sym.Size = f.ByteOrder.Uint32(data[8:12])
		sym.Info = data[12]
		sym.Other = data[13]
		sym.Shndx = f.ByteOrder.Uint16(data[14:16])
		str, _ := getString(strdata, int(sym.Name))
		symbols[i].Name = str
		symbols[i].Info = sym.Info
		symbols[i].Other = sym.Other
		symbols[i].Section = SectionIndex(sym.Shndx)
		symbols[i].Value = uint64(sym.Value)
		symbols[i].Size = uint64(sym.Size)
		i++
		data = data[Sym32Size:]
	}

	return symbols, strdata, nil
}

func (f *File) getSymbols64(typ SectionType) ([]Symbol, []byte, error) {
	symtabSection := f.SectionByType(typ)
	if symtabSection == nil {
		return nil, nil, ErrNoSymbols
	}

	data, err := symtabSection.Data()
	if err != nil {
		return nil, nil, fmt.Errorf("cannot load symbol section: %w", err)
	}
	if len(data) == 0 {
		return nil, nil, ErrNoSymbols
	}
	if len(data)%Sym64Size != 0 {
		return nil, nil, errors.New("length of symbol section is not a multiple of Sym64Size")
	}

	strdata, err := f.stringTable(symtabSection.Link)
	if err != nil {
		return nil, nil, fmt.Errorf("cannot load string table section: %w", err)
	}

	// The first entry is all zeros.
	data = data[Sym64Size:]

	symbols := make([]Symbol, len(data)/Sym64Size)

	i := 0
	var sym Sym64
	for len(data) > 0 {
		sym.Name = f.ByteOrder.Uint32(data[0:4])
		sym.Info = data[4]
		sym.Other = data[5]
		sym.Shndx = f.ByteOrder.Uint16(data[6:8])
		sym.Value = f.ByteOrder.Uint64(data[8:16])
		sym.Size = f.ByteOrder.Uint64(data[16:24])
		str, _ := getString(strdata, int(sym.Name))
		symbols[i].Name = str
		symbols[i].Info = sym.Info
		symbols[i].Other = sym.Other
		symbols[i].Section = SectionIndex(sym.Shndx)
		symbols[i].Value = sym.Value
		symbols[i].Size = sym.Size
		i++
		data = data[Sym64Size:]
	}

	return symbols, strdata, nil
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
// in REL or RELA format.
func (f *File) applyRelocations(dst []byte, rels []byte) error {
	switch {
	case f.Class == ELFCLASS64 && f.Machine == EM_X86_64:
		return f.applyRelocationsAMD64(dst, rels)
	case f.Class == ELFCLASS32 && f.Machine == EM_386:
		return f.applyRelocations386(dst, rels)
	case f.Class == ELFCLASS32 && f.Machine == EM_ARM:
		return f.applyRelocationsARM(dst, rels)
	case f.Class == ELFCLASS64 && f.Machine == EM_AARCH64:
		return f.applyRelocationsARM64(dst, rels)
	case f.Class == ELFCLASS32 && f.Machine == EM_PPC:
		return f.applyRelocationsPPC(dst, rels)
	case f.Class == ELFCLASS64 && f.Machine == EM_PPC64:
		return f.applyRelocationsPPC64(dst, rels)
	case f.Class == ELFCLASS32 && f.Machine == EM_MIPS:
		return f.applyRelocationsMIPS(dst, rels)
	case f.Class == ELFCLASS64 && f.Machine == EM_MIPS:
		return f.applyRelocationsMIPS64(dst, rels)
	case f.Class == ELFCLASS64 && f.Machine == EM_LOONGARCH:
		return f.applyRelocationsLOONG64(dst, rels)
	case f.Class == ELFCLASS64 && f.Machine == EM_RISCV:
		return f.applyRelocationsRISCV64(dst, rels)
	case f.Class == ELFCLASS64 && f.Machine == EM_S390:
		return f.applyRelocationss390x(dst, rels)
	case f.Class == ELFCLASS64 && f.Machine == EM_SPARCV9:
		return f.applyRelocationsSPARC64(dst, rels)
	default:
		return errors.New("applyRelocations: not implemented")
	}
}

// canApplyRelocation reports whether we should try to apply a
// relocation to a DWARF data section, given a pointer to the symbol
// targeted by the relocation.
// Most relocations in DWARF data tend to be section-relative, but
// some target non-section symbols (for example, low_PC attrs on
// subprogram or compilation unit DIEs that target function symbols).
func canApplyRelocation(sym *Symbol) bool {
	return sym.Section != SHN_UNDEF && sym.Section < SHN_LORESERVE
}

func (f *File) applyRelocationsAMD64(dst []byte, rels []byte) error {
	// 24 is the size of Rela64.
	if len(rels)%24 != 0 {
		return errors.New("length of relocation section is not a multiple of 24")
	}

	symbols, _, err := f.getSymbols(SHT_SYMTAB)
	if err != nil {
		return err
	}

	b := bytes.NewReader(rels)
	var rela Rela64

	for b.Len() > 0 {
		binary.Read(b, f.ByteOrder, &rela)
		symNo := rela.Info >> 32
		t := R_X86_64(rela.Info & 0xffff)

		if symNo == 0 || symNo > uint64(len(symbols)) {
			continue
		}
		sym := &symbols[symNo-1]
		if !canApplyRelocation(sym) {
			continue
		}

		// There are relocations, so this must be a normal
		// object file.  The code below handles only basic relocations
		// of the form S + A (symbol plus addend).

		switch t {
		case R_X86_64_64:
			putUint(f.ByteOrder, dst, rela.Off, 8, sym.Value, rela.Addend, false)
		case R_X86_64_32:
			putUint(f.ByteOrder, dst, rela.Off, 4, sym.Value, rela.Addend, false)
		}
	}

	return nil
}

func (f *File) applyRelocations386(dst []byte, rels []byte) error {
	// 8 is the size of Rel32.
	if len(rels)%8 != 0 {
		return errors.New("length of relocation section is not a multiple of 8")
	}

	symbols, _, err := f.getSymbols(SHT_SYMTAB)
	if err != nil {
		return err
	}

	b := bytes.NewReader(rels)
	var rel Rel32

	for b.Len() > 0 {
		binary.Read(b, f.ByteOrder, &rel)
		symNo := rel.Info >> 8
		t := R_386(rel.Info & 0xff)

		if symNo == 0 || symNo > uint32(len(symbols)) {
			continue
		}
		sym := &symbols[symNo-1]

		if t == R_386_32 {
			putUint(f.ByteOrder, dst, uint64(rel.Off), 4, sym.Value, 0, true)
		}
	}

	return nil
}

func (f *File) applyRelocationsARM(dst []byte, rels []byte) error {
	// 8 is the size of Rel32.
	if len(rels)%8 != 0 {
		return errors.New("length of relocation section is not a multiple of 8")
	}

	symbols, _, err := f.getSymbols(SHT_SYMTAB)
	if err != nil {
		return err
	}

	b := bytes.NewReader(rels)
	var rel Rel32

	for b.Len() > 0 {
		binary.Read(b, f.ByteOrder, &rel)
		symNo := rel.Info >> 8
		t := R_ARM(rel.Info & 0xff)

		if symNo == 0 || symNo > uint32(len(symbols)) {
			continue
		}
		sym := &symbols[symNo-1]

		switch t {
		case R_ARM_ABS32:
			putUint(f.ByteOrder, dst, uint64(rel.Off), 4, sym.Value, 0, true)
		}
	}

	return nil
}

func (f *File) applyRelocationsARM64(dst []byte, rels []byte) error {
	// 24 is the size of Rela64.
	if len(rels)%24 != 0 {
		return errors.New("length of relocation section is not a multiple of 24")
	}

	symbols, _, err := f.getSymbols(SHT_SYMTAB)
	if err != nil {
		return err
	}

	b := bytes.NewReader(rels)
	var rela Rela64

	for b.Len() > 0 {
		binary.Read(b, f.ByteOrder, &rela)
		symNo := rela.Info >> 32
		t := R_AARCH64(rela.Info & 0xffff)

		if symNo == 0 || symNo > uint64(len(symbols)) {
			continue
		}
		sym := &symbols[symNo-1]
		if !canApplyRelocation(sym) {
			continue
		}

		// There are relocations, so this must be a normal
		// object file.  The code below handles only basic relocations
		// of the form S + A (symbol plus addend).

		switch t {
		case R_AARCH64_ABS64:
			putUint(f.ByteOrder, dst, rela.Off, 8, sym.Value, rela.Addend, false)
		case R_AARCH64_ABS32:
			putUint(f.ByteOrder, dst, rela.Off, 4, sym.Value, rela.Addend, false)
		}
	}

	return nil
}

func (f *File) applyRelocationsPPC(dst []byte, rels []byte) error {
	// 12 is the size of Rela32.
	if len(rels)%12 != 0 {
		return errors.New("length of relocation section is not a multiple of 12")
	}

	symbols, _, err := f.getSymbols(SHT_SYMTAB)
	if err != nil {
		return err
	}

	b := bytes.NewReader(rels)
	var rela Rela32

	for b.Len() > 0 {
		binary.Read(b, f.ByteOrder, &rela)
		symNo := rela.Info >> 8
		t := R_PPC(rela.Info & 0xff)

		if symNo == 0 || symNo > uint32(len(symbols)) {
			continue
		}
		sym := &symbols[symNo-1]
		if !canApplyRelocation(sym) {
			continue
		}

		switch t {
		case R_PPC_ADDR32:
			putUint(f.ByteOrder, dst, uint64(rela.Off), 4, sym.Value, 0, false)
		}
	}

	return nil
}

func (f *File) applyRelocationsPPC64(dst []byte, rels []byte) error {
	// 24 is the size of Rela64.
	if len(rels)%24 != 0 {
		return errors.New("length of relocation section is not a multiple of 24")
	}

	symbols, _, err := f.getSymbols(SHT_SYMTAB)
	if err != nil {
		return err
	}

	b := bytes.NewReader(rels)
	var rela Rela64

	for b.Len() > 0 {
		binary.Read(b, f.ByteOrder, &rela)
		symNo := rela.Info >> 32
		t := R_PPC64(rela.Info & 0xffff)

		if symNo == 0 || symNo > uint64(len(symbols)) {
			continue
		}
		sym := &symbols[symNo-1]
		if !canApplyRelocation(sym) {
			continue
		}

		switch t {
		case R_PPC64_ADDR64:
			putUint(f.ByteOrder, dst, rela.Off, 8, sym.Value, rela.Addend, false)
		case R_PPC64_ADDR32:
			putUint(f.ByteOrder, dst, rela.Off, 4, sym.Value, rela.Addend, false)
		}
	}

	return nil
}

func (f *File) applyRelocationsMIPS(dst []byte, rels []byte) error {
	// 8 is the size of Rel32.
	if len(rels)%8 != 0 {
		return errors.New("length of relocation section is not a multiple of 8")
	}

	symbols, _, err := f.getSymbols(SHT_SYMTAB)
	if err != nil {
		return err
	}

	b := bytes.NewReader(rels)
	var rel Rel32

	for b.Len() > 0 {
		binary.Read(b, f.ByteOrder, &rel)
		symNo := rel.Info >> 8
		t := R_MIPS(rel.Info & 0xff)

		if symNo == 0 || symNo > uint32(len(symbols)) {
			continue
		}
		sym := &symbols[symNo-1]

		switch t {
		case R_MIPS_32:
			putUint(f.ByteOrder, dst, uint64(rel.Off), 4, sym.Value, 0, true)
		}
	}

	return nil
}

func (f *File) applyRelocationsMIPS64(dst []byte, rels []byte) error {
	// 24 is the size of Rela64.
	if len(rels)%24 != 0 {
		return errors.New("length of relocation section is not a multiple of 24")
	}

	symbols, _, err := f.getSymbols(SHT_SYMTAB)
	if err != nil {
		return err
	}

	b := bytes.NewReader(rels)
	var rela Rela64

	for b.Len() > 0 {
		binary.Read(b, f.ByteOrder, &rela)
		var symNo uint64
		var t R_MIPS
		if f.ByteOrder == binary.BigEndian {
			symNo = rela.Info >> 32
			t = R_MIPS(rela.Info & 0xff)
		} else {
			symNo = rela.Info & 0xffffffff
			t = R_MIPS(rela.Info >> 56)
		}

		if symNo == 0 || symNo > uint64(len(symbols)) {
			continue
		}
		sym := &symbols[symNo-1]
		if !canApplyRelocation(sym) {
			continue
		}

		switch t {
		case R_MIPS_64:
			putUint(f.ByteOrder, dst, rela.Off, 8, sym.Value, rela.Addend, false)
		case R_MIPS_32:
			putUint(f.ByteOrder, dst, rela.Off, 4, sym.Value, rela.Addend, false)
		}
	}

	return nil
}

func (f *File) applyRelocationsLOONG64(dst []byte, rels []byte) error {
	// 24 is the size of Rela64.
	if len(rels)%24 != 0 {
		return errors.New("length of relocation section is not a multiple of 24")
	}

	symbols, _, err := f.getSymbols(SHT_SYMTAB)
	if err != nil {
		return err
	}

	b := bytes.NewReader(rels)
	var rela Rela64

	for b.Len() > 0 {
		binary.Read(b, f.ByteOrder, &rela)
		var symNo uint64
		var t R_LARCH
		symNo = rela.Info >> 32
		t = R_LARCH(rela.Info & 0xffff)

		if symNo == 0 || symNo > uint64(len(symbols)) {
			continue
		}
		sym := &symbols[symNo-1]
		if !canApplyRelocation(sym) {
			continue
		}

		switch t {
		case R_LARCH_64:
			putUint(f.ByteOrder, dst, rela.Off, 8, sym.Value, rela.Addend, false)
		case R_LARCH_32:
			putUint(f.ByteOrder, dst, rela.Off, 4, sym.Value, rela.Addend, false)
		}
	}

	return nil
}

func (f *File) applyRelocationsRISCV64(dst []byte, rels []byte) error {
	// 24 is the size of Rela64.
	if len(rels)%24 != 0 {
		return errors.New("length of relocation section is not a multiple of 24")
	}

	symbols, _, err := f.getSymbols(SHT_SYMTAB)
	if err != nil {
		return err
	}

	b := bytes.NewReader(rels)
	var rela Rela64

	for b.Len() > 0 {
		binary.Read(b, f.ByteOrder, &rela)
		symNo := rela.Info >> 32
		t := R_RISCV(rela.Info & 0xffff)

		if symNo == 0 || symNo > uint64(len(symbols)) {
			continue
		}
		sym := &symbols[symNo-1]
		if !canApplyRelocation(sym) {
			continue
		}

		switch t {
		case R_RISCV_64:
			putUint(f.ByteOrder, dst, rela.Off, 8, sym.Value, rela.Addend, false)
		case R_RISCV_32:
			putUint(f.ByteOrder, dst, rela.Off, 4, sym.Value, rela.Addend, false)
		}
	}

	return nil
}

func (f *File) applyRelocationss390x(dst []byte, rels []byte) error {
	// 24 is the size of Rela64.
	if len(rels)%24 != 0 {
		return errors.New("length of relocation section is not a multiple of 24")
	}

	symbols, _, err := f.getSymbols(SHT_SYMTAB)
	if err != nil {
		return err
	}

	b := bytes.NewReader(rels)
	var rela Rela64

	for b.Len() > 0 {
		binary.Read(b, f.ByteOrder, &rela)
		symNo := rela.Info >> 32
		t := R_390(rela.Info & 0xffff)

		if symNo == 0 || symNo > uint64(len(symbols)) {
			continue
		}
		sym := &symbols[symNo-1]
		if !canApplyRelocation(sym) {
			continue
		}

		switch t {
		case R_390_64:
			putUint(f.ByteOrder, dst, rela.Off, 8, sym.Value, rela.Addend, false)
		case R_390_32:
			putUint(f.ByteOrder, dst, rela.Off, 4, sym.Value, rela.Addend, false)
		}
	}

	return nil
}

func (f *File) applyRelocationsSPARC64(dst []byte, rels []byte) error {
	// 24 is the size of Rela64.
	if len(rels)%24 != 0 {
		return errors.New("length of relocation section is not a multiple of 24")
	}

	symbols, _, err := f.getSymbols(SHT_SYMTAB)
	if err != nil {
		return err
	}

	b := bytes.NewReader(rels)
	var rela Rela64

	for b.Len() > 0 {
		binary.Read(b, f.ByteOrder, &rela)
		symNo := rela.Info >> 32
		t := R_SPARC(rela.Info & 0xff)

		if symNo == 0 || symNo > uint64(len(symbols)) {
			continue
		}
		sym := &symbols[symNo-1]
		if !canApplyRelocation(sym) {
			continue
		}

		switch t {
		case R_SPARC_64, R_SPARC_UA64:
			putUint(f.ByteOrder, dst, rela.Off, 8, sym.Value, rela.Addend, false)

		case R_SPARC_32, R_SPARC_UA32:
			putUint(f.ByteOrder, dst, rela.Off, 4, sym.Value, rela.Addend, false)
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
	// sectionData gets the data for s, checks its size, and
	// applies any applicable relations.
	sectionData := func(i int, s *Section) ([]byte, error) {
		b, err := s.Data()
		if err != nil && uint64(len(b)) < s.Size {
			return nil, err
		}

		if f.Type == ET_EXEC {
			// Do not apply relocations to DWARF sections for ET_EXEC binaries.
			// Relocations should already be applied, and .rela sections may
			// contain incorrect data.
			return b, nil
		}

		for _, r := range f.Sections {
			if r.Type != SHT_RELA && r.Type != SHT_REL {
				continue
			}
			if int(r.Info) != i {
				continue
			}
			rd, err := r.Data()
			if err != nil {
				return nil, err
			}
			err = f.applyRelocations(b, rd)
			if err != nil {
				return nil, err
			}
		}
		return b, nil
	}

	// There are many DWARF sections, but these are the ones
	// the debug/dwarf package started with.
	var dat = map[string][]byte{"abbrev": nil, "info": nil, "str": nil, "line": nil, "ranges": nil}
	for i, s := range f.Sections {
		suffix := dwarfSuffix(s)
		if suffix == "" {
			continue
		}
		if _, ok := dat[suffix]; !ok {
			continue
		}
		b, err := sectionData(i, s)
		if err != nil {
			return nil, err
		}
		dat[suffix] = b
	}

	d, err := dwarf.New(dat["abbrev"], nil, nil, dat["info"], dat["line"], nil, dat["ranges"], dat["str"])
	if err != nil {
		return nil, err
	}

	// Look for DWARF4 .debug_types sections and DWARF5 sections.
	for i, s := range f.Sections {
		suffix := dwarfSuffix(s)
		if suffix == "" {
			continue
		}
		if _, ok := dat[suffix]; ok {
			// Already handled.
			continue
		}

		b, err := sectionData(i, s)
		if err != nil {
			return nil, err
		}

		if suffix == "types" {
			if err := d.AddTypes(fmt.Sprintf("types-%d", i), b); err != nil {
				return nil, err
			}
		} else {
			if err := d.AddSection(".debug_"+suffix, b); err != nil {
				return nil, err
			}
		}
	}

	return d, nil
}

// Symbols returns the symbol table for f. The symbols will be listed in the order
// they appear in f.
//
// For compatibility with Go 1.0, Symbols omits the null symbol at index 0.
// After retrieving the symbols as symtab, an externally supplied index x
// corresponds to symtab[x-1], not symtab[x].
func (f *File) Symbols() ([]Symbol, error) {
	sym, _, err := f.getSymbols(SHT_SYMTAB)
	return sym, err
}

// DynamicSymbols returns the dynamic symbol table for f. The symbols
// will be listed in the order they appear in f.
//
// If f has a symbol version table, the returned [File.Symbols] will have
// initialized Version and Library fields.
//
// For compatibility with [File.Symbols], [File.DynamicSymbols] omits the null symbol at index 0.
// After retrieving the symbols as symtab, an externally supplied index x
// corresponds to symtab[x-1], not symtab[x].
func (f *File) DynamicSymbols() ([]Symbol, error) {
	sym, str, err := f.getSymbols(SHT_DYNSYM)
	if err != nil {
		return nil, err
	}
	hasVersions, err := f.gnuVersionInit(str)
	if err != nil {
		return nil, err
	}
	if hasVersions {
		for i := range sym {
			sym[i].HasVersion, sym[i].VersionIndex, sym[i].Version, sym[i].Library = f.gnuVersion(i)
		}
	}
	return sym, nil
}

type ImportedSymbol struct {
	Name    string
	Version string
	Library string
}

// ImportedSymbols returns the names of all symbols
// referred to by the binary f that are expected to be
// satisfied by other libraries at dynamic load time.
// It does not return weak symbols.
func (f *File) ImportedSymbols() ([]ImportedSymbol, error) {
	sym, str, err := f.getSymbols(SHT_DYNSYM)
	if err != nil {
		return nil, err
	}
	if _, err := f.gnuVersionInit(str); err != nil {
		return nil, err
	}
	var all []ImportedSymbol
	for i, s := range sym {
		if ST_BIND(s.Info) == STB_GLOBAL && s.Section == SHN_UNDEF {
			all = append(all, ImportedSymbol{Name: s.Name})
			sym := &all[len(all)-1]
			_, _, sym.Version, sym.Library = f.gnuVersion(i)
		}
	}
	return all, nil
}

// VersionIndex is the type of a [Symbol] version index.
type VersionIndex uint16

// IsHidden reports whether the symbol is hidden within the version.
// This means that the symbol can only be seen by specifying the exact version.
func (vi VersionIndex) IsHidden() bool {
	return vi&0x8000 != 0
}

// Index returns the version index.
// If this is the value 0, it means that the symbol is local,
// and is not visible externally.
// If this is the value 1, it means that the symbol is in the base version,
// and has no specific version; it may or may not match a
// [DynamicVersion.Index] in the slice returned by [File.DynamicVersions].
// Other values will match either [DynamicVersion.Index]
// in the slice returned by [File.DynamicVersions],
// or [DynamicVersionDep.Index] in the Needs field
// of the elements of the slice returned by [File.DynamicVersionNeeds].
// In general, a defined symbol will have an index referring
// to DynamicVersions, and an undefined symbol will have an index
// referring to some version in DynamicVersionNeeds.
func (vi VersionIndex) Index() uint16 {
	return uint16(vi & 0x7fff)
}

// DynamicVersion is a version defined by a dynamic object.
// This describes entries in the ELF SHT_GNU_verdef section.
// We assume that the vd_version field is 1.
// Note that the name of the version appears here;
// it is not in the first Deps entry as it is in the ELF file.
type DynamicVersion struct {
	Name  string // Name of version defined by this index.
	Index uint16 // Version index.
	Flags DynamicVersionFlag
	Deps  []string // Names of versions that this version depends upon.
}

// DynamicVersionNeed describes a shared library needed by a dynamic object,
// with a list of the versions needed from that shared library.
// This describes entries in the ELF SHT_GNU_verneed section.
// We assume that the vn_version field is 1.
type DynamicVersionNeed struct {
	Name  string              // Shared library name.
	Needs []DynamicVersionDep // Dependencies.
}

// DynamicVersionDep is a version needed from some shared library.
type DynamicVersionDep struct {
	Flags DynamicVersionFlag
	Index uint16 // Version index.
	Dep   string // Name of required version.
}

// dynamicVersions returns version information for a dynamic object.
func (f *File) dynamicVersions(str []byte) error {
	if f.dynVers != nil {
		// Already initialized.
		return nil
	}

	// Accumulate verdef information.
	vd := f.SectionByType(SHT_GNU_VERDEF)
	if vd == nil {
		return nil
	}
	d, _ := vd.Data()

	var dynVers []DynamicVersion
	i := 0
	for {
		if i+20 > len(d) {
			break
		}
		version := f.ByteOrder.Uint16(d[i : i+2])
		if version != 1 {
			return &FormatError{int64(vd.Offset + uint64(i)), "unexpected dynamic version", version}
		}
		flags := DynamicVersionFlag(f.ByteOrder.Uint16(d[i+2 : i+4]))
		ndx := f.ByteOrder.Uint16(d[i+4 : i+6])
		cnt := f.ByteOrder.Uint16(d[i+6 : i+8])
		aux := f.ByteOrder.Uint32(d[i+12 : i+16])
		next := f.ByteOrder.Uint32(d[i+16 : i+20])

		if cnt == 0 {
			return &FormatError{int64(vd.Offset + uint64(i)), "dynamic version has no name", nil}
		}

		var name string
		var depName string
		var deps []string
		j := i + int(aux)
		for c := 0; c < int(cnt); c++ {
			if j+8 > len(d) {
				break
			}
			vname := f.ByteOrder.Uint32(d[j : j+4])
			vnext := f.ByteOrder.Uint32(d[j+4 : j+8])
			depName, _ = getString(str, int(vname))

			if c == 0 {
				name = depName
			} else {
				deps = append(deps, depName)
			}

			j += int(vnext)
		}

		dynVers = append(dynVers, DynamicVersion{
			Name:  name,
			Index: ndx,
			Flags: flags,
			Deps:  deps,
		})

		if next == 0 {
			break
		}
		i += int(next)
	}

	f.dynVers = dynVers

	return nil
}

// DynamicVersions returns version information for a dynamic object.
func (f *File) DynamicVersions() ([]DynamicVersion, error) {
	if f.dynVers == nil {
		_, str, err := f.getSymbols(SHT_DYNSYM)
		if err != nil {
			return nil, err
		}
		hasVersions, err := f.gnuVersionInit(str)
		if err != nil {
			return nil, err
		}
		if !hasVersions {
			return nil, errors.New("DynamicVersions: missing version table")
		}
	}

	return f.dynVers, nil
}

// dynamicVersionNeeds returns version dependencies for a dynamic object.
func (f *File) dynamicVersionNeeds(str []byte) error {
	if f.dynVerNeeds != nil {
		// Already initialized.
		return nil
	}

	// Accumulate verneed information.
	vn := f.SectionByType(SHT_GNU_VERNEED)
	if vn == nil {
		return nil
	}
	d, _ := vn.Data()

	var dynVerNeeds []DynamicVersionNeed
	i := 0
	for {
		if i+16 > len(d) {
			break
		}
		vers := f.ByteOrder.Uint16(d[i : i+2])
		if vers != 1 {
			return &FormatError{int64(vn.Offset + uint64(i)), "unexpected dynamic need version", vers}
		}
		cnt := f.ByteOrder.Uint16(d[i+2 : i+4])
		fileoff := f.ByteOrder.Uint32(d[i+4 : i+8])
		aux := f.ByteOrder.Uint32(d[i+8 : i+12])
		next := f.ByteOrder.Uint32(d[i+12 : i+16])
		file, _ := getString(str, int(fileoff))

		var deps []DynamicVersionDep
		j := i + int(aux)
		for c := 0; c < int(cnt); c++ {
			if j+16 > len(d) {
				break
			}
			flags := DynamicVersionFlag(f.ByteOrder.Uint16(d[j+4 : j+6]))
			index := f.ByteOrder.Uint16(d[j+6 : j+8])
			nameoff := f.ByteOrder.Uint32(d[j+8 : j+12])
			next := f.ByteOrder.Uint32(d[j+12 : j+16])
			depName, _ := getString(str, int(nameoff))

			deps = append(deps, DynamicVersionDep{
				Flags: flags,
				Index: index,
				Dep:   depName,
			})

			if next == 0 {
				break
			}
			j += int(next)
		}

		dynVerNeeds = append(dynVerNeeds, DynamicVersionNeed{
			Name:  file,
			Needs: deps,
		})

		if next == 0 {
			break
		}
		i += int(next)
	}

	f.dynVerNeeds = dynVerNeeds

	return nil
}

// DynamicVersionNeeds returns version dependencies for a dynamic object.
func (f *File) DynamicVersionNeeds() ([]DynamicVersionNeed, error) {
	if f.dynVerNeeds == nil {
		_, str, err := f.getSymbols(SHT_DYNSYM)
		if err != nil {
			return nil, err
		}
		hasVersions, err := f.gnuVersionInit(str)
		if err != nil {
			return nil, err
		}
		if !hasVersions {
			return nil, errors.New("DynamicVersionNeeds: missing version table")
		}
	}

	return f.dynVerNeeds, nil
}

// gnuVersionInit parses the GNU version tables
// for use by calls to gnuVersion.
// It reports whether any version tables were found.
func (f *File) gnuVersionInit(str []byte) (bool, error) {
	// Versym parallels symbol table, indexing into verneed.
	vs := f.SectionByType(SHT_GNU_VERSYM)
	if vs == nil {
		return false, nil
	}
	d, _ := vs.Data()

	f.gnuVersym = d
	if err := f.dynamicVersions(str); err != nil {
		return false, err
	}
	if err := f.dynamicVersionNeeds(str); err != nil {
		return false, err
	}
	return true, nil
}

// gnuVersion adds Library and Version information to sym,
// which came from offset i of the symbol table.
func (f *File) gnuVersion(i int) (hasVersion bool, versionIndex VersionIndex, version string, library string) {
	// Each entry is two bytes; skip undef entry at beginning.
	i = (i + 1) * 2
	if i >= len(f.gnuVersym) {
		return false, 0, "", ""
	}
	s := f.gnuVersym[i:]
	if len(s) < 2 {
		return false, 0, "", ""
	}
	vi := VersionIndex(f.ByteOrder.Uint16(s))
	ndx := vi.Index()

	if ndx == 0 || ndx == 1 {
		return true, vi, "", ""
	}

	for _, v := range f.dynVerNeeds {
		for _, n := range v.Needs {
			if ndx == n.Index {
				return true, vi, n.Dep, v.Name
			}
		}
	}

	for _, v := range f.dynVers {
		if ndx == v.Index {
			return true, vi, v.Name, ""
		}
	}

	return false, 0, "", ""
}

// ImportedLibraries returns the names of all libraries
// referred to by the binary f that are expected to be
// linked with the binary at dynamic link time.
func (f *File) ImportedLibraries() ([]string, error) {
	return f.DynString(DT_NEEDED)
}

// DynString returns the strings listed for the given tag in the file's dynamic
// section.
//
// The tag must be one that takes string values: [DT_NEEDED], [DT_SONAME], [DT_RPATH], or
// [DT_RUNPATH].
func (f *File) DynString(tag DynTag) ([]string, error) {
	switch tag {
	case DT_NEEDED, DT_SONAME, DT_RPATH, DT_RUNPATH:
	default:
		return nil, fmt.Errorf("non-string-valued tag %v", tag)
	}
	ds := f.SectionByType(SHT_DYNAMIC)
	if ds == nil {
		// not dynamic, so no libraries
		return nil, nil
	}
	d, err := ds.Data()
	if err != nil {
		return nil, err
	}

	dynSize := 8
	if f.Class == ELFCLASS64 {
		dynSize = 16
	}
	if len(d)%dynSize != 0 {
		return nil, errors.New("length of dynamic section is not a multiple of dynamic entry size")
	}

	str, err := f.stringTable(ds.Link)
	if err != nil {
		return nil, err
	}
	var all []string
	for len(d) > 0 {
		var t DynTag
		var v uint64
		switch f.Class {
		case ELFCLASS32:
			t = DynTag(f.ByteOrder.Uint32(d[0:4]))
			v = uint64(f.ByteOrder.Uint32(d[4:8]))
			d = d[8:]
		case ELFCLASS64:
			t = DynTag(f.ByteOrder.Uint64(d[0:8]))
			v = f.ByteOrder.Uint64(d[8:16])
			d = d[16:]
		}
		if t == tag {
			s, ok := getString(str, int(v))
			if ok {
				all = append(all, s)
			}
		}
	}
	return all, nil
}

// DynValue returns the values listed for the given tag in the file's dynamic
// section.
func (f *File) DynValue(tag DynTag) ([]uint64, error) {
	ds := f.SectionByType(SHT_DYNAMIC)
	if ds == nil {
		return nil, nil
	}
	d, err := ds.Data()
	if err != nil {
		return nil, err
	}

	dynSize := 8
	if f.Class == ELFCLASS64 {
		dynSize = 16
	}
	if len(d)%dynSize != 0 {
		return nil, errors.New("length of dynamic section is not a multiple of dynamic entry size")
	}

	// Parse the .dynamic section as a string of bytes.
	var vals []uint64
	for len(d) > 0 {
		var t DynTag
		var v uint64
		switch f.Class {
		case ELFCLASS32:
			t = DynTag(f.ByteOrder.Uint32(d[0:4]))
			v = uint64(f.ByteOrder.Uint32(d[4:8]))
			d = d[8:]
		case ELFCLASS64:
			t = DynTag(f.ByteOrder.Uint64(d[0:8]))
			v = f.ByteOrder.Uint64(d[8:16])
			d = d[16:]
		}
		if t == tag {
			vals = append(vals, v)
		}
	}
	return vals, nil
}

type nobitsSectionReader struct{}

func (*nobitsSectionReader) ReadAt(p []byte, off int64) (n int, err error) {
	return 0, errors.New("unexpected read from SHT_NOBITS section")
}

// putUint writes a relocation to slice
// at offset start of length length (4 or 8 bytes),
// adding sym+addend to the existing value if readUint is true,
// or just writing sym+addend if readUint is false.
// If the write would extend beyond the end of slice, putUint does nothing.
// If the addend is negative, putUint does nothing.
// If the addition would overflow, putUint does nothing.
func putUint(byteOrder binary.ByteOrder, slice []byte, start, length, sym uint64, addend int64, readUint bool) {
	if start+length > uint64(len(slice)) || math.MaxUint64-start < length {
		return
	}
	if addend < 0 {
		return
	}

	s := slice[start : start+length]

	switch length {
	case 4:
		ae := uint32(addend)
		if readUint {
			ae += byteOrder.Uint32(s)
		}
		byteOrder.PutUint32(s, uint32(sym)+ae)
	case 8:
		ae := uint64(addend)
		if readUint {
			ae += byteOrder.Uint64(s)
		}
		byteOrder.PutUint64(s, sym+ae)
	default:
		panic("can't happen")
	}
}
