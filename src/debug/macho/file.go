// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package macho implements access to Mach-O object files.

# Security

This package is not designed to be hardened against adversarial inputs, and is
outside the scope of https://go.dev/security/policy. In particular, only basic
validation is done when parsing object files. As such, care should be taken when
parsing untrusted inputs, as parsing malformed files may consume significant
resources, or cause panics.
*/
package macho

// High level access to low level data structures.

import (
	"bytes"
	"compress/zlib"
	"debug/dwarf"
	"encoding/binary"
	"fmt"
	"internal/saferio"
	"io"
	"os"
	"strings"
)

// A File represents an open Mach-O file.
type File struct {
	FileHeader
	ByteOrder binary.ByteOrder
	Loads     []Load
	Sections  []*Section

	Symtab   *Symtab
	Dysymtab *Dysymtab

	closer io.Closer
}

// A Load represents any Mach-O load command.
type Load interface {
	Raw() []byte
}

// A LoadBytes is the uninterpreted bytes of a Mach-O load command.
type LoadBytes []byte

func (b LoadBytes) Raw() []byte { return b }

// A SegmentHeader is the header for a Mach-O 32-bit or 64-bit load segment command.
type SegmentHeader struct {
	Cmd     LoadCmd
	Len     uint32
	Name    string
	Addr    uint64
	Memsz   uint64
	Offset  uint64
	Filesz  uint64
	Maxprot uint32
	Prot    uint32
	Nsect   uint32
	Flag    uint32
}

// A Segment represents a Mach-O 32-bit or 64-bit load segment command.
type Segment struct {
	LoadBytes
	SegmentHeader

	// Embed ReaderAt for ReadAt method.
	// Do not embed SectionReader directly
	// to avoid having Read and Seek.
	// If a client wants Read and Seek it must use
	// Open() to avoid fighting over the seek offset
	// with other clients.
	io.ReaderAt
	sr *io.SectionReader
}

// Data reads and returns the contents of the segment.
func (s *Segment) Data() ([]byte, error) {
	return saferio.ReadDataAt(s.sr, s.Filesz, 0)
}

// Open returns a new ReadSeeker reading the segment.
func (s *Segment) Open() io.ReadSeeker { return io.NewSectionReader(s.sr, 0, 1<<63-1) }

type SectionHeader struct {
	Name   string
	Seg    string
	Addr   uint64
	Size   uint64
	Offset uint32
	Align  uint32
	Reloff uint32
	Nreloc uint32
	Flags  uint32
}

// A Reloc represents a Mach-O relocation.
type Reloc struct {
	Addr  uint32
	Value uint32
	// when Scattered == false && Extern == true, Value is the symbol number.
	// when Scattered == false && Extern == false, Value is the section number.
	// when Scattered == true, Value is the value that this reloc refers to.
	Type      uint8
	Len       uint8 // 0=byte, 1=word, 2=long, 3=quad
	Pcrel     bool
	Extern    bool // valid if Scattered == false
	Scattered bool
}

type Section struct {
	SectionHeader
	Relocs []Reloc

	// Embed ReaderAt for ReadAt method.
	// Do not embed SectionReader directly
	// to avoid having Read and Seek.
	// If a client wants Read and Seek it must use
	// Open() to avoid fighting over the seek offset
	// with other clients.
	io.ReaderAt
	sr *io.SectionReader
}

// Data reads and returns the contents of the Mach-O section.
func (s *Section) Data() ([]byte, error) {
	return saferio.ReadDataAt(s.sr, s.Size, 0)
}

// Open returns a new ReadSeeker reading the Mach-O section.
func (s *Section) Open() io.ReadSeeker { return io.NewSectionReader(s.sr, 0, 1<<63-1) }

// A Dylib represents a Mach-O load dynamic library command.
type Dylib struct {
	LoadBytes
	Name           string
	Time           uint32
	CurrentVersion uint32
	CompatVersion  uint32
}

// A Symtab represents a Mach-O symbol table command.
type Symtab struct {
	LoadBytes
	SymtabCmd
	Syms []Symbol
}

// A Dysymtab represents a Mach-O dynamic symbol table command.
type Dysymtab struct {
	LoadBytes
	DysymtabCmd
	IndirectSyms []uint32 // indices into Symtab.Syms
}

// A Rpath represents a Mach-O rpath command.
type Rpath struct {
	LoadBytes
	Path string
}

// A Symbol is a Mach-O 32-bit or 64-bit symbol table entry.
type Symbol struct {
	Name  string
	Type  uint8
	Sect  uint8
	Desc  uint16
	Value uint64
}

/*
 * Mach-O reader
 */

// FormatError is returned by some operations if the data does
// not have the correct format for an object file.
type FormatError struct {
	off int64
	msg string
	val any
}

func (e *FormatError) Error() string {
	msg := e.msg
	if e.val != nil {
		msg += fmt.Sprintf(" '%v'", e.val)
	}
	msg += fmt.Sprintf(" in record at byte %#x", e.off)
	return msg
}

// Open opens the named file using [os.Open] and prepares it for use as a Mach-O binary.
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

// NewFile creates a new [File] for accessing a Mach-O binary in an underlying reader.
// The Mach-O binary is expected to start at position 0 in the ReaderAt.
func NewFile(r io.ReaderAt) (*File, error) {
	f := new(File)
	sr := io.NewSectionReader(r, 0, 1<<63-1)

	// Read and decode Mach magic to determine byte order, size.
	// Magic32 and Magic64 differ only in the bottom bit.
	var ident [4]byte
	if _, err := r.ReadAt(ident[0:], 0); err != nil {
		return nil, err
	}
	be := binary.BigEndian.Uint32(ident[0:])
	le := binary.LittleEndian.Uint32(ident[0:])
	switch Magic32 &^ 1 {
	case be &^ 1:
		f.ByteOrder = binary.BigEndian
		f.Magic = be
	case le &^ 1:
		f.ByteOrder = binary.LittleEndian
		f.Magic = le
	default:
		return nil, &FormatError{0, "invalid magic number", nil}
	}

	// Read entire file header.
	if err := binary.Read(sr, f.ByteOrder, &f.FileHeader); err != nil {
		return nil, err
	}

	// Then load commands.
	offset := int64(fileHeaderSize32)
	if f.Magic == Magic64 {
		offset = fileHeaderSize64
	}
	dat, err := saferio.ReadDataAt(r, uint64(f.Cmdsz), offset)
	if err != nil {
		return nil, err
	}
	c := saferio.SliceCap[Load](uint64(f.Ncmd))
	if c < 0 {
		return nil, &FormatError{offset, "too many load commands", nil}
	}
	f.Loads = make([]Load, 0, c)
	bo := f.ByteOrder
	for i := uint32(0); i < f.Ncmd; i++ {
		// Each load command begins with uint32 command and length.
		if len(dat) < 8 {
			return nil, &FormatError{offset, "command block too small", nil}
		}
		cmd, siz := LoadCmd(bo.Uint32(dat[0:4])), bo.Uint32(dat[4:8])
		if siz < 8 || siz > uint32(len(dat)) {
			return nil, &FormatError{offset, "invalid command block size", nil}
		}
		var cmddat []byte
		cmddat, dat = dat[0:siz], dat[siz:]
		offset += int64(siz)
		var s *Segment
		switch cmd {
		default:
			f.Loads = append(f.Loads, LoadBytes(cmddat))

		case LoadCmdRpath:
			var hdr RpathCmd
			b := bytes.NewReader(cmddat)
			if err := binary.Read(b, bo, &hdr); err != nil {
				return nil, err
			}
			l := new(Rpath)
			if hdr.Path >= uint32(len(cmddat)) {
				return nil, &FormatError{offset, "invalid path in rpath command", hdr.Path}
			}
			l.Path = cstring(cmddat[hdr.Path:])
			l.LoadBytes = LoadBytes(cmddat)
			f.Loads = append(f.Loads, l)

		case LoadCmdDylib:
			var hdr DylibCmd
			b := bytes.NewReader(cmddat)
			if err := binary.Read(b, bo, &hdr); err != nil {
				return nil, err
			}
			l := new(Dylib)
			if hdr.Name >= uint32(len(cmddat)) {
				return nil, &FormatError{offset, "invalid name in dynamic library command", hdr.Name}
			}
			l.Name = cstring(cmddat[hdr.Name:])
			l.Time = hdr.Time
			l.CurrentVersion = hdr.CurrentVersion
			l.CompatVersion = hdr.CompatVersion
			l.LoadBytes = LoadBytes(cmddat)
			f.Loads = append(f.Loads, l)

		case LoadCmdSymtab:
			var hdr SymtabCmd
			b := bytes.NewReader(cmddat)
			if err := binary.Read(b, bo, &hdr); err != nil {
				return nil, err
			}
			strtab, err := saferio.ReadDataAt(r, uint64(hdr.Strsize), int64(hdr.Stroff))
			if err != nil {
				return nil, err
			}
			var symsz int
			if f.Magic == Magic64 {
				symsz = 16
			} else {
				symsz = 12
			}
			symdat, err := saferio.ReadDataAt(r, uint64(hdr.Nsyms)*uint64(symsz), int64(hdr.Symoff))
			if err != nil {
				return nil, err
			}
			st, err := f.parseSymtab(symdat, strtab, cmddat, &hdr, offset)
			if err != nil {
				return nil, err
			}
			f.Loads = append(f.Loads, st)
			f.Symtab = st

		case LoadCmdDysymtab:
			var hdr DysymtabCmd
			b := bytes.NewReader(cmddat)
			if err := binary.Read(b, bo, &hdr); err != nil {
				return nil, err
			}
			if f.Symtab == nil {
				return nil, &FormatError{offset, "dynamic symbol table seen before any ordinary symbol table", nil}
			} else if hdr.Iundefsym > uint32(len(f.Symtab.Syms)) {
				return nil, &FormatError{offset, fmt.Sprintf(
					"undefined symbols index in dynamic symbol table command is greater than symbol table length (%d > %d)",
					hdr.Iundefsym, len(f.Symtab.Syms)), nil}
			} else if hdr.Iundefsym+hdr.Nundefsym > uint32(len(f.Symtab.Syms)) {
				return nil, &FormatError{offset, fmt.Sprintf(
					"number of undefined symbols after index in dynamic symbol table command is greater than symbol table length (%d > %d)",
					hdr.Iundefsym+hdr.Nundefsym, len(f.Symtab.Syms)), nil}
			}
			dat, err := saferio.ReadDataAt(r, uint64(hdr.Nindirectsyms)*4, int64(hdr.Indirectsymoff))
			if err != nil {
				return nil, err
			}
			x := make([]uint32, hdr.Nindirectsyms)
			if err := binary.Read(bytes.NewReader(dat), bo, x); err != nil {
				return nil, err
			}
			st := new(Dysymtab)
			st.LoadBytes = LoadBytes(cmddat)
			st.DysymtabCmd = hdr
			st.IndirectSyms = x
			f.Loads = append(f.Loads, st)
			f.Dysymtab = st

		case LoadCmdSegment:
			var seg32 Segment32
			b := bytes.NewReader(cmddat)
			if err := binary.Read(b, bo, &seg32); err != nil {
				return nil, err
			}
			s = new(Segment)
			s.LoadBytes = cmddat
			s.Cmd = cmd
			s.Len = siz
			s.Name = cstring(seg32.Name[0:])
			s.Addr = uint64(seg32.Addr)
			s.Memsz = uint64(seg32.Memsz)
			s.Offset = uint64(seg32.Offset)
			s.Filesz = uint64(seg32.Filesz)
			s.Maxprot = seg32.Maxprot
			s.Prot = seg32.Prot
			s.Nsect = seg32.Nsect
			s.Flag = seg32.Flag
			f.Loads = append(f.Loads, s)
			for i := 0; i < int(s.Nsect); i++ {
				var sh32 Section32
				if err := binary.Read(b, bo, &sh32); err != nil {
					return nil, err
				}
				sh := new(Section)
				sh.Name = cstring(sh32.Name[0:])
				sh.Seg = cstring(sh32.Seg[0:])
				sh.Addr = uint64(sh32.Addr)
				sh.Size = uint64(sh32.Size)
				sh.Offset = sh32.Offset
				sh.Align = sh32.Align
				sh.Reloff = sh32.Reloff
				sh.Nreloc = sh32.Nreloc
				sh.Flags = sh32.Flags
				if err := f.pushSection(sh, r); err != nil {
					return nil, err
				}
			}

		case LoadCmdSegment64:
			var seg64 Segment64
			b := bytes.NewReader(cmddat)
			if err := binary.Read(b, bo, &seg64); err != nil {
				return nil, err
			}
			s = new(Segment)
			s.LoadBytes = cmddat
			s.Cmd = cmd
			s.Len = siz
			s.Name = cstring(seg64.Name[0:])
			s.Addr = seg64.Addr
			s.Memsz = seg64.Memsz
			s.Offset = seg64.Offset
			s.Filesz = seg64.Filesz
			s.Maxprot = seg64.Maxprot
			s.Prot = seg64.Prot
			s.Nsect = seg64.Nsect
			s.Flag = seg64.Flag
			f.Loads = append(f.Loads, s)
			for i := 0; i < int(s.Nsect); i++ {
				var sh64 Section64
				if err := binary.Read(b, bo, &sh64); err != nil {
					return nil, err
				}
				sh := new(Section)
				sh.Name = cstring(sh64.Name[0:])
				sh.Seg = cstring(sh64.Seg[0:])
				sh.Addr = sh64.Addr
				sh.Size = sh64.Size
				sh.Offset = sh64.Offset
				sh.Align = sh64.Align
				sh.Reloff = sh64.Reloff
				sh.Nreloc = sh64.Nreloc
				sh.Flags = sh64.Flags
				if err := f.pushSection(sh, r); err != nil {
					return nil, err
				}
			}
		}
		if s != nil {
			if int64(s.Offset) < 0 {
				return nil, &FormatError{offset, "invalid section offset", s.Offset}
			}
			if int64(s.Filesz) < 0 {
				return nil, &FormatError{offset, "invalid section file size", s.Filesz}
			}
			s.sr = io.NewSectionReader(r, int64(s.Offset), int64(s.Filesz))
			s.ReaderAt = s.sr
		}
	}
	return f, nil
}

func (f *File) parseSymtab(symdat, strtab, cmddat []byte, hdr *SymtabCmd, offset int64) (*Symtab, error) {
	bo := f.ByteOrder
	c := saferio.SliceCap[Symbol](uint64(hdr.Nsyms))
	if c < 0 {
		return nil, &FormatError{offset, "too many symbols", nil}
	}
	symtab := make([]Symbol, 0, c)
	b := bytes.NewReader(symdat)
	for i := 0; i < int(hdr.Nsyms); i++ {
		var n Nlist64
		if f.Magic == Magic64 {
			if err := binary.Read(b, bo, &n); err != nil {
				return nil, err
			}
		} else {
			var n32 Nlist32
			if err := binary.Read(b, bo, &n32); err != nil {
				return nil, err
			}
			n.Name = n32.Name
			n.Type = n32.Type
			n.Sect = n32.Sect
			n.Desc = n32.Desc
			n.Value = uint64(n32.Value)
		}
		if n.Name >= uint32(len(strtab)) {
			return nil, &FormatError{offset, "invalid name in symbol table", n.Name}
		}
		// We add "_" to Go symbols. Strip it here. See issue 33808.
		name := cstring(strtab[n.Name:])
		if strings.Contains(name, ".") && name[0] == '_' {
			name = name[1:]
		}
		symtab = append(symtab, Symbol{
			Name:  name,
			Type:  n.Type,
			Sect:  n.Sect,
			Desc:  n.Desc,
			Value: n.Value,
		})
	}
	st := new(Symtab)
	st.LoadBytes = LoadBytes(cmddat)
	st.Syms = symtab
	return st, nil
}

type relocInfo struct {
	Addr   uint32
	Symnum uint32
}

func (f *File) pushSection(sh *Section, r io.ReaderAt) error {
	f.Sections = append(f.Sections, sh)
	sh.sr = io.NewSectionReader(r, int64(sh.Offset), int64(sh.Size))
	sh.ReaderAt = sh.sr

	if sh.Nreloc > 0 {
		reldat, err := saferio.ReadDataAt(r, uint64(sh.Nreloc)*8, int64(sh.Reloff))
		if err != nil {
			return err
		}
		b := bytes.NewReader(reldat)

		bo := f.ByteOrder

		sh.Relocs = make([]Reloc, sh.Nreloc)
		for i := range sh.Relocs {
			rel := &sh.Relocs[i]

			var ri relocInfo
			if err := binary.Read(b, bo, &ri); err != nil {
				return err
			}

			if ri.Addr&(1<<31) != 0 { // scattered
				rel.Addr = ri.Addr & (1<<24 - 1)
				rel.Type = uint8((ri.Addr >> 24) & (1<<4 - 1))
				rel.Len = uint8((ri.Addr >> 28) & (1<<2 - 1))
				rel.Pcrel = ri.Addr&(1<<30) != 0
				rel.Value = ri.Symnum
				rel.Scattered = true
			} else {
				switch bo {
				case binary.LittleEndian:
					rel.Addr = ri.Addr
					rel.Value = ri.Symnum & (1<<24 - 1)
					rel.Pcrel = ri.Symnum&(1<<24) != 0
					rel.Len = uint8((ri.Symnum >> 25) & (1<<2 - 1))
					rel.Extern = ri.Symnum&(1<<27) != 0
					rel.Type = uint8((ri.Symnum >> 28) & (1<<4 - 1))
				case binary.BigEndian:
					rel.Addr = ri.Addr
					rel.Value = ri.Symnum >> 8
					rel.Pcrel = ri.Symnum&(1<<7) != 0
					rel.Len = uint8((ri.Symnum >> 5) & (1<<2 - 1))
					rel.Extern = ri.Symnum&(1<<4) != 0
					rel.Type = uint8(ri.Symnum & (1<<4 - 1))
				default:
					panic("unreachable")
				}
			}
		}
	}

	return nil
}

func cstring(b []byte) string {
	i := bytes.IndexByte(b, 0)
	if i == -1 {
		i = len(b)
	}
	return string(b[0:i])
}

// Segment returns the first Segment with the given name, or nil if no such segment exists.
func (f *File) Segment(name string) *Segment {
	for _, l := range f.Loads {
		if s, ok := l.(*Segment); ok && s.Name == name {
			return s
		}
	}
	return nil
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

// DWARF returns the DWARF debug information for the Mach-O file.
func (f *File) DWARF() (*dwarf.Data, error) {
	dwarfSuffix := func(s *Section) string {
		switch {
		case strings.HasPrefix(s.Name, "__debug_"):
			return s.Name[8:]
		case strings.HasPrefix(s.Name, "__zdebug_"):
			return s.Name[9:]
		default:
			return ""
		}

	}
	sectionData := func(s *Section) ([]byte, error) {
		b, err := s.Data()
		if err != nil && uint64(len(b)) < s.Size {
			return nil, err
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

		b, err := sectionData(s)
		if err != nil {
			return nil, err
		}

		if suffix == "types" {
			err = d.AddTypes(fmt.Sprintf("types-%d", i), b)
		} else {
			err = d.AddSection(".debug_"+suffix, b)
		}
		if err != nil {
			return nil, err
		}
	}

	return d, nil
}

// ImportedSymbols returns the names of all symbols
// referred to by the binary f that are expected to be
// satisfied by other libraries at dynamic load time.
func (f *File) ImportedSymbols() ([]string, error) {
	if f.Dysymtab == nil || f.Symtab == nil {
		return nil, &FormatError{0, "missing symbol table", nil}
	}

	st := f.Symtab
	dt := f.Dysymtab
	var all []string
	for _, s := range st.Syms[dt.Iundefsym : dt.Iundefsym+dt.Nundefsym] {
		all = append(all, s.Name)
	}
	return all, nil
}

// ImportedLibraries returns the paths of all libraries
// referred to by the binary f that are expected to be
// linked with the binary at dynamic link time.
func (f *File) ImportedLibraries() ([]string, error) {
	var all []string
	for _, l := range f.Loads {
		if lib, ok := l.(*Dylib); ok {
			all = append(all, lib.Name)
		}
	}
	return all, nil
}
