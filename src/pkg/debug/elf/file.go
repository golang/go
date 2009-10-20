// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package elf implements access to ELF object files.
package elf

import (
	"debug/dwarf";
	"encoding/binary";
	"fmt";
	"io";
	"os";
)

// TODO: error reporting detail

/*
 * Internal ELF representation
 */

// A FileHeader represents an ELF file header.
type FileHeader struct {
	Class		Class;
	Data		Data;
	Version		Version;
	OSABI		OSABI;
	ABIVersion	uint8;
	ByteOrder	binary.ByteOrder;
	Type		Type;
	Machine		Machine;
}

// A File represents an open ELF file.
type File struct {
	FileHeader;
	Sections	[]*Section;
	Progs		[]*Prog;
	closer		io.Closer;
}

// A SectionHeader represents a single ELF section header.
type SectionHeader struct {
	Name		string;
	Type		SectionType;
	Flags		SectionFlag;
	Addr		uint64;
	Offset		uint64;
	Size		uint64;
	Link		uint32;
	Info		uint32;
	Addralign	uint64;
	Entsize		uint64;
}

// A Section represents a single section in an ELF file.
type Section struct {
	SectionHeader;

	// Embed ReaderAt for ReadAt method.
	// Do not embed SectionReader directly
	// to avoid having Read and Seek.
	// If a client wants Read and Seek it must use
	// Open() to avoid fighting over the seek offset
	// with other clients.
	io.ReaderAt;
	sr	*io.SectionReader;
}

// Data reads and returns the contents of the ELF section.
func (s *Section) Data() ([]byte, os.Error) {
	dat := make([]byte, s.sr.Size());
	n, err := s.sr.ReadAt(dat, 0);
	return dat[0:n], err;
}

// Open returns a new ReadSeeker reading the ELF section.
func (s *Section) Open() io.ReadSeeker {
	return io.NewSectionReader(s.sr, 0, 1<<63 - 1);
}

// A ProgHeader represents a single ELF program header.
type ProgHeader struct {
	Type	ProgType;
	Flags	ProgFlag;
	Vaddr	uint64;
	Paddr	uint64;
	Filesz	uint64;
	Memsz	uint64;
	Align	uint64;
}

// A Prog represents a single ELF program header in an ELF binary.
type Prog struct {
	ProgHeader;

	// Embed ReaderAt for ReadAt method.
	// Do not embed SectionReader directly
	// to avoid having Read and Seek.
	// If a client wants Read and Seek it must use
	// Open() to avoid fighting over the seek offset
	// with other clients.
	io.ReaderAt;
	sr	*io.SectionReader;
}

// Open returns a new ReadSeeker reading the ELF program body.
func (p *Prog) Open() io.ReadSeeker {
	return io.NewSectionReader(p.sr, 0, 1<<63 - 1);
}


/*
 * ELF reader
 */

type FormatError struct {
	off	int64;
	msg	string;
	val	interface{};
}

func (e *FormatError) String() string {
	msg := e.msg;
	if e.val != nil {
		msg += fmt.Sprintf(" '%v' ", e.val);
	}
	msg += fmt.Sprintf("in record at byte %#x", e.off);
	return msg;
}

// Open opens the named file using os.Open and prepares it for use as an ELF binary.
func Open(name string) (*File, os.Error) {
	f, err := os.Open(name, os.O_RDONLY, 0);
	if err != nil {
		return nil, err;
	}
	ff, err := NewFile(f);
	if err != nil {
		f.Close();
		return nil, err;
	}
	ff.closer = f;
	return ff, nil;
}

// Close closes the File.
// If the File was created using NewFile directly instead of Open,
// Close has no effect.
func (f *File) Close() os.Error {
	var err os.Error;
	if f.closer != nil {
		err = f.closer.Close();
		f.closer = nil;
	}
	return err;
}

// NewFile creates a new File for acecssing an ELF binary in an underlying reader.
// The ELF binary is expected to start at position 0 in the ReaderAt.
func NewFile(r io.ReaderAt) (*File, os.Error) {
	sr := io.NewSectionReader(r, 0, 1<<63 - 1);
	// Read and decode ELF identifier
	var ident [16]uint8;
	if _, err := r.ReadAt(&ident, 0); err != nil {
		return nil, err;
	}
	if ident[0] != '\x7f' || ident[1] != 'E' || ident[2] != 'L' || ident[3] != 'F' {
		return nil, &FormatError{0, "bad magic number", ident[0:4]};
	}

	f := new(File);
	f.Class = Class(ident[EI_CLASS]);
	switch f.Class {
	case ELFCLASS32:
	case ELFCLASS64:
	// ok
	default:
		return nil, &FormatError{0, "unknown ELF class", f.Class};
	}

	f.Data = Data(ident[EI_DATA]);
	switch f.Data {
	case ELFDATA2LSB:
		f.ByteOrder = binary.LittleEndian;
	case ELFDATA2MSB:
		f.ByteOrder = binary.BigEndian;
	default:
		return nil, &FormatError{0, "unknown ELF data encoding", f.Data};
	}

	f.Version = Version(ident[EI_VERSION]);
	if f.Version != EV_CURRENT {
		return nil, &FormatError{0, "unknown ELF version", f.Version};
	}

	f.OSABI = OSABI(ident[EI_OSABI]);
	f.ABIVersion = ident[EI_ABIVERSION];

	// Read ELF file header
	var shoff int64;
	var shentsize, shnum, shstrndx int;
	shstrndx = -1;
	switch f.Class {
	case ELFCLASS32:
		hdr := new(Header32);
		sr.Seek(0, 0);
		if err := binary.Read(sr, f.ByteOrder, hdr); err != nil {
			return nil, err;
		}
		f.Type = Type(hdr.Type);
		f.Machine = Machine(hdr.Machine);
		if v := Version(hdr.Version); v != f.Version {
			return nil, &FormatError{0, "mismatched ELF version", v};
		}
		shoff = int64(hdr.Shoff);
		shentsize = int(hdr.Shentsize);
		shnum = int(hdr.Shnum);
		shstrndx = int(hdr.Shstrndx);
	case ELFCLASS64:
		hdr := new(Header64);
		sr.Seek(0, 0);
		if err := binary.Read(sr, f.ByteOrder, hdr); err != nil {
			return nil, err;
		}
		f.Type = Type(hdr.Type);
		f.Machine = Machine(hdr.Machine);
		if v := Version(hdr.Version); v != f.Version {
			return nil, &FormatError{0, "mismatched ELF version", v};
		}
		shoff = int64(hdr.Shoff);
		shentsize = int(hdr.Shentsize);
		shnum = int(hdr.Shnum);
		shstrndx = int(hdr.Shstrndx);
	}
	if shstrndx < 0 || shstrndx >= shnum {
		return nil, &FormatError{0, "invalid ELF shstrndx", shstrndx};
	}

	// Read program headers
	// TODO

	// Read section headers
	f.Sections = make([]*Section, shnum);
	names := make([]uint32, shnum);
	for i := 0; i < shnum; i++ {
		off := shoff + int64(i)*int64(shentsize);
		sr.Seek(off, 0);
		s := new(Section);
		switch f.Class {
		case ELFCLASS32:
			sh := new(Section32);
			if err := binary.Read(sr, f.ByteOrder, sh); err != nil {
				return nil, err;
			}
			names[i] = sh.Name;
			s.SectionHeader = SectionHeader{
				Type: SectionType(sh.Type),
				Flags: SectionFlag(sh.Flags),
				Addr: uint64(sh.Addr),
				Offset: uint64(sh.Off),
				Size: uint64(sh.Size),
				Link: uint32(sh.Link),
				Info: uint32(sh.Info),
				Addralign: uint64(sh.Addralign),
				Entsize: uint64(sh.Entsize),
			};
		case ELFCLASS64:
			sh := new(Section64);
			if err := binary.Read(sr, f.ByteOrder, sh); err != nil {
				return nil, err;
			}
			names[i] = sh.Name;
			s.SectionHeader = SectionHeader{
				Type: SectionType(sh.Type),
				Flags: SectionFlag(sh.Flags),
				Offset: uint64(sh.Off),
				Size: uint64(sh.Size),
				Addr: uint64(sh.Addr),
				Link: uint32(sh.Link),
				Info: uint32(sh.Info),
				Addralign: uint64(sh.Addralign),
				Entsize: uint64(sh.Entsize),
			};
		}
		s.sr = io.NewSectionReader(r, int64(s.Offset), int64(s.Size));
		s.ReaderAt = s.sr;
		f.Sections[i] = s;
	}

	// Load section header string table.
	s := f.Sections[shstrndx];
	shstrtab := make([]byte, s.Size);
	if _, err := r.ReadAt(shstrtab, int64(s.Offset)); err != nil {
		return nil, err;
	}
	for i, s := range f.Sections {
		var ok bool;
		s.Name, ok = getString(shstrtab, int(names[i]));
		if !ok {
			return nil, &FormatError{shoff+int64(i * shentsize), "bad section name index", names[i]};
		}
	}

	return f, nil;
}

// getString extracts a string from an ELF string table.
func getString(section []byte, start int) (string, bool) {
	if start < 0 || start >= len(section) {
		return "", false;
	}

	for end := start; end < len(section); end++ {
		if section[end] == 0 {
			return string(section[start:end]), true;
		}
	}
	return "", false;
}

// Section returns a section with the given name, or nil if no such
// section exists.
func (f *File) Section(name string) *Section {
	for _, s := range f.Sections {
		if s.Name == name {
			return s;
		}
	}
	return nil;
}

func (f *File) DWARF() (*dwarf.Data, os.Error) {
	// There are many other DWARF sections, but these
	// are the required ones, and the debug/dwarf package
	// does not use the others, so don't bother loading them.
	var names = [...]string{"abbrev", "info", "str"};
	var dat [len(names)][]byte;
	for i, name := range names {
		name = ".debug_" + name;
		s := f.Section(name);
		if s == nil {
			continue;
		}
		b, err := s.Data();
		if err != nil && uint64(len(b)) < s.Size {
			return nil, err;
		}
		dat[i] = b;
	}

	abbrev, info, str := dat[0], dat[1], dat[2];
	return dwarf.New(abbrev, nil, nil, info, nil, nil, nil, str);
}
