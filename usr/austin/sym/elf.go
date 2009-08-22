// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sym

import (
	"fmt";
	"io";
	"log";
	"os";
)

/*
 * Internal ELF representation
 */

// Elf represents a decoded ELF binary.
type Elf struct {
	class int;
	data byteOrder;
	Type ElfType;
	Machine ElfMachine;
	Sections []*Section;
}

// Section represents a single section in an ELF binary.
type Section struct {
	r io.ReadSeeker;
	Name string;
	offset int64;
	Size uint64;
	Addr uint64;
}

/*
 * ELF reader
 */

type FormatError struct {
	off int64;
	msg string;
	val interface{};
}

func (e *FormatError) String() string {
	msg := e.msg;
	if e.val != nil {
		msg += fmt.Sprintf(" '%v' ", e.val);
	}
	msg += fmt.Sprintf("in record at byte %#x", e.off);
	return msg;
}

// NewElf reads and decodes an ELF binary.  The ELF binary is expected
// to start where the reader is currently positioned.
func NewElf(r io.ReadSeeker) (*Elf, os.Error) {
	// Read ELF identifier
	var ident [eiNIdent]uint8;
	off, err := r.Seek(0, 0);
	if err != nil {
		return nil, err;
	}
	start := off;
	n, err := io.ReadFull(r, &ident);
	if err != nil {
		if err == os.EOF {
			err = io.ErrUnexpectedEOF;
		}
		return nil, err;
	}

	// Decode identifier
	if ident[eiMag0] != '\x7f' || ident[eiMag1] != 'E' || ident[eiMag2] != 'L' || ident[eiMag3] != 'F' {
		return nil, &FormatError{off, "bad magic number", string(ident[eiMag0:eiMag3])};
	}
	e := &Elf{};

	switch ident[eiClass] {
	case elfClass32:
		e.class = 32;
	case elfClass64:
		e.class = 64;
	default:
		return nil, &FormatError{off, "unknown ELF class", ident[eiClass]};
	}

	switch ident[eiData] {
	case elfData2LSB:
		e.data = lsb;
	case elfData2MSB:
		e.data = msb;
	default:
		return nil, &FormatError{off, "unknown ELF data encoding", ident[eiData]};
	}

	if ident[eiVersion] != evCurrent {
		return nil, &FormatError{off, "unknown ELF version", ident[eiVersion]};
	}

	// TODO(austin) Do something with ABI?

	// Read ELF file header
	var shoff int64;
	var shentsize, shnum, shstrndx int;

	br := newBinaryReader(r, e.data);
	switch e.class {
	case 32:
		return nil, &FormatError{off, "ELF32 not implemented", nil};
	case 64:
		hdr := &elf64Ehdr{};
		br.ReadAny(hdr);
		if err := br.Error(); err != nil {
			return nil, err;
		}

		if hdr.Type > etCore && hdr.Type < etLoOS {
			return nil, &FormatError{off, "unknown ELF file type", hdr.Type};
		}
		e.Type = ElfType(hdr.Type);
		e.Machine = ElfMachine(hdr.Machine);
		if hdr.Version != evCurrent {
			return nil, &FormatError{off, "unknown second ELF version", hdr.Version};
		}

		shoff = int64(hdr.Shoff);
		shentsize = int(hdr.Shentsize);
		shnum = int(hdr.Shnum);
		shstrndx = int(hdr.Shstrndx);
	}

	// Read section headers
	e.Sections = make([]*Section, shnum);
	secNames := make([]uint32, shnum);
	for i := 0; i < shnum; i++ {
		off, err = r.Seek(start + shoff + int64(i*shentsize), 0);
		if err != nil {
			return nil, err;
		}

		br = newBinaryReader(r, e.data);
		switch e.class {
		case 32:
			panic("not reached");
		case 64:
			shdr := &elf64Shdr{};
			br.ReadAny(shdr);
			if err := br.Error(); err != nil {
				return nil, err;
			}

			s := &Section{
				r: r,
				offset: start + int64(shdr.Off),
				Size: shdr.Size,
				Addr: uint64(shdr.Addr),
			};
			secNames[i] = shdr.Name;
			e.Sections[i] = s;
		}
	}

	// Resolve section names
	off, err = r.Seek(start + e.Sections[shstrndx].offset, 0);
	if err != nil {
		return nil, err;
	}
	blob := make([]byte, e.Sections[shstrndx].Size);
	n, err = io.ReadFull(r, blob);

	for i, s := range e.Sections {
		var ok bool;
		s.Name, ok = getString(blob, int(secNames[i]));
		if !ok {
			return nil, &FormatError{start + shoff + int64(i*shentsize), "bad section name", secNames[i]};
		}
	}

	return e, nil;
}

// getString extracts a string from an ELF string table.
func getString(section []byte, index int) (string, bool) {
	if index < 0 || index >= len(section) {
		return "", false;
	}

	for end := index; end < len(section); end++ {
		if section[end] == 0 {
			return string(section[index:end]), true;
		}
	}
	return "", false;
}

// Section returns a section with the given name, or nil if no such
// section exists.
func (e *Elf) Section(name string) *Section {
	for _, s := range e.Sections {
		if s.Name == name {
			return s;
		}
	}
	return nil;
}

/*
 * Sections
 */

type subReader struct {
	r io.Reader;
	rem uint64;
}

func (r *subReader) Read(b []byte) (ret int, err os.Error) {
	if r.rem == 0 {
		return 0, os.EOF;
	}
	if uint64(len(b)) > r.rem {
		b = b[0:r.rem];
	}
	ret, err = r.r.Read(b);
	r.rem -= uint64(ret);
	if err == os.EOF {
		err = io.ErrUnexpectedEOF;
	}
	return ret, err;
}

// Open returns a reader backed by the data in this section.
// The original ELF file must still be open for this to work.
// The returned reader assumes there will be no seeks on the
// underlying file or any other opened section between the Open call
// and the last call to Read.
func (s *Section) Open() (io.Reader, os.Error) {
	_, err := s.r.Seek(s.offset, 0);
	if err != nil {
		return nil, err;
	}
	return &subReader{s.r, s.Size}, nil;
}
