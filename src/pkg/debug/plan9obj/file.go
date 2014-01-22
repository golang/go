// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package plan9obj implements access to Plan 9 a.out object files.
package plan9obj

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

// A FileHeader represents an Plan 9 a.out file header.
type FileHeader struct {
	Ptrsz int
}

// A File represents an open Plan 9 a.out file.
type File struct {
	FileHeader
	Sections []*Section
	closer   io.Closer
}

type SectionHeader struct {
	Name   string
	Size   uint32
	Offset uint32
}

// A Section represents a single section in an Plan 9 a.out file.
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

// Data reads and returns the contents of the Plan 9 a.out section.
func (s *Section) Data() ([]byte, error) {
	dat := make([]byte, s.sr.Size())
	n, err := s.sr.ReadAt(dat, 0)
	return dat[0:n], err
}

// Open returns a new ReadSeeker reading the Plan 9 a.out section.
func (s *Section) Open() io.ReadSeeker { return io.NewSectionReader(s.sr, 0, 1<<63-1) }

// A ProgHeader represents a single Plan 9 a.out program header.
type ProgHeader struct {
	Magic uint32
	Text  uint32
	Data  uint32
	Bss   uint32
	Syms  uint32
	Entry uint64
	Spsz  uint32
	Pcsz  uint32
}

// A Prog represents the program header in an Plan 9 a.out binary.
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

// Open returns a new ReadSeeker reading the Plan 9 a.out program body.
func (p *Prog) Open() io.ReadSeeker { return io.NewSectionReader(p.sr, 0, 1<<63-1) }

// A Symbol represents an entry in a Plan 9 a.out symbol table section.
type Sym struct {
	Value uint64
	Type  rune
	Name  string
}

/*
 * Plan 9 a.out reader
 */

type FormatError struct {
	off int
	msg string
	val interface{}
}

func (e *FormatError) Error() string {
	msg := e.msg
	if e.val != nil {
		msg += fmt.Sprintf(" '%v'", e.val)
	}
	msg += fmt.Sprintf(" in record at byte %#x", e.off)
	return msg
}

// Open opens the named file using os.Open and prepares it for use as an Plan 9 a.out binary.
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

func parseMagic(magic [4]byte) (*ExecTable, error) {
	for _, e := range exectab {
		if string(magic[:]) == e.Magic {
			return &e, nil
		}
	}
	return nil, &FormatError{0, "bad magic number", magic[:]}
}

// NewFile creates a new File for accessing an Plan 9 binary in an underlying reader.
// The Plan 9 binary is expected to start at position 0 in the ReaderAt.
func NewFile(r io.ReaderAt) (*File, error) {
	sr := io.NewSectionReader(r, 0, 1<<63-1)
	// Read and decode Plan 9 magic
	var magic [4]byte
	if _, err := r.ReadAt(magic[:], 0); err != nil {
		return nil, err
	}
	mp, err := parseMagic(magic)
	if err != nil {
		return nil, err
	}

	f := &File{FileHeader{mp.Ptrsz}, nil, nil}

	ph := new(prog)
	if err := binary.Read(sr, binary.BigEndian, ph); err != nil {
		return nil, err
	}

	p := new(Prog)
	p.ProgHeader = ProgHeader{
		Magic: ph.Magic,
		Text:  ph.Text,
		Data:  ph.Data,
		Bss:   ph.Bss,
		Syms:  ph.Syms,
		Entry: uint64(ph.Entry),
		Spsz:  ph.Spsz,
		Pcsz:  ph.Pcsz,
	}

	if mp.Ptrsz == 8 {
		if err := binary.Read(sr, binary.BigEndian, &p.Entry); err != nil {
			return nil, err
		}
	}

	var sects = []struct {
		name string
		size uint32
	}{
		{"text", ph.Text},
		{"data", ph.Data},
		{"syms", ph.Syms},
		{"spsz", ph.Spsz},
		{"pcsz", ph.Pcsz},
	}

	f.Sections = make([]*Section, 5)

	off := mp.Hsize

	for i, sect := range sects {
		s := new(Section)
		s.SectionHeader = SectionHeader{
			Name:   sect.name,
			Size:   sect.size,
			Offset: off,
		}
		off += sect.size
		s.sr = io.NewSectionReader(r, int64(s.SectionHeader.Offset), int64(s.SectionHeader.Size))
		s.ReaderAt = s.sr
		f.Sections[i] = s
	}

	return f, nil
}

func walksymtab(data []byte, ptrsz int, fn func(sym) error) error {
	var order binary.ByteOrder = binary.BigEndian
	var s sym
	p := data
	for len(p) >= 4 {
		// Symbol type, value.
		if len(p) < ptrsz {
			return &FormatError{len(data), "unexpected EOF", nil}
		}
		// fixed-width value
		if ptrsz == 8 {
			s.value = order.Uint64(p[0:8])
			p = p[8:]
		} else {
			s.value = uint64(order.Uint32(p[0:4]))
			p = p[4:]
		}

		var typ byte
		typ = p[0] & 0x7F
		s.typ = typ
		p = p[1:]

		// Name.
		var i int
		var nnul int
		for i = 0; i < len(p); i++ {
			if p[i] == 0 {
				nnul = 1
				break
			}
		}
		switch typ {
		case 'z', 'Z':
			p = p[i+nnul:]
			for i = 0; i+2 <= len(p); i += 2 {
				if p[i] == 0 && p[i+1] == 0 {
					nnul = 2
					break
				}
			}
		}
		if len(p) < i+nnul {
			return &FormatError{len(data), "unexpected EOF", nil}
		}
		s.name = p[0:i]
		i += nnul
		p = p[i:]

		fn(s)
	}
	return nil
}

// NewTable decodes the Go symbol table in data,
// returning an in-memory representation.
func newTable(symtab []byte, ptrsz int) ([]Sym, error) {
	var n int
	err := walksymtab(symtab, ptrsz, func(s sym) error {
		n++
		return nil
	})
	if err != nil {
		return nil, err
	}

	fname := make(map[uint16]string)
	syms := make([]Sym, 0, n)
	err = walksymtab(symtab, ptrsz, func(s sym) error {
		n := len(syms)
		syms = syms[0 : n+1]
		ts := &syms[n]
		ts.Type = rune(s.typ)
		ts.Value = s.value
		switch s.typ {
		default:
			ts.Name = string(s.name[:])
		case 'z', 'Z':
			for i := 0; i < len(s.name); i += 2 {
				eltIdx := binary.BigEndian.Uint16(s.name[i : i+2])
				elt, ok := fname[eltIdx]
				if !ok {
					return &FormatError{-1, "bad filename code", eltIdx}
				}
				if n := len(ts.Name); n > 0 && ts.Name[n-1] != '/' {
					ts.Name += "/"
				}
				ts.Name += elt
			}
		}
		switch s.typ {
		case 'f':
			fname[uint16(s.value)] = ts.Name
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	return syms, nil
}

// Symbols returns the symbol table for f.
func (f *File) Symbols() ([]Sym, error) {
	symtabSection := f.Section("syms")
	if symtabSection == nil {
		return nil, errors.New("no symbol section")
	}

	symtab, err := symtabSection.Data()
	if err != nil {
		return nil, errors.New("cannot load symbol section")
	}

	return newTable(symtab, f.Ptrsz)
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
