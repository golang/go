// Copyright 2014 The Go Authors. All rights reserved.
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

// A FileHeader represents a Plan 9 a.out file header.
type FileHeader struct {
	Magic       uint32
	Bss         uint32
	Entry       uint64
	PtrSize     int
	LoadAddress uint64
	HdrSize     uint64
}

// A File represents an open Plan 9 a.out file.
type File struct {
	FileHeader
	Sections []*Section
	closer   io.Closer
}

// A SectionHeader represents a single Plan 9 a.out section header.
// This structure doesn't exist on-disk, but eases navigation
// through the object file.
type SectionHeader struct {
	Name   string
	Size   uint32
	Offset uint32
}

// A Section represents a single section in a Plan 9 a.out file.
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
	if n == len(dat) {
		err = nil
	}
	return dat[0:n], err
}

// Open returns a new ReadSeeker reading the Plan 9 a.out section.
func (s *Section) Open() io.ReadSeeker { return io.NewSectionReader(s.sr, 0, 1<<63-1) }

// A Symbol represents an entry in a Plan 9 a.out symbol table section.
type Sym struct {
	Value uint64
	Type  rune
	Name  string
}

/*
 * Plan 9 a.out reader
 */

// formatError is returned by some operations if the data does
// not have the correct format for an object file.
type formatError struct {
	off int
	msg string
	val interface{}
}

func (e *formatError) Error() string {
	msg := e.msg
	if e.val != nil {
		msg += fmt.Sprintf(" '%v'", e.val)
	}
	msg += fmt.Sprintf(" in record at byte %#x", e.off)
	return msg
}

// Open opens the named file using os.Open and prepares it for use as a Plan 9 a.out binary.
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

func parseMagic(magic []byte) (uint32, error) {
	m := binary.BigEndian.Uint32(magic)
	switch m {
	case Magic386, MagicAMD64, MagicARM:
		return m, nil
	}
	return 0, &formatError{0, "bad magic number", magic}
}

// NewFile creates a new File for accessing a Plan 9 binary in an underlying reader.
// The Plan 9 binary is expected to start at position 0 in the ReaderAt.
func NewFile(r io.ReaderAt) (*File, error) {
	sr := io.NewSectionReader(r, 0, 1<<63-1)
	// Read and decode Plan 9 magic
	var magic [4]byte
	if _, err := r.ReadAt(magic[:], 0); err != nil {
		return nil, err
	}
	_, err := parseMagic(magic[:])
	if err != nil {
		return nil, err
	}

	ph := new(prog)
	if err := binary.Read(sr, binary.BigEndian, ph); err != nil {
		return nil, err
	}

	f := &File{FileHeader: FileHeader{
		Magic:       ph.Magic,
		Bss:         ph.Bss,
		Entry:       uint64(ph.Entry),
		PtrSize:     4,
		LoadAddress: 0x1000,
		HdrSize:     4 * 8,
	}}

	if ph.Magic&Magic64 != 0 {
		if err := binary.Read(sr, binary.BigEndian, &f.Entry); err != nil {
			return nil, err
		}
		f.PtrSize = 8
		f.LoadAddress = 0x200000
		f.HdrSize += 8
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

	off := uint32(f.HdrSize)

	for i, sect := range sects {
		s := new(Section)
		s.SectionHeader = SectionHeader{
			Name:   sect.name,
			Size:   sect.size,
			Offset: off,
		}
		off += sect.size
		s.sr = io.NewSectionReader(r, int64(s.Offset), int64(s.Size))
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
			return &formatError{len(data), "unexpected EOF", nil}
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
			return &formatError{len(data), "unexpected EOF", nil}
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
			ts.Name = string(s.name)
		case 'z', 'Z':
			for i := 0; i < len(s.name); i += 2 {
				eltIdx := binary.BigEndian.Uint16(s.name[i : i+2])
				elt, ok := fname[eltIdx]
				if !ok {
					return &formatError{-1, "bad filename code", eltIdx}
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

	return newTable(symtab, f.PtrSize)
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
