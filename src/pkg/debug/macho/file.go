// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package macho implements access to Mach-O object files, as defined by
// http://developer.apple.com/mac/library/documentation/DeveloperTools/Conceptual/MachORuntime/Reference/reference.html.
package macho

// High level access to low level data structures.

import (
	"bytes";
	"debug/binary";
	"debug/dwarf";
	"fmt";
	"io";
	"os";
)

// A File represents an open Mach-O file.
type File struct {
	FileHeader;
	ByteOrder binary.ByteOrder;
	Loads []Load;
	Sections []*Section;

	closer io.Closer;
}

// A Load represents any Mach-O load command.
type Load interface {
	Raw() []byte
}

// A LoadBytes is the uninterpreted bytes of a Mach-O load command.
type LoadBytes []byte

func (b LoadBytes) Raw() []byte {
	return b
}

// A SegmentHeader is the header for a Mach-O 32-bit or 64-bit load segment command.
type SegmentHeader struct {
	Cmd LoadCmd;
	Len uint32;
	Name string;
	Addr uint64;
	Memsz uint64;
	Offset uint64;
	Filesz uint64;
	Maxprot uint32;
	Prot uint32;
	Nsect uint32;
	Flag uint32;
}

// A Segment represents a Mach-O 32-bit or 64-bit load segment command.
type Segment struct {
	LoadBytes;
	SegmentHeader;

	// Embed ReaderAt for ReadAt method.
	// Do not embed SectionReader directly
	// to avoid having Read and Seek.
	// If a client wants Read and Seek it must use
	// Open() to avoid fighting over the seek offset
	// with other clients.
	io.ReaderAt;
	sr *io.SectionReader;
}

// Data reads and returns the contents of the segment.
func (s *Segment) Data() ([]byte, os.Error) {
	dat := make([]byte, s.sr.Size());
	n, err := s.sr.ReadAt(dat, 0);
	return dat[0:n], err;
}

// Open returns a new ReadSeeker reading the segment.
func (s *Segment) Open() io.ReadSeeker {
	return io.NewSectionReader(s.sr, 0, 1<<63 - 1);
}

type SectionHeader struct {
	Name	string;
	Seg	string;
	Addr	uint64;
	Size	uint64;
	Offset	uint32;
	Align	uint32;
	Reloff	uint32;
	Nreloc	uint32;
	Flags	uint32;
}

type Section struct {
	SectionHeader;

	// Embed ReaderAt for ReadAt method.
	// Do not embed SectionReader directly
	// to avoid having Read and Seek.
	// If a client wants Read and Seek it must use
	// Open() to avoid fighting over the seek offset
	// with other clients.
	io.ReaderAt;
	sr *io.SectionReader;
}

// Data reads and returns the contents of the Mach-O section.
func (s *Section) Data() ([]byte, os.Error) {
	dat := make([]byte, s.sr.Size());
	n, err := s.sr.ReadAt(dat, 0);
	return dat[0:n], err;
}

// Open returns a new ReadSeeker reading the Mach-O section.
func (s *Section) Open() io.ReadSeeker {
	return io.NewSectionReader(s.sr, 0, 1<<63 - 1);
}


/*
 * Mach-O reader
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

// Open opens the named file using os.Open and prepares it for use as a Mach-O binary.
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

// NewFile creates a new File for acecssing a Mach-O binary in an underlying reader.
// The Mach-O binary is expected to start at position 0 in the ReaderAt.
func NewFile(r io.ReaderAt) (*File, os.Error) {
	f := new(File);
	sr := io.NewSectionReader(r, 0, 1<<63 - 1);

	// Read and decode Mach magic to determine byte order, size.
	// Magic32 and Magic64 differ only in the bottom bit.
	var ident [4]uint8;
	if _, err := r.ReadAt(&ident, 0); err != nil {
		return nil, err;
	}
	be := binary.BigEndian.Uint32(&ident);
	le := binary.LittleEndian.Uint32(&ident);
	switch Magic32&^1 {
	case be&^1:
		f.ByteOrder = binary.BigEndian;
		f.Magic = be;
	case le&^1:
		f.ByteOrder = binary.LittleEndian;
		f.Magic = le;
	}

	// Read entire file header.
	if err := binary.Read(sr, f.ByteOrder, &f.FileHeader); err != nil {
		return nil, err;
	}

	// Then load commands.
	offset := int64(fileHeaderSize32);
	if f.Magic == Magic64 {
		offset = fileHeaderSize64;
	}
	dat := make([]byte, f.Cmdsz);
	if _, err := r.ReadAt(dat, offset); err != nil {
		return nil, err;
	}
	f.Loads = make([]Load, f.Ncmd);
	bo := f.ByteOrder;
	for i := range f.Loads {
		// Each load command begins with uint32 command and length.
		if len(dat) < 8 {
			return nil, &FormatError{offset, "command block too small", nil};
		}
		cmd, siz := LoadCmd(bo.Uint32(dat[0:4])), bo.Uint32(dat[4:8]);
		if siz < 8 || siz > uint32(len(dat)) {
			return nil, &FormatError{offset, "invalid command block size", nil};
		}
		var cmddat []byte;
		cmddat, dat = dat[0:siz], dat[siz:len(dat)];
		offset += int64(siz);
		var s *Segment;
		switch cmd {
		default:
			f.Loads[i] = LoadBytes(cmddat);

		case LoadCmdSegment:
			var seg32 Segment32;
			b := bytes.NewBuffer(cmddat);
			if err := binary.Read(b, bo, &seg32); err != nil {
				return nil, err;
			}
			s = new(Segment);
			s.LoadBytes = cmddat;
			s.Cmd = cmd;
			s.Len = siz;
			s.Name = cstring(&seg32.Name);
			s.Addr = uint64(seg32.Addr);
			s.Memsz = uint64(seg32.Memsz);
			s.Offset = uint64(seg32.Offset);
			s.Filesz = uint64(seg32.Filesz);
			s.Maxprot = seg32.Maxprot;
			s.Prot = seg32.Prot;
			s.Nsect = seg32.Nsect;
			s.Flag = seg32.Flag;
			f.Loads[i] = s;
			for i := 0; i < int(s.Nsect); i++ {
				var sh32 Section32;
				if err := binary.Read(b, bo, &sh32); err != nil {
					return nil, err;
				}
				sh := new(Section);
				sh.Name = cstring(&sh32.Name);
				sh.Seg = cstring(&sh32.Seg);
				sh.Addr = uint64(sh32.Addr);
				sh.Size = uint64(sh32.Size);
				sh.Offset = sh32.Offset;
				sh.Align = sh32.Align;
				sh.Reloff = sh32.Reloff;
				sh.Nreloc = sh32.Nreloc;
				sh.Flags = sh32.Flags;
				f.pushSection(sh, r);
			}

		case LoadCmdSegment64:
			var seg64 Segment64;
			b := bytes.NewBuffer(cmddat);
			if err := binary.Read(b, bo, &seg64); err != nil {
				return nil, err;
			}
			s = new(Segment);
			s.LoadBytes = cmddat;
			s.Cmd = cmd;
			s.Len = siz;
			s.Name = cstring(&seg64.Name);
			s.Addr = seg64.Addr;
			s.Memsz = seg64.Memsz;
			s.Offset = seg64.Offset;
			s.Filesz = seg64.Filesz;
			s.Maxprot = seg64.Maxprot;
			s.Prot = seg64.Prot;
			s.Nsect = seg64.Nsect;
			s.Flag = seg64.Flag;
			f.Loads[i] = s;
			for i := 0; i < int(s.Nsect); i++ {
				var sh64 Section64;
				if err := binary.Read(b, bo, &sh64); err != nil {
					return nil, err;
				}
				sh := new(Section);
				sh.Name = cstring(&sh64.Name);
				sh.Seg = cstring(&sh64.Seg);
				sh.Addr = sh64.Addr;
				sh.Size = sh64.Size;
				sh.Offset = sh64.Offset;
				sh.Align = sh64.Align;
				sh.Reloff = sh64.Reloff;
				sh.Nreloc = sh64.Nreloc;
				sh.Flags = sh64.Flags;
				f.pushSection(sh, r);
			}
		}
		if s != nil {
			s.sr = io.NewSectionReader(r, int64(s.Offset), int64(s.Filesz));
			s.ReaderAt = s.sr;
		}
	}
	return f, nil;
}

func (f *File) pushSection(sh *Section, r io.ReaderAt) {
	n := len(f.Sections);
	if n >= cap(f.Sections) {
		m := (n+1)*2;
		new := make([]*Section, n, m);
		for i, sh := range f.Sections {
			new[i] = sh;
		}
		f.Sections = new;
	}
	f.Sections = f.Sections[0:n+1];
	f.Sections[n] = sh;
	sh.sr = io.NewSectionReader(r, int64(sh.Offset), int64(sh.Size));
	sh.ReaderAt = sh.sr;
}

func cstring(b []byte) string {
	var i int;
	for i=0; i<len(b) && b[i] != 0; i++ {
	}
	return string(b[0:i]);
}

// Segment returns the first Segment with the given name, or nil if no such segment exists.
func (f *File) Segment(name string) *Segment {
	for _, l := range f.Loads {
		if s, ok := l.(*Segment); ok && s.Name == name {
			return s;
		}
	}
	return nil;
}

// Section returns the first section with the given name, or nil if no such
// section exists.
func (f *File) Section(name string) *Section {
	for _, s := range f.Sections {
		if s.Name == name {
			return s;
		}
	}
	return nil;
}

// DWARF returns the DWARF debug information for the Mach-O file.
func (f *File) DWARF() (*dwarf.Data, os.Error) {
	// There are many other DWARF sections, but these
	// are the required ones, and the debug/dwarf package
	// does not use the others, so don't bother loading them.
	var names = [...]string{"abbrev", "info", "str"};
	var dat [len(names)][]byte;
	for i, name := range names {
		name = "__debug_" + name;
		s := f.Section(name);
		if s == nil {
			return nil, os.NewError("missing Mach-O section " + name);
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
