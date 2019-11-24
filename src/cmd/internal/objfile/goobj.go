// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parsing of Go intermediate object files and archives.

package objfile

import (
	"cmd/internal/goobj"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"debug/dwarf"
	"debug/gosym"
	"errors"
	"fmt"
	"os"
)

type goobjFile struct {
	goobj *goobj.Package
	f     *os.File // the underlying .o or .a file
}

func openGoFile(r *os.File) (*File, error) {
	f, err := goobj.Parse(r, `""`)
	if err != nil {
		return nil, err
	}
	rf := &goobjFile{goobj: f, f: r}
	if len(f.Native) == 0 {
		return &File{r, []*Entry{{raw: rf}}}, nil
	}
	entries := make([]*Entry, len(f.Native)+1)
	entries[0] = &Entry{
		raw: rf,
	}
L:
	for i, nr := range f.Native {
		for _, try := range openers {
			if raw, err := try(nr); err == nil {
				entries[i+1] = &Entry{
					name: nr.Name,
					raw:  raw,
				}
				continue L
			}
		}
		return nil, fmt.Errorf("open %s: unrecognized archive member %s", r.Name(), nr.Name)
	}
	return &File{r, entries}, nil
}

func goobjName(id goobj.SymID) string {
	if id.Version == 0 {
		return id.Name
	}
	return fmt.Sprintf("%s<%d>", id.Name, id.Version)
}

func (f *goobjFile) symbols() ([]Sym, error) {
	seen := make(map[goobj.SymID]bool)

	var syms []Sym
	for _, s := range f.goobj.Syms {
		seen[s.SymID] = true
		sym := Sym{Addr: uint64(s.Data.Offset), Name: goobjName(s.SymID), Size: s.Size, Type: s.Type.Name, Code: '?'}
		switch s.Kind {
		case objabi.STEXT:
			sym.Code = 'T'
		case objabi.SRODATA:
			sym.Code = 'R'
		case objabi.SDATA:
			sym.Code = 'D'
		case objabi.SBSS, objabi.SNOPTRBSS, objabi.STLSBSS:
			sym.Code = 'B'
		}
		if s.Version != 0 {
			sym.Code += 'a' - 'A'
		}
		for i, r := range s.Reloc {
			sym.Relocs = append(sym.Relocs, Reloc{Addr: uint64(s.Data.Offset) + uint64(r.Offset), Size: uint64(r.Size), Stringer: &s.Reloc[i]})
		}
		syms = append(syms, sym)
	}

	for _, s := range f.goobj.Syms {
		for _, r := range s.Reloc {
			if !seen[r.Sym] {
				seen[r.Sym] = true
				sym := Sym{Name: goobjName(r.Sym), Code: 'U'}
				if s.Version != 0 {
					// should not happen but handle anyway
					sym.Code = 'u'
				}
				syms = append(syms, sym)
			}
		}
	}

	return syms, nil
}

func (f *goobjFile) pcln() (textStart uint64, symtab, pclntab []byte, err error) {
	// Should never be called. We implement Liner below, callers
	// should use that instead.
	return 0, nil, nil, fmt.Errorf("pcln not available in go object file")
}

// Find returns the file name, line, and function data for the given pc.
// Returns "",0,nil if unknown.
// This function implements the Liner interface in preference to pcln() above.
func (f *goobjFile) PCToLine(pc uint64) (string, int, *gosym.Func) {
	// TODO: this is really inefficient. Binary search? Memoize last result?
	var arch *sys.Arch
	for _, a := range sys.Archs {
		if a.Name == f.goobj.Arch {
			arch = a
			break
		}
	}
	if arch == nil {
		return "", 0, nil
	}
	for _, s := range f.goobj.Syms {
		if pc < uint64(s.Data.Offset) || pc >= uint64(s.Data.Offset+s.Data.Size) {
			continue
		}
		if s.Func == nil {
			return "", 0, nil
		}
		pcfile := make([]byte, s.Func.PCFile.Size)
		_, err := f.f.ReadAt(pcfile, s.Func.PCFile.Offset)
		if err != nil {
			return "", 0, nil
		}
		fileID := int(pcValue(pcfile, pc-uint64(s.Data.Offset), arch))
		fileName := s.Func.File[fileID]
		pcline := make([]byte, s.Func.PCLine.Size)
		_, err = f.f.ReadAt(pcline, s.Func.PCLine.Offset)
		if err != nil {
			return "", 0, nil
		}
		line := int(pcValue(pcline, pc-uint64(s.Data.Offset), arch))
		// Note: we provide only the name in the Func structure.
		// We could provide more if needed.
		return fileName, line, &gosym.Func{Sym: &gosym.Sym{Name: s.Name}}
	}
	return "", 0, nil
}

// pcValue looks up the given PC in a pc value table. target is the
// offset of the pc from the entry point.
func pcValue(tab []byte, target uint64, arch *sys.Arch) int32 {
	val := int32(-1)
	var pc uint64
	for step(&tab, &pc, &val, pc == 0, arch) {
		if target < pc {
			return val
		}
	}
	return -1
}

// step advances to the next pc, value pair in the encoded table.
func step(p *[]byte, pc *uint64, val *int32, first bool, arch *sys.Arch) bool {
	uvdelta := readvarint(p)
	if uvdelta == 0 && !first {
		return false
	}
	if uvdelta&1 != 0 {
		uvdelta = ^(uvdelta >> 1)
	} else {
		uvdelta >>= 1
	}
	vdelta := int32(uvdelta)
	pcdelta := readvarint(p) * uint32(arch.MinLC)
	*pc += uint64(pcdelta)
	*val += vdelta
	return true
}

// readvarint reads, removes, and returns a varint from *p.
func readvarint(p *[]byte) uint32 {
	var v, shift uint32
	s := *p
	for shift = 0; ; shift += 7 {
		b := s[0]
		s = s[1:]
		v |= (uint32(b) & 0x7F) << shift
		if b&0x80 == 0 {
			break
		}
	}
	*p = s
	return v
}

// We treat the whole object file as the text section.
func (f *goobjFile) text() (textStart uint64, text []byte, err error) {
	var info os.FileInfo
	info, err = f.f.Stat()
	if err != nil {
		return
	}
	text = make([]byte, info.Size())
	_, err = f.f.ReadAt(text, 0)
	return
}

func (f *goobjFile) goarch() string {
	return f.goobj.Arch
}

func (f *goobjFile) loadAddress() (uint64, error) {
	return 0, fmt.Errorf("unknown load address")
}

func (f *goobjFile) dwarf() (*dwarf.Data, error) {
	return nil, errors.New("no DWARF data in go object file")
}
