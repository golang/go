// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parsing of Go intermediate object files and archives.

package objfile

import (
	"cmd/internal/goobj"
	"fmt"
	"os"
)

type goobjFile struct {
	goobj *goobj.Package
}

func openGoobj(r *os.File) (rawFile, error) {
	f, err := goobj.Parse(r, `""`)
	if err != nil {
		return nil, err
	}
	return &goobjFile{f}, nil
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
		sym := Sym{Addr: uint64(s.Data.Offset), Name: goobjName(s.SymID), Size: int64(s.Size), Type: s.Type.Name, Code: '?'}
		switch s.Kind {
		case goobj.STEXT, goobj.SELFRXSECT:
			sym.Code = 'T'
		case goobj.STYPE, goobj.SSTRING, goobj.SGOSTRING, goobj.SGOFUNC, goobj.SRODATA, goobj.SFUNCTAB, goobj.STYPELINK, goobj.SSYMTAB, goobj.SPCLNTAB, goobj.SELFROSECT:
			sym.Code = 'R'
		case goobj.SMACHOPLT, goobj.SELFSECT, goobj.SMACHO, goobj.SMACHOGOT, goobj.SNOPTRDATA, goobj.SINITARR, goobj.SDATA, goobj.SWINDOWS:
			sym.Code = 'D'
		case goobj.SBSS, goobj.SNOPTRBSS, goobj.STLSBSS:
			sym.Code = 'B'
		case goobj.SXREF, goobj.SMACHOSYMSTR, goobj.SMACHOSYMTAB, goobj.SMACHOINDIRECTPLT, goobj.SMACHOINDIRECTGOT, goobj.SFILE, goobj.SFILEPATH, goobj.SCONST, goobj.SDYNIMPORT, goobj.SHOSTOBJ:
			sym.Code = 'X' // should not see
		}
		if s.Version != 0 {
			sym.Code += 'a' - 'A'
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

// pcln does not make sense for Go object files, because each
// symbol has its own individual pcln table, so there is no global
// space of addresses to map.
func (f *goobjFile) pcln() (textStart uint64, symtab, pclntab []byte, err error) {
	return 0, nil, nil, fmt.Errorf("pcln not available in go object file")
}

// text does not make sense for Go object files, because
// each function has a separate section.
func (f *goobjFile) text() (textStart uint64, text []byte, err error) {
	return 0, nil, fmt.Errorf("text not available in go object file")
}

// goarch makes sense but is not exposed in debug/goobj's API,
// and we don't need it yet for any users of internal/objfile.
func (f *goobjFile) goarch() string {
	return "GOARCH unimplemented for debug/goobj files"
}
