// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parsing of Go intermediate object files and archives.

package main

import (
	"fmt"
	"os"
)

func goobjName(id goobj_SymID) string {
	if id.Version == 0 {
		return id.Name
	}
	return fmt.Sprintf("%s<%d>", id.Name, id.Version)
}

func goobjSymbols(f *os.File) []Sym {
	pkg, err := goobj_Parse(f, `""`)
	if err != nil {
		errorf("parsing %s: %v", f.Name(), err)
		return nil
	}

	seen := make(map[goobj_SymID]bool)

	var syms []Sym
	for _, s := range pkg.Syms {
		seen[s.goobj_SymID] = true
		sym := Sym{Addr: uint64(s.Data.Offset), Name: goobjName(s.goobj_SymID), Size: int64(s.Size), Type: s.Type.Name, Code: '?'}
		switch s.Kind {
		case goobj_STEXT, goobj_SELFRXSECT:
			sym.Code = 'T'
		case goobj_STYPE, goobj_SSTRING, goobj_SGOSTRING, goobj_SGOFUNC, goobj_SRODATA, goobj_SFUNCTAB, goobj_STYPELINK, goobj_SSYMTAB, goobj_SPCLNTAB, goobj_SELFROSECT:
			sym.Code = 'R'
		case goobj_SMACHOPLT, goobj_SELFSECT, goobj_SMACHO, goobj_SMACHOGOT, goobj_SNOPTRDATA, goobj_SINITARR, goobj_SDATA, goobj_SWINDOWS:
			sym.Code = 'D'
		case goobj_SBSS, goobj_SNOPTRBSS, goobj_STLSBSS:
			sym.Code = 'B'
		case goobj_SXREF, goobj_SMACHOSYMSTR, goobj_SMACHOSYMTAB, goobj_SMACHOINDIRECTPLT, goobj_SMACHOINDIRECTGOT, goobj_SFILE, goobj_SFILEPATH, goobj_SCONST, goobj_SDYNIMPORT, goobj_SHOSTOBJ:
			sym.Code = 'X' // should not see
		}
		if s.Version != 0 {
			sym.Code += 'a' - 'A'
		}
		syms = append(syms, sym)
	}

	for _, s := range pkg.Syms {
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

	return syms
}
