// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parsing of Go intermediate object files and archives.

package main

import (
	"debug/goobj"
	"fmt"
	"os"
)

func goobjName(id goobj.SymID) string {
	if id.Version == 0 {
		return id.Name
	}
	return fmt.Sprintf("%s<%d>", id.Name, id.Version)
}

func goobjSymbols(f *os.File) []Sym {
	pkg, err := goobj.Parse(f, `""`)
	if err != nil {
		errorf("parsing %s: %v", f.Name(), err)
		return nil
	}

	seen := make(map[goobj.SymID]bool)

	var syms []Sym
	for _, s := range pkg.Syms {
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
