// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parsing of Plan 9 a.out executables.

package objfile

import (
	"debug/dwarf"
	"debug/plan9obj"
	"errors"
	"fmt"
	"io"
	"slices"
	"sort"
)

var validSymType = map[rune]bool{
	'T': true,
	't': true,
	'D': true,
	'd': true,
	'B': true,
	'b': true,
}

type plan9File struct {
	plan9 *plan9obj.File
}

func openPlan9(r io.ReaderAt) (rawFile, error) {
	f, err := plan9obj.NewFile(r)
	if err != nil {
		return nil, err
	}
	return &plan9File{f}, nil
}

func (f *plan9File) symbols() ([]Sym, error) {
	plan9Syms, err := f.plan9.Symbols()
	if err != nil {
		return nil, err
	}

	// Build sorted list of addresses of all symbols.
	// We infer the size of a symbol by looking at where the next symbol begins.
	var addrs []uint64
	for _, s := range plan9Syms {
		if !validSymType[s.Type] {
			continue
		}
		addrs = append(addrs, s.Value)
	}
	slices.Sort(addrs)

	var syms []Sym

	for _, s := range plan9Syms {
		if !validSymType[s.Type] {
			continue
		}
		sym := Sym{Addr: s.Value, Name: s.Name, Code: s.Type}
		i := sort.Search(len(addrs), func(x int) bool { return addrs[x] > s.Value })
		if i < len(addrs) {
			sym.Size = int64(addrs[i] - s.Value)
		}
		syms = append(syms, sym)
	}

	return syms, nil
}

func (f *plan9File) pcln() (textStart uint64, symtab, pclntab []byte, err error) {
	textStart = f.plan9.LoadAddress + f.plan9.HdrSize
	if pclntab, err = loadPlan9Table(f.plan9, "runtime.pclntab", "runtime.epclntab"); err != nil {
		// We didn't find the symbols, so look for the names used in 1.3 and earlier.
		// TODO: Remove code looking for the old symbols when we no longer care about 1.3.
		var err2 error
		if pclntab, err2 = loadPlan9Table(f.plan9, "pclntab", "epclntab"); err2 != nil {
			return 0, nil, nil, err
		}
	}
	if symtab, err = loadPlan9Table(f.plan9, "runtime.symtab", "runtime.esymtab"); err != nil {
		// Same as above.
		var err2 error
		if symtab, err2 = loadPlan9Table(f.plan9, "symtab", "esymtab"); err2 != nil {
			return 0, nil, nil, err
		}
	}
	return textStart, symtab, pclntab, nil
}

func (f *plan9File) text() (textStart uint64, text []byte, err error) {
	sect := f.plan9.Section("text")
	if sect == nil {
		return 0, nil, fmt.Errorf("text section not found")
	}
	textStart = f.plan9.LoadAddress + f.plan9.HdrSize
	text, err = sect.Data()
	return
}

func findPlan9Symbol(f *plan9obj.File, name string) (*plan9obj.Sym, error) {
	syms, err := f.Symbols()
	if err != nil {
		return nil, err
	}
	for _, s := range syms {
		if s.Name != name {
			continue
		}
		return &s, nil
	}
	return nil, fmt.Errorf("no %s symbol found", name)
}

func loadPlan9Table(f *plan9obj.File, sname, ename string) ([]byte, error) {
	ssym, err := findPlan9Symbol(f, sname)
	if err != nil {
		return nil, err
	}
	esym, err := findPlan9Symbol(f, ename)
	if err != nil {
		return nil, err
	}
	sect := f.Section("text")
	if sect == nil {
		return nil, err
	}
	data, err := sect.Data()
	if err != nil {
		return nil, err
	}
	textStart := f.LoadAddress + f.HdrSize
	return data[ssym.Value-textStart : esym.Value-textStart], nil
}

func (f *plan9File) goarch() string {
	switch f.plan9.Magic {
	case plan9obj.Magic386:
		return "386"
	case plan9obj.MagicAMD64:
		return "amd64"
	case plan9obj.MagicARM:
		return "arm"
	}
	return ""
}

func (f *plan9File) loadAddress() (uint64, error) {
	return 0, fmt.Errorf("unknown load address")
}

func (f *plan9File) dwarf() (*dwarf.Data, error) {
	return nil, errors.New("no DWARF data in Plan 9 file")
}
