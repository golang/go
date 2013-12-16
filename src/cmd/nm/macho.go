// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parsing of Mach-O executables (OS X).

package main

import (
	"debug/macho"
	"os"
	"sort"
)

func machoSymbols(f *os.File) []Sym {
	p, err := macho.NewFile(f)
	if err != nil {
		errorf("parsing %s: %v", f.Name(), err)
		return nil
	}

	if p.Symtab == nil {
		errorf("%s: no symbol table", f.Name())
		return nil
	}

	// Build sorted list of addresses of all symbols.
	// We infer the size of a symbol by looking at where the next symbol begins.
	var addrs []uint64
	for _, s := range p.Symtab.Syms {
		addrs = append(addrs, s.Value)
	}
	sort.Sort(uint64s(addrs))

	var syms []Sym
	for _, s := range p.Symtab.Syms {
		sym := Sym{Name: s.Name, Addr: s.Value, Code: '?'}
		i := sort.Search(len(addrs), func(x int) bool { return addrs[x] > s.Value })
		if i < len(addrs) {
			sym.Size = int64(addrs[i] - s.Value)
		}
		if s.Sect == 0 {
			sym.Code = 'U'
		} else if int(s.Sect) <= len(p.Sections) {
			sect := p.Sections[s.Sect-1]
			switch sect.Seg {
			case "__TEXT":
				sym.Code = 'R'
			case "__DATA":
				sym.Code = 'D'
			}
			switch sect.Seg + " " + sect.Name {
			case "__TEXT __text":
				sym.Code = 'T'
			case "__DATA __bss", "__DATA __noptrbss":
				sym.Code = 'B'
			}
		}
		syms = append(syms, sym)
	}

	return syms
}

type uint64s []uint64

func (x uint64s) Len() int           { return len(x) }
func (x uint64s) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x uint64s) Less(i, j int) bool { return x[i] < x[j] }
