// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parsing of Plan 9 a.out executables.

package main

import (
	"debug/plan9obj"
	"os"
	"sort"
)

func plan9Symbols(f *os.File) (syms []Sym, goarch string) {
	p, err := plan9obj.NewFile(f)
	if err != nil {
		errorf("parsing %s: %v", f.Name(), err)
		return
	}

	plan9Syms, err := p.Symbols()
	if err != nil {
		errorf("parsing %s: %v", f.Name(), err)
		return
	}

	goarch = "386"

	// Build sorted list of addresses of all symbols.
	// We infer the size of a symbol by looking at where the next symbol begins.
	var addrs []uint64
	for _, s := range plan9Syms {
		addrs = append(addrs, s.Value)
	}
	sort.Sort(uint64s(addrs))

	for _, s := range plan9Syms {
		sym := Sym{Addr: s.Value, Name: s.Name, Code: rune(s.Type)}
		i := sort.Search(len(addrs), func(x int) bool { return addrs[x] > s.Value })
		if i < len(addrs) {
			sym.Size = int64(addrs[i] - s.Value)
		}
		syms = append(syms, sym)
	}

	return
}
