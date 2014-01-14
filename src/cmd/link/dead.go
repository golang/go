// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Removal of dead code and data.

package main

import "debug/goobj"

// dead removes unreachable code and data from the program.
// It is basically a mark-sweep garbage collection: traverse all the
// symbols reachable from the entry (startSymID) and then delete
// the rest.
func (p *Prog) dead() {
	p.Dead = make(map[goobj.SymID]bool)
	reachable := make(map[goobj.SymID]bool)
	p.walkDead(p.startSym, reachable)

	for sym := range p.Syms {
		if !reachable[sym] {
			delete(p.Syms, sym)
			p.Dead[sym] = true
		}
	}

	for sym := range p.Missing {
		if !reachable[sym] {
			delete(p.Missing, sym)
			p.Dead[sym] = true
		}
	}

	p.SymOrder = removeDead(p.SymOrder, reachable)

	for _, pkg := range p.Packages {
		pkg.Syms = removeDead(pkg.Syms, reachable)
	}
}

// walkDead traverses the symbols reachable from sym, adding them to reachable.
// The caller has verified that reachable[sym] = false.
func (p *Prog) walkDead(sym goobj.SymID, reachable map[goobj.SymID]bool) {
	reachable[sym] = true
	s := p.Syms[sym]
	if s == nil {
		return
	}
	for i := range s.Reloc {
		r := &s.Reloc[i]
		if !reachable[r.Sym] {
			p.walkDead(r.Sym, reachable)
		}
	}
	if s.Func != nil {
		for _, fdata := range s.Func.FuncData {
			if fdata.Sym.Name != "" && !reachable[fdata.Sym] {
				p.walkDead(fdata.Sym, reachable)
			}
		}
	}
}

// removeDead removes unreachable (dead) symbols from syms,
// returning a shortened slice using the same underlying array.
func removeDead(syms []*Sym, reachable map[goobj.SymID]bool) []*Sym {
	keep := syms[:0]
	for _, sym := range syms {
		if reachable[sym.SymID] {
			keep = append(keep, sym)
		}
	}
	return keep
}
