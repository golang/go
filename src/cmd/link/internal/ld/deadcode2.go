// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/objfile"
	"cmd/link/internal/sym"
	"fmt"
	"strings"
)

var _ = fmt.Print

// TODO:
// - Live method tracking:
//   Prune methods that are not directly called and cannot
//   be potentially called by interface or reflect call.
//   For now, all the methods from reachable type are alive.
// - Shared object support:
//   It basically marks everything. We could consider using
//   a different mechanism to represent it.
// - Field tracking support:
//   It needs to record from where the symbol is referenced.

type workQueue []objfile.Sym

func (q *workQueue) push(i objfile.Sym) { *q = append(*q, i) }
func (q *workQueue) pop() objfile.Sym   { i := (*q)[len(*q)-1]; *q = (*q)[:len(*q)-1]; return i }
func (q *workQueue) empty() bool        { return len(*q) == 0 }

type deadcodePass2 struct {
	ctxt   *Link
	loader *objfile.Loader
	wq     workQueue
}

func (d *deadcodePass2) init() {
	d.loader.InitReachable()

	var names []string

	// In a normal binary, start at main.main and the init
	// functions and mark what is reachable from there.
	if d.ctxt.linkShared && (d.ctxt.BuildMode == BuildModeExe || d.ctxt.BuildMode == BuildModePIE) {
		names = append(names, "main.main", "main..inittask")
	} else {
		// The external linker refers main symbol directly.
		if d.ctxt.LinkMode == LinkExternal && (d.ctxt.BuildMode == BuildModeExe || d.ctxt.BuildMode == BuildModePIE) {
			if d.ctxt.HeadType == objabi.Hwindows && d.ctxt.Arch.Family == sys.I386 {
				*flagEntrySymbol = "_main"
			} else {
				*flagEntrySymbol = "main"
			}
		}
		names = append(names, *flagEntrySymbol)
		if d.ctxt.BuildMode == BuildModePlugin {
			names = append(names, objabi.PathToPrefix(*flagPluginPath)+"..inittask", objabi.PathToPrefix(*flagPluginPath)+".main", "go.plugin.tabs")

			// We don't keep the go.plugin.exports symbol,
			// but we do keep the symbols it refers to.
			exportsIdx := d.loader.Lookup("go.plugin.exports", 0)
			if exportsIdx != 0 {
				relocs := d.loader.Relocs(exportsIdx)
				for i := 0; i < relocs.Count; i++ {
					d.mark(relocs.At(i).Sym)
				}
			}
		}
	}
	for _, s := range dynexp {
		d.mark(d.loader.Lookup(s.Name, int(s.Version)))
	}

	for _, name := range names {
		// Mark symbol as an data/ABI0 symbol.
		d.mark(d.loader.Lookup(name, 0))
		// Also mark any Go functions (internal ABI).
		d.mark(d.loader.Lookup(name, sym.SymVerABIInternal))
	}
}

func (d *deadcodePass2) flood() {
	for !d.wq.empty() {
		symIdx := d.wq.pop()
		relocs := d.loader.Relocs(symIdx)
		for i := 0; i < relocs.Count; i++ {
			r := relocs.At(i)
			if r.Type == objabi.R_WEAKADDROFF {
				continue
			}
			if r.Type == objabi.R_METHODOFF {
				// TODO: we should do something about it
				// For now, all the methods are considered live
			}
			d.mark(r.Sym)
		}
		naux := d.loader.NAux(symIdx)
		for i := 0; i < naux; i++ {
			d.mark(d.loader.AuxSym(symIdx, i))
		}
	}
}

func (d *deadcodePass2) mark(symIdx objfile.Sym) {
	if symIdx != 0 && !d.loader.Reachable.Has(symIdx) {
		d.wq.push(symIdx)
		d.loader.Reachable.Set(symIdx)
	}
}

func deadcode2(ctxt *Link) {
	loader := ctxt.loader
	d := deadcodePass2{ctxt: ctxt, loader: loader}
	d.init()
	d.flood()

	n := loader.NSym()
	if ctxt.BuildMode != BuildModeShared {
		// Keep a itablink if the symbol it points at is being kept.
		// (When BuildModeShared, always keep itablinks.)
		for i := 1; i < n; i++ {
			s := objfile.Sym(i)
			if strings.HasPrefix(loader.RawSymName(s), "go.itablink.") {
				relocs := loader.Relocs(s)
				if relocs.Count > 0 && loader.Reachable.Has(relocs.At(0).Sym) {
					loader.Reachable.Set(s)
				}
			}
		}
	}

	// Set reachable attr for now.
	for i := 1; i < n; i++ {
		if loader.Reachable.Has(objfile.Sym(i)) {
			s := loader.Syms[i]
			if s != nil && s.Name != "" {
				s.Attr.Set(sym.AttrReachable, true)
			}
		}
	}
}
