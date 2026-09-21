// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"cmp"
	"slices"
)

var sehp struct {
	pdata []sym.LoaderSym
	xdata []sym.LoaderSym
}

// collectSEH collects the SEH unwind information for all functions and organizes
// it into .pdata and .xdata sections.
func collectSEH(ctxt *Link) {
	switch ctxt.Arch.Family {
	case sys.AMD64:
		collectSEHAMD64(ctxt)
	}
}

func collectSEHAMD64(ctxt *Link) {
	ldr := ctxt.loader
	mkSecSym := func(name string, kind sym.SymKind) *loader.SymbolBuilder {
		s := ldr.CreateSymForUpdate(name, 0)
		s.SetType(kind)
		s.SetAlign(4)
		return s
	}
	pdata := mkSecSym(".pdata", sym.SSEHSECT)
	xdata := mkSecSym(".xdata", sym.SSEHSECT)
	// The .xdata entries have very low cardinality
	// as it only contains frame pointer operations,
	// which are very similar across functions.
	// These are referenced by .pdata entries using
	// an RVA, so it is possible, and binary-size wise,
	// to deduplicate .xdata entries.
	uwcache := make(map[string]int64) // aux symbol name --> .xdata offset
	type pdataEntry struct {
		start    sym.LoaderSym
		xdataOff int64
	}
	var entries []pdataEntry
	for _, s := range ctxt.Textp {
		if fi := ldr.FuncInfo(s); !fi.Valid() {
			continue
		}
		uw := ldr.SEHUnwindSym(s)
		if uw == 0 {
			continue
		}
		name := ctxt.SymName(uw)
		off, cached := uwcache[name]
		if !cached {
			off = xdata.Size()
			uwcache[name] = off
			xdata.AddBytes(ldr.Data(uw))
			// The SEH unwind data can contain relocations,
			// make sure those are copied over.
			rels := ldr.Relocs(uw)
			for i := 0; i < rels.Count(); i++ {
				r := rels.At(i)
				rel, _ := xdata.AddRel(r.Type())
				rel.SetOff(int32(off) + r.Off())
				rel.SetSiz(r.Siz())
				rel.SetSym(r.Sym())
				rel.SetAdd(r.Add())
			}
		}

		entries = append(entries, pdataEntry{start: s, xdataOff: off})
	}
	slices.SortFunc(entries, func(a, b pdataEntry) int {
		return cmp.Compare(ldr.SymAddr(a.start), ldr.SymAddr(b.start))
	})
	for _, ent := range entries {
		// Reference:
		// https://learn.microsoft.com/en-us/cpp/build/exception-handling-x64#struct-runtime_function
		pdata.AddPEImageRelativeAddrPlus(ctxt.Arch, ent.start, 0)                      // function start address
		pdata.AddPEImageRelativeAddrPlus(ctxt.Arch, ent.start, ldr.SymSize(ent.start)) // function end address
		pdata.AddPEImageRelativeAddrPlus(ctxt.Arch, xdata.Sym(), ent.xdataOff)         // xdata symbol offset
	}
	if pdata.Size() > 0 {
		sehp.pdata = append(sehp.pdata, pdata.Sym())
	}
	if xdata.Size() > 0 {
		sehp.xdata = append(sehp.xdata, xdata.Sym())
	}
}
