// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/objabi"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
)

// Temporary dumping around for sym.Symbol version of helper
// functions in xcoff.go, still being used for some archs/oses.
// FIXME: get rid of this file when dodata() is completely
// converted.

// xcoffUpdateOuterSize stores the size of outer symbols in order to have it
// in the symbol table.
func xcoffUpdateOuterSize(ctxt *Link, size int64, stype sym.SymKind) {
	if size == 0 {
		return
	}

	switch stype {
	default:
		Errorf(nil, "unknown XCOFF outer symbol for type %s", stype.String())
	case sym.SRODATA, sym.SRODATARELRO, sym.SFUNCTAB, sym.SSTRING:
		// Nothing to do
	case sym.STYPERELRO:
		if ctxt.UseRelro() && (ctxt.BuildMode == BuildModeCArchive || ctxt.BuildMode == BuildModeCShared || ctxt.BuildMode == BuildModePIE) {
			// runtime.types size must be removed, as it's a real symbol.
			outerSymSize["typerel.*"] = size - ctxt.Syms.ROLookup("runtime.types", 0).Size
			return
		}
		fallthrough
	case sym.STYPE:
		if !ctxt.DynlinkingGo() {
			// runtime.types size must be removed, as it's a real symbol.
			outerSymSize["type.*"] = size - ctxt.Syms.ROLookup("runtime.types", 0).Size
		}
	case sym.SGOSTRING:
		outerSymSize["go.string.*"] = size
	case sym.SGOFUNC:
		if !ctxt.DynlinkingGo() {
			outerSymSize["go.func.*"] = size
		}
	case sym.SGOFUNCRELRO:
		outerSymSize["go.funcrel.*"] = size
	case sym.SGCBITS:
		outerSymSize["runtime.gcbits.*"] = size
	case sym.SITABLINK:
		outerSymSize["runtime.itablink"] = size

	}
}

// Xcoffadddynrel adds a dynamic relocation in a XCOFF file.
// This relocation will be made by the loader.
func Xcoffadddynrel(target *Target, ldr *loader.Loader, s *sym.Symbol, r *sym.Reloc) bool {
	if target.IsExternal() {
		return true
	}
	if s.Type <= sym.SPCLNTAB {
		Errorf(s, "cannot have a relocation to %s in a text section symbol", r.Sym.Name)
		return false
	}

	xldr := &xcoffLoaderReloc{
		sym:  s,
		roff: r.Off,
	}

	switch r.Type {
	default:
		Errorf(s, "unexpected .loader relocation to symbol: %s (type: %s)", r.Sym.Name, r.Type.String())
		return false
	case objabi.R_ADDR:
		if s.Type == sym.SXCOFFTOC && r.Sym.Type == sym.SDYNIMPORT {
			// Imported symbol relocation
			for i, dynsym := range xfile.loaderSymbols {
				if ldr.Syms[dynsym.sym].Name == r.Sym.Name {
					xldr.symndx = int32(i + 3) // +3 because of 3 section symbols
					break
				}
			}
		} else if s.Type == sym.SDATA || s.Type == sym.SNOPTRDATA || s.Type == sym.SBUILDINFO || s.Type == sym.SXCOFFTOC {
			switch r.Sym.Sect.Seg {
			default:
				Errorf(s, "unknown segment for .loader relocation with symbol %s", r.Sym.Name)
			case &Segtext:
			case &Segrodata:
				xldr.symndx = 0 // .text
			case &Segdata:
				if r.Sym.Type == sym.SBSS || r.Sym.Type == sym.SNOPTRBSS {
					xldr.symndx = 2 // .bss
				} else {
					xldr.symndx = 1 // .data
				}

			}

		} else {
			Errorf(s, "unexpected type for .loader relocation R_ADDR for symbol %s: %s to %s", r.Sym.Name, s.Type, r.Sym.Type)
			return false
		}

		xldr.rtype = 0x3F<<8 + XCOFF_R_POS
	}

	xfile.loaderReloc = append(xfile.loaderReloc, xldr)
	return true
}
