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
// functions in lib.go, still being used for some archs/oses.

func Entryvalue(ctxt *Link) int64 {
	a := *flagEntrySymbol
	if a[0] >= '0' && a[0] <= '9' {
		return atolwhex(a)
	}
	s := ctxt.Syms.Lookup(a, 0)
	if s.Type == 0 {
		return *FlagTextAddr
	}
	if ctxt.HeadType != objabi.Haix && s.Type != sym.STEXT {
		Errorf(s, "entry not text")
	}
	return s.Value
}

func datoff2(s *sym.Symbol, addr int64) int64 {
	if uint64(addr) >= Segdata.Vaddr {
		return int64(uint64(addr) - Segdata.Vaddr + Segdata.Fileoff)
	}
	if uint64(addr) >= Segtext.Vaddr {
		return int64(uint64(addr) - Segtext.Vaddr + Segtext.Fileoff)
	}
	Errorf(s, "invalid datoff %#x", addr)
	return 0
}

func ElfSymForReloc(ctxt *Link, s *sym.Symbol) int32 {
	// If putelfsym created a local version of this symbol, use that in all
	// relocations.
	les := ctxt.loader.SymLocalElfSym(loader.Sym(s.SymIdx))
	if les != 0 {
		return les
	} else {
		return ctxt.loader.SymElfSym(loader.Sym(s.SymIdx))
	}
}
