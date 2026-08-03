// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objfile

import (
	"debug/macho"
	"testing"
)

func TestMachoSymbolsUseSectionEndForLastTextSymbol(t *testing.T) {
	f := &machoFile{macho: &macho.File{
		Symtab: &macho.Symtab{Syms: []macho.Symbol{
			{Name: "_f", Sect: 1, Value: 0x1000},
			{Name: "_g", Sect: 1, Value: 0x1010},
		}},
		Sections: []*macho.Section{
			{SectionHeader: macho.SectionHeader{Name: "__text", Seg: "__TEXT", Addr: 0x1000, Size: 0x30}},
		},
	}}

	syms, err := f.symbols()
	if err != nil {
		t.Fatal(err)
	}

	got := map[string]Sym{}
	for _, sym := range syms {
		got[sym.Name] = sym
	}

	if got["_f"].Size != 0x10 {
		t.Fatalf("_f size = %#x, want %#x", got["_f"].Size, int64(0x10))
	}
	if got["_g"].Size != 0x20 {
		t.Fatalf("_g size = %#x, want %#x", got["_g"].Size, int64(0x20))
	}
	if got["_g"].Code != 'T' {
		t.Fatalf("_g code = %q, want %q", got["_g"].Code, 'T')
	}
}

func TestMachoSymbolsDoNotCrossSectionBoundaries(t *testing.T) {
	f := &machoFile{macho: &macho.File{
		Symtab: &macho.Symtab{Syms: []macho.Symbol{
			{Name: "_f", Sect: 1, Value: 0x1000},
			{Name: "_g", Sect: 1, Value: 0x1010},
			{Name: "_data", Sect: 2, Value: 0x2000},
		}},
		Sections: []*macho.Section{
			{SectionHeader: macho.SectionHeader{Name: "__text", Seg: "__TEXT", Addr: 0x1000, Size: 0x30}},
			{SectionHeader: macho.SectionHeader{Name: "__data", Seg: "__DATA", Addr: 0x2000, Size: 0x10}},
		},
	}}

	syms, err := f.symbols()
	if err != nil {
		t.Fatal(err)
	}

	got := map[string]Sym{}
	for _, sym := range syms {
		got[sym.Name] = sym
	}

	if got["_g"].Size != 0x20 {
		t.Fatalf("_g size = %#x, want %#x", got["_g"].Size, int64(0x20))
	}
	if got["_data"].Size != 0x10 {
		t.Fatalf("_data size = %#x, want %#x", got["_data"].Size, int64(0x10))
	}
}
