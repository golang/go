// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pe

import (
	"reflect"
	"testing"
)

type fileTest struct {
	file     string
	hdr      FileHeader
	sections []*SectionHeader
}

var fileTests = []fileTest{
	{
		"testdata/gcc-386-mingw-obj",
		FileHeader{0x014c, 0x000c, 0x0, 0x64a, 0x1e, 0x0, 0x104},
		[]*SectionHeader{
			&SectionHeader{".text", 0, 0, 36, 500, 1440, 0, 3, 0, 0x60300020},
			&SectionHeader{".data", 0, 0, 0, 0, 0, 0, 0, 0, 3224371264},
			&SectionHeader{".bss", 0, 0, 0, 0, 0, 0, 0, 0, 3224371328},
			&SectionHeader{".debug_abbrev", 0, 0, 137, 536, 0, 0, 0, 0, 0x42100000},
			&SectionHeader{".debug_info", 0, 0, 418, 673, 1470, 0, 7, 0, 1108344832},
			&SectionHeader{".debug_line", 0, 0, 128, 1091, 1540, 0, 1, 0, 1108344832},
			&SectionHeader{".rdata", 0, 0, 16, 1219, 0, 0, 0, 0, 1076887616},
			&SectionHeader{".debug_frame", 0, 0, 52, 1235, 1550, 0, 2, 0, 1110441984},
			&SectionHeader{".debug_loc", 0, 0, 56, 1287, 0, 0, 0, 0, 1108344832},
			&SectionHeader{".debug_pubnames", 0, 0, 27, 1343, 1570, 0, 1, 0, 1108344832},
			&SectionHeader{".debug_pubtypes", 0, 0, 38, 1370, 1580, 0, 1, 0, 1108344832},
			&SectionHeader{".debug_aranges", 0, 0, 32, 1408, 1590, 0, 2, 0, 1108344832},
		},
	},
	{
		"testdata/gcc-386-mingw-exec",
		FileHeader{0x014c, 0x000f, 0x4c6a1b60, 0x3c00, 0x282, 0xe0, 0x107},
		[]*SectionHeader{
			&SectionHeader{Name: ".text", VirtualSize: 0xcd8, VirtualAddress: 0x1000, Size: 0xe00, Offset: 0x400, PointerToRelocations: 0x0, PointerToLineNumbers: 0x0, NumberOfRelocations: 0x0, NumberOfLineNumbers: 0x0, Characteristics: 0x60500060},
			&SectionHeader{Name: ".data", VirtualSize: 0x10, VirtualAddress: 0x2000, Size: 0x200, Offset: 0x1200, PointerToRelocations: 0x0, PointerToLineNumbers: 0x0, NumberOfRelocations: 0x0, NumberOfLineNumbers: 0x0, Characteristics: 0xc0300040},
			&SectionHeader{Name: ".rdata", VirtualSize: 0x120, VirtualAddress: 0x3000, Size: 0x200, Offset: 0x1400, PointerToRelocations: 0x0, PointerToLineNumbers: 0x0, NumberOfRelocations: 0x0, NumberOfLineNumbers: 0x0, Characteristics: 0x40300040},
			&SectionHeader{Name: ".bss", VirtualSize: 0xdc, VirtualAddress: 0x4000, Size: 0x0, Offset: 0x0, PointerToRelocations: 0x0, PointerToLineNumbers: 0x0, NumberOfRelocations: 0x0, NumberOfLineNumbers: 0x0, Characteristics: 0xc0400080},
			&SectionHeader{Name: ".idata", VirtualSize: 0x3c8, VirtualAddress: 0x5000, Size: 0x400, Offset: 0x1600, PointerToRelocations: 0x0, PointerToLineNumbers: 0x0, NumberOfRelocations: 0x0, NumberOfLineNumbers: 0x0, Characteristics: 0xc0300040},
			&SectionHeader{Name: ".CRT", VirtualSize: 0x18, VirtualAddress: 0x6000, Size: 0x200, Offset: 0x1a00, PointerToRelocations: 0x0, PointerToLineNumbers: 0x0, NumberOfRelocations: 0x0, NumberOfLineNumbers: 0x0, Characteristics: 0xc0300040},
			&SectionHeader{Name: ".tls", VirtualSize: 0x20, VirtualAddress: 0x7000, Size: 0x200, Offset: 0x1c00, PointerToRelocations: 0x0, PointerToLineNumbers: 0x0, NumberOfRelocations: 0x0, NumberOfLineNumbers: 0x0, Characteristics: 0xc0300040},
			&SectionHeader{Name: ".debug_aranges", VirtualSize: 0x20, VirtualAddress: 0x8000, Size: 0x200, Offset: 0x1e00, PointerToRelocations: 0x0, PointerToLineNumbers: 0x0, NumberOfRelocations: 0x0, NumberOfLineNumbers: 0x0, Characteristics: 0x42100000},
			&SectionHeader{Name: ".debug_pubnames", VirtualSize: 0x51, VirtualAddress: 0x9000, Size: 0x200, Offset: 0x2000, PointerToRelocations: 0x0, PointerToLineNumbers: 0x0, NumberOfRelocations: 0x0, NumberOfLineNumbers: 0x0, Characteristics: 0x42100000},
			&SectionHeader{Name: ".debug_pubtypes", VirtualSize: 0x91, VirtualAddress: 0xa000, Size: 0x200, Offset: 0x2200, PointerToRelocations: 0x0, PointerToLineNumbers: 0x0, NumberOfRelocations: 0x0, NumberOfLineNumbers: 0x0, Characteristics: 0x42100000},
			&SectionHeader{Name: ".debug_info", VirtualSize: 0xe22, VirtualAddress: 0xb000, Size: 0x1000, Offset: 0x2400, PointerToRelocations: 0x0, PointerToLineNumbers: 0x0, NumberOfRelocations: 0x0, NumberOfLineNumbers: 0x0, Characteristics: 0x42100000},
			&SectionHeader{Name: ".debug_abbrev", VirtualSize: 0x157, VirtualAddress: 0xc000, Size: 0x200, Offset: 0x3400, PointerToRelocations: 0x0, PointerToLineNumbers: 0x0, NumberOfRelocations: 0x0, NumberOfLineNumbers: 0x0, Characteristics: 0x42100000},
			&SectionHeader{Name: ".debug_line", VirtualSize: 0x144, VirtualAddress: 0xd000, Size: 0x200, Offset: 0x3600, PointerToRelocations: 0x0, PointerToLineNumbers: 0x0, NumberOfRelocations: 0x0, NumberOfLineNumbers: 0x0, Characteristics: 0x42100000},
			&SectionHeader{Name: ".debug_frame", VirtualSize: 0x34, VirtualAddress: 0xe000, Size: 0x200, Offset: 0x3800, PointerToRelocations: 0x0, PointerToLineNumbers: 0x0, NumberOfRelocations: 0x0, NumberOfLineNumbers: 0x0, Characteristics: 0x42300000},
			&SectionHeader{Name: ".debug_loc", VirtualSize: 0x38, VirtualAddress: 0xf000, Size: 0x200, Offset: 0x3a00, PointerToRelocations: 0x0, PointerToLineNumbers: 0x0, NumberOfRelocations: 0x0, NumberOfLineNumbers: 0x0, Characteristics: 0x42100000},
		},
	},
}

func TestOpen(t *testing.T) {
	for i := range fileTests {
		tt := &fileTests[i]

		f, err := Open(tt.file)
		if err != nil {
			t.Error(err)
			continue
		}
		if !reflect.DeepEqual(f.FileHeader, tt.hdr) {
			t.Errorf("open %s:\n\thave %#v\n\twant %#v\n", tt.file, f.FileHeader, tt.hdr)
			continue
		}

		for i, sh := range f.Sections {
			if i >= len(tt.sections) {
				break
			}
			have := &sh.SectionHeader
			want := tt.sections[i]
			if !reflect.DeepEqual(have, want) {
				t.Errorf("open %s, section %d:\n\thave %#v\n\twant %#v\n", tt.file, i, have, want)
			}
		}
		tn := len(tt.sections)
		fn := len(f.Sections)
		if tn != fn {
			t.Errorf("open %s: len(Sections) = %d, want %d", tt.file, fn, tn)
		}

	}
}

func TestOpenFailure(t *testing.T) {
	filename := "file.go"    // not a PE file
	_, err := Open(filename) // don't crash
	if err == nil {
		t.Errorf("open %s: succeeded unexpectedly", filename)
	}
}
