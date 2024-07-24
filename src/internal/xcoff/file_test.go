// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xcoff

import (
	"reflect"
	"slices"
	"testing"
)

type fileTest struct {
	file     string
	hdr      FileHeader
	sections []*SectionHeader
	needed   []string
}

var fileTests = []fileTest{
	{
		"testdata/gcc-ppc32-aix-dwarf2-exec",
		FileHeader{U802TOCMAGIC},
		[]*SectionHeader{
			{".text", 0x10000290, 0x00000bbd, STYP_TEXT, 0x7ae6, 0x36},
			{".data", 0x20000e4d, 0x00000437, STYP_DATA, 0x7d02, 0x2b},
			{".bss", 0x20001284, 0x0000021c, STYP_BSS, 0, 0},
			{".loader", 0x00000000, 0x000004b3, STYP_LOADER, 0, 0},
			{".dwline", 0x00000000, 0x000000df, STYP_DWARF | SSUBTYP_DWLINE, 0x7eb0, 0x7},
			{".dwinfo", 0x00000000, 0x00000314, STYP_DWARF | SSUBTYP_DWINFO, 0x7ef6, 0xa},
			{".dwabrev", 0x00000000, 0x000000d6, STYP_DWARF | SSUBTYP_DWABREV, 0, 0},
			{".dwarnge", 0x00000000, 0x00000020, STYP_DWARF | SSUBTYP_DWARNGE, 0x7f5a, 0x2},
			{".dwloc", 0x00000000, 0x00000074, STYP_DWARF | SSUBTYP_DWLOC, 0, 0},
			{".debug", 0x00000000, 0x00005e4f, STYP_DEBUG, 0, 0},
		},
		[]string{"libc.a/shr.o"},
	},
	{
		"testdata/gcc-ppc64-aix-dwarf2-exec",
		FileHeader{U64_TOCMAGIC},
		[]*SectionHeader{
			{".text", 0x10000480, 0x00000afd, STYP_TEXT, 0x8322, 0x34},
			{".data", 0x20000f7d, 0x000002f3, STYP_DATA, 0x85fa, 0x25},
			{".bss", 0x20001270, 0x00000428, STYP_BSS, 0, 0},
			{".loader", 0x00000000, 0x00000535, STYP_LOADER, 0, 0},
			{".dwline", 0x00000000, 0x000000b4, STYP_DWARF | SSUBTYP_DWLINE, 0x8800, 0x4},
			{".dwinfo", 0x00000000, 0x0000036a, STYP_DWARF | SSUBTYP_DWINFO, 0x8838, 0x7},
			{".dwabrev", 0x00000000, 0x000000b5, STYP_DWARF | SSUBTYP_DWABREV, 0, 0},
			{".dwarnge", 0x00000000, 0x00000040, STYP_DWARF | SSUBTYP_DWARNGE, 0x889a, 0x2},
			{".dwloc", 0x00000000, 0x00000062, STYP_DWARF | SSUBTYP_DWLOC, 0, 0},
			{".debug", 0x00000000, 0x00006605, STYP_DEBUG, 0, 0},
		},
		[]string{"libc.a/shr_64.o"},
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
		tl := tt.needed
		fl, err := f.ImportedLibraries()
		if err != nil {
			t.Error(err)
		}
		if !slices.Equal(tl, fl) {
			t.Errorf("open %s: loader import = %v, want %v", tt.file, tl, fl)
		}
	}
}

func TestOpenFailure(t *testing.T) {
	filename := "file.go"    // not an XCOFF object file
	_, err := Open(filename) // don't crash
	if err == nil {
		t.Errorf("open %s: succeeded unexpectedly", filename)
	}
}
