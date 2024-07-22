// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package macho

import (
	"bytes"
	"errors"
	"internal/obscuretestdata"
	"io"
	"reflect"
	"testing"
)

type fileTest struct {
	file        string
	hdr         FileHeader
	loads       []any
	sections    []*SectionHeader
	relocations map[string][]Reloc
}

var fileTests = []fileTest{
	{
		"testdata/gcc-386-darwin-exec.base64",
		FileHeader{0xfeedface, Cpu386, 0x3, 0x2, 0xc, 0x3c0, 0x85},
		[]any{
			&SegmentHeader{LoadCmdSegment, 0x38, "__PAGEZERO", 0x0, 0x1000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			&SegmentHeader{LoadCmdSegment, 0xc0, "__TEXT", 0x1000, 0x1000, 0x0, 0x1000, 0x7, 0x5, 0x2, 0x0},
			&SegmentHeader{LoadCmdSegment, 0xc0, "__DATA", 0x2000, 0x1000, 0x1000, 0x1000, 0x7, 0x3, 0x2, 0x0},
			&SegmentHeader{LoadCmdSegment, 0x7c, "__IMPORT", 0x3000, 0x1000, 0x2000, 0x1000, 0x7, 0x7, 0x1, 0x0},
			&SegmentHeader{LoadCmdSegment, 0x38, "__LINKEDIT", 0x4000, 0x1000, 0x3000, 0x12c, 0x7, 0x1, 0x0, 0x0},
			nil, // LC_SYMTAB
			nil, // LC_DYSYMTAB
			nil, // LC_LOAD_DYLINKER
			nil, // LC_UUID
			nil, // LC_UNIXTHREAD
			&Dylib{nil, "/usr/lib/libgcc_s.1.dylib", 0x2, 0x10000, 0x10000},
			&Dylib{nil, "/usr/lib/libSystem.B.dylib", 0x2, 0x6f0104, 0x10000},
		},
		[]*SectionHeader{
			{"__text", "__TEXT", 0x1f68, 0x88, 0xf68, 0x2, 0x0, 0x0, 0x80000400},
			{"__cstring", "__TEXT", 0x1ff0, 0xd, 0xff0, 0x0, 0x0, 0x0, 0x2},
			{"__data", "__DATA", 0x2000, 0x14, 0x1000, 0x2, 0x0, 0x0, 0x0},
			{"__dyld", "__DATA", 0x2014, 0x1c, 0x1014, 0x2, 0x0, 0x0, 0x0},
			{"__jump_table", "__IMPORT", 0x3000, 0xa, 0x2000, 0x6, 0x0, 0x0, 0x4000008},
		},
		nil,
	},
	{
		"testdata/gcc-amd64-darwin-exec.base64",
		FileHeader{0xfeedfacf, CpuAmd64, 0x80000003, 0x2, 0xb, 0x568, 0x85},
		[]any{
			&SegmentHeader{LoadCmdSegment64, 0x48, "__PAGEZERO", 0x0, 0x100000000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			&SegmentHeader{LoadCmdSegment64, 0x1d8, "__TEXT", 0x100000000, 0x1000, 0x0, 0x1000, 0x7, 0x5, 0x5, 0x0},
			&SegmentHeader{LoadCmdSegment64, 0x138, "__DATA", 0x100001000, 0x1000, 0x1000, 0x1000, 0x7, 0x3, 0x3, 0x0},
			&SegmentHeader{LoadCmdSegment64, 0x48, "__LINKEDIT", 0x100002000, 0x1000, 0x2000, 0x140, 0x7, 0x1, 0x0, 0x0},
			nil, // LC_SYMTAB
			nil, // LC_DYSYMTAB
			nil, // LC_LOAD_DYLINKER
			nil, // LC_UUID
			nil, // LC_UNIXTHREAD
			&Dylib{nil, "/usr/lib/libgcc_s.1.dylib", 0x2, 0x10000, 0x10000},
			&Dylib{nil, "/usr/lib/libSystem.B.dylib", 0x2, 0x6f0104, 0x10000},
		},
		[]*SectionHeader{
			{"__text", "__TEXT", 0x100000f14, 0x6d, 0xf14, 0x2, 0x0, 0x0, 0x80000400},
			{"__symbol_stub1", "__TEXT", 0x100000f81, 0xc, 0xf81, 0x0, 0x0, 0x0, 0x80000408},
			{"__stub_helper", "__TEXT", 0x100000f90, 0x18, 0xf90, 0x2, 0x0, 0x0, 0x0},
			{"__cstring", "__TEXT", 0x100000fa8, 0xd, 0xfa8, 0x0, 0x0, 0x0, 0x2},
			{"__eh_frame", "__TEXT", 0x100000fb8, 0x48, 0xfb8, 0x3, 0x0, 0x0, 0x6000000b},
			{"__data", "__DATA", 0x100001000, 0x1c, 0x1000, 0x3, 0x0, 0x0, 0x0},
			{"__dyld", "__DATA", 0x100001020, 0x38, 0x1020, 0x3, 0x0, 0x0, 0x0},
			{"__la_symbol_ptr", "__DATA", 0x100001058, 0x10, 0x1058, 0x2, 0x0, 0x0, 0x7},
		},
		nil,
	},
	{
		"testdata/gcc-amd64-darwin-exec-debug.base64",
		FileHeader{0xfeedfacf, CpuAmd64, 0x80000003, 0xa, 0x4, 0x5a0, 0},
		[]any{
			nil, // LC_UUID
			&SegmentHeader{LoadCmdSegment64, 0x1d8, "__TEXT", 0x100000000, 0x1000, 0x0, 0x0, 0x7, 0x5, 0x5, 0x0},
			&SegmentHeader{LoadCmdSegment64, 0x138, "__DATA", 0x100001000, 0x1000, 0x0, 0x0, 0x7, 0x3, 0x3, 0x0},
			&SegmentHeader{LoadCmdSegment64, 0x278, "__DWARF", 0x100002000, 0x1000, 0x1000, 0x1bc, 0x7, 0x3, 0x7, 0x0},
		},
		[]*SectionHeader{
			{"__text", "__TEXT", 0x100000f14, 0x0, 0x0, 0x2, 0x0, 0x0, 0x80000400},
			{"__symbol_stub1", "__TEXT", 0x100000f81, 0x0, 0x0, 0x0, 0x0, 0x0, 0x80000408},
			{"__stub_helper", "__TEXT", 0x100000f90, 0x0, 0x0, 0x2, 0x0, 0x0, 0x0},
			{"__cstring", "__TEXT", 0x100000fa8, 0x0, 0x0, 0x0, 0x0, 0x0, 0x2},
			{"__eh_frame", "__TEXT", 0x100000fb8, 0x0, 0x0, 0x3, 0x0, 0x0, 0x6000000b},
			{"__data", "__DATA", 0x100001000, 0x0, 0x0, 0x3, 0x0, 0x0, 0x0},
			{"__dyld", "__DATA", 0x100001020, 0x0, 0x0, 0x3, 0x0, 0x0, 0x0},
			{"__la_symbol_ptr", "__DATA", 0x100001058, 0x0, 0x0, 0x2, 0x0, 0x0, 0x7},
			{"__debug_abbrev", "__DWARF", 0x100002000, 0x36, 0x1000, 0x0, 0x0, 0x0, 0x0},
			{"__debug_aranges", "__DWARF", 0x100002036, 0x30, 0x1036, 0x0, 0x0, 0x0, 0x0},
			{"__debug_frame", "__DWARF", 0x100002066, 0x40, 0x1066, 0x0, 0x0, 0x0, 0x0},
			{"__debug_info", "__DWARF", 0x1000020a6, 0x54, 0x10a6, 0x0, 0x0, 0x0, 0x0},
			{"__debug_line", "__DWARF", 0x1000020fa, 0x47, 0x10fa, 0x0, 0x0, 0x0, 0x0},
			{"__debug_pubnames", "__DWARF", 0x100002141, 0x1b, 0x1141, 0x0, 0x0, 0x0, 0x0},
			{"__debug_str", "__DWARF", 0x10000215c, 0x60, 0x115c, 0x0, 0x0, 0x0, 0x0},
		},
		nil,
	},
	{
		"testdata/clang-386-darwin-exec-with-rpath.base64",
		FileHeader{0xfeedface, Cpu386, 0x3, 0x2, 0x10, 0x42c, 0x1200085},
		[]any{
			nil, // LC_SEGMENT
			nil, // LC_SEGMENT
			nil, // LC_SEGMENT
			nil, // LC_SEGMENT
			nil, // LC_DYLD_INFO_ONLY
			nil, // LC_SYMTAB
			nil, // LC_DYSYMTAB
			nil, // LC_LOAD_DYLINKER
			nil, // LC_UUID
			nil, // LC_VERSION_MIN_MACOSX
			nil, // LC_SOURCE_VERSION
			nil, // LC_MAIN
			nil, // LC_LOAD_DYLIB
			&Rpath{nil, "/my/rpath"},
			nil, // LC_FUNCTION_STARTS
			nil, // LC_DATA_IN_CODE
		},
		nil,
		nil,
	},
	{
		"testdata/clang-amd64-darwin-exec-with-rpath.base64",
		FileHeader{0xfeedfacf, CpuAmd64, 0x80000003, 0x2, 0x10, 0x4c8, 0x200085},
		[]any{
			nil, // LC_SEGMENT
			nil, // LC_SEGMENT
			nil, // LC_SEGMENT
			nil, // LC_SEGMENT
			nil, // LC_DYLD_INFO_ONLY
			nil, // LC_SYMTAB
			nil, // LC_DYSYMTAB
			nil, // LC_LOAD_DYLINKER
			nil, // LC_UUID
			nil, // LC_VERSION_MIN_MACOSX
			nil, // LC_SOURCE_VERSION
			nil, // LC_MAIN
			nil, // LC_LOAD_DYLIB
			&Rpath{nil, "/my/rpath"},
			nil, // LC_FUNCTION_STARTS
			nil, // LC_DATA_IN_CODE
		},
		nil,
		nil,
	},
	{
		"testdata/clang-386-darwin.obj.base64",
		FileHeader{0xfeedface, Cpu386, 0x3, 0x1, 0x4, 0x138, 0x2000},
		nil,
		nil,
		map[string][]Reloc{
			"__text": {
				{
					Addr:      0x1d,
					Type:      uint8(GENERIC_RELOC_VANILLA),
					Len:       2,
					Pcrel:     true,
					Extern:    true,
					Value:     1,
					Scattered: false,
				},
				{
					Addr:      0xe,
					Type:      uint8(GENERIC_RELOC_LOCAL_SECTDIFF),
					Len:       2,
					Pcrel:     false,
					Value:     0x2d,
					Scattered: true,
				},
				{
					Addr:      0x0,
					Type:      uint8(GENERIC_RELOC_PAIR),
					Len:       2,
					Pcrel:     false,
					Value:     0xb,
					Scattered: true,
				},
			},
		},
	},
	{
		"testdata/clang-amd64-darwin.obj.base64",
		FileHeader{0xfeedfacf, CpuAmd64, 0x3, 0x1, 0x4, 0x200, 0x2000},
		nil,
		nil,
		map[string][]Reloc{
			"__text": {
				{
					Addr:   0x19,
					Type:   uint8(X86_64_RELOC_BRANCH),
					Len:    2,
					Pcrel:  true,
					Extern: true,
					Value:  1,
				},
				{
					Addr:   0xb,
					Type:   uint8(X86_64_RELOC_SIGNED),
					Len:    2,
					Pcrel:  true,
					Extern: false,
					Value:  2,
				},
			},
			"__compact_unwind": {
				{
					Addr:   0x0,
					Type:   uint8(X86_64_RELOC_UNSIGNED),
					Len:    3,
					Pcrel:  false,
					Extern: false,
					Value:  1,
				},
			},
		},
	},
}

func readerAtFromObscured(name string) (io.ReaderAt, error) {
	b, err := obscuretestdata.ReadFile(name)
	if err != nil {
		return nil, err
	}
	return bytes.NewReader(b), nil
}

func openObscured(name string) (*File, error) {
	ra, err := readerAtFromObscured(name)
	if err != nil {
		return nil, err
	}
	ff, err := NewFile(ra)
	if err != nil {
		return nil, err
	}
	return ff, nil
}

func openFatObscured(name string) (*FatFile, error) {
	ra, err := readerAtFromObscured(name)
	if err != nil {
		return nil, err
	}
	ff, err := NewFatFile(ra)
	if err != nil {
		return nil, err
	}
	return ff, nil
}

func TestOpen(t *testing.T) {
	for i := range fileTests {
		tt := &fileTests[i]

		// Use obscured files to prevent Apple’s notarization service from
		// mistaking them as candidates for notarization and rejecting the entire
		// toolchain.
		// See golang.org/issue/34986
		f, err := openObscured(tt.file)
		if err != nil {
			t.Error(err)
			continue
		}
		if !reflect.DeepEqual(f.FileHeader, tt.hdr) {
			t.Errorf("open %s:\n\thave %#v\n\twant %#v\n", tt.file, f.FileHeader, tt.hdr)
			continue
		}
		for i, l := range f.Loads {
			if len(l.Raw()) < 8 {
				t.Errorf("open %s, command %d:\n\tload command %T don't have enough data\n", tt.file, i, l)
			}
		}
		if tt.loads != nil {
			for i, l := range f.Loads {
				if i >= len(tt.loads) {
					break
				}

				want := tt.loads[i]
				if want == nil {
					continue
				}

				switch l := l.(type) {
				case *Segment:
					have := &l.SegmentHeader
					if !reflect.DeepEqual(have, want) {
						t.Errorf("open %s, command %d:\n\thave %#v\n\twant %#v\n", tt.file, i, have, want)
					}
				case *Dylib:
					have := l
					have.LoadBytes = nil
					if !reflect.DeepEqual(have, want) {
						t.Errorf("open %s, command %d:\n\thave %#v\n\twant %#v\n", tt.file, i, have, want)
					}
				case *Rpath:
					have := l
					have.LoadBytes = nil
					if !reflect.DeepEqual(have, want) {
						t.Errorf("open %s, command %d:\n\thave %#v\n\twant %#v\n", tt.file, i, have, want)
					}
				default:
					t.Errorf("open %s, command %d: unknown load command\n\thave %#v\n\twant %#v\n", tt.file, i, l, want)
				}
			}
			tn := len(tt.loads)
			fn := len(f.Loads)
			if tn != fn {
				t.Errorf("open %s: len(Loads) = %d, want %d", tt.file, fn, tn)
			}
		}

		if tt.sections != nil {
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

		if tt.relocations != nil {
			for i, sh := range f.Sections {
				have := sh.Relocs
				want := tt.relocations[sh.Name]
				if !reflect.DeepEqual(have, want) {
					t.Errorf("open %s, relocations in section %d (%s):\n\thave %#v\n\twant %#v\n", tt.file, i, sh.Name, have, want)
				}
			}
		}
	}
}

func TestOpenFailure(t *testing.T) {
	filename := "file.go"    // not a Mach-O file
	_, err := Open(filename) // don't crash
	if err == nil {
		t.Errorf("open %s: succeeded unexpectedly", filename)
	}
}

func TestOpenFat(t *testing.T) {
	ff, err := openFatObscured("testdata/fat-gcc-386-amd64-darwin-exec.base64")
	if err != nil {
		t.Fatal(err)
	}

	if ff.Magic != MagicFat {
		t.Errorf("OpenFat: got magic number %#x, want %#x", ff.Magic, MagicFat)
	}
	if len(ff.Arches) != 2 {
		t.Errorf("OpenFat: got %d architectures, want 2", len(ff.Arches))
	}

	for i := range ff.Arches {
		arch := &ff.Arches[i]
		ftArch := &fileTests[i]

		if arch.Cpu != ftArch.hdr.Cpu || arch.SubCpu != ftArch.hdr.SubCpu {
			t.Errorf("OpenFat: architecture #%d got cpu=%#x subtype=%#x, expected cpu=%#x, subtype=%#x", i, arch.Cpu, arch.SubCpu, ftArch.hdr.Cpu, ftArch.hdr.SubCpu)
		}

		if !reflect.DeepEqual(arch.FileHeader, ftArch.hdr) {
			t.Errorf("OpenFat header:\n\tgot %#v\n\twant %#v\n", arch.FileHeader, ftArch.hdr)
		}
	}
}

func TestOpenFatFailure(t *testing.T) {
	filename := "file.go" // not a Mach-O file
	if _, err := OpenFat(filename); err == nil {
		t.Errorf("OpenFat %s: succeeded unexpectedly", filename)
	}

	filename = "testdata/gcc-386-darwin-exec.base64" // not a fat Mach-O
	ff, err := openFatObscured(filename)
	if !errors.Is(err, ErrNotFat) {
		t.Errorf("OpenFat %s: got %v, want ErrNotFat", filename, err)
	}
	if ff != nil {
		t.Errorf("OpenFat %s: got %v, want nil", filename, ff)
	}
}

func TestRelocTypeString(t *testing.T) {
	if X86_64_RELOC_BRANCH.String() != "X86_64_RELOC_BRANCH" {
		t.Errorf("got %v, want %v", X86_64_RELOC_BRANCH.String(), "X86_64_RELOC_BRANCH")
	}
	if X86_64_RELOC_BRANCH.GoString() != "macho.X86_64_RELOC_BRANCH" {
		t.Errorf("got %v, want %v", X86_64_RELOC_BRANCH.GoString(), "macho.X86_64_RELOC_BRANCH")
	}
}

func TestTypeString(t *testing.T) {
	if TypeExec.String() != "Exec" {
		t.Errorf("got %v, want %v", TypeExec.String(), "Exec")
	}
	if TypeExec.GoString() != "macho.Exec" {
		t.Errorf("got %v, want %v", TypeExec.GoString(), "macho.Exec")
	}
}

func TestOpenBadDysymCmd(t *testing.T) {
	_, err := openObscured("testdata/gcc-amd64-darwin-exec-with-bad-dysym.base64")
	if err == nil {
		t.Fatal("openObscured did not fail when opening a file with an invalid dynamic symbol table command")
	}
}
