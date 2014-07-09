// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package plan9obj

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
		"testdata/386-plan9-exec",
		FileHeader{Magic386, 0x324, 0x14, 4, 0x1000, 32},
		[]*SectionHeader{
			{"text", 0x4c5f, 0x20},
			{"data", 0x94c, 0x4c7f},
			{"syms", 0x2c2b, 0x55cb},
			{"spsz", 0x0, 0x81f6},
			{"pcsz", 0xf7a, 0x81f6},
		},
	},
	{
		"testdata/amd64-plan9-exec",
		FileHeader{MagicAMD64, 0x618, 0x13, 8, 0x200000, 40},
		[]*SectionHeader{
			{"text", 0x4213, 0x28},
			{"data", 0xa80, 0x423b},
			{"syms", 0x2c8c, 0x4cbb},
			{"spsz", 0x0, 0x7947},
			{"pcsz", 0xca0, 0x7947},
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
	filename := "file.go"    // not a Plan 9 a.out file
	_, err := Open(filename) // don't crash
	if err == nil {
		t.Errorf("open %s: succeeded unexpectedly", filename)
	}
}
