// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xcoff

import (
	"reflect"
	"testing"
)

type archiveTest struct {
	file              string
	hdr               ArchiveHeader
	members           []*MemberHeader
	membersFileHeader []FileHeader
}

var archTest = []archiveTest{
	{
		"testdata/bigar-ppc64",
		ArchiveHeader{AIAMAGBIG},
		[]*MemberHeader{
			{"printbye.o", 836},
			{"printhello.o", 860},
		},
		[]FileHeader{
			FileHeader{U64_TOCMAGIC},
			FileHeader{U64_TOCMAGIC},
		},
	},
	{
		"testdata/bigar-empty",
		ArchiveHeader{AIAMAGBIG},
		[]*MemberHeader{},
		[]FileHeader{},
	},
}

func TestOpenArchive(t *testing.T) {
	for i := range archTest {
		tt := &archTest[i]
		arch, err := OpenArchive(tt.file)
		if err != nil {
			t.Error(err)
			continue
		}
		if !reflect.DeepEqual(arch.ArchiveHeader, tt.hdr) {
			t.Errorf("open archive %s:\n\thave %#v\n\twant %#v\n", tt.file, arch.ArchiveHeader, tt.hdr)
			continue
		}

		for i, mem := range arch.Members {
			if i >= len(tt.members) {
				break
			}
			have := &mem.MemberHeader
			want := tt.members[i]
			if !reflect.DeepEqual(have, want) {
				t.Errorf("open %s, member %d:\n\thave %#v\n\twant %#v\n", tt.file, i, have, want)
			}

			f, err := arch.GetFile(mem.Name)
			if err != nil {
				t.Error(err)
				continue
			}
			if !reflect.DeepEqual(f.FileHeader, tt.membersFileHeader[i]) {
				t.Errorf("open %s, member file header %d:\n\thave %#v\n\twant %#v\n", tt.file, i, f.FileHeader, tt.membersFileHeader[i])
			}
		}
		tn := len(tt.members)
		an := len(arch.Members)
		if tn != an {
			t.Errorf("open %s: len(Members) = %d, want %d", tt.file, an, tn)
		}

	}
}
