// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd !android,linux nacl netbsd openbsd solaris
// +build !cgo

package user

import (
	"strings"
	"testing"
)

const testGroupFile = `# See the opendirectoryd(8) man page for additional 
# information about Open Directory.
##
nobody:*:-2:
nogroup:*:-1:
wheel:*:0:root
emptyid:*::root
      
daemon:*:1:root
    indented:*:7:
# comment:*:4:found
     # comment:*:4:found
kmem:*:2:root
`

var groupTests = []struct {
	in   string
	name string
	gid  string
}{
	{testGroupFile, "nobody", "-2"},
	{testGroupFile, "kmem", "2"},
	{testGroupFile, "notinthefile", ""},
	{testGroupFile, "comment", ""},
	{testGroupFile, "emptyid", ""},
	{testGroupFile, "indented", "7"},
	{testGroupFile, "# comment", ""},
	{"", "emptyfile", ""},
}

func TestFindGroupName(t *testing.T) {
	for _, tt := range groupTests {
		got, err := findGroupName(tt.name, strings.NewReader(tt.in))
		if tt.gid == "" {
			if err == nil {
				t.Errorf("findGroupName(%s): got nil error, expected err", tt.name)
				continue
			}
			switch terr := err.(type) {
			case UnknownGroupError:
				if terr.Error() != "group: unknown group "+tt.name {
					t.Errorf("findGroupName(%s): got %v, want %v", tt.name, terr, tt.name)
				}
			default:
				t.Errorf("findGroupName(%s): got unexpected error %v", tt.name, terr)
			}
		} else {
			if err != nil {
				t.Fatalf("findGroupName(%s): got unexpected error %v", tt.name, err)
			}
			if got.Gid != tt.gid {
				t.Errorf("findGroupName(%s): got gid %v, want %s", tt.name, got.Gid, tt.gid)
			}
			if got.Name != tt.name {
				t.Errorf("findGroupName(%s): got name %s, want %s", tt.name, got.Name, tt.name)
			}
		}
	}
}

var groupIdTests = []struct {
	in   string
	gid  string
	name string
}{
	{testGroupFile, "-2", "nobody"},
	{testGroupFile, "2", "kmem"},
	{testGroupFile, "notinthefile", ""},
	{testGroupFile, "comment", ""},
	{testGroupFile, "7", "indented"},
	{testGroupFile, "4", ""},
	{"", "emptyfile", ""},
}

func TestFindGroupId(t *testing.T) {
	for _, tt := range groupIdTests {
		got, err := findGroupId(tt.gid, strings.NewReader(tt.in))
		if tt.name == "" {
			if err == nil {
				t.Errorf("findGroupId(%s): got nil error, expected err", tt.gid)
				continue
			}
			switch terr := err.(type) {
			case UnknownGroupIdError:
				if terr.Error() != "group: unknown groupid "+tt.gid {
					t.Errorf("findGroupId(%s): got %v, want %v", tt.name, terr, tt.name)
				}
			default:
				t.Errorf("findGroupId(%s): got unexpected error %v", tt.name, terr)
			}
		} else {
			if err != nil {
				t.Fatalf("findGroupId(%s): got unexpected error %v", tt.name, err)
			}
			if got.Gid != tt.gid {
				t.Errorf("findGroupId(%s): got gid %v, want %s", tt.name, got.Gid, tt.gid)
			}
			if got.Name != tt.name {
				t.Errorf("findGroupId(%s): got name %s, want %s", tt.name, got.Name, tt.name)
			}
		}
	}
}
