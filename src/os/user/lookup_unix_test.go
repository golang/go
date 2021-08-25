// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (aix || darwin || dragonfly || freebsd || (!android && linux) || netbsd || openbsd || solaris) && !cgo

package user

import (
	"reflect"
	"strings"
	"testing"
)

var groupTests = []struct {
	in   string
	name string
	gid  string
}{
	{testGroupFile, "nobody", "-2"},
	{testGroupFile, "kmem", "2"},
	{testGroupFile, "notinthefile", ""},
	{testGroupFile, "comment", ""},
	{testGroupFile, "plussign", ""},
	{testGroupFile, "+plussign", ""},
	{testGroupFile, "-minussign", ""},
	{testGroupFile, "minussign", ""},
	{testGroupFile, "emptyid", ""},
	{testGroupFile, "invalidgid", ""},
	{testGroupFile, "indented", "7"},
	{testGroupFile, "# comment", ""},
	{testGroupFile, "largegroup", "1000"},
	{testGroupFile, "manymembers", "777"},
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
	{testGroupFile, "20", ""}, // row starts with a plus
	{testGroupFile, "21", ""}, // row starts with a minus
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

const testUserFile = `   # Example user file
root:x:0:0:root:/root:/bin/bash
daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
bin:x:2:3:bin:/bin:/usr/sbin/nologin
     indented:x:3:3:indented:/dev:/usr/sbin/nologin
sync:x:4:65534:sync:/bin:/bin/sync
negative:x:-5:60:games:/usr/games:/usr/sbin/nologin
man:x:6:12:man:/var/cache/man:/usr/sbin/nologin
allfields:x:6:12:mansplit,man2,man3,man4:/home/allfields:/usr/sbin/nologin
+plussign:x:8:10:man:/var/cache/man:/usr/sbin/nologin
-minussign:x:9:10:man:/var/cache/man:/usr/sbin/nologin

malformed:x:27:12 # more:colons:after:comment

struid:x:notanumber:12 # more:colons:after:comment

# commented:x:28:12:commented:/var/cache/man:/usr/sbin/nologin
      # commentindented:x:29:12:commentindented:/var/cache/man:/usr/sbin/nologin

struid2:x:30:badgid:struid2name:/home/struid:/usr/sbin/nologin
`

var userIdTests = []struct {
	in   string
	uid  string
	name string
}{
	{testUserFile, "-5", "negative"},
	{testUserFile, "2", "bin"},
	{testUserFile, "100", ""}, // not in the file
	{testUserFile, "8", ""},   // plus sign, glibc doesn't find it
	{testUserFile, "9", ""},   // minus sign, glibc doesn't find it
	{testUserFile, "27", ""},  // malformed
	{testUserFile, "28", ""},  // commented out
	{testUserFile, "29", ""},  // commented out, indented
	{testUserFile, "3", "indented"},
	{testUserFile, "30", ""}, // the Gid is not valid, shouldn't match
	{"", "1", ""},
}

func TestInvalidUserId(t *testing.T) {
	_, err := findUserId("notanumber", strings.NewReader(""))
	if err == nil {
		t.Fatalf("findUserId('notanumber'): got nil error")
	}
	if want := "user: invalid userid notanumber"; err.Error() != want {
		t.Errorf("findUserId('notanumber'): got %v, want %s", err, want)
	}
}

func TestLookupUserId(t *testing.T) {
	for _, tt := range userIdTests {
		got, err := findUserId(tt.uid, strings.NewReader(tt.in))
		if tt.name == "" {
			if err == nil {
				t.Errorf("findUserId(%s): got nil error, expected err", tt.uid)
				continue
			}
			switch terr := err.(type) {
			case UnknownUserIdError:
				if want := "user: unknown userid " + tt.uid; terr.Error() != want {
					t.Errorf("findUserId(%s): got %v, want %v", tt.name, terr, want)
				}
			default:
				t.Errorf("findUserId(%s): got unexpected error %v", tt.name, terr)
			}
		} else {
			if err != nil {
				t.Fatalf("findUserId(%s): got unexpected error %v", tt.name, err)
			}
			if got.Uid != tt.uid {
				t.Errorf("findUserId(%s): got uid %v, want %s", tt.name, got.Uid, tt.uid)
			}
			if got.Username != tt.name {
				t.Errorf("findUserId(%s): got name %s, want %s", tt.name, got.Username, tt.name)
			}
		}
	}
}

func TestLookupUserPopulatesAllFields(t *testing.T) {
	u, err := findUsername("allfields", strings.NewReader(testUserFile))
	if err != nil {
		t.Fatal(err)
	}
	want := &User{
		Username: "allfields",
		Uid:      "6",
		Gid:      "12",
		Name:     "mansplit",
		HomeDir:  "/home/allfields",
	}
	if !reflect.DeepEqual(u, want) {
		t.Errorf("findUsername: got %#v, want %#v", u, want)
	}
}

var userTests = []struct {
	in   string
	name string
	uid  string
}{
	{testUserFile, "negative", "-5"},
	{testUserFile, "bin", "2"},
	{testUserFile, "notinthefile", ""},
	{testUserFile, "indented", "3"},
	{testUserFile, "plussign", ""},
	{testUserFile, "+plussign", ""},
	{testUserFile, "minussign", ""},
	{testUserFile, "-minussign", ""},
	{testUserFile, "   indented", ""},
	{testUserFile, "commented", ""},
	{testUserFile, "commentindented", ""},
	{testUserFile, "malformed", ""},
	{testUserFile, "# commented", ""},
	{"", "emptyfile", ""},
}

func TestLookupUser(t *testing.T) {
	for _, tt := range userTests {
		got, err := findUsername(tt.name, strings.NewReader(tt.in))
		if tt.uid == "" {
			if err == nil {
				t.Errorf("lookupUser(%s): got nil error, expected err", tt.uid)
				continue
			}
			switch terr := err.(type) {
			case UnknownUserError:
				if want := "user: unknown user " + tt.name; terr.Error() != want {
					t.Errorf("lookupUser(%s): got %v, want %v", tt.name, terr, want)
				}
			default:
				t.Errorf("lookupUser(%s): got unexpected error %v", tt.name, terr)
			}
		} else {
			if err != nil {
				t.Fatalf("lookupUser(%s): got unexpected error %v", tt.name, err)
			}
			if got.Uid != tt.uid {
				t.Errorf("lookupUser(%s): got uid %v, want %s", tt.name, got.Uid, tt.uid)
			}
			if got.Username != tt.name {
				t.Errorf("lookupUser(%s): got name %s, want %s", tt.name, got.Username, tt.name)
			}
		}
	}
}
