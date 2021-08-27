//go:build (aix || darwin || dragonfly || freebsd || (js && wasm) || (!android && linux) || netbsd || openbsd || solaris) && (!cgo || osusergo)
// +build aix darwin dragonfly freebsd js,wasm !android,linux netbsd openbsd solaris
// +build !cgo osusergo

package user

import (
	"strings"
	"testing"
)

func TestIterateGroups(t *testing.T) {
	const testGroupFile = `# See the opendirectoryd(8) man page for additional 
# information about Open Directory.
##
nobody:*:-2:
nogroup:*:-1:
invalidgid:*:notanumber:root
+plussign:*:20:root
    indented:*:7:
# comment:*:4:found
     # comment:*:4:found
kmem:*:2:root
`
	// Ordered list of correctly parsed wantGroups from testGroupFile
	var wantGroups = []*Group{
		{Gid: "-2", Name: "nobody"},
		{Gid: "-1", Name: "nogroup"},
		{Gid: "7", Name: "indented"},
		{Gid: "2", Name: "kmem"},
	}

	r := strings.NewReader(testGroupFile)

	gotGroups := make([]*Group, 0, len(wantGroups))
	_, err := readColonFile(r, groupsIterator(func(g *Group) error {
		gotGroups = append(gotGroups, g)
		return nil
	}), 3)

	if len(gotGroups) != len(wantGroups) {
		t.Errorf("wantGroups could not be retrieved correctly: parsed %d/%d", len(gotGroups), len(wantGroups))
	}

	for i, g := range gotGroups {
		if *g != *wantGroups[i] {
			t.Errorf("iterate wantGroups result is incorrect: got: %+v, want: %+v", g, wantGroups[i])
		}
	}

	if err != nil {
		t.Errorf("readEtcFile error: %v", err)
	}
}

func TestIterateUsers(t *testing.T) {
	const testUserFile = `   # Example user file
root:x:0:0:root:/root:/bin/bash
     indented:x:3:3:indented with a name:/dev:/usr/sbin/nologin
negative:x:-5:60:games:/usr/games:/usr/sbin/nologin
allfields:x:6:12:mansplit,man2,man3,man4:/home/allfields:/usr/sbin/nologin
+plussign:x:8:10:man:/var/cache/man:/usr/sbin/nologin

malformed:x:27:12 # more:colons:after:comment

struid:x:notanumber:12 # more:colons:after:comment

# commented:x:28:12:commented:/var/cache/man:/usr/sbin/nologin
      # commentindented:x:29:12:commentindented:/var/cache/man:/usr/sbin/nologin

struid2:x:30:badgid:struid2name:/home/struid:/usr/sbin/nologin
`
	var wantUsers = []*User{
		{Username: "root", Name: "root", Uid: "0", Gid: "0", HomeDir: "/root"},
		{Username: "indented", Name: "indented with a name", Uid: "3", Gid: "3", HomeDir: "/dev"},
		{Username: "negative", Name: "games", Uid: "-5", Gid: "60", HomeDir: "/usr/games"},
		{Username: "allfields", Name: "mansplit", Uid: "6", Gid: "12", HomeDir: "/home/allfields"},
	}

	gotUsers := make([]*User, 0, len(wantUsers))
	r := strings.NewReader(testUserFile)
	_, err := readColonFile(r, usersIterator(func(u *User) error {
		gotUsers = append(gotUsers, u)
		return nil
	}), 6)

	if len(gotUsers) != len(wantUsers) {
		t.Errorf("wantUsers could not be parsed correctly: parsed %d/%d", len(gotUsers), len(wantUsers))
	}

	for i, u := range gotUsers {
		if *u != *wantUsers[i] {
			t.Errorf("iterate wantUsers result is incorrect: got: %+v, want: %+v", u, wantUsers[i])
		}
	}

	if err != nil {
		t.Errorf("readEtcFile error: %v", err)
	}
}
