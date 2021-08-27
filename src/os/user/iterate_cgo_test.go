//go:build (aix || dragonfly || (!android && linux) || solaris) && cgo && !osusergo
// +build aix dragonfly !android,linux solaris
// +build cgo
// +build !osusergo

package user

import (
	"reflect"
	"syscall"
	"testing"
)

// This file is used for testing cgo based unix implementation of users and
// groups iterators. Only unix based systems which support fgetpwent(3) and
// fgetgrent(3) can run tests from this file.

func TestIterateUser(t *testing.T) {
	var wantsUsers = []*User{
		{Uid: "0", Gid: "0", Username: "root", Name: "System Administrator", HomeDir: "/var/root"},
		{Uid: "1", Gid: "1", Username: "daemon", Name: "System Services", HomeDir: "/var/root"},
		{Uid: "4", Gid: "4", Username: "_uucp", Name: "Unix to Unix Copy Protocol", HomeDir: "/var/spool/uucp"},
		{Uid: "13", Gid: "13", Username: "_taskgated", Name: "Task Gate Daemon", HomeDir: "/var/empty"},
		{Uid: "24", Gid: "24", Username: "_networkd", Name: "Network Services", HomeDir: "/var/networkd"},
		{Uid: "25", Gid: "25", Username: "_installassistant", Name: "Install Assistant", HomeDir: "/var/empty"},
		{Uid: "26", Gid: "26", Username: "_lp", Name: "Printing Services", HomeDir: "/var/spool/cups"},
		{Uid: "27", Gid: "27", Username: "_postfix", Name: "Postfix Mail Server", HomeDir: "/var/spool/postfix"},
	}

	userIterator = &iterateUsersHelperTest{}

	// Test that users are retrieved in same order as defined
	gotUsers := make([]*User, 0, len(wantsUsers))
	err := iterateUsers(func(user *User) error {
		gotUsers = append(gotUsers, user)
		return nil
	})

	if err != syscall.ENOENT {
		t.Errorf("iterating users: %v", err)
	}

	if !reflect.DeepEqual(wantsUsers, gotUsers) {
		t.Errorf("iterate users result is incorrect: got: %+v, want: %+v", gotUsers, wantsUsers)
	}
}

func TestIterateGroup(t *testing.T) {
	var wantsGroups = []*Group{
		{Gid: "0", Name: "wheel"},
		{Gid: "1", Name: "daemon"},
		{Gid: "2", Name: "kmem"},
		{Gid: "3", Name: "sys"},
		{Gid: "5", Name: "operator"},
		{Gid: "6", Name: "mail"},
		{Gid: "4", Name: "tty"},
		{Gid: "7", Name: "bin"},
		{Gid: "8", Name: "procview"},
		{Gid: "9", Name: "procmod"},
		{Gid: "10", Name: "owner"},
		{Gid: "12", Name: "everyone"},
	}

	// Use testdata fixture
	groupIterator = &iterateGroupsHelperTest{}

	// Test that groups are retrieved in same order as defined
	gotGroups := make([]*Group, 0, len(wantsGroups))
	err := iterateGroups(func(g *Group) error {
		gotGroups = append(gotGroups, g)
		return nil
	})

	if err != syscall.ENOENT {
		t.Errorf("iterating groups: %v", err)
	}

	if !reflect.DeepEqual(wantsGroups, gotGroups) {
		t.Errorf("iterate groups result is incorrect: got: %+v, want: %+v", gotGroups, wantsGroups)
	}
}
