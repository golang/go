// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package user

import (
	"testing"
)

func checkUser(t *testing.T) {
	t.Helper()
	if !userImplemented {
		t.Skip("user: not implemented; skipping tests")
	}
}

func TestCurrent(t *testing.T) {
	u, err := Current()
	if err != nil {
		t.Fatalf("Current: %v (got %#v)", err, u)
	}
	if u.HomeDir == "" {
		t.Errorf("didn't get a HomeDir")
	}
	if u.Username == "" {
		t.Errorf("didn't get a username")
	}
}

func BenchmarkCurrent(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Current()
	}
}

func compare(t *testing.T, want, got *User) {
	if want.Uid != got.Uid {
		t.Errorf("got Uid=%q; want %q", got.Uid, want.Uid)
	}
	if want.Username != got.Username {
		t.Errorf("got Username=%q; want %q", got.Username, want.Username)
	}
	if want.Name != got.Name {
		t.Errorf("got Name=%q; want %q", got.Name, want.Name)
	}
	if want.HomeDir != got.HomeDir {
		t.Errorf("got HomeDir=%q; want %q", got.HomeDir, want.HomeDir)
	}
	if want.Gid != got.Gid {
		t.Errorf("got Gid=%q; want %q", got.Gid, want.Gid)
	}
}

func TestLookup(t *testing.T) {
	checkUser(t)

	want, err := Current()
	if err != nil {
		t.Fatalf("Current: %v", err)
	}
	// TODO: Lookup() has a fast path that calls Current() and returns if the
	// usernames match, so this test does not exercise very much. It would be
	// good to try and test finding a different user than the current user.
	got, err := Lookup(want.Username)
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	compare(t, want, got)
}

func TestLookupId(t *testing.T) {
	checkUser(t)

	want, err := Current()
	if err != nil {
		t.Fatalf("Current: %v", err)
	}
	got, err := LookupId(want.Uid)
	if err != nil {
		t.Fatalf("LookupId: %v", err)
	}
	compare(t, want, got)
}

func checkGroup(t *testing.T) {
	t.Helper()
	if !groupImplemented {
		t.Skip("user: group not implemented; skipping test")
	}
}

func TestLookupGroup(t *testing.T) {
	checkGroup(t)
	user, err := Current()
	if err != nil {
		t.Fatalf("Current(): %v", err)
	}

	g1, err := LookupGroupId(user.Gid)
	if err != nil {
		// NOTE(rsc): Maybe the group isn't defined. That's fine.
		// On my OS X laptop, rsc logs in with group 5000 even
		// though there's no name for group 5000. Such is Unix.
		t.Logf("LookupGroupId(%q): %v", user.Gid, err)
		return
	}
	if g1.Gid != user.Gid {
		t.Errorf("LookupGroupId(%q).Gid = %s; want %s", user.Gid, g1.Gid, user.Gid)
	}

	g2, err := LookupGroup(g1.Name)
	if err != nil {
		t.Fatalf("LookupGroup(%q): %v", g1.Name, err)
	}
	if g1.Gid != g2.Gid || g1.Name != g2.Name {
		t.Errorf("LookupGroup(%q) = %+v; want %+v", g1.Name, g2, g1)
	}
}

func checkGroupList(t *testing.T) {
	t.Helper()
	if !groupListImplemented {
		t.Skip("user: group list not implemented; skipping test")
	}
}

func TestGroupIds(t *testing.T) {
	checkGroupList(t)
	user, err := Current()
	if err != nil {
		t.Fatalf("Current(): %v", err)
	}
	gids, err := user.GroupIds()
	if err != nil {
		t.Fatalf("%+v.GroupIds(): %v", user, err)
	}
	if !containsID(gids, user.Gid) {
		t.Errorf("%+v.GroupIds() = %v; does not contain user GID %s", user, gids, user.Gid)
	}
}

func containsID(ids []string, id string) bool {
	for _, x := range ids {
		if x == id {
			return true
		}
	}
	return false
}
