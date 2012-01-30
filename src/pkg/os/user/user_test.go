// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package user

import (
	"os"
	"runtime"
	"testing"
)

func skip(t *testing.T) bool {
	if !implemented {
		t.Logf("user: not implemented; skipping tests")
		return true
	}

	switch runtime.GOOS {
	case "linux", "freebsd", "darwin", "windows":
		return false
	}

	t.Logf("user: Lookup not implemented on %s; skipping test", runtime.GOOS)
	return true
}

func TestCurrent(t *testing.T) {
	if skip(t) {
		return
	}

	u, err := Current()
	if err != nil {
		t.Fatalf("Current: %v", err)
	}
	fi, err := os.Stat(u.HomeDir)
	if err != nil || !fi.IsDir() {
		t.Errorf("expected a valid HomeDir; stat(%q): err=%v", u.HomeDir, err)
	}
	if u.Username == "" {
		t.Fatalf("didn't get a username")
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
	// TODO(brainman): fix it once we know how.
	if runtime.GOOS == "windows" {
		t.Log("skipping Gid and HomeDir comparisons")
		return
	}
	if want.Gid != got.Gid {
		t.Errorf("got Gid=%q; want %q", got.Gid, want.Gid)
	}
	if want.HomeDir != got.HomeDir {
		t.Errorf("got HomeDir=%q; want %q", got.HomeDir, want.HomeDir)
	}
}

func TestLookup(t *testing.T) {
	if skip(t) {
		return
	}

	want, err := Current()
	if err != nil {
		t.Fatalf("Current: %v", err)
	}
	got, err := Lookup(want.Username)
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	compare(t, want, got)
}

func TestLookupId(t *testing.T) {
	if skip(t) {
		return
	}

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
