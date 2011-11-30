// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package user

import (
	"os"
	"reflect"
	"runtime"
	"syscall"
	"testing"
)

func skip(t *testing.T) bool {
	if !implemented {
		t.Logf("user: not implemented; skipping tests")
		return true
	}

	if runtime.GOOS == "linux" || runtime.GOOS == "freebsd" || runtime.GOOS == "darwin" {
		return false
	}

	t.Logf("user: Lookup not implemented on %s; skipping test", runtime.GOOS)
	return true
}

func TestLookup(t *testing.T) {
	if skip(t) {
		return
	}

	// Test LookupId on the current user
	uid := syscall.Getuid()
	u, err := LookupId(uid)
	if err != nil {
		t.Fatalf("LookupId: %v", err)
	}
	if e, g := uid, u.Uid; e != g {
		t.Errorf("expected Uid of %d; got %d", e, g)
	}
	fi, err := os.Stat(u.HomeDir)
	if err != nil || !fi.IsDir() {
		t.Errorf("expected a valid HomeDir; stat(%q): err=%v, IsDir=%v", u.HomeDir, err, fi.IsDir())
	}
	if u.Username == "" {
		t.Fatalf("didn't get a username")
	}

	// Test Lookup by username, using the username from LookupId
	un, err := Lookup(u.Username)
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if !reflect.DeepEqual(u, un) {
		t.Errorf("Lookup by userid vs. name didn't match\n"+
			"LookupId(%d): %#v\n"+
			"Lookup(%q): %#v\n", uid, u, u.Username, un)
	}
}
