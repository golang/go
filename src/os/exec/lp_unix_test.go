// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package exec_test

import (
	"os"
	"os/exec"
	"testing"
)

func TestLookPathUnixEmptyPath(t *testing.T) {
	// Not parallel: uses Chdir and Setenv.

	tmp := t.TempDir()
	chdir(t, tmp)

	f, err := os.OpenFile("exec_me", os.O_CREATE|os.O_EXCL, 0700)
	if err != nil {
		t.Fatal("OpenFile failed: ", err)
	}
	err = f.Close()
	if err != nil {
		t.Fatal("Close failed: ", err)
	}

	t.Setenv("PATH", "")

	path, err := exec.LookPath("exec_me")
	if err == nil {
		t.Fatal("LookPath found exec_me in empty $PATH")
	}
	if path != "" {
		t.Fatalf("LookPath path == %q when err != nil", path)
	}
}
