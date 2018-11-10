// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package exec

import (
	"io/ioutil"
	"os"
	"testing"
)

func TestLookPathUnixEmptyPath(t *testing.T) {
	tmp, err := ioutil.TempDir("", "TestLookPathUnixEmptyPath")
	if err != nil {
		t.Fatal("TempDir failed: ", err)
	}
	defer os.RemoveAll(tmp)
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal("Getwd failed: ", err)
	}
	err = os.Chdir(tmp)
	if err != nil {
		t.Fatal("Chdir failed: ", err)
	}
	defer os.Chdir(wd)

	f, err := os.OpenFile("exec_me", os.O_CREATE|os.O_EXCL, 0700)
	if err != nil {
		t.Fatal("OpenFile failed: ", err)
	}
	err = f.Close()
	if err != nil {
		t.Fatal("Close failed: ", err)
	}

	pathenv := os.Getenv("PATH")
	defer os.Setenv("PATH", pathenv)

	err = os.Setenv("PATH", "")
	if err != nil {
		t.Fatal("Setenv failed: ", err)
	}

	path, err := LookPath("exec_me")
	if err == nil {
		t.Fatal("LookPath found exec_me in empty $PATH")
	}
	if path != "" {
		t.Fatalf("LookPath path == %q when err != nil", path)
	}
}
