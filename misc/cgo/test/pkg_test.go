// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestCrossPackageTests compiles and runs tests that depend on imports of other
// local packages, using source code stored in the testdata directory.
//
// The tests in the misc directory tree do not have a valid import path in
// GOPATH mode, so they previously used relative imports. However, relative
// imports do not work in module mode. In order to make the test work in both
// modes, we synthesize a GOPATH in which the module paths are equivalent, and
// run the tests as a subprocess.
//
// If and when we no longer support these tests in GOPATH mode, we can remove
// this shim and move the tests currently located in testdata back into the
// parent directory.
func TestCrossPackageTests(t *testing.T) {
	switch runtime.GOOS {
	case "android":
		t.Skip("Can't exec cmd/go subprocess on Android.")
	case "ios":
		switch runtime.GOARCH {
		case "arm64":
			t.Skip("Can't exec cmd/go subprocess on iOS.")
		}
	}

	GOPATH, err := os.MkdirTemp("", "cgotest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(GOPATH)

	modRoot := filepath.Join(GOPATH, "src", "cgotest")
	if err := overlayDir(modRoot, "testdata"); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modRoot, "go.mod"), []byte("module cgotest\n"), 0666); err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command("go", "test")
	if testing.Verbose() {
		cmd.Args = append(cmd.Args, "-v")
	}
	if testing.Short() {
		cmd.Args = append(cmd.Args, "-short")
	}
	cmd.Dir = modRoot
	cmd.Env = append(os.Environ(), "GOPATH="+GOPATH, "PWD="+cmd.Dir)
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Logf("%s:\n%s", strings.Join(cmd.Args, " "), out)
	} else {
		t.Fatalf("%s: %s\n%s", strings.Join(cmd.Args, " "), err, out)
	}
}
