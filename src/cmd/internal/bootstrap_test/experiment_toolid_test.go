// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build explicit

package bootstrap_test

import (
	"bytes"
	"errors"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

// TestExperimentToolID verifies that GOEXPERIMENT settings built
// into the toolchain influence tool ids in the Go command.
// This test requires bootstrapping the toolchain twice, so it's very expensive.
// It must be run explicitly with -tags=explicit.
// Verifies go.dev/issue/33091.
func TestExperimentToolID(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test that rebuilds the entire toolchain twice")
	}
	switch runtime.GOOS {
	case "android", "ios", "js", "wasip1":
		t.Skipf("skipping because the toolchain does not have to bootstrap on GOOS=%s", runtime.GOOS)
	}

	realGoroot := testenv.GOROOT(t)

	// Set up GOROOT.
	goroot := t.TempDir()
	gorootSrc := filepath.Join(goroot, "src")
	if err := overlayDir(gorootSrc, filepath.Join(realGoroot, "src")); err != nil {
		t.Fatal(err)
	}
	gorootLib := filepath.Join(goroot, "lib")
	if err := overlayDir(gorootLib, filepath.Join(realGoroot, "lib")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(goroot, "VERSION"), []byte("go1.999"), 0666); err != nil {
		t.Fatal(err)
	}
	env := append(os.Environ(), "GOROOT=", "GOROOT_BOOTSTRAP="+realGoroot)

	// Use a clean cache.
	gocache := t.TempDir()
	env = append(env, "GOCACHE="+gocache)

	// Build the toolchain without GOEXPERIMENT.
	var makeScript string
	switch runtime.GOOS {
	case "windows":
		makeScript = "make.bat"
	case "plan9":
		makeScript = "make.rc"
	default:
		makeScript = "make.bash"
	}
	makeScriptPath := filepath.Join(realGoroot, "src", makeScript)
	runCmd(t, gorootSrc, env, makeScriptPath)

	// Verify compiler version string.
	goCmdPath := filepath.Join(goroot, "bin", "go")
	gotVersion := bytes.TrimSpace(runCmd(t, gorootSrc, env, goCmdPath, "tool", "compile", "-V=full"))
	wantVersion := []byte(`compile version go1.999`)
	if !bytes.Equal(gotVersion, wantVersion) {
		t.Errorf("compile version without experiment is unexpected:\ngot  %q\nwant %q", gotVersion, wantVersion)
	}

	// Build a package in a mode not handled by the make script.
	runCmd(t, gorootSrc, env, goCmdPath, "build", "-race", "archive/tar")

	// Rebuild the toolchain with GOEXPERIMENT.
	env = append(env, "GOEXPERIMENT=fieldtrack")
	runCmd(t, gorootSrc, env, makeScriptPath)

	// Verify compiler version string.
	gotVersion = bytes.TrimSpace(runCmd(t, gorootSrc, env, goCmdPath, "tool", "compile", "-V=full"))
	wantVersion = []byte(`compile version go1.999 X:fieldtrack`)
	if !bytes.Equal(gotVersion, wantVersion) {
		t.Errorf("compile version with experiment is unexpected:\ngot  %q\nwant %q", gotVersion, wantVersion)
	}

	// Build the same package. We should not get a cache conflict.
	runCmd(t, gorootSrc, env, goCmdPath, "build", "-race", "archive/tar")
}

func runCmd(t *testing.T, dir string, env []string, path string, args ...string) []byte {
	cmd := exec.Command(path, args...)
	cmd.Dir = dir
	cmd.Env = env
	out, err := cmd.Output()
	if err != nil {
		if ee := (*exec.ExitError)(nil); errors.As(err, &ee) {
			out = append(out, ee.Stderr...)
		}
		t.Fatalf("%s failed:\n%s\n%s", cmd, out, err)
	}
	return out
}
