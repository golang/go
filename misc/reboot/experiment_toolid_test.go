// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build explicit
// +build explicit

// Package experiment_toolid_test verifies that GOEXPERIMENT settings built
// into the toolchain influence tool ids in the Go command.
// This test requires bootstrapping the toolchain twice, so it's very expensive.
// It must be run explicitly with -tags=explicit.
// Verifies golang.org/issue/33091.
package reboot_test

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

func TestExperimentToolID(t *testing.T) {
	// Set up GOROOT
	goroot, err := os.MkdirTemp("", "experiment-goroot")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(goroot)

	gorootSrc := filepath.Join(goroot, "src")
	if err := overlayDir(gorootSrc, filepath.Join(runtime.GOROOT(), "src")); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(goroot, "VERSION"), []byte("go1.999"), 0666); err != nil {
		t.Fatal(err)
	}
	env := append(os.Environ(), "GOROOT=", "GOROOT_BOOTSTRAP="+runtime.GOROOT())

	// Use a clean cache.
	gocache, err := os.MkdirTemp("", "experiment-gocache")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(gocache)
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
	makeScriptPath := filepath.Join(runtime.GOROOT(), "src", makeScript)
	runCmd(t, gorootSrc, env, makeScriptPath)

	// Verify compiler version string.
	goCmdPath := filepath.Join(goroot, "bin", "go")
	if runtime.GOOS == "windows" {
		goCmdPath += ".exe"
	}
	gotVersion := bytes.TrimSpace(runCmd(t, gorootSrc, env, goCmdPath, "tool", "compile", "-V=full"))
	wantVersion := []byte(`compile version go1.999`)
	if !bytes.Equal(gotVersion, wantVersion) {
		t.Errorf("compile version without experiment: got %q, want %q", gotVersion, wantVersion)
	}

	// Build a package in a mode not handled by the make script.
	runCmd(t, gorootSrc, env, goCmdPath, "build", "-race", "archive/tar")

	// Rebuild the toolchain with GOEXPERIMENT.
	env = append(env, "GOEXPERIMENT=fieldtrack")
	runCmd(t, gorootSrc, env, makeScriptPath)

	// Verify compiler version string.
	gotVersion = bytes.TrimSpace(runCmd(t, gorootSrc, env, goCmdPath, "tool", "compile", "-V=full"))
	wantVersion = []byte(`compile version go1.999 X:fieldtrack,framepointer`)
	if !bytes.Equal(gotVersion, wantVersion) {
		t.Errorf("compile version with experiment: got %q, want %q", gotVersion, wantVersion)
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
		t.Fatal(err)
	}
	return out
}
