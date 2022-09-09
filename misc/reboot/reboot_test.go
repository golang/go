// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package reboot_test verifies that the current GOROOT can be used to bootstrap
// itself.
package reboot_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func TestRepeatBootstrap(t *testing.T) {
	if testing.Short() {
		t.Skipf("skipping test that rebuilds the entire toolchain")
	}

	realGoroot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}

	// To ensure that bootstrapping doesn't unexpectedly depend
	// on the Go repo's git metadata, add a fake (unreadable) git
	// directory above the simulated GOROOT.
	// This mimics the configuration one much have when
	// building from distro-packaged source code
	// (see https://go.dev/issue/54852).
	parent := t.TempDir()
	dotGit := filepath.Join(parent, ".git")
	if err := os.Mkdir(dotGit, 000); err != nil {
		t.Fatal(err)
	}
	goroot := filepath.Join(parent, "goroot")

	gorootSrc := filepath.Join(goroot, "src")
	overlayStart := time.Now()
	if err := overlayDir(gorootSrc, filepath.Join(realGoroot, "src")); err != nil {
		t.Fatal(err)
	}
	t.Logf("GOROOT/src overlay set up in %s", time.Since(overlayStart))

	if err := os.WriteFile(filepath.Join(goroot, "VERSION"), []byte(runtime.Version()), 0666); err != nil {
		t.Fatal(err)
	}

	var makeScript string
	switch runtime.GOOS {
	case "windows":
		makeScript = "make.bat"
	case "plan9":
		makeScript = "make.rc"
	default:
		makeScript = "make.bash"
	}

	cmd := exec.Command(filepath.Join(goroot, "src", makeScript))
	cmd.Dir = gorootSrc
	cmd.Env = append(cmd.Environ(), "GOROOT=", "GOROOT_BOOTSTRAP="+realGoroot)
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stdout
	if err := cmd.Run(); err != nil {
		t.Fatal(err)
	}
}
