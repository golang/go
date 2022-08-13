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

	goroot := t.TempDir()

	gorootSrc := filepath.Join(goroot, "src")
	overlayStart := time.Now()
	if err := overlayDir(gorootSrc, filepath.Join(runtime.GOROOT(), "src")); err != nil {
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

	cmd := exec.Command(filepath.Join(runtime.GOROOT(), "src", makeScript))
	cmd.Dir = gorootSrc
	cmd.Env = append(os.Environ(), "GOROOT=", "GOROOT_BOOTSTRAP="+runtime.GOROOT())
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stdout
	if err := cmd.Run(); err != nil {
		t.Fatal(err)
	}
}
