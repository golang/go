// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bootstrap_test verifies that the current GOROOT can be used to bootstrap
// itself.
package bootstrap_test

import (
	"fmt"
	"internal/goexperiment"
	"internal/platform"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestRepeatBootstrap(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test that rebuilds the entire toolchain")
	}
	switch runtime.GOOS {
	case "android", "ios", "js", "wasip1":
		t.Skipf("skipping because the toolchain does not have to bootstrap on GOOS=%s", runtime.GOOS)
	}

	realGoroot := testenv.GOROOT(t)

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

	overlayStart := time.Now()

	goroot := filepath.Join(parent, "goroot")

	gorootSrc := filepath.Join(goroot, "src")
	if err := overlayDir(gorootSrc, filepath.Join(realGoroot, "src")); err != nil {
		t.Fatal(err)
	}

	gorootLib := filepath.Join(goroot, "lib")
	if err := overlayDir(gorootLib, filepath.Join(realGoroot, "lib")); err != nil {
		t.Fatal(err)
	}

	t.Logf("GOROOT overlay set up in %s", time.Since(overlayStart))

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

	var stdout strings.Builder
	cmd := exec.Command(filepath.Join(goroot, "src", makeScript))
	cmd.Dir = gorootSrc
	cmd.Env = append(cmd.Environ(), "GOROOT=", "GOROOT_BOOTSTRAP="+realGoroot, "GOFIPS140=off")
	cmd.Stderr = os.Stderr
	cmd.Stdout = io.MultiWriter(os.Stdout, &stdout)
	if err := cmd.Run(); err != nil {
		t.Fatal(err)
	}

	t.Run("GOFIPS140=latest", func(t *testing.T) {
		if !platform.FIPS140Supported(runtime.GOOS, runtime.GOARCH) {
			t.Skipf("GOFIPS140 is not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
		}
		if goexperiment.BoringCrypto {
			t.Skip("GOEXPERIMENT=boringcrypto is not compatible with GOFIPS140")
		}

		cmd := exec.Command(filepath.Join(goroot, "src", makeScript))
		cmd.Dir = gorootSrc
		cmd.Env = append(cmd.Environ(), "GOROOT=", "GOROOT_BOOTSTRAP="+realGoroot, "GOFIPS140=latest")
		cmd.Stderr = os.Stderr
		cmd.Stdout = os.Stdout
		if err := cmd.Run(); err != nil {
			t.Fatal(err)
		}

		cmd = exec.Command(filepath.Join(goroot, "bin", "go"), "env", "GOFIPS140")
		cmd.Env = append(cmd.Environ(), "GOROOT=", "GOFIPS140=")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%s failed: %v\n%s", cmd, err, out)
		}
		if got := strings.TrimSpace(string(out)); got != "latest" {
			t.Errorf("default GOFIPS140 = %q, want latest", got)
		}

		mainGo := filepath.Join(t.TempDir(), "main.go")
		const program = `package main

import "crypto/fips140"

func main() {
	if !fips140.Enabled() {
		panic("FIPS 140-3 mode is not enabled")
	}
}
`
		if err := os.WriteFile(mainGo, []byte(program), 0666); err != nil {
			t.Fatal(err)
		}
		cmd = exec.Command(filepath.Join(goroot, "bin", "go"), "run", mainGo)
		cmd.Env = append(cmd.Environ(), "GOROOT=", "GOFIPS140=")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("%s failed: %v\n%s", cmd, err, out)
		}
	})

	// Test that go.dev/issue/42563 hasn't regressed.
	t.Run("PATH reminder", func(t *testing.T) {
		var want string
		switch gorootBin := filepath.Join(goroot, "bin"); runtime.GOOS {
		default:
			want = fmt.Sprintf("*** You need to add %s to your PATH.", gorootBin)
		case "plan9":
			want = fmt.Sprintf("*** You need to bind %s before /bin.", gorootBin)
		}
		if got := stdout.String(); !strings.Contains(got, want) {
			t.Errorf("reminder %q is missing from %s stdout:\n%s", want, makeScript, got)
		}
	})
}
