// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testenv_test

import (
	"internal/platform"
	"internal/testenv"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestGoToolLocation(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	var exeSuffix string
	if runtime.GOOS == "windows" {
		exeSuffix = ".exe"
	}

	// Tests are defined to run within their package source directory,
	// and this package's source directory is $GOROOT/src/internal/testenv.
	// The 'go' command is installed at $GOROOT/bin/go, so if the environment
	// is correct then testenv.GoTool() should be identical to ../../../bin/go.

	relWant := "../../../bin/go" + exeSuffix
	absWant, err := filepath.Abs(relWant)
	if err != nil {
		t.Fatal(err)
	}

	wantInfo, err := os.Stat(absWant)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("found go tool at %q (%q)", relWant, absWant)

	goTool, err := testenv.GoTool()
	if err != nil {
		t.Fatalf("testenv.GoTool(): %v", err)
	}
	t.Logf("testenv.GoTool() = %q", goTool)

	gotInfo, err := os.Stat(goTool)
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(wantInfo, gotInfo) {
		t.Fatalf("%q is not the same file as %q", absWant, goTool)
	}
}

func TestHasGoBuild(t *testing.T) {
	if !testenv.HasGoBuild() {
		switch runtime.GOOS {
		case "js", "wasip1":
			// No exec syscall, so these shouldn't be able to 'go build'.
			t.Logf("HasGoBuild is false on %s", runtime.GOOS)
			return
		}

		b := testenv.Builder()
		if b == "" {
			// We shouldn't make assumptions about what kind of sandbox or build
			// environment external Go users may be running in.
			t.Skipf("skipping: 'go build' unavailable")
		}

		// Since we control the Go builders, we know which ones ought
		// to be able to run 'go build'. Check that they can.
		//
		// (Note that we don't verify that any builders *can't* run 'go build'.
		// If a builder starts running 'go build' tests when it shouldn't,
		// we will presumably find out about it when those tests fail.)
		switch runtime.GOOS {
		case "ios":
			if strings.HasSuffix(b, "-corellium") {
				// The corellium environment is self-hosting, so it should be able
				// to build even though real "ios" devices can't exec.
			} else {
				// The usual iOS sandbox does not allow the app to start another
				// process. If we add builders on stock iOS devices, they presumably
				// will not be able to exec, so we may as well allow that now.
				t.Logf("HasGoBuild is false on %s", b)
				return
			}
		case "android":
			if strings.HasSuffix(b, "-emu") && platform.MustLinkExternal(runtime.GOOS, runtime.GOARCH, false) {
				// As of 2023-05-02, the test environment on the emulated builders is
				// missing a C linker.
				t.Logf("HasGoBuild is false on %s", b)
				return
			}
		}

		if strings.HasSuffix(b, "-noopt") {
			// The -noopt builder sets GO_GCFLAGS, which causes tests of 'go build' to
			// be skipped.
			t.Logf("HasGoBuild is false on %s", b)
			return
		}

		t.Fatalf("HasGoBuild unexpectedly false on %s", b)
	}

	t.Logf("HasGoBuild is true; checking consistency with other functions")

	hasExec := false
	hasExecGo := false
	t.Run("MustHaveExec", func(t *testing.T) {
		testenv.MustHaveExec(t)
		hasExec = true
	})
	t.Run("MustHaveExecPath", func(t *testing.T) {
		testenv.MustHaveExecPath(t, "go")
		hasExecGo = true
	})
	if !hasExec {
		t.Errorf(`MustHaveExec(t) skipped unexpectedly`)
	}
	if !hasExecGo {
		t.Errorf(`MustHaveExecPath(t, "go") skipped unexpectedly`)
	}

	dir := t.TempDir()
	mainGo := filepath.Join(dir, "main.go")
	if err := os.WriteFile(mainGo, []byte("package main\nfunc main() {}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	cmd := testenv.Command(t, "go", "build", "-o", os.DevNull, mainGo)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v: %v\n%s", cmd, err, out)
	}
}

func TestMustHaveExec(t *testing.T) {
	hasExec := false
	t.Run("MustHaveExec", func(t *testing.T) {
		testenv.MustHaveExec(t)
		t.Logf("MustHaveExec did not skip")
		hasExec = true
	})

	switch runtime.GOOS {
	case "js", "wasip1":
		if hasExec {
			// js and wasip1 lack an “exec” syscall.
			t.Errorf("expected MustHaveExec to skip on %v", runtime.GOOS)
		}
	case "ios":
		if b := testenv.Builder(); strings.HasSuffix(b, "-corellium") && !hasExec {
			// Most ios environments can't exec, but the corellium builder can.
			t.Errorf("expected MustHaveExec not to skip on %v", b)
		}
	default:
		if b := testenv.Builder(); b != "" && !hasExec {
			t.Errorf("expected MustHaveExec not to skip on %v", b)
		}
	}
}

func TestCleanCmdEnvPWD(t *testing.T) {
	// Test that CleanCmdEnv sets PWD if cmd.Dir is set.
	switch runtime.GOOS {
	case "plan9", "windows":
		t.Skipf("PWD is not used on %s", runtime.GOOS)
	}
	dir := t.TempDir()
	cmd := testenv.Command(t, testenv.GoToolPath(t), "help")
	cmd.Dir = dir
	cmd = testenv.CleanCmdEnv(cmd)

	for _, env := range cmd.Env {
		if strings.HasPrefix(env, "PWD=") {
			pwd := strings.TrimPrefix(env, "PWD=")
			if pwd != dir {
				t.Errorf("unexpected PWD: want %s, got %s", dir, pwd)
			}
			return
		}
	}
	t.Error("PWD not set in cmd.Env")
}
