// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9
// +build !plan9

package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"testing"
)

func TestMain(m *testing.M) {
	if os.Getenv("GO_GETGO_TEST_IS_GETGO") != "" {
		main()
		os.Exit(0)
	}

	if os.Getenv("GOGET_INTEGRATION") == "" {
		fmt.Fprintln(os.Stderr, "main_test: Skipping integration tests with GOGET_INTEGRATION unset")
		return
	}

	// Don't let these environment variables confuse the test.
	os.Unsetenv("GOBIN")
	os.Unsetenv("GOPATH")
	os.Unsetenv("GIT_ALLOW_PROTOCOL")
	os.Unsetenv("PATH")

	os.Exit(m.Run())
}

func createTmpHome(t *testing.T) string {
	tmpd, err := os.MkdirTemp("", "testgetgo")
	if err != nil {
		t.Fatalf("creating test tempdir failed: %v", err)
	}

	os.Setenv("HOME", tmpd)
	return tmpd
}

// doRun runs the test getgo command, recording stdout and stderr and
// returning exit status.
func doRun(t *testing.T, args ...string) error {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Helper()

	t.Logf("running getgo %v", args)
	var stdout, stderr bytes.Buffer
	cmd := exec.Command(exe, args...)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	cmd.Env = append(os.Environ(), "GO_GETGO_TEST_IS_GETGO=1")
	status := cmd.Run()
	if stdout.Len() > 0 {
		t.Log("standard output:")
		t.Log(stdout.String())
	}
	if stderr.Len() > 0 {
		t.Log("standard error:")
		t.Log(stderr.String())
	}
	return status
}

func TestCommandVerbose(t *testing.T) {
	tmpd := createTmpHome(t)
	defer os.RemoveAll(tmpd)

	err := doRun(t, "-v")
	if err != nil {
		t.Fatal(err)
	}
	// make sure things are in path
	shellConfig, err := shellConfigFile()
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(shellConfig)
	if err != nil {
		t.Fatal(err)
	}
	home, err := getHomeDir()
	if err != nil {
		t.Fatal(err)
	}

	expected := fmt.Sprintf(`
export PATH=$PATH:%s/.go/bin

export GOPATH=%s/go

export PATH=$PATH:%s/go/bin
`, home, home, home)

	if string(b) != expected {
		t.Fatalf("%s expected %q, got %q", shellConfig, expected, string(b))
	}
}

func TestCommandPathExists(t *testing.T) {
	tmpd := createTmpHome(t)
	defer os.RemoveAll(tmpd)

	// run once
	err := doRun(t, "-skip-dl")
	if err != nil {
		t.Fatal(err)
	}
	// make sure things are in path
	shellConfig, err := shellConfigFile()
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(shellConfig)
	if err != nil {
		t.Fatal(err)
	}
	home, err := getHomeDir()
	if err != nil {
		t.Fatal(err)
	}

	expected := fmt.Sprintf(`
export GOPATH=%s/go

export PATH=$PATH:%s/go/bin
`, home, home)

	if string(b) != expected {
		t.Fatalf("%s expected %q, got %q", shellConfig, expected, string(b))
	}

	// run twice
	if err := doRun(t, "-skip-dl"); err != nil {
		t.Fatal(err)
	}

	b, err = os.ReadFile(shellConfig)
	if err != nil {
		t.Fatal(err)
	}

	if string(b) != expected {
		t.Fatalf("%s expected %q, got %q", shellConfig, expected, string(b))
	}
}
