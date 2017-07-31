// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"runtime"
	"testing"
)

const (
	testbin = "testgetgo"
)

var (
	exeSuffix string // ".exe" on Windows
)

func init() {
	if runtime.GOOS == "windows" {
		exeSuffix = ".exe"
	}
}

// TestMain creates a getgo command for testing purposes and
// deletes it after the tests have been run.
func TestMain(m *testing.M) {
	if os.Getenv("GOGET_INTEGRATION") == "" {
		fmt.Fprintln(os.Stderr, "main_test: Skipping integration tests with GOGET_INTEGRATION unset")
		return
	}

	args := []string{"build", "-tags", testbin, "-o", testbin + exeSuffix}
	out, err := exec.Command("go", args...).CombinedOutput()
	if err != nil {
		fmt.Fprintf(os.Stderr, "building %s failed: %v\n%s", testbin, err, out)
		os.Exit(2)
	}

	// Don't let these environment variables confuse the test.
	os.Unsetenv("GOBIN")
	os.Unsetenv("GOPATH")
	os.Unsetenv("GIT_ALLOW_PROTOCOL")
	os.Unsetenv("PATH")

	r := m.Run()

	os.Remove(testbin + exeSuffix)

	os.Exit(r)
}

func createTmpHome(t *testing.T) string {
	tmpd, err := ioutil.TempDir("", "testgetgo")
	if err != nil {
		t.Fatalf("creating test tempdir failed: %v", err)
	}

	os.Setenv("HOME", tmpd)
	return tmpd
}

// doRun runs the test getgo command, recording stdout and stderr and
// returning exit status.
func doRun(t *testing.T, args ...string) error {
	var stdout, stderr bytes.Buffer
	t.Logf("running %s %v", testbin, args)
	cmd := exec.Command("./"+testbin+exeSuffix, args...)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	cmd.Env = os.Environ()
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
	b, err := ioutil.ReadFile(shellConfig)
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
	b, err := ioutil.ReadFile(shellConfig)
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

	b, err = ioutil.ReadFile(shellConfig)
	if err != nil {
		t.Fatal(err)
	}

	if string(b) != expected {
		t.Fatalf("%s expected %q, got %q", shellConfig, expected, string(b))
	}
}
