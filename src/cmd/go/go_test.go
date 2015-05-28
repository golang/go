// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// Whether we can run go or ./testgo.
var canRun = true

func init() {
	switch runtime.GOOS {
	case "android", "nacl":
		canRun = false
	case "darwin":
		switch runtime.GOARCH {
		case "arm", "arm64":
			canRun = false
		}
	}
}

// The TestMain function creates a go command for testing purposes and
// deletes it after the tests have been run.
func TestMain(m *testing.M) {
	flag.Parse()

	if canRun {
		// We give the executable a .exe extension because Windows.
		out, err := exec.Command("go", "build", "-tags", "testgo", "-o", "testgo.exe").CombinedOutput()
		if err != nil {
			fmt.Fprintf(os.Stderr, "building testgo failed: %v\n%s", err, out)
			os.Exit(2)
		}
	}

	// Don't let these environment variables confuse the test.
	os.Unsetenv("GOBIN")
	os.Unsetenv("GOPATH")
	os.Unsetenv("GOROOT")

	r := m.Run()

	if canRun {
		os.Remove("testgo.exe")
	}

	os.Exit(r)
}

// Skip a test if we can't run ./testgo.
func checkTestGo(t *testing.T) {
	if !canRun {
		switch runtime.GOOS {
		case "android", "nacl":
			t.Skipf("skipping on %s", runtime.GOOS)
		case "darwin":
			switch runtime.GOARCH {
			case "arm", "arm64":
				t.Skipf("skipping on %s/%s, no fork", runtime.GOOS, runtime.GOARCH)
			}
		default:
			t.Skip("skipping for unknown reason")
		}
	}
}

// runTestGo runs the test go command, returning stdout, stderr, and
// status.  The contents of addEnv are added to the environment.
func runTestGo(addEnv []string, args ...string) (stdout, stderr string, err error) {
	if !canRun {
		panic("runTestGo called but canRun false")
	}

	cmd := exec.Command("./testgo.exe", args...)
	var ob, eb bytes.Buffer
	cmd.Stdout = &ob
	cmd.Stderr = &eb
	if len(addEnv) > 0 {
		cmd.Env = append(addEnv, os.Environ()...)
	}
	err = cmd.Run()
	return ob.String(), eb.String(), err
}

// tempFile describes a file to put into a temporary directory.
type tempFile struct {
	path     string
	contents string
}

// tempDir describes a temporary directory created for a single test.
type tempDir struct {
	name string
}

// makeTempDir creates a temporary directory for a single test.
func makeTempDir(t *testing.T, files []tempFile) *tempDir {
	dir, err := ioutil.TempDir("", "gotest")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range files {
		if err := ioutil.WriteFile(filepath.Join(dir, f.path), []byte(f.contents), 0666); err != nil {
			t.Fatal(err)
		}
	}
	return &tempDir{dir}
}

// path returns the absolute pathname to file within tempDir.
func (td *tempDir) path(name string) string {
	return filepath.Join(td.name, name)
}

// Remove a temporary directory after a test completes.  This is
// normally called via defer.
func (td *tempDir) remove(t *testing.T) {
	if err := os.RemoveAll(td.name); err != nil {
		fmt.Fprintln(os.Stderr, err)
	}
}

func TestFileLineInErrorMessages(t *testing.T) {
	checkTestGo(t)
	td := makeTempDir(t, []tempFile{{"err.go", `package main; import "bar"`}})
	defer td.remove(t)
	path := td.path("err.go")
	stdout, stderr, err := runTestGo(nil, "run", path)
	if err == nil {
		t.Fatal("go command did not fail")
	}
	lines := strings.Split(stderr, "\n")
	for _, ln := range lines {
		if strings.HasPrefix(ln, path+":") {
			// Test has passed.
			return
		}
	}
	t.Log(err)
	t.Log(stdout)
	t.Log(stderr)
	t.Error("missing file:line in error message")
}
