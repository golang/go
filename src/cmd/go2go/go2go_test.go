// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"sync"
	"testing"
)

func TestMain(m *testing.M) {
	os.Exit(testMain(m))
}

func testMain(m *testing.M) int {
	dir, err := ioutil.TempDir("", "go2gotest")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer os.RemoveAll(dir)
	testTempDir = dir

	return m.Run()
}

// testTempDir is a temporary directory the tests can use.
var testTempDir string

// testGo2go is the version of cmd/go2go run by the tests.
var testGo2go string

// testGo2goOnce ensures that testGo2go is built only once.
var testGo2goOnce sync.Once

// testGo2goErr is an error that occurred when building testGo2go.
// In the normal case this is nil.
var testGo2goErr error

// buildGo2go builds an up-to-date version of cmd/go2go.
// This is not run from TestMain because it's simpler if it has a *testing.T.
func buildGo2go(t *testing.T) {
	t.Helper()
	testenv.MustHaveGoBuild(t)
	testGo2goOnce.Do(func() {
		testGo2go = filepath.Join(testTempDir, "go2go.exe")
		t.Logf("running [go build -o %s]", testGo2go)
		out, err := exec.Command(testenv.GoToolPath(t), "build", "-o", testGo2go).CombinedOutput()
		if len(out) > 0 {
			t.Logf("%s", out)
		}
		testGo2goErr = err
	})
	if testGo2goErr != nil {
		t.Fatal("failed to build testgo2go program:", testGo2goErr)
	}
}

func TestGO2PATH(t *testing.T) {
	buildGo2go(t)

	copyFile := func(path string, info os.FileInfo, err error) error {
		if err != nil {
			t.Fatal(err)
		}
		newPath := filepath.Join(testTempDir, path)
		if info.IsDir() {
			if err := os.MkdirAll(newPath, 0755); err != nil {
				t.Fatal(err)
			}
			return nil
		}
		data, err := ioutil.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := ioutil.WriteFile(newPath, data, 0444); err != nil {
			t.Fatal(err)
		}
		return nil
	}

	if err := filepath.Walk("testdata/go2path/src", copyFile); err != nil {
		t.Fatal(err)
	}

	d, err := os.Open(filepath.Join(testTempDir, "testdata/go2path/src"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	dirs, err := d.Readdirnames(-1)
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(dirs)

	for _, dir := range dirs {
		t.Run(dir, func(t *testing.T) {
			cmd := exec.Command(testGo2go, "test")
			cmd.Dir = filepath.Join(testTempDir, "testdata", "go2path", "src", dir)
			cmd.Env = append(os.Environ(), "GO2PATH=" + filepath.Join(testTempDir, "testdata", "go2path"))
			t.Logf("running [%s test] in %s", testGo2go, cmd.Dir)
			out, err := cmd.CombinedOutput()
			if len(out) > 0 {
				t.Log(dir)
				t.Logf("%s", out)
			}
			if err != nil {
				t.Errorf("error testing %s: %v", dir, err)
			}
		})
	}
}
