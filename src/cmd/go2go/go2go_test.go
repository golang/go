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
	"reflect"
	"runtime"
	"sort"
	"strings"
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
	t.Parallel()
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
			cmd.Env = append(os.Environ(), "GO2PATH="+filepath.Join(testTempDir, "testdata", "go2path"))
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

func TestGO2PATHEqGOPATH(t *testing.T) {
	t.Parallel()
	buildGo2go(t)

	pathDir := t.TempDir()
	pkgDir := filepath.Join(pathDir, "src", "pkg")
	if err := os.MkdirAll(pkgDir, 0755); err != nil {
		t.Fatal(err)
	}
	const pkgSrc = `package pkg; func F() {}`
	if err := ioutil.WriteFile(filepath.Join(pkgDir, "p.go"), []byte(pkgSrc), 0644); err != nil {
		t.Fatal(err)
	}
	cmdDir := filepath.Join(pathDir, "src", "cmd")
	if err := os.MkdirAll(cmdDir, 0755); err != nil {
		t.Fatal(err)
	}
	const cmdSrc = `package main; import "pkg"; func main() { pkg.F() }`
	if err := ioutil.WriteFile(filepath.Join(cmdDir, "cmd.go2"), []byte(cmdSrc), 0644); err != nil {
		t.Fatal(err)
	}

	t.Log("go2go build")
	cmd := exec.Command(testGo2go, "build")
	cmd.Dir = cmdDir
	cmd.Env = append(os.Environ(),
		"GOPATH="+pathDir,
		"GO2PATH="+pathDir,
		"GO111MODULE=off",
	)
	out, err := cmd.CombinedOutput()
	if len(out) > 0 {
		t.Logf("%s", out)
	}
	if err != nil {
		t.Fatalf(`error running "go2go build": %v`, err)
	}

	t.Log("./cmd")
	cmdName := "./cmd"
	if runtime.GOOS == "windows" {
		cmdName += ".exe"
	}
	cmd = exec.Command(cmdName)
	cmd.Dir = cmdDir
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("error running %q: %v", cmdName, err)
	}
	if len(out) != 0 {
		t.Errorf("unexpected output: %q", out)
	}
}

const buildSource = `
package main

import "fmt"

func PrintSlice(type Elem)(s []Elem) {
	for _, v := range s {
		fmt.Println(v)
	}
}

func main() {
	PrintSlice([]string{"hello", "world"})
}
`

func TestBuild(t *testing.T) {
	t.Parallel()
	buildGo2go(t)

	dir := filepath.Join(t.TempDir(), "hello")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(dir, "hello.go2"), []byte(buildSource), 0444); err != nil {
		t.Fatal(err)
	}
	t.Log("go2go build")
	cmd := exec.Command(testGo2go, "build")
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if len(out) > 0 {
		t.Logf("%s", out)
	}
	if err != nil {
		t.Fatalf(`error running "go2go build": %v`, err)
	}

	runHello(t, dir)
}

func runHello(t *testing.T, dir string) {
	cmdName := "./hello"
	if runtime.GOOS == "windows" {
		cmdName += ".exe"
	}
	cmd := exec.Command(cmdName)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	t.Log("./hello")
	if len(out) > 0 {
		t.Logf("%s", out)
	}
	if err != nil {
		t.Fatalf("error running hello: %v", err)
	}
	got := strings.Split(strings.TrimSpace(string(out)), "\n")
	want := []string{"hello", "world"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("hello output %v, want %v", got, want)
	}
}

func TestBuildPackage(t *testing.T) {
	t.Parallel()
	buildGo2go(t)

	gopath := t.TempDir()
	dir := filepath.Join(gopath, "src", "cmd", "hello")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(dir, "hello.go2"), []byte(buildSource), 0444); err != nil {
		t.Fatal(err)
	}
	t.Log("go2go build")
	cmd := exec.Command(testGo2go, "build", "cmd/hello")
	cmd.Dir = gopath
	cmd.Env = append(os.Environ(),
		"GO2PATH="+gopath,
	)
	out, err := cmd.CombinedOutput()
	if len(out) > 0 {
		t.Logf("%s", out)
	}
	if err != nil {
		t.Fatalf(`error running "go2go build": %v`, err)
	}

	runHello(t, gopath)
}
