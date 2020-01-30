// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo

package packages_test

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/testenv"
)

func TestLoadImportsC(t *testing.T) {
	// This test checks that when a package depends on the
	// test variant of "syscall", "unsafe", or "runtime/cgo", that dependency
	// is not removed when those packages are added when it imports "C".
	//
	// For this test to work, the external test of syscall must have a dependency
	// on net, and net must import "syscall" and "C".
	if runtime.GOOS == "windows" {
		t.Skipf("skipping on windows; packages on windows do not satisfy conditions for test.")
	}
	if runtime.GOOS == "plan9" {
		// See https://golang.org/issue/27100.
		t.Skip(`skipping on plan9; for some reason "net [syscall.test]" is not loaded`)
	}
	testenv.NeedsGoPackages(t)

	cfg := &packages.Config{
		Context: testCtx,
		Mode:    packages.LoadImports,
		Tests:   true,
	}
	initial, err := packages.Load(cfg, "syscall", "net")
	if err != nil {
		t.Fatalf("failed to load imports: %v", err)
	}

	_, all := importGraph(initial)

	for _, test := range []struct {
		pattern    string
		wantImport string // an import to check for
	}{
		{"net", "syscall:syscall"},
		{"net [syscall.test]", "syscall:syscall [syscall.test]"},
		{"syscall_test [syscall.test]", "net:net [syscall.test]"},
	} {
		// Test the import paths.
		pkg := all[test.pattern]
		if pkg == nil {
			t.Errorf("package %q not loaded", test.pattern)
			continue
		}
		if imports := strings.Join(imports(pkg), " "); !strings.Contains(imports, test.wantImport) {
			t.Errorf("package %q: got \n%s, \nwant to have %s", test.pattern, imports, test.wantImport)
		}
	}
}

// Stolen from internal/testenv package in core.
// hasGoBuild reports whether the current system can build programs with ``go build''
// and then run them with os.StartProcess or exec.Command.
func hasGoBuild() bool {
	if os.Getenv("GO_GCFLAGS") != "" {
		// It's too much work to require every caller of the go command
		// to pass along "-gcflags="+os.Getenv("GO_GCFLAGS").
		// For now, if $GO_GCFLAGS is set, report that we simply can't
		// run go build.
		return false
	}
	switch runtime.GOOS {
	case "android", "js":
		return false
	case "darwin":
		if strings.HasPrefix(runtime.GOARCH, "arm") {
			return false
		}
	}
	return true
}

func TestCgoNoSyntax(t *testing.T) {
	packagestest.TestAll(t, testCgoNoSyntax)
}
func testCgoNoSyntax(t *testing.T, exporter packagestest.Exporter) {
	// The android builders have a complex setup which causes this test to fail. See discussion on
	// golang.org/cl/214943 for more details.
	if !hasGoBuild() {
		t.Skip("this test can't run on platforms without go build. See discussion on golang.org/cl/214943 for more details.")
	}

	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"c/c.go": `package c; import "C"`,
		},
	}})

	// Explicitly enable cgo.
	exported.Config.Env = append(exported.Config.Env, "CGO_ENABLED=1")

	modes := []packages.LoadMode{
		packages.NeedTypes,
		packages.NeedName | packages.NeedTypes,
		packages.NeedName | packages.NeedTypes | packages.NeedImports,
		packages.NeedName | packages.NeedTypes | packages.NeedImports | packages.NeedDeps,
		packages.NeedName | packages.NeedImports,
	}
	for _, mode := range modes {
		t.Run(fmt.Sprint(mode), func(t *testing.T) {
			exported.Config.Mode = mode
			pkgs, err := packages.Load(exported.Config, "golang.org/fake/c")
			if err != nil {
				t.Fatal(err)
			}
			if len(pkgs) != 1 {
				t.Fatalf("Expected 1 package, got %v", pkgs)
			}
			pkg := pkgs[0]
			if len(pkg.Errors) != 0 {
				t.Fatalf("Expected no errors in package, got %v", pkg.Errors)
			}
		})
	}
}

func TestCgoBadPkgConfig(t *testing.T) {
	packagestest.TestAll(t, testCgoBadPkgConfig)
}
func testCgoBadPkgConfig(t *testing.T, exporter packagestest.Exporter) {
	if !hasGoBuild() {
		t.Skip("this test can't run on platforms without go build. See discussion on golang.org/cl/214943 for more details.")
	}

	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"c/c.go": `package c

// #cgo pkg-config: --cflags --  foo
import "C"`,
		},
	}})

	dir := buildFakePkgconfig(t, exported.Config.Env)
	defer os.RemoveAll(dir)
	env := exported.Config.Env
	for i, v := range env {
		if strings.HasPrefix(v, "PATH=") {
			env[i] = "PATH=" + dir + string(os.PathListSeparator) + v[len("PATH="):]
		}
	}

	exported.Config.Env = append(exported.Config.Env, "CGO_ENABLED=1")

	exported.Config.Mode = packages.NeedName | packages.NeedCompiledGoFiles
	pkgs, err := packages.Load(exported.Config, "golang.org/fake/c")
	if err != nil {
		t.Fatal(err)
	}
	if len(pkgs) != 1 {
		t.Fatalf("Expected 1 package, got %v", pkgs)
	}
	if pkgs[0].Name != "c" {
		t.Fatalf("Expected package to have name \"c\", got %q", pkgs[0].Name)
	}
}

func buildFakePkgconfig(t *testing.T, env []string) string {
	tmpdir, err := ioutil.TempDir("", "fakepkgconfig")
	if err != nil {
		t.Fatal(err)
	}
	err = ioutil.WriteFile(filepath.Join(tmpdir, "pkg-config.go"), []byte(`
package main

import "fmt"
import "os"

func main() {
	fmt.Fprintln(os.Stderr, "bad")
	os.Exit(2)
}
`), 0644)
	if err != nil {
		os.RemoveAll(tmpdir)
		t.Fatal(err)
	}
	cmd := exec.Command("go", "build", "-o", "pkg-config", "pkg-config.go")
	cmd.Dir = tmpdir
	cmd.Env = env

	if b, err := cmd.CombinedOutput(); err != nil {
		os.RemoveAll(tmpdir)
		fmt.Println(os.Environ())
		t.Log(string(b))
		t.Fatal(err)
	}
	return tmpdir
}
