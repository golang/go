// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcimporter

import (
	"go/build"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"code.google.com/p/go.tools/go/types"
)

var gcPath string // Go compiler path

func init() {
	// determine compiler
	var gc string
	switch runtime.GOARCH {
	case "386":
		gc = "8g"
	case "amd64":
		gc = "6g"
	case "arm":
		gc = "5g"
	default:
		gcPath = "unknown-GOARCH-compiler"
		return
	}
	gcPath = filepath.Join(build.ToolDir, gc)
}

func compile(t *testing.T, dirname, filename string) string {
	cmd := exec.Command(gcPath, filename)
	cmd.Dir = dirname
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Logf("%s", out)
		t.Fatalf("%s %s failed: %s", gcPath, filename, err)
	}
	archCh, _ := build.ArchChar(runtime.GOARCH)
	// filename should end with ".go"
	return filepath.Join(dirname, filename[:len(filename)-2]+archCh)
}

// Use the same global imports map for all tests. The effect is
// as if all tested packages were imported into a single package.
var imports = make(map[string]*types.Package)

func testPath(t *testing.T, path string) bool {
	t0 := time.Now()
	_, err := Import(imports, path)
	if err != nil {
		t.Errorf("testPath(%s): %s", path, err)
		return false
	}
	t.Logf("testPath(%s): %v", path, time.Since(t0))
	return true
}

const maxTime = 30 * time.Second

func testDir(t *testing.T, dir string, endTime time.Time) (nimports int) {
	dirname := filepath.Join(runtime.GOROOT(), "pkg", runtime.GOOS+"_"+runtime.GOARCH, dir)
	list, err := ioutil.ReadDir(dirname)
	if err != nil {
		t.Fatalf("testDir(%s): %s", dirname, err)
	}
	for _, f := range list {
		if time.Now().After(endTime) {
			t.Log("testing time used up")
			return
		}
		switch {
		case !f.IsDir():
			// try extensions
			for _, ext := range pkgExts {
				if strings.HasSuffix(f.Name(), ext) {
					name := f.Name()[0 : len(f.Name())-len(ext)] // remove extension
					if testPath(t, filepath.Join(dir, name)) {
						nimports++
					}
				}
			}
		case f.IsDir():
			nimports += testDir(t, filepath.Join(dir, f.Name()), endTime)
		}
	}
	return
}

func TestImport(t *testing.T) {
	// This package does not handle gccgo export data.
	if runtime.Compiler == "gccgo" {
		return
	}

	// On cross-compile builds, the path will not exist.
	// Need to use GOHOSTOS, which is not available.
	if _, err := os.Stat(gcPath); err != nil {
		t.Skipf("skipping test: %v", err)
	}

	if outFn := compile(t, "testdata", "exports.go"); outFn != "" {
		defer os.Remove(outFn)
	}

	nimports := 0
	if testPath(t, "./testdata/exports") {
		nimports++
	}
	nimports += testDir(t, "", time.Now().Add(maxTime)) // installed packages
	t.Logf("tested %d imports", nimports)
}

var importedObjectTests = []struct {
	name string
	want string
}{
	{"unsafe.Pointer", "type Pointer unsafe.Pointer"},
	{"math.Pi", "const Pi untyped float"},
	{"io.Reader", "type Reader interface{Read(p []byte) (n int, err error)}"},
	{"io.ReadWriter", "type ReadWriter interface{Read(p []byte) (n int, err error); Write(p []byte) (n int, err error)}"},
	{"math.Sin", "func Sin(x float64) float64"},
	// TODO(gri) add more tests
}

func TestImportedTypes(t *testing.T) {
	// This package does not handle gccgo export data.
	if runtime.Compiler == "gccgo" {
		return
	}
	for _, test := range importedObjectTests {
		s := strings.Split(test.name, ".")
		if len(s) != 2 {
			t.Fatal("inconsistent test data")
		}
		importPath := s[0]
		objName := s[1]

		pkg, err := Import(imports, importPath)
		if err != nil {
			t.Error(err)
			continue
		}

		obj := pkg.Scope().Lookup(objName)
		if obj == nil {
			t.Errorf("%s: object not found", test.name)
			continue
		}

		got := types.ObjectString(pkg, obj)
		if got != test.want {
			t.Errorf("%s: got %q; want %q", test.name, got, test.want)
		}
	}
}

func TestIssue5815(t *testing.T) {
	// This package does not handle gccgo export data.
	if runtime.Compiler == "gccgo" {
		return
	}

	pkg, err := Import(make(map[string]*types.Package), "strings")
	if err != nil {
		t.Fatal(err)
	}

	scope := pkg.Scope()
	for _, name := range scope.Names() {
		obj := scope.Lookup(name)
		if obj.Pkg() == nil {
			t.Errorf("no pkg for %s", obj)
		}
		if tname, _ := obj.(*types.TypeName); tname != nil {
			named := tname.Type().(*types.Named)
			for i := 0; i < named.NumMethods(); i++ {
				m := named.Method(i)
				if m.Pkg() == nil {
					t.Errorf("no pkg for %s", m)
				}
			}
		}
	}
}
