// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/ast"
	"go/build"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
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
var imports = make(map[string]*Package)

func testPath(t *testing.T, path string) bool {
	t0 := time.Now()
	_, err := GcImport(imports, path)
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

func TestGcImport(t *testing.T) {
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
	kind ast.ObjKind
	typ  string
}{
	{"unsafe.Pointer", ast.Typ, "Pointer"},
	{"math.Pi", ast.Con, "untyped float"},
	{"io.Reader", ast.Typ, "interface{Read(p []byte) (n int, err error)}"},
	{"io.ReadWriter", ast.Typ, "interface{Read(p []byte) (n int, err error); Write(p []byte) (n int, err error)}"},
	{"math.Sin", ast.Fun, "func(x·2 float64) (_ float64)"},
	// TODO(gri) add more tests
}

func TestGcImportedTypes(t *testing.T) {
	// This package does not yet know how to read gccgo export data.
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

		pkg, err := GcImport(imports, importPath)
		if err != nil {
			t.Error(err)
			continue
		}

		obj := pkg.Scope.Lookup(objName)

		// TODO(gri) should define an accessor on Object
		var kind ast.ObjKind
		var typ Type
		switch obj := obj.(type) {
		case *Const:
			kind = ast.Con
			typ = obj.Type
		case *TypeName:
			kind = ast.Typ
			typ = obj.Type
		case *Var:
			kind = ast.Var
			typ = obj.Type
		case *Func:
			kind = ast.Fun
			typ = obj.Type
		default:
			unreachable()
		}

		if kind != test.kind {
			t.Errorf("%s: got kind = %q; want %q", test.name, kind, test.kind)
		}

		str := typeString(underlying(typ))
		if str != test.typ {
			t.Errorf("%s: got type = %q; want %q", test.name, typ, test.typ)
		}
	}
}
