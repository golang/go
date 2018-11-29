// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package srcimporter

import (
	"go/build"
	"go/token"
	"go/types"
	"internal/testenv"
	"io/ioutil"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

const maxTime = 2 * time.Second

var importer = New(&build.Default, token.NewFileSet(), make(map[string]*types.Package))

func doImport(t *testing.T, path, srcDir string) {
	t0 := time.Now()
	if _, err := importer.ImportFrom(path, srcDir, 0); err != nil {
		// don't report an error if there's no buildable Go files
		if _, nogo := err.(*build.NoGoError); !nogo {
			t.Errorf("import %q failed (%v)", path, err)
		}
		return
	}
	t.Logf("import %q: %v", path, time.Since(t0))
}

// walkDir imports the all the packages with the given path
// prefix recursively. It returns the number of packages
// imported and whether importing was aborted because time
// has passed endTime.
func walkDir(t *testing.T, path string, endTime time.Time) (int, bool) {
	if time.Now().After(endTime) {
		t.Log("testing time used up")
		return 0, true
	}

	// ignore fake packages and testdata directories
	if path == "builtin" || path == "unsafe" || strings.HasSuffix(path, "testdata") {
		return 0, false
	}

	list, err := ioutil.ReadDir(filepath.Join(runtime.GOROOT(), "src", path))
	if err != nil {
		t.Fatalf("walkDir %s failed (%v)", path, err)
	}

	nimports := 0
	hasGoFiles := false
	for _, f := range list {
		if f.IsDir() {
			n, abort := walkDir(t, filepath.Join(path, f.Name()), endTime)
			nimports += n
			if abort {
				return nimports, true
			}
		} else if strings.HasSuffix(f.Name(), ".go") {
			hasGoFiles = true
		}
	}

	if hasGoFiles {
		doImport(t, path, "")
		nimports++
	}

	return nimports, false
}

func TestImportStdLib(t *testing.T) {
	if !testenv.HasSrc() {
		t.Skip("no source code available")
	}

	dt := maxTime
	if testing.Short() && testenv.Builder() == "" {
		dt = 500 * time.Millisecond
	}
	nimports, _ := walkDir(t, "", time.Now().Add(dt)) // installed packages
	t.Logf("tested %d imports", nimports)
}

var importedObjectTests = []struct {
	name string
	want string
}{
	{"flag.Bool", "func Bool(name string, value bool, usage string) *bool"},
	{"io.Reader", "type Reader interface{Read(p []byte) (n int, err error)}"},
	{"io.ReadWriter", "type ReadWriter interface{Reader; Writer}"}, // go/types.gcCompatibilityMode is off => interface not flattened
	{"math.Pi", "const Pi untyped float"},
	{"math.Sin", "func Sin(x float64) float64"},
	{"math/big.Int", "type Int struct{neg bool; abs nat}"},
	{"internal/x/text/unicode/norm.MaxSegmentSize", "const MaxSegmentSize untyped int"},
}

func TestImportedTypes(t *testing.T) {
	if !testenv.HasSrc() {
		t.Skip("no source code available")
	}

	for _, test := range importedObjectTests {
		s := strings.Split(test.name, ".")
		if len(s) != 2 {
			t.Fatal("invalid test data format")
		}
		importPath := s[0]
		objName := s[1]

		pkg, err := importer.ImportFrom(importPath, ".", 0)
		if err != nil {
			t.Error(err)
			continue
		}

		obj := pkg.Scope().Lookup(objName)
		if obj == nil {
			t.Errorf("%s: object not found", test.name)
			continue
		}

		got := types.ObjectString(obj, types.RelativeTo(pkg))
		if got != test.want {
			t.Errorf("%s: got %q; want %q", test.name, got, test.want)
		}

		if named, _ := obj.Type().(*types.Named); named != nil {
			verifyInterfaceMethodRecvs(t, named, 0)
		}
	}
}

// verifyInterfaceMethodRecvs verifies that method receiver types
// are named if the methods belong to a named interface type.
func verifyInterfaceMethodRecvs(t *testing.T, named *types.Named, level int) {
	// avoid endless recursion in case of an embedding bug that lead to a cycle
	if level > 10 {
		t.Errorf("%s: embeds itself", named)
		return
	}

	iface, _ := named.Underlying().(*types.Interface)
	if iface == nil {
		return // not an interface
	}

	// check explicitly declared methods
	for i := 0; i < iface.NumExplicitMethods(); i++ {
		m := iface.ExplicitMethod(i)
		recv := m.Type().(*types.Signature).Recv()
		if recv == nil {
			t.Errorf("%s: missing receiver type", m)
			continue
		}
		if recv.Type() != named {
			t.Errorf("%s: got recv type %s; want %s", m, recv.Type(), named)
		}
	}

	// check embedded interfaces (they are named, too)
	for i := 0; i < iface.NumEmbeddeds(); i++ {
		// embedding of interfaces cannot have cycles; recursion will terminate
		verifyInterfaceMethodRecvs(t, iface.Embedded(i), level+1)
	}
}

func TestReimport(t *testing.T) {
	if !testenv.HasSrc() {
		t.Skip("no source code available")
	}

	// Reimporting a partially imported (incomplete) package is not supported (see issue #19337).
	// Make sure we recognize the situation and report an error.

	mathPkg := types.NewPackage("math", "math") // incomplete package
	importer := New(&build.Default, token.NewFileSet(), map[string]*types.Package{mathPkg.Path(): mathPkg})
	_, err := importer.ImportFrom("math", ".", 0)
	if err == nil || !strings.HasPrefix(err.Error(), "reimport") {
		t.Errorf("got %v; want reimport error", err)
	}
}

func TestIssue20855(t *testing.T) {
	if !testenv.HasSrc() {
		t.Skip("no source code available")
	}

	pkg, err := importer.ImportFrom("go/internal/srcimporter/testdata/issue20855", ".", 0)
	if err == nil || !strings.Contains(err.Error(), "missing function body") {
		t.Fatalf("got unexpected or no error: %v", err)
	}
	if pkg == nil {
		t.Error("got no package despite no hard errors")
	}
}

func testImportPath(t *testing.T, pkgPath string) {
	if !testenv.HasSrc() {
		t.Skip("no source code available")
	}

	pkgName := path.Base(pkgPath)

	pkg, err := importer.Import(pkgPath)
	if err != nil {
		t.Fatal(err)
	}

	if pkg.Name() != pkgName {
		t.Errorf("got %q; want %q", pkg.Name(), pkgName)
	}

	if pkg.Path() != pkgPath {
		t.Errorf("got %q; want %q", pkg.Path(), pkgPath)
	}
}

// TestIssue23092 tests relative imports.
func TestIssue23092(t *testing.T) {
	testImportPath(t, "./testdata/issue23092")
}

// TestIssue24392 tests imports against a path containing 'testdata'.
func TestIssue24392(t *testing.T) {
	testImportPath(t, "go/internal/srcimporter/testdata/issue24392")
}
