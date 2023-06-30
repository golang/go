// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objectpath_test

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"strings"
	"testing"

	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/go/gcexportdata"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/types/objectpath"
)

func TestPaths(t *testing.T) {
	pkgs := map[string]map[string]string{
		"b": {"b.go": `
package b

import "a"

const C = a.Int(0)

func F(a, b, c int, d a.T)

type T struct{ A int; b int; a.T }

func (T) M() *interface{ f() }

type U T

type A = struct{ x int }

var V []*a.T

type M map[struct{x int}]struct{y int}

func unexportedFunc()
type unexportedType struct{}

func (unexportedType) F() {} // not reachable from package's public API (export data)

type S struct{t struct{x int}}
type R []struct{y int}
type Q [2]struct{z int}
`},
		"a": {"a.go": `
package a

type Int int

type T struct{x, y int}

`},
	}
	paths := []pathTest{
		// Good paths
		{"b", "C", "const b.C a.Int", ""},
		{"b", "F", "func b.F(a int, b int, c int, d a.T)", ""},
		{"b", "F.PA0", "var a int", ""},
		{"b", "F.PA1", "var b int", ""},
		{"b", "F.PA2", "var c int", ""},
		{"b", "F.PA3", "var d a.T", ""},
		{"b", "T", "type b.T struct{A int; b int; a.T}", ""},
		{"b", "T.O", "type b.T struct{A int; b int; a.T}", ""},
		{"b", "T.UF0", "field A int", ""},
		{"b", "T.UF1", "field b int", ""},
		{"b", "T.UF2", "field T a.T", ""},
		{"b", "U.UF2", "field T a.T", ""}, // U.U... are aliases for T.U...
		{"b", "A", "type b.A = struct{x int}", ""},
		{"b", "A.F0", "field x int", ""},
		{"b", "V", "var b.V []*a.T", ""},
		{"b", "M", "type b.M map[struct{x int}]struct{y int}", ""},
		{"b", "M.UKF0", "field x int", ""},
		{"b", "M.UEF0", "field y int", ""},
		{"b", "T.M0", "func (b.T).M() *interface{f()}", ""}, // concrete method
		{"b", "T.M0.RA0", "var  *interface{f()}", ""},       // parameter
		{"b", "T.M0.RA0.EM0", "func (interface).f()", ""},   // interface method
		{"b", "unexportedType", "type b.unexportedType struct{}", ""},
		{"b", "unexportedType.M0", "func (b.unexportedType).F()", ""},
		{"b", "S.UF0.F0", "field x int", ""},
		{"b", "R.UEF0", "field y int", ""},
		{"b", "Q.UEF0", "field z int", ""},
		{"a", "T", "type a.T struct{x int; y int}", ""},
		{"a", "T.UF0", "field x int", ""},

		// Bad paths
		{"b", "", "", "empty path"},
		{"b", "missing", "", `package b does not contain "missing"`},
		{"b", "F.U", "", "invalid path: ends with 'U', want [AFMO]"},
		{"b", "F.PA3.O", "", "path denotes type a.T struct{x int; y int}, which belongs to a different package"},
		{"b", "F.PA!", "", `invalid path: bad numeric operand "" for code 'A'`},
		{"b", "F.PA3.UF0", "", "path denotes field x int, which belongs to a different package"},
		{"b", "F.PA3.UF5", "", "field index 5 out of range [0-2)"},
		{"b", "V.EE", "", "invalid path: ends with 'E', want [AFMO]"},
		{"b", "F..O", "", "invalid path: unexpected '.' in type context"},
		{"b", "T.OO", "", "invalid path: code 'O' in object context"},
		{"b", "T.EO", "", "cannot apply 'E' to b.T (got *types.Named, want pointer, slice, array, chan or map)"},
		{"b", "A.O", "", "cannot apply 'O' to struct{x int} (got *types.Struct, want named or type param)"},
		{"b", "A.UF0", "", "cannot apply 'U' to struct{x int} (got *types.Struct, want named)"},
		{"b", "M.UPO", "", "cannot apply 'P' to map[struct{x int}]struct{y int} (got *types.Map, want signature)"},
		{"b", "C.O", "", "path denotes type a.Int int, which belongs to a different package"},
		{"b", "T.M9", "", "method index 9 out of range [0-1)"},
		{"b", "M.UF0", "", "cannot apply 'F' to map[struct{x int}]struct{y int} (got *types.Map, want struct)"},
		{"b", "V.KO", "", "cannot apply 'K' to []*a.T (got *types.Slice, want map)"},
		{"b", "V.A4", "", "cannot apply 'A' to []*a.T (got *types.Slice, want tuple)"},
		{"b", "V.RA0", "", "cannot apply 'R' to []*a.T (got *types.Slice, want signature)"},
		{"b", "F.PA4", "", "tuple index 4 out of range [0-4)"},
		{"b", "F.XO", "", "invalid path: unknown code 'X'"},
	}
	conf := loader.Config{Build: buildutil.FakeContext(pkgs)}
	conf.Import("a")
	conf.Import("b")
	prog, err := conf.Load()
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range paths {
		if err := testPath(prog, test); err != nil {
			t.Error(err)
		}
	}

	// bad objects
	bInfo := prog.Imported["b"]
	for _, test := range []struct {
		obj     types.Object
		wantErr string
	}{
		{types.Universe.Lookup("nil"), "predeclared nil has no path"},
		{types.Universe.Lookup("len"), "predeclared builtin len has no path"},
		{types.Universe.Lookup("int"), "predeclared type int has no path"},
		{bInfo.Implicits[bInfo.Files[0].Imports[0]], "no path for package a"}, // import "a"
		{bInfo.Pkg.Scope().Lookup("unexportedFunc"), "no path for non-exported func b.unexportedFunc()"},
	} {
		path, err := objectpath.For(test.obj)
		if err == nil {
			t.Errorf("Object(%s) = %q, want error", test.obj, path)
			continue
		}
		if err.Error() != test.wantErr {
			t.Errorf("Object(%s) error was %q, want %q", test.obj, err, test.wantErr)
			continue
		}
	}
}

type pathTest struct {
	pkg     string
	path    objectpath.Path
	wantobj string
	wantErr string
}

func testPath(prog *loader.Program, test pathTest) error {
	// We test objectpath by enumerating a set of paths
	// and ensuring that Path(pkg, Object(pkg, path)) == path.
	//
	// It might seem more natural to invert the test:
	// identify a set of objects and for each one,
	// ensure that Object(pkg, Path(pkg, obj)) == obj.
	// However, for most interesting test cases there is no
	// easy way to identify the object short of applying
	// a series of destructuring operations to pkg---which
	// is essentially what objectpath.Object does.
	// (We do a little of that when testing bad paths, below.)
	//
	// The downside is that the test depends on the path encoding.
	// The upside is that the test exercises the encoding.

	pkg := prog.Imported[test.pkg].Pkg
	// check path -> object
	obj, err := objectpath.Object(pkg, test.path)
	if (test.wantErr != "") != (err != nil) {
		return fmt.Errorf("Object(%s, %q) returned error %q, want %q", pkg.Path(), test.path, err, test.wantErr)
	}
	if test.wantErr != "" {
		if got := err.Error(); got != test.wantErr {
			return fmt.Errorf("Object(%s, %q) error was %q, want %q",
				pkg.Path(), test.path, got, test.wantErr)
		}
		return nil
	}
	// Inv: err == nil

	if objString := obj.String(); objString != test.wantobj {
		return fmt.Errorf("Object(%s, %q) = %s, want %s", pkg.Path(), test.path, objString, test.wantobj)
	}
	if obj.Pkg() != pkg {
		return fmt.Errorf("Object(%s, %q) = %v, which belongs to package %s",
			pkg.Path(), test.path, obj, obj.Pkg().Path())
	}

	// check object -> path
	path2, err := objectpath.For(obj)
	if err != nil {
		return fmt.Errorf("For(%v) failed: %v, want %q", obj, err, test.path)
	}
	// We do not require that test.path == path2. Aliases are legal.
	// But we do require that Object(path2) finds the same object.
	obj2, err := objectpath.Object(pkg, path2)
	if err != nil {
		return fmt.Errorf("Object(%s, %q) failed: %v (roundtrip from %q)", pkg.Path(), path2, err, test.path)
	}
	if obj2 != obj {
		return fmt.Errorf("Object(%s, For(obj)) != obj: got %s, obj is %s (path1=%q, path2=%q)", pkg.Path(), obj2, obj, test.path, path2)
	}
	return nil
}

// TestSourceAndExportData uses objectpath to compute a correspondence
// of objects between two versions of the same package, one loaded from
// source, the other from export data.
func TestSourceAndExportData(t *testing.T) {
	const src = `
package p

type I int

func (I) F() *struct{ X, Y int } {
	return nil
}

type Foo interface {
	Method() (string, func(int) struct{ X int })
}

var X chan struct{ Z int }
var Z map[string]struct{ A int }

var V unexported
type unexported struct{}
func (unexported) F() {} // reachable via V

// The name 'unreachable' has special meaning to the test.
type unreachable struct{}
func (unreachable) F() {} // not reachable in export data
`

	// Parse source file and type-check it as a package, "src".
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "src.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	conf := types.Config{Importer: importer.For("source", nil)}
	info := &types.Info{
		Defs: make(map[*ast.Ident]types.Object),
	}
	srcpkg, err := conf.Check("src/p", fset, []*ast.File{f}, info)
	if err != nil {
		t.Fatal(err)
	}

	// Export binary export data then reload it as a new package, "bin".
	var buf bytes.Buffer
	if err := gcexportdata.Write(&buf, fset, srcpkg); err != nil {
		t.Fatal(err)
	}

	imports := make(map[string]*types.Package)
	binpkg, err := gcexportdata.Read(&buf, fset, imports, "bin/p")
	if err != nil {
		t.Fatal(err)
	}

	// Now find the correspondences between them.
	for _, srcobj := range info.Defs {
		if srcobj == nil {
			continue // e.g. package declaration
		}
		if _, ok := srcobj.(*types.PkgName); ok {
			continue // PkgName has no objectpath
		}

		path, err := objectpath.For(srcobj)
		if err != nil {
			t.Errorf("For(%v): %v", srcobj, err)
			continue
		}

		// Do we expect to find this object in the export data?
		reachable := !strings.Contains(string(path), "unreachable")

		binobj, err := objectpath.Object(binpkg, path)
		if err != nil {
			if reachable {
				t.Errorf("Object(%s, %q): %v", binpkg.Path(), path, err)
			}
			continue
		}
		if !reachable {
			t.Errorf("Object(%s, %q) = %v (unexpectedly reachable)", binpkg.Path(), path, binobj)
		}

		// Check the object strings match.
		// (We can't check that types are identical because the
		// objects belong to different type-checker realms.)
		srcstr := objectString(srcobj)
		binstr := objectString(binobj)
		if srcstr != binstr {
			t.Errorf("ObjectStrings do not match: Object(For(%q)) = %s, want %s",
				path, srcstr, binstr)
			continue
		}
	}
}

func objectString(obj types.Object) string {
	s := types.ObjectString(obj, (*types.Package).Name)

	// The printing of interface methods changed in go1.11.
	// This work-around makes the specific test pass with earlier versions.
	s = strings.Replace(s, "func (interface).Method", "func (p.Foo).Method", -1)

	return s
}

// TestOrdering uses objectpath over two Named types with the same method
// names but in a different source order and checks that objectpath is the
// same for methods with the same name.
func TestOrdering(t *testing.T) {
	pkgs := map[string]map[string]string{
		"p": {"p.go": `
package p

type T struct{ A int }

func (T) M() { }
func (T) N() { }
func (T) X() { }
func (T) Y() { }
`},
		"q": {"q.go": `
package q

type T struct{ A int }

func (T) N() { }
func (T) M() { }
func (T) Y() { }
func (T) X() { }
`}}
	conf := loader.Config{Build: buildutil.FakeContext(pkgs)}
	conf.Import("p")
	conf.Import("q")
	prog, err := conf.Load()
	if err != nil {
		t.Fatal(err)
	}
	p := prog.Imported["p"].Pkg
	q := prog.Imported["q"].Pkg

	// From here, the objectpaths generated for p and q should be the
	// same. If they are not, then we are generating an ordering that is
	// dependent on the declaration of the types within the file.
	for _, test := range []struct {
		path objectpath.Path
	}{
		{"T.M0"},
		{"T.M1"},
		{"T.M2"},
		{"T.M3"},
	} {
		pobj, err := objectpath.Object(p, test.path)
		if err != nil {
			t.Errorf("Object(%s) failed in a1: %v", test.path, err)
			continue
		}
		qobj, err := objectpath.Object(q, test.path)
		if err != nil {
			t.Errorf("Object(%s) failed in a2: %v", test.path, err)
			continue
		}
		if pobj.Name() != pobj.Name() {
			t.Errorf("Objects(%s) not equal, got a1 = %v, a2 = %v", test.path, pobj.Name(), qobj.Name())
		}
	}
}
