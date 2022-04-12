// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts_test

import (
	"encoding/gob"
	"fmt"
	"go/token"
	"go/types"
	"os"
	"reflect"
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/internal/facts"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/testenv"
	"golang.org/x/tools/internal/typeparams"
)

type myFact struct {
	S string
}

func (f *myFact) String() string { return fmt.Sprintf("myFact(%s)", f.S) }
func (f *myFact) AFact()         {}

func init() {
	gob.Register(new(myFact))
}

func TestEncodeDecode(t *testing.T) {
	tests := []struct {
		name       string
		typeparams bool // requires typeparams to be enabled
		files      map[string]string
		plookups   []pkgLookups // see testEncodeDecode for details
	}{
		{
			name: "loading-order",
			// c -> b -> a, a2
			// c does not directly depend on a, but it indirectly uses a.T.
			//
			// Package a2 is never loaded directly so it is incomplete.
			//
			// We use only types in this example because we rely on
			// types.Eval to resolve the lookup expressions, and it only
			// works for types. This is a definite gap in the typechecker API.
			files: map[string]string{
				"a/a.go":  `package a; type A int; type T int`,
				"a2/a.go": `package a2; type A2 int; type Unneeded int`,
				"b/b.go":  `package b; import ("a"; "a2"); type B chan a2.A2; type F func() a.T`,
				"c/c.go":  `package c; import "b"; type C []b.B`,
			},
			// In the following table, we analyze packages (a, b, c) in order,
			// look up various objects accessible within each package,
			// and see if they have a fact.  The "analysis" exports a fact
			// for every object at package level.
			//
			// Note: Loop iterations are not independent test cases;
			// order matters, as we populate factmap.
			plookups: []pkgLookups{
				{"a", []lookup{
					{"A", "myFact(a.A)"},
				}},
				{"b", []lookup{
					{"a.A", "myFact(a.A)"},
					{"a.T", "myFact(a.T)"},
					{"B", "myFact(b.B)"},
					{"F", "myFact(b.F)"},
					{"F(nil)()", "myFact(a.T)"}, // (result type of b.F)
				}},
				{"c", []lookup{
					{"b.B", "myFact(b.B)"},
					{"b.F", "myFact(b.F)"},
					//{"b.F(nil)()", "myFact(a.T)"}, // no fact; TODO(adonovan): investigate
					{"C", "myFact(c.C)"},
					{"C{}[0]", "myFact(b.B)"},
					{"<-(C{}[0])", "no fact"}, // object but no fact (we never "analyze" a2)
				}},
			},
		},
		{
			name: "globals",
			files: map[string]string{
				"a/a.go": `package a;
				type T1 int
				type T2 int
				type T3 int
				type T4 int
				type T5 int
				type K int; type V string
				`,
				"b/b.go": `package b
				import "a"
				var (
					G1 []a.T1
					G2 [7]a.T2
					G3 chan a.T3
					G4 *a.T4
					G5 struct{ F a.T5 }
					G6 map[a.K]a.V
				)
				`,
				"c/c.go": `package c; import "b";
				var (
					v1 = b.G1
					v2 = b.G2
					v3 = b.G3
					v4 = b.G4
					v5 = b.G5
					v6 = b.G6
				)
				`,
			},
			plookups: []pkgLookups{
				{"a", []lookup{}},
				{"b", []lookup{}},
				{"c", []lookup{
					{"v1[0]", "myFact(a.T1)"},
					{"v2[0]", "myFact(a.T2)"},
					{"<-v3", "myFact(a.T3)"},
					{"*v4", "myFact(a.T4)"},
					{"v5.F", "myFact(a.T5)"},
					{"v6[0]", "myFact(a.V)"},
				}},
			},
		},
		{
			name:       "typeparams",
			typeparams: true,
			files: map[string]string{
				"a/a.go": `package a
				  type T1 int
				  type T2 int
				  type T3 interface{Foo()}
				  type T4 int
				  type T5 int
				  type T6 interface{Foo()}
				`,
				"b/b.go": `package b
				  import "a"
				  type N1[T a.T1|int8] func() T
				  type N2[T any] struct{ F T }
				  type N3[T a.T3] func() T
				  type N4[T a.T4|int8] func() T
				  type N5[T interface{Bar() a.T5} ] func() T
		
				  type t5 struct{}; func (t5) Bar() a.T5
		
				  var G1 N1[a.T1]
				  var G2 func() N2[a.T2]
				  var G3 N3[a.T3]
				  var G4 N4[a.T4]
				  var G5 N5[t5]
		
				  func F6[T a.T6]() T { var x T; return x }
				  `,
				"c/c.go": `package c; import "b";
				  var (
					  v1 = b.G1
					  v2 = b.G2
					  v3 = b.G3
					  v4 = b.G4
					  v5 = b.G5
					  v6 = b.F6[t6]
				  )
		
				  type t6 struct{}; func (t6) Foo() {}
				`,
			},
			plookups: []pkgLookups{
				{"a", []lookup{}},
				{"b", []lookup{}},
				{"c", []lookup{
					{"v1", "myFact(b.N1)"},
					{"v1()", "myFact(a.T1)"},
					{"v2()", "myFact(b.N2)"},
					{"v2().F", "myFact(a.T2)"},
					{"v3", "myFact(b.N3)"},
					{"v4", "myFact(b.N4)"},
					{"v4()", "myFact(a.T4)"},
					{"v5", "myFact(b.N5)"},
					{"v5()", "myFact(b.t5)"},
					{"v6()", "myFact(c.t6)"},
				}},
			},
		},
	}

	for i := range tests {
		test := tests[i]
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if test.typeparams && !typeparams.Enabled {
				t.Skip("type parameters are not enabled")
			}
			testEncodeDecode(t, test.files, test.plookups)
		})
	}
}

type lookup struct {
	objexpr string
	want    string
}

type pkgLookups struct {
	path    string
	lookups []lookup
}

// testEncodeDecode tests fact encoding and decoding and simulates how package facts
// are passed during analysis. It operates on a group of Go file contents. Then
// for each <package, []lookup> in tests it does the following:
//  1. loads and type checks the package,
//  2. calls facts.Decode to loads the facts exported by its imports,
//  3. exports a myFact Fact for all of package level objects,
//  4. For each lookup for the current package:
//     4.a) lookup the types.Object for an Go source expression in the curent package
//     (or confirms one is not expected want=="no object"),
//     4.b) finds a Fact for the object (or confirms one is not expected want=="no fact"),
//     4.c) compares the content of the Fact to want.
//  5. encodes the Facts of the package.
//
// Note: tests are not independent test cases; order matters (as does a package being
// skipped). It changes what Facts can be imported.
//
// Failures are reported on t.
func testEncodeDecode(t *testing.T, files map[string]string, tests []pkgLookups) {
	dir, cleanup, err := analysistest.WriteFiles(files)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()

	// factmap represents the passing of encoded facts from one
	// package to another. In practice one would use the file system.
	factmap := make(map[string][]byte)
	read := func(path string) ([]byte, error) { return factmap[path], nil }

	// Analyze packages in order, look up various objects accessible within
	// each package, and see if they have a fact.  The "analysis" exports a
	// fact for every object at package level.
	//
	// Note: Loop iterations are not independent test cases;
	// order matters, as we populate factmap.
	for _, test := range tests {
		// load package
		pkg, err := load(t, dir, test.path)
		if err != nil {
			t.Fatal(err)
		}

		// decode
		facts, err := facts.Decode(pkg, read)
		if err != nil {
			t.Fatalf("Decode failed: %v", err)
		}
		t.Logf("decode %s facts = %v", pkg.Path(), facts) // show all facts

		// export
		// (one fact for each package-level object)
		for _, name := range pkg.Scope().Names() {
			obj := pkg.Scope().Lookup(name)
			fact := &myFact{obj.Pkg().Name() + "." + obj.Name()}
			facts.ExportObjectFact(obj, fact)
		}
		t.Logf("exported %s facts = %v", pkg.Path(), facts) // show all facts

		// import
		// (after export, because an analyzer may import its own facts)
		for _, lookup := range test.lookups {
			fact := new(myFact)
			var got string
			if obj := find(pkg, lookup.objexpr); obj == nil {
				got = "no object"
			} else if facts.ImportObjectFact(obj, fact) {
				got = fact.String()
			} else {
				got = "no fact"
			}
			if got != lookup.want {
				t.Errorf("in %s, ImportObjectFact(%s, %T) = %s, want %s",
					pkg.Path(), lookup.objexpr, fact, got, lookup.want)
			}
		}

		// encode
		factmap[pkg.Path()] = facts.Encode()
	}
}

func find(p *types.Package, expr string) types.Object {
	// types.Eval only allows us to compute a TypeName object for an expression.
	// TODO(adonovan): support other expressions that denote an object:
	// - an identifier (or qualified ident) for a func, const, or var
	// - new(T).f for a field or method
	// I've added CheckExpr in https://go-review.googlesource.com/c/go/+/144677.
	// If that becomes available, use it.

	// Choose an arbitrary position within the (single-file) package
	// so that we are within the scope of its import declarations.
	somepos := p.Scope().Lookup(p.Scope().Names()[0]).Pos()
	tv, err := types.Eval(token.NewFileSet(), p, somepos, expr)
	if err != nil {
		return nil
	}
	if n, ok := tv.Type.(*types.Named); ok {
		return n.Obj()
	}
	return nil
}

func load(t *testing.T, dir string, path string) (*types.Package, error) {
	cfg := &packages.Config{
		Mode: packages.LoadSyntax,
		Dir:  dir,
		Env:  append(os.Environ(), "GOPATH="+dir, "GO111MODULE=off", "GOPROXY=off"),
	}
	testenv.NeedsGoPackagesEnv(t, cfg.Env)
	pkgs, err := packages.Load(cfg, path)
	if err != nil {
		return nil, err
	}
	if packages.PrintErrors(pkgs) > 0 {
		return nil, fmt.Errorf("packages had errors")
	}
	if len(pkgs) == 0 {
		return nil, fmt.Errorf("no package matched %s", path)
	}
	return pkgs[0].Types, nil
}

type otherFact struct {
	S string
}

func (f *otherFact) String() string { return fmt.Sprintf("otherFact(%s)", f.S) }
func (f *otherFact) AFact()         {}

func TestFactFilter(t *testing.T) {
	files := map[string]string{
		"a/a.go": `package a; type A int`,
	}
	dir, cleanup, err := analysistest.WriteFiles(files)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()

	pkg, err := load(t, dir, "a")
	if err != nil {
		t.Fatal(err)
	}

	obj := pkg.Scope().Lookup("A")
	s, err := facts.Decode(pkg, func(string) ([]byte, error) { return nil, nil })
	if err != nil {
		t.Fatal(err)
	}
	s.ExportObjectFact(obj, &myFact{"good object fact"})
	s.ExportPackageFact(&myFact{"good package fact"})
	s.ExportObjectFact(obj, &otherFact{"bad object fact"})
	s.ExportPackageFact(&otherFact{"bad package fact"})

	filter := map[reflect.Type]bool{
		reflect.TypeOf(&myFact{}): true,
	}

	pkgFacts := s.AllPackageFacts(filter)
	wantPkgFacts := `[{package a ("a") myFact(good package fact)}]`
	if got := fmt.Sprintf("%v", pkgFacts); got != wantPkgFacts {
		t.Errorf("AllPackageFacts: got %v, want %v", got, wantPkgFacts)
	}

	objFacts := s.AllObjectFacts(filter)
	wantObjFacts := "[{type a.A int myFact(good object fact)}]"
	if got := fmt.Sprintf("%v", objFacts); got != wantObjFacts {
		t.Errorf("AllObjectFacts: got %v, want %v", got, wantObjFacts)
	}
}
