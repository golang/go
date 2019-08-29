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
)

type myFact struct {
	S string
}

func (f *myFact) String() string { return fmt.Sprintf("myFact(%s)", f.S) }
func (f *myFact) AFact()         {}

func TestEncodeDecode(t *testing.T) {
	gob.Register(new(myFact))

	// c -> b -> a, a2
	// c does not directly depend on a, but it indirectly uses a.T.
	//
	// Package a2 is never loaded directly so it is incomplete.
	//
	// We use only types in this example because we rely on
	// types.Eval to resolve the lookup expressions, and it only
	// works for types. This is a definite gap in the typechecker API.
	files := map[string]string{
		"a/a.go":  `package a; type A int; type T int`,
		"a2/a.go": `package a2; type A2 int; type Unneeded int`,
		"b/b.go":  `package b; import ("a"; "a2"); type B chan a2.A2; type F func() a.T`,
		"c/c.go":  `package c; import "b"; type C []b.B`,
	}
	dir, cleanup, err := analysistest.WriteFiles(files)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()

	// factmap represents the passing of encoded facts from one
	// package to another. In practice one would use the file system.
	factmap := make(map[string][]byte)
	read := func(path string) ([]byte, error) { return factmap[path], nil }

	// In the following table, we analyze packages (a, b, c) in order,
	// look up various objects accessible within each package,
	// and see if they have a fact.  The "analysis" exports a fact
	// for every object at package level.
	//
	// Note: Loop iterations are not independent test cases;
	// order matters, as we populate factmap.
	type lookups []struct {
		objexpr string
		want    string
	}
	for _, test := range []struct {
		path    string
		lookups lookups
	}{
		{"a", lookups{
			{"A", "myFact(a.A)"},
		}},
		{"b", lookups{
			{"a.A", "myFact(a.A)"},
			{"a.T", "myFact(a.T)"},
			{"B", "myFact(b.B)"},
			{"F", "myFact(b.F)"},
			{"F(nil)()", "myFact(a.T)"}, // (result type of b.F)
		}},
		{"c", lookups{
			{"b.B", "myFact(b.B)"},
			{"b.F", "myFact(b.F)"},
			//{"b.F(nil)()", "myFact(a.T)"}, // no fact; TODO(adonovan): investigate
			{"C", "myFact(c.C)"},
			{"C{}[0]", "myFact(b.B)"},
			{"<-(C{}[0])", "no fact"}, // object but no fact (we never "analyze" a2)
		}},
	} {
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
		if true {
			t.Logf("decode %s facts = %v", pkg.Path(), facts) // show all facts
		}

		// export
		// (one fact for each package-level object)
		scope := pkg.Scope()
		for _, name := range scope.Names() {
			obj := scope.Lookup(name)
			fact := &myFact{obj.Pkg().Name() + "." + obj.Name()}
			facts.ExportObjectFact(obj, fact)
		}

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
