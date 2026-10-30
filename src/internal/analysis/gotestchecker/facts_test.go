// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gotestchecker_test

import (
	"encoding/gob"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"reflect"
	"slices"
	"strings"
	"testing"

	"internal/analysis/gotestchecker"
)

type testFact struct{ N int }

func (*testFact) AFact() {}

func (f *testFact) String() string { return fmt.Sprintf("testFact(%d)", f.N) }

// An ifaceFact is a fact with an interface-typed field.
type ifaceFact struct{ V any }

func (*ifaceFact) AFact() {}

func (f *ifaceFact) String() string { return fmt.Sprintf("ifaceFact(%v)", f.V) }

type ifaceValue struct{ N int }

// TestFactsTwoHops checks the encoding of fact sets (see facts.go)
// across two hops of the import graph, c -> b -> a: facts exported by
// a, decoded by b, re-exported by b, and decoded by c; and the pruning
// of facts that are not re-exported (package facts, and facts about
// package-level non-type objects, of dependencies).
//
// TODO(adonovan): make a more systematic coverage of fact behavior
// through the go script test mechanism, when we can.
func TestFactsTwoHops(t *testing.T) {
	gotestchecker.RegisterFactType(new(testFact))
	gotestchecker.RegisterFactType(new(ifaceFact))
	gob.Register(ifaceValue{})

	// Type-check the packages from source.
	fset := token.NewFileSet()
	pkgs := make(map[string]*types.Package)
	check := func(path, src string) *types.Package {
		t.Helper()
		f, err := parser.ParseFile(fset, path+".go", src, 0)
		if err != nil {
			t.Fatal(err)
		}
		conf := types.Config{Importer: importerFunc(func(path string) (*types.Package, error) {
			return pkgs[path], nil
		})}
		pkg, err := conf.Check(path, fset, []*ast.File{f}, nil)
		if err != nil {
			t.Fatal(err)
		}
		pkgs[path] = pkg
		return pkg
	}
	a := check("a", `package a; type T struct{ F int }; func (T) M() {}; func G() {}`)
	b := check("b", `package b; import "a"; type U struct{ a.T }; func H() a.T { return a.T{} }`)
	c := check("c", `package c; import "b"; var _ b.U`)

	T := a.Scope().Lookup("T").(*types.TypeName)
	F := T.Type().Underlying().(*types.Struct).Field(0)
	M, _, _ := types.LookupFieldOrMethod(T.Type(), false, a, "M")
	G := a.Scope().Lookup("G")
	H := b.Scope().Lookup("H")
	names := map[types.Object]string{T: "a.T", F: "a.T.F", M: "a.T.M", G: "a.G", H: "b.H"}

	// decode returns the fact set of pkg, given the encoded fact
	// sets of its imports.
	decode := func(pkg *types.Package, imports map[string][]byte) *gotestchecker.FactSet {
		t.Helper()
		getPackage := func(path string) *types.Package { return pkgs[path] }
		set, err := gotestchecker.DecodeFacts(pkg, getPackage, func(path string) ([]byte, error) {
			return imports[path], nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return set
	}

	// encode encodes the fact set, and checks the numbers of facts
	// about the package itself (own) and its dependencies (deps).
	encode := func(set *gotestchecker.FactSet, wantOwn, wantDeps int) []byte {
		t.Helper()
		data, own, deps := set.Encode()
		if own != wantOwn || deps != wantDeps {
			t.Errorf("Encode: own=%d deps=%d, want %d, %d", own, deps, wantOwn, wantDeps)
		}
		return data
	}

	// want checks that the fact set contains exactly the facts
	// described by want, one per line, in any order.
	want := func(set *gotestchecker.FactSet, want string) {
		t.Helper()
		allTypes := map[reflect.Type]bool{
			reflect.TypeFor[*testFact]():  true,
			reflect.TypeFor[*ifaceFact](): true,
		}
		var got []string
		for _, f := range set.AllObjectFacts(allTypes) {
			got = append(got, fmt.Sprintf("%s: %v", names[f.Object], f.Fact))
		}
		for _, f := range set.AllPackageFacts(allTypes) {
			got = append(got, fmt.Sprintf("package %s: %v", f.Package.Path(), f.Fact))
		}
		slices.Sort(got)
		var wantLines []string
		for line := range strings.Lines(want) {
			if line := strings.TrimSpace(line); line != "" {
				wantLines = append(wantLines, line)
			}
		}
		slices.Sort(wantLines)
		if !slices.Equal(got, wantLines) {
			t.Errorf("got facts:\n%s\nwant:\n%s", strings.Join(got, "\n"), strings.Join(wantLines, "\n"))
		}
	}

	// Package a exports facts about its objects and itself.
	setA := decode(a, nil)
	setA.ExportObjectFact(T, &ifaceFact{V: ifaceValue{N: 1}})
	setA.ExportObjectFact(F, &testFact{2})
	setA.ExportObjectFact(M, &testFact{3})
	setA.ExportObjectFact(G, &testFact{4})
	setA.ExportPackageFact(&testFact{5})
	dataA := encode(setA, 5, 0)

	// Package b sees all of a's facts, and exports one of its own.
	setB := decode(b, map[string][]byte{"a": dataA})
	setB.ExportObjectFact(H, &testFact{6})
	dataB := encode(setB, 1, 3)
	want(setB, `
		a.G: testFact(4)
		a.T: ifaceFact({1})
		a.T.F: testFact(2)
		a.T.M: testFact(3)
		b.H: testFact(6)
		package a: testFact(5)`)

	// Package c sees a's facts about types and their members
	// (re-exported by b), and b's own, but not a's facts about
	// other package-level objects, nor its package facts.
	setC := decode(c, map[string][]byte{"b": dataB})
	want(setC, `
		a.T: ifaceFact({1})
		a.T.F: testFact(2)
		a.T.M: testFact(3)
		b.H: testFact(6)`)
}

type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }
