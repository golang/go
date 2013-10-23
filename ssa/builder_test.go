// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"go/parser"
	"reflect"
	"sort"
	"strings"
	"testing"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/importer"
	"code.google.com/p/go.tools/ssa"
)

func isEmpty(f *ssa.Function) bool { return f.Blocks == nil }

// Tests that programs partially loaded from gc object files contain
// functions with no code for the external portions, but are otherwise ok.
func TestExternalPackages(t *testing.T) {
	test := `
package main

import (
	"bytes"
	"io"
	"testing"
)

func main() {
        var t testing.T
	t.Parallel()    // static call to external declared method
        t.Fail()        // static call to promoted external declared method
        testing.Short() // static call to external package-level function

        var w io.Writer = new(bytes.Buffer)
        w.Write(nil)    // interface invoke of external declared method
}
`
	imp := importer.New(new(importer.Config)) // no go/build.Context; uses GC importer

	f, err := parser.ParseFile(imp.Fset, "<input>", test, 0)
	if err != nil {
		t.Error(err)
		return
	}

	mainInfo := imp.CreatePackage("main", f)

	prog := ssa.NewProgram(imp.Fset, ssa.SanityCheckFunctions)
	if err := prog.CreatePackages(imp); err != nil {
		t.Error(err)
		return
	}
	mainPkg := prog.Package(mainInfo.Pkg)
	mainPkg.Build()

	// Only the main package and its immediate dependencies are loaded.
	deps := []string{"bytes", "io", "testing"}
	all := prog.AllPackages()
	if len(all) != 1+len(deps) {
		t.Errorf("unexpected set of loaded packages: %q", all)
	}
	for _, path := range deps {
		pkg := prog.ImportedPackage(path)
		if pkg == nil {
			t.Errorf("package not loaded: %q", path)
			continue
		}

		// External packages should have no function bodies (except for wrappers).
		isExt := pkg != mainPkg

		// init()
		if isExt && !isEmpty(pkg.Func("init")) {
			t.Errorf("external package %s has non-empty init", pkg)
		} else if !isExt && isEmpty(pkg.Func("init")) {
			t.Errorf("main package %s has empty init", pkg)
		}

		for _, mem := range pkg.Members {
			switch mem := mem.(type) {
			case *ssa.Function:
				// Functions at package level.
				if isExt && !isEmpty(mem) {
					t.Errorf("external function %s is non-empty", mem)
				} else if !isExt && isEmpty(mem) {
					t.Errorf("function %s is empty", mem)
				}

			case *ssa.Type:
				// Methods of named types T.
				// (In this test, all exported methods belong to *T not T.)
				if !isExt {
					t.Fatalf("unexpected name type in main package: %s", mem)
				}
				mset := types.NewPointer(mem.Type()).MethodSet()
				for i, n := 0, mset.Len(); i < n; i++ {
					m := prog.Method(mset.At(i))
					// For external types, only synthetic wrappers have code.
					expExt := !strings.Contains(m.Synthetic, "wrapper")
					if expExt && !isEmpty(m) {
						t.Errorf("external method %s is non-empty: %s",
							m, m.Synthetic)
					} else if !expExt && isEmpty(m) {
						t.Errorf("method function %s is empty: %s",
							m, m.Synthetic)
					}
				}
			}
		}
	}

	expectedCallee := []string{
		"(*testing.T).Parallel",
		"(*testing.common).Fail",
		"testing.Short",
		"N/A",
	}
	callNum := 0
	for _, b := range mainPkg.Func("main").Blocks {
		for _, instr := range b.Instrs {
			switch instr := instr.(type) {
			case ssa.CallInstruction:
				call := instr.Common()
				if want := expectedCallee[callNum]; want != "N/A" {
					got := call.StaticCallee().String()
					if want != got {
						t.Errorf("call #%d from main.main: got callee %s, want %s",
							callNum, got, want)
					}
				}
				callNum++
			}
		}
	}
	if callNum != 4 {
		t.Errorf("in main.main: got %d calls, want %d", callNum, 4)
	}
}

// TestMethodSets tests that Package.TypesWithMethodSets includes all necessary types.
func TestTypesWithMethodSets(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		// An exported package-level type is needed.
		{`package p; type T struct{}; func (T) f() {}`,
			[]string{"*p.T", "p.T"},
		},
		// An unexported package-level type is not needed.
		{`package p; type t struct{}; func (t) f() {}`,
			nil,
		},
		// Subcomponents of type of exported package-level var are needed.
		{`package p; import "bytes"; var V struct {*bytes.Buffer}`,
			[]string{"struct{*bytes.Buffer}"},
		},
		// Subcomponents of type of unexported package-level var are not needed.
		{`package p; import "bytes"; var v struct {*bytes.Buffer}`,
			nil,
		},
		// Subcomponents of type of exported package-level function are needed.
		{`package p; import "bytes"; func F(struct {*bytes.Buffer}) {}`,
			[]string{"struct{*bytes.Buffer}"},
		},
		// Subcomponents of type of unexported package-level function are not needed.
		{`package p; import "bytes"; func f(struct {*bytes.Buffer}) {}`,
			nil,
		},
		// Subcomponents of type of exported method are needed.
		{`package p; import "bytes"; type x struct{}; func (x) G(struct {*bytes.Buffer}) {}`,
			[]string{"*p.x", "p.x", "struct{*bytes.Buffer}"},
		},
		// Subcomponents of type of unexported method are not needed.
		{`package p; import "bytes"; type X struct{}; func (X) G(struct {*bytes.Buffer}) {}`,
			[]string{"*p.X", "p.X", "struct{*bytes.Buffer}"},
		},
		// Local types aren't needed.
		{`package p; import "bytes"; func f() { type T struct {*bytes.Buffer}; var t T; _ = t }`,
			nil,
		},
		// ...unless used by MakeInterface.
		{`package p; import "bytes"; func f() { type T struct {*bytes.Buffer}; _ = interface{}(T{}) }`,
			[]string{"*p.T", "p.T"},
		},
		// Types used as operand of MakeInterface are needed.
		{`package p; import "bytes"; func f() { _ = interface{}(struct{*bytes.Buffer}{}) }`,
			[]string{"struct{*bytes.Buffer}"},
		},
		// MakeInterface is optimized away when storing to a blank.
		{`package p; import "bytes"; var _ interface{} = struct{*bytes.Buffer}{}`,
			nil,
		},
	}
	for i, test := range tests {
		imp := importer.New(new(importer.Config)) // no go/build.Context; uses GC importer

		f, err := parser.ParseFile(imp.Fset, "<input>", test.input, 0)
		if err != nil {
			t.Errorf("test %d: %s", i, err)
			continue
		}

		mainInfo := imp.CreatePackage("p", f)
		prog := ssa.NewProgram(imp.Fset, ssa.SanityCheckFunctions)
		if err := prog.CreatePackages(imp); err != nil {
			t.Errorf("test %d: %s", i, err)
			continue
		}
		mainPkg := prog.Package(mainInfo.Pkg)
		prog.BuildAll()

		var typstrs []string
		for _, T := range mainPkg.TypesWithMethodSets() {
			typstrs = append(typstrs, T.String())
		}
		sort.Strings(typstrs)

		if !reflect.DeepEqual(typstrs, test.want) {
			t.Errorf("test %d: got %q, want %q", i, typstrs, test.want)
		}
	}
}
