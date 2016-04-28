// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

package ssa_test

import (
	"bytes"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"reflect"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

func isEmpty(f *ssa.Function) bool { return f.Blocks == nil }

// Tests that programs partially loaded from gc object files contain
// functions with no code for the external portions, but are otherwise ok.
func TestBuildPackage(t *testing.T) {
	input := `
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

	// Parse the file.
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", input, 0)
	if err != nil {
		t.Error(err)
		return
	}

	// Build an SSA program from the parsed file.
	// Load its dependencies from gc binary export data.
	mainPkg, _, err := ssautil.BuildPackage(&types.Config{Importer: importer.Default()}, fset,
		types.NewPackage("main", ""), []*ast.File{f}, ssa.SanityCheckFunctions)
	if err != nil {
		t.Error(err)
		return
	}

	// The main package, its direct and indirect dependencies are loaded.
	deps := []string{
		// directly imported dependencies:
		"bytes", "io", "testing",
		// indirect dependencies mentioned by
		// the direct imports' export data
		"sync", "unicode", "time",
	}

	prog := mainPkg.Prog
	all := prog.AllPackages()
	if len(all) <= len(deps) {
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
				mset := prog.MethodSets.MethodSet(types.NewPointer(mem.Type()))
				for i, n := 0, mset.Len(); i < n; i++ {
					m := prog.MethodValue(mset.At(i))
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

// TestRuntimeTypes tests that (*Program).RuntimeTypes() includes all necessary types.
func TestRuntimeTypes(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		// An exported package-level type is needed.
		{`package A; type T struct{}; func (T) f() {}`,
			[]string{"*p.T", "p.T"},
		},
		// An unexported package-level type is not needed.
		{`package B; type t struct{}; func (t) f() {}`,
			nil,
		},
		// Subcomponents of type of exported package-level var are needed.
		{`package C; import "bytes"; var V struct {*bytes.Buffer}`,
			[]string{"*bytes.Buffer", "*struct{*bytes.Buffer}", "struct{*bytes.Buffer}"},
		},
		// Subcomponents of type of unexported package-level var are not needed.
		{`package D; import "bytes"; var v struct {*bytes.Buffer}`,
			nil,
		},
		// Subcomponents of type of exported package-level function are needed.
		{`package E; import "bytes"; func F(struct {*bytes.Buffer}) {}`,
			[]string{"*bytes.Buffer", "struct{*bytes.Buffer}"},
		},
		// Subcomponents of type of unexported package-level function are not needed.
		{`package F; import "bytes"; func f(struct {*bytes.Buffer}) {}`,
			nil,
		},
		// Subcomponents of type of exported method of uninstantiated unexported type are not needed.
		{`package G; import "bytes"; type x struct{}; func (x) G(struct {*bytes.Buffer}) {}; var v x`,
			nil,
		},
		// ...unless used by MakeInterface.
		{`package G2; import "bytes"; type x struct{}; func (x) G(struct {*bytes.Buffer}) {}; var v interface{} = x{}`,
			[]string{"*bytes.Buffer", "*p.x", "p.x", "struct{*bytes.Buffer}"},
		},
		// Subcomponents of type of unexported method are not needed.
		{`package I; import "bytes"; type X struct{}; func (X) G(struct {*bytes.Buffer}) {}`,
			[]string{"*bytes.Buffer", "*p.X", "p.X", "struct{*bytes.Buffer}"},
		},
		// Local types aren't needed.
		{`package J; import "bytes"; func f() { type T struct {*bytes.Buffer}; var t T; _ = t }`,
			nil,
		},
		// ...unless used by MakeInterface.
		{`package K; import "bytes"; func f() { type T struct {*bytes.Buffer}; _ = interface{}(T{}) }`,
			[]string{"*bytes.Buffer", "*p.T", "p.T"},
		},
		// Types used as operand of MakeInterface are needed.
		{`package L; import "bytes"; func f() { _ = interface{}(struct{*bytes.Buffer}{}) }`,
			[]string{"*bytes.Buffer", "struct{*bytes.Buffer}"},
		},
		// MakeInterface is optimized away when storing to a blank.
		{`package M; import "bytes"; var _ interface{} = struct{*bytes.Buffer}{}`,
			nil,
		},
	}
	for _, test := range tests {
		// Parse the file.
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, "input.go", test.input, 0)
		if err != nil {
			t.Errorf("test %q: %s", test.input[:15], err)
			continue
		}

		// Create a single-file main package.
		// Load dependencies from gc binary export data.
		ssapkg, _, err := ssautil.BuildPackage(&types.Config{Importer: importer.Default()}, fset,
			types.NewPackage("p", ""), []*ast.File{f}, ssa.SanityCheckFunctions)
		if err != nil {
			t.Errorf("test %q: %s", test.input[:15], err)
			continue
		}

		var typstrs []string
		for _, T := range ssapkg.Prog.RuntimeTypes() {
			typstrs = append(typstrs, T.String())
		}
		sort.Strings(typstrs)

		if !reflect.DeepEqual(typstrs, test.want) {
			t.Errorf("test 'package %s': got %q, want %q",
				f.Name.Name, typstrs, test.want)
		}
	}
}

// TestInit tests that synthesized init functions are correctly formed.
// Bare init functions omit calls to dependent init functions and the use of
// an init guard. They are useful in cases where the client uses a different
// calling convention for init functions, or cases where it is easier for a
// client to analyze bare init functions. Both of these aspects are used by
// the llgo compiler for simpler integration with gccgo's runtime library,
// and to simplify the analysis whereby it deduces which stores to globals
// can be lowered to global initializers.
func TestInit(t *testing.T) {
	tests := []struct {
		mode        ssa.BuilderMode
		input, want string
	}{
		{0, `package A; import _ "errors"; var i int = 42`,
			`# Name: A.init
# Package: A
# Synthetic: package initializer
func init():
0:                                                                entry P:0 S:2
	t0 = *init$guard                                                   bool
	if t0 goto 2 else 1
1:                                                           init.start P:1 S:1
	*init$guard = true:bool
	t1 = errors.init()                                                   ()
	*i = 42:int
	jump 2
2:                                                            init.done P:2 S:0
	return

`},
		{ssa.BareInits, `package B; import _ "errors"; var i int = 42`,
			`# Name: B.init
# Package: B
# Synthetic: package initializer
func init():
0:                                                                entry P:0 S:0
	*i = 42:int
	return

`},
	}
	for _, test := range tests {
		// Create a single-file main package.
		var conf loader.Config
		f, err := conf.ParseFile("<input>", test.input)
		if err != nil {
			t.Errorf("test %q: %s", test.input[:15], err)
			continue
		}
		conf.CreateFromFiles(f.Name.Name, f)

		lprog, err := conf.Load()
		if err != nil {
			t.Errorf("test 'package %s': Load: %s", f.Name.Name, err)
			continue
		}
		prog := ssautil.CreateProgram(lprog, test.mode)
		mainPkg := prog.Package(lprog.Created[0].Pkg)
		prog.Build()
		initFunc := mainPkg.Func("init")
		if initFunc == nil {
			t.Errorf("test 'package %s': no init function", f.Name.Name)
			continue
		}

		var initbuf bytes.Buffer
		_, err = initFunc.WriteTo(&initbuf)
		if err != nil {
			t.Errorf("test 'package %s': WriteTo: %s", f.Name.Name, err)
			continue
		}

		if initbuf.String() != test.want {
			t.Errorf("test 'package %s': got %s, want %s", f.Name.Name, initbuf.String(), test.want)
		}
	}
}

// TestSyntheticFuncs checks that the expected synthetic functions are
// created, reachable, and not duplicated.
func TestSyntheticFuncs(t *testing.T) {
	const input = `package P
type T int
func (T) f() int
func (*T) g() int
var (
	// thunks
	a = T.f
	b = T.f
	c = (struct{T}).f
	d = (struct{T}).f
	e = (*T).g
	f = (*T).g
	g = (struct{*T}).g
	h = (struct{*T}).g

	// bounds
	i = T(0).f
	j = T(0).f
	k = new(T).g
	l = new(T).g

	// wrappers
	m interface{} = struct{T}{}
	n interface{} = struct{T}{}
	o interface{} = struct{*T}{}
	p interface{} = struct{*T}{}
	q interface{} = new(struct{T})
	r interface{} = new(struct{T})
	s interface{} = new(struct{*T})
	t interface{} = new(struct{*T})
)
`
	// Parse
	var conf loader.Config
	f, err := conf.ParseFile("<input>", input)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	conf.CreateFromFiles(f.Name.Name, f)

	// Load
	lprog, err := conf.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Create and build SSA
	prog := ssautil.CreateProgram(lprog, 0)
	prog.Build()

	// Enumerate reachable synthetic functions
	want := map[string]string{
		"(*P.T).g$bound": "bound method wrapper for func (*P.T).g() int",
		"(P.T).f$bound":  "bound method wrapper for func (P.T).f() int",

		"(*P.T).g$thunk":         "thunk for func (*P.T).g() int",
		"(P.T).f$thunk":          "thunk for func (P.T).f() int",
		"(struct{*P.T}).g$thunk": "thunk for func (*P.T).g() int",
		"(struct{P.T}).f$thunk":  "thunk for func (P.T).f() int",

		"(*P.T).f":          "wrapper for func (P.T).f() int",
		"(*struct{*P.T}).f": "wrapper for func (P.T).f() int",
		"(*struct{*P.T}).g": "wrapper for func (*P.T).g() int",
		"(*struct{P.T}).f":  "wrapper for func (P.T).f() int",
		"(*struct{P.T}).g":  "wrapper for func (*P.T).g() int",
		"(struct{*P.T}).f":  "wrapper for func (P.T).f() int",
		"(struct{*P.T}).g":  "wrapper for func (*P.T).g() int",
		"(struct{P.T}).f":   "wrapper for func (P.T).f() int",

		"P.init": "package initializer",
	}
	for fn := range ssautil.AllFunctions(prog) {
		if fn.Synthetic == "" {
			continue
		}
		name := fn.String()
		wantDescr, ok := want[name]
		if !ok {
			t.Errorf("got unexpected/duplicate func: %q: %q", name, fn.Synthetic)
			continue
		}
		delete(want, name)

		if wantDescr != fn.Synthetic {
			t.Errorf("(%s).Synthetic = %q, want %q", name, fn.Synthetic, wantDescr)
		}
	}
	for fn, descr := range want {
		t.Errorf("want func: %q: %q", fn, descr)
	}
}
