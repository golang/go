// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/build"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
	"golang.org/x/tools/internal/typeparams"
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
	mode := ssa.SanityCheckFunctions
	mainPkg, _, err := ssautil.BuildPackage(&types.Config{Importer: importer.Default()}, fset,
		types.NewPackage("main", ""), []*ast.File{f}, mode)
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
		mode := ssa.SanityCheckFunctions
		ssapkg, _, err := ssautil.BuildPackage(&types.Config{Importer: importer.Default()}, fset,
			types.NewPackage("p", ""), []*ast.File{f}, mode)
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
	prog := ssautil.CreateProgram(lprog, ssa.BuilderMode(0))
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

// TestPhiElimination ensures that dead phis, including those that
// participate in a cycle, are properly eliminated.
func TestPhiElimination(t *testing.T) {
	const input = `
package p

func f() error

func g(slice []int) {
	for {
		for range slice {
			// e should not be lifted to a dead φ-node.
			e := f()
			h(e)
		}
	}
}

func h(error)
`
	// The SSA code for this function should look something like this:
	// 0:
	//         jump 1
	// 1:
	//         t0 = len(slice)
	//         jump 2
	// 2:
	//         t1 = phi [1: -1:int, 3: t2]
	//         t2 = t1 + 1:int
	//         t3 = t2 < t0
	//         if t3 goto 3 else 1
	// 3:
	//         t4 = f()
	//         t5 = h(t4)
	//         jump 2
	//
	// But earlier versions of the SSA construction algorithm would
	// additionally generate this cycle of dead phis:
	//
	// 1:
	//         t7 = phi [0: nil:error, 2: t8] #e
	//         ...
	// 2:
	//         t8 = phi [1: t7, 3: t4] #e
	//         ...

	// Parse
	var conf loader.Config
	f, err := conf.ParseFile("<input>", input)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	conf.CreateFromFiles("p", f)

	// Load
	lprog, err := conf.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Create and build SSA
	prog := ssautil.CreateProgram(lprog, ssa.BuilderMode(0))
	p := prog.Package(lprog.Package("p").Pkg)
	p.Build()
	g := p.Func("g")

	phis := 0
	for _, b := range g.Blocks {
		for _, instr := range b.Instrs {
			if _, ok := instr.(*ssa.Phi); ok {
				phis++
			}
		}
	}
	if phis != 1 {
		g.WriteTo(os.Stderr)
		t.Errorf("expected a single Phi (for the range index), got %d", phis)
	}
}

// TestGenericDecls ensures that *unused* generic types, methods and functions
// signatures can be built.
//
// TODO(taking): Add calls from non-generic functions to instantiations of generic functions.
// TODO(taking): Add globals with types that are instantiations of generic functions.
func TestGenericDecls(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("TestGenericDecls only works with type parameters enabled.")
	}
	const input = `
package p

import "unsafe"

type Pointer[T any] struct {
	v unsafe.Pointer
}

func (x *Pointer[T]) Load() *T {
	return (*T)(LoadPointer(&x.v))
}

func Load[T any](x *Pointer[T]) *T {
	return x.Load()
}

func LoadPointer(addr *unsafe.Pointer) (val unsafe.Pointer)
`
	// The SSA members for this package should look something like this:
	//          func  LoadPointer func(addr *unsafe.Pointer) (val unsafe.Pointer)
	//      type  Pointer     struct{v unsafe.Pointer}
	//        method (*Pointer[T any]) Load() *T
	//      func  init        func()
	//      var   init$guard  bool

	// Parse
	var conf loader.Config
	f, err := conf.ParseFile("<input>", input)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	conf.CreateFromFiles("p", f)

	// Load
	lprog, err := conf.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Create and build SSA
	prog := ssautil.CreateProgram(lprog, ssa.BuilderMode(0))
	p := prog.Package(lprog.Package("p").Pkg)
	p.Build()

	if load := p.Func("Load"); typeparams.ForSignature(load.Signature).Len() != 1 {
		t.Errorf("expected a single type param T for Load got %q", load.Signature)
	}
	if ptr := p.Type("Pointer"); typeparams.ForNamed(ptr.Type().(*types.Named)).Len() != 1 {
		t.Errorf("expected a single type param T for Pointer got %q", ptr.Type())
	}
}

func TestGenericWrappers(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("TestGenericWrappers only works with type parameters enabled.")
	}
	const input = `
package p

type S[T any] struct {
	t *T
}

func (x S[T]) M() T {
	return *(x.t)
}

var thunk = S[int].M

var g S[int]
var bound = g.M

type R[T any] struct{ S[T] }

var indirect = R[int].M
`
	// The relevant SSA members for this package should look something like this:
	// var   bound      func() int
	// var   thunk      func(S[int]) int
	// var   wrapper    func(R[int]) int

	// Parse
	var conf loader.Config
	f, err := conf.ParseFile("<input>", input)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	conf.CreateFromFiles("p", f)

	// Load
	lprog, err := conf.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	for _, mode := range []ssa.BuilderMode{ssa.BuilderMode(0), ssa.InstantiateGenerics} {
		// Create and build SSA
		prog := ssautil.CreateProgram(lprog, mode)
		p := prog.Package(lprog.Package("p").Pkg)
		p.Build()

		for _, entry := range []struct {
			name    string // name of the package variable
			typ     string // type of the package variable
			wrapper string // wrapper function to which the package variable is set
			callee  string // callee within the wrapper function
		}{
			{
				"bound",
				"*func() int",
				"(p.S[int]).M$bound",
				"(p.S[int]).M[int]",
			},
			{
				"thunk",
				"*func(p.S[int]) int",
				"(p.S[int]).M$thunk",
				"(p.S[int]).M[int]",
			},
			{
				"indirect",
				"*func(p.R[int]) int",
				"(p.R[int]).M$thunk",
				"(p.S[int]).M[int]",
			},
		} {
			entry := entry
			t.Run(entry.name, func(t *testing.T) {
				v := p.Var(entry.name)
				if v == nil {
					t.Fatalf("Did not find variable for %q in %s", entry.name, p.String())
				}
				if v.Type().String() != entry.typ {
					t.Errorf("Expected type for variable %s: %q. got %q", v, entry.typ, v.Type())
				}

				// Find the wrapper for v. This is stored exactly once in init.
				var wrapper *ssa.Function
				for _, bb := range p.Func("init").Blocks {
					for _, i := range bb.Instrs {
						if store, ok := i.(*ssa.Store); ok && v == store.Addr {
							switch val := store.Val.(type) {
							case *ssa.Function:
								wrapper = val
							case *ssa.MakeClosure:
								wrapper = val.Fn.(*ssa.Function)
							}
						}
					}
				}
				if wrapper == nil {
					t.Fatalf("failed to find wrapper function for %s", entry.name)
				}
				if wrapper.String() != entry.wrapper {
					t.Errorf("Expected wrapper function %q. got %q", wrapper, entry.wrapper)
				}

				// Find the callee within the wrapper. There should be exactly one call.
				var callee *ssa.Function
				for _, bb := range wrapper.Blocks {
					for _, i := range bb.Instrs {
						if call, ok := i.(*ssa.Call); ok {
							callee = call.Call.StaticCallee()
						}
					}
				}
				if callee == nil {
					t.Fatalf("failed to find callee within wrapper %s", wrapper)
				}
				if callee.String() != entry.callee {
					t.Errorf("Expected callee in wrapper %q is %q. got %q", v, entry.callee, callee)
				}
			})
		}
	}
}

// TestTypeparamTest builds SSA over compilable examples in $GOROOT/test/typeparam/*.go.

func TestTypeparamTest(t *testing.T) {
	if !typeparams.Enabled {
		return
	}

	// Tests use a fake goroot to stub out standard libraries with delcarations in
	// testdata/src. Decreases runtime from ~80s to ~1s.

	dir := filepath.Join(build.Default.GOROOT, "test", "typeparam")

	// Collect all of the .go files in
	list, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}

	for _, entry := range list {
		if entry.Name() == "issue376214.go" {
			continue // investigate variadic + New signature.
		}
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") {
			continue // Consider standalone go files.
		}
		input := filepath.Join(dir, entry.Name())
		t.Run(entry.Name(), func(t *testing.T) {
			src, err := os.ReadFile(input)
			if err != nil {
				t.Fatal(err)
			}
			// Only build test files that can be compiled, or compiled and run.
			if !bytes.HasPrefix(src, []byte("// run")) && !bytes.HasPrefix(src, []byte("// compile")) {
				t.Skipf("not detected as a run test")
			}

			t.Logf("Input: %s\n", input)

			ctx := build.Default    // copy
			ctx.GOROOT = "testdata" // fake goroot. Makes tests ~1s. tests take ~80s.

			reportErr := func(err error) {
				t.Error(err)
			}
			conf := loader.Config{Build: &ctx, TypeChecker: types.Config{Error: reportErr}}
			if _, err := conf.FromArgs([]string{input}, true); err != nil {
				t.Fatalf("FromArgs(%s) failed: %s", input, err)
			}

			iprog, err := conf.Load()
			if iprog != nil {
				for _, pkg := range iprog.Created {
					for i, e := range pkg.Errors {
						t.Errorf("Loading pkg %s error[%d]=%s", pkg, i, e)
					}
				}
			}
			if err != nil {
				t.Fatalf("conf.Load(%s) failed: %s", input, err)
			}

			mode := ssa.SanityCheckFunctions | ssa.InstantiateGenerics
			prog := ssautil.CreateProgram(iprog, mode)
			prog.Build()
		})
	}
}

// TestOrderOfOperations ensures order of operations are as intended.
func TestOrderOfOperations(t *testing.T) {
	// Testing for the order of operations within an expression is done
	// by collecting the sequence of direct function calls within a *Function.
	// Callees are all external functions so they cannot be safely re-ordered by ssa.
	const input = `
package p

func a() int
func b() int
func c() int

func slice(s []int) []int { return s[a():b()] }
func sliceMax(s []int) []int { return s[a():b():c()] }

`

	// Parse
	var conf loader.Config
	f, err := conf.ParseFile("<input>", input)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	conf.CreateFromFiles("p", f)

	// Load
	lprog, err := conf.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Create and build SSA
	prog := ssautil.CreateProgram(lprog, ssa.BuilderMode(0))
	p := prog.Package(lprog.Package("p").Pkg)
	p.Build()

	for _, item := range []struct {
		fn   string
		want string // sequence of calls within the function.
	}{
		{"sliceMax", "[a() b() c()]"},
		{"slice", "[a() b()]"},
	} {
		fn := p.Func(item.fn)
		want := item.want
		t.Run(item.fn, func(t *testing.T) {
			t.Parallel()

			var calls []string
			for _, b := range fn.Blocks {
				for _, instr := range b.Instrs {
					if call, ok := instr.(ssa.CallInstruction); ok {
						calls = append(calls, call.String())
					}
				}
			}
			if got := fmt.Sprint(calls); got != want {
				fn.WriteTo(os.Stderr)
				t.Errorf("Expected sequence of function calls in %s was %s. got %s", fn, want, got)
			}
		})
	}
}

// TestGenericFunctionSelector ensures generic functions from other packages can be selected.
func TestGenericFunctionSelector(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("TestGenericFunctionSelector uses type parameters.")
	}

	pkgs := map[string]map[string]string{
		"main": {"m.go": `package main; import "a"; func main() { a.F[int](); a.G[int,string](); a.H(0) }`},
		"a":    {"a.go": `package a; func F[T any](){}; func G[S, T any](){}; func H[T any](a T){} `},
	}

	for _, mode := range []ssa.BuilderMode{
		ssa.SanityCheckFunctions,
		ssa.SanityCheckFunctions | ssa.InstantiateGenerics,
	} {
		conf := loader.Config{
			Build: buildutil.FakeContext(pkgs),
		}
		conf.Import("main")

		lprog, err := conf.Load()
		if err != nil {
			t.Errorf("Load failed: %s", err)
		}
		if lprog == nil {
			t.Fatalf("Load returned nil *Program")
		}
		// Create and build SSA
		prog := ssautil.CreateProgram(lprog, mode)
		p := prog.Package(lprog.Package("main").Pkg)
		p.Build()

		var callees []string // callees of the CallInstruction.String() in main().
		for _, b := range p.Func("main").Blocks {
			for _, i := range b.Instrs {
				if call, ok := i.(ssa.CallInstruction); ok {
					if callee := call.Common().StaticCallee(); call != nil {
						callees = append(callees, callee.String())
					} else {
						t.Errorf("CallInstruction without StaticCallee() %q", call)
					}
				}
			}
		}
		sort.Strings(callees) // ignore the order in the code.

		want := "[a.F[int] a.G[int string] a.H[int]]"
		if got := fmt.Sprint(callees); got != want {
			t.Errorf("Expected main() to contain calls %v. got %v", want, got)
		}
	}
}
