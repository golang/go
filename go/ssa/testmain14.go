// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.5

package ssa

// CreateTestMainPackage synthesizes a main package that runs all the
// tests of the supplied packages.
// It is closely coupled to $GOROOT/src/cmd/go/test.go and $GOROOT/src/testing.

import (
	"go/ast"
	"go/token"
	"os"
	"sort"
	"strings"

	"golang.org/x/tools/go/exact"
	"golang.org/x/tools/go/types"
)

// FindTests returns the list of packages that define at least one Test,
// Example or Benchmark function (as defined by "go test"), and the
// lists of all such functions.
//
func FindTests(pkgs []*Package) (testpkgs []*Package, tests, benchmarks, examples []*Function) {
	if len(pkgs) == 0 {
		return
	}
	prog := pkgs[0].Prog

	// The first two of these may be nil: if the program doesn't import "testing",
	// it can't contain any tests, but it may yet contain Examples.
	var testSig *types.Signature                              // func(*testing.T)
	var benchmarkSig *types.Signature                         // func(*testing.B)
	var exampleSig = types.NewSignature(nil, nil, nil, false) // func()

	// Obtain the types from the parameters of testing.Main().
	if testingPkg := prog.ImportedPackage("testing"); testingPkg != nil {
		params := testingPkg.Func("Main").Signature.Params()
		testSig = funcField(params.At(1).Type())
		benchmarkSig = funcField(params.At(2).Type())
	}

	seen := make(map[*Package]bool)
	for _, pkg := range pkgs {
		if pkg.Prog != prog {
			panic("wrong Program")
		}

		// TODO(adonovan): use a stable order, e.g. lexical.
		for _, mem := range pkg.Members {
			if f, ok := mem.(*Function); ok &&
				ast.IsExported(f.Name()) &&
				strings.HasSuffix(prog.Fset.Position(f.Pos()).Filename, "_test.go") {

				switch {
				case testSig != nil && isTestSig(f, "Test", testSig):
					tests = append(tests, f)
				case benchmarkSig != nil && isTestSig(f, "Benchmark", benchmarkSig):
					benchmarks = append(benchmarks, f)
				case isTestSig(f, "Example", exampleSig):
					examples = append(examples, f)
				default:
					continue
				}

				if !seen[pkg] {
					seen[pkg] = true
					testpkgs = append(testpkgs, pkg)
				}
			}
		}
	}
	return
}

// Like isTest, but checks the signature too.
func isTestSig(f *Function, prefix string, sig *types.Signature) bool {
	return isTest(f.Name(), prefix) && types.Identical(f.Signature, sig)
}

// If non-nil, testMainStartBodyHook is called immediately after
// startBody for main.init and main.main, making it easy for users to
// add custom imports and initialization steps for proprietary build
// systems that don't exactly follow 'go test' conventions.
var testMainStartBodyHook func(*Function)

// CreateTestMainPackage creates and returns a synthetic "main"
// package that runs all the tests of the supplied packages, similar
// to the one that would be created by the 'go test' tool.
//
// It returns nil if the program contains no tests.
//
func (prog *Program) CreateTestMainPackage(pkgs ...*Package) *Package {
	pkgs, tests, benchmarks, examples := FindTests(pkgs)
	if len(pkgs) == 0 {
		return nil
	}

	testmain := &Package{
		Prog:    prog,
		Members: make(map[string]Member),
		values:  make(map[types.Object]Value),
		Pkg:     types.NewPackage("test$main", "main"),
	}

	// Build package's init function.
	init := &Function{
		name:      "init",
		Signature: new(types.Signature),
		Synthetic: "package initializer",
		Pkg:       testmain,
		Prog:      prog,
	}
	init.startBody()

	if testMainStartBodyHook != nil {
		testMainStartBodyHook(init)
	}

	// Initialize packages to test.
	var pkgpaths []string
	for _, pkg := range pkgs {
		var v Call
		v.Call.Value = pkg.init
		v.setType(types.NewTuple())
		init.emit(&v)

		pkgpaths = append(pkgpaths, pkg.Pkg.Path())
	}
	sort.Strings(pkgpaths)
	init.emit(new(Return))
	init.finishBody()
	testmain.init = init
	testmain.Pkg.MarkComplete()
	testmain.Members[init.name] = init

	// For debugging convenience, define an unexported const
	// that enumerates the packages.
	packagesConst := types.NewConst(token.NoPos, testmain.Pkg, "packages", tString,
		exact.MakeString(strings.Join(pkgpaths, " ")))
	memberFromObject(testmain, packagesConst, nil)

	// Create main *types.Func and *ssa.Function
	mainFunc := types.NewFunc(token.NoPos, testmain.Pkg, "main", new(types.Signature))
	memberFromObject(testmain, mainFunc, nil)
	main := testmain.Func("main")
	main.Synthetic = "test main function"

	main.startBody()

	if testMainStartBodyHook != nil {
		testMainStartBodyHook(main)
	}

	if testingPkg := prog.ImportedPackage("testing"); testingPkg != nil {
		testingMain := testingPkg.Func("Main")
		testingMainParams := testingMain.Signature.Params()

		// The generated code is as if compiled from this:
		//
		// func main() {
		//      match      := func(_, _ string) (bool, error) { return true, nil }
		//      tests      := []testing.InternalTest{{"TestFoo", TestFoo}, ...}
		//      benchmarks := []testing.InternalBenchmark{...}
		//      examples   := []testing.InternalExample{...}
		// 	testing.Main(match, tests, benchmarks, examples)
		// }

		matcher := &Function{
			name:      "matcher",
			Signature: testingMainParams.At(0).Type().(*types.Signature),
			Synthetic: "test matcher predicate",
			parent:    main,
			Pkg:       testmain,
			Prog:      prog,
		}
		main.AnonFuncs = append(main.AnonFuncs, matcher)
		matcher.startBody()
		matcher.emit(&Return{Results: []Value{vTrue, nilConst(types.Universe.Lookup("error").Type())}})
		matcher.finishBody()

		// Emit call: testing.Main(matcher, tests, benchmarks, examples).
		var c Call
		c.Call.Value = testingMain
		c.Call.Args = []Value{
			matcher,
			testMainSlice(main, tests, testingMainParams.At(1).Type()),
			testMainSlice(main, benchmarks, testingMainParams.At(2).Type()),
			testMainSlice(main, examples, testingMainParams.At(3).Type()),
		}
		emitTailCall(main, &c)
	} else {
		// The program does not import "testing", but FindTests
		// returned non-nil, which must mean there were Examples
		// but no Tests or Benchmarks.
		// We'll simply call them from testmain.main; this will
		// ensure they don't panic, but will not check any
		// "Output:" comments.
		for _, eg := range examples {
			var c Call
			c.Call.Value = eg
			c.setType(types.NewTuple())
			main.emit(&c)
		}
		main.emit(&Return{})
		main.currentBlock = nil
	}

	main.finishBody()

	testmain.Members["main"] = main

	if prog.mode&PrintPackages != 0 {
		printMu.Lock()
		testmain.WriteTo(os.Stdout)
		printMu.Unlock()
	}

	if prog.mode&SanityCheckFunctions != 0 {
		sanityCheckPackage(testmain)
	}

	prog.packages[testmain.Pkg] = testmain

	return testmain
}

// testMainSlice emits to fn code to construct a slice of type slice
// (one of []testing.Internal{Test,Benchmark,Example}) for all
// functions in testfuncs.  It returns the slice value.
//
func testMainSlice(fn *Function, testfuncs []*Function, slice types.Type) Value {
	if testfuncs == nil {
		return nilConst(slice)
	}

	tElem := slice.(*types.Slice).Elem()
	tPtrString := types.NewPointer(tString)
	tPtrElem := types.NewPointer(tElem)
	tPtrFunc := types.NewPointer(funcField(slice))

	// TODO(adonovan): fix: populate the
	// testing.InternalExample.Output field correctly so that tests
	// work correctly under the interpreter.  This requires that we
	// do this step using ASTs, not *ssa.Functions---quite a
	// redesign.  See also the fake runExample in go/ssa/interp.

	// Emit: array = new [n]testing.InternalTest
	tArray := types.NewArray(tElem, int64(len(testfuncs)))
	array := emitNew(fn, tArray, token.NoPos)
	array.Comment = "test main"
	for i, testfunc := range testfuncs {
		// Emit: pitem = &array[i]
		ia := &IndexAddr{X: array, Index: intConst(int64(i))}
		ia.setType(tPtrElem)
		pitem := fn.emit(ia)

		// Emit: pname = &pitem.Name
		fa := &FieldAddr{X: pitem, Field: 0} // .Name
		fa.setType(tPtrString)
		pname := fn.emit(fa)

		// Emit: *pname = "testfunc"
		emitStore(fn, pname, stringConst(testfunc.Name()), token.NoPos)

		// Emit: pfunc = &pitem.F
		fa = &FieldAddr{X: pitem, Field: 1} // .F
		fa.setType(tPtrFunc)
		pfunc := fn.emit(fa)

		// Emit: *pfunc = testfunc
		emitStore(fn, pfunc, testfunc, token.NoPos)
	}

	// Emit: slice array[:]
	sl := &Slice{X: array}
	sl.setType(slice)
	return fn.emit(sl)
}

// Given the type of one of the three slice parameters of testing.Main,
// returns the function type.
func funcField(slice types.Type) *types.Signature {
	return slice.(*types.Slice).Elem().Underlying().(*types.Struct).Field(1).Type().(*types.Signature)
}

// Plundered from $GOROOT/src/cmd/go/test.go

// isTest tells whether name looks like a test (or benchmark, according to prefix).
// It is a Test (say) if there is a character after Test that is not a lower-case letter.
// We don't want TesticularCancer.
func isTest(name, prefix string) bool {
	if !strings.HasPrefix(name, prefix) {
		return false
	}
	if len(name) == len(prefix) { // "Test" is ok
		return true
	}
	return ast.IsExported(name[len(prefix):])
}
