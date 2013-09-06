// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// CreateTestMainFunction synthesizes main functions for tests.
// It is closely coupled to src/cmd/go/test.go and src/pkg/testing.

import (
	"go/token"
	"strings"
	"unicode"
	"unicode/utf8"

	"code.google.com/p/go.tools/go/exact"
	"code.google.com/p/go.tools/go/types"
)

// CreateTestMainFunction adds to pkg (a test package) a synthetic
// main function similar to the one that would be created by the 'go
// test' tool.  The synthetic function is returned.
//
// If the package already has a member named "main", the package
// remains unchanged; the result is that member if it's a function,
// nil if not.
//
func (pkg *Package) CreateTestMainFunction() *Function {
	if main := pkg.Members["main"]; main != nil {
		main, _ := main.(*Function)
		return main
	}
	fn := &Function{
		name:      "main",
		Signature: new(types.Signature),
		Synthetic: "test main function",
		Prog:      pkg.Prog,
		Pkg:       pkg,
	}

	testingPkg := pkg.Prog.ImportedPackage("testing")
	if testingPkg == nil {
		// If it doesn't import "testing", it can't be a test.
		// TODO(adonovan): but it might contain Examples.
		// Support them (by just calling them directly).
		return nil
	}
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
		Enclosing: fn,
		Pkg:       fn.Pkg,
		Prog:      fn.Prog,
	}
	fn.AnonFuncs = append(fn.AnonFuncs, matcher)
	matcher.startBody()
	matcher.emit(&Ret{Results: []Value{vTrue, nilConst(types.Universe.Lookup("error").Type())}})
	matcher.finishBody()

	fn.startBody()
	var c Call
	c.Call.Value = testingMain
	c.Call.Args = []Value{
		matcher,
		testMainSlice(fn, "Test", testingMainParams.At(1).Type()),
		testMainSlice(fn, "Benchmark", testingMainParams.At(2).Type()),
		testMainSlice(fn, "Example", testingMainParams.At(3).Type()),
	}
	// Emit: testing.Main(nil, tests, benchmarks, examples)
	emitTailCall(fn, &c)
	fn.finishBody()

	pkg.Members["main"] = fn
	return fn
}

// testMainSlice emits to fn code to construct a slice of type slice
// (one of []testing.Internal{Test,Benchmark,Example}) for all
// functions in this package whose name starts with prefix (one of
// "Test", "Benchmark" or "Example") and whose type is appropriate.
// It returns the slice value.
//
func testMainSlice(fn *Function, prefix string, slice types.Type) Value {
	tElem := slice.(*types.Slice).Elem()
	tFunc := tElem.Underlying().(*types.Struct).Field(1).Type()

	var testfuncs []*Function
	for name, mem := range fn.Pkg.Members {
		if fn, ok := mem.(*Function); ok && isTest(name, prefix) && types.IsIdentical(fn.Signature, tFunc) {
			testfuncs = append(testfuncs, fn)
		}
	}
	if testfuncs == nil {
		return nilConst(slice)
	}

	tString := types.Typ[types.String]
	tPtrString := types.NewPointer(tString)
	tPtrElem := types.NewPointer(tElem)
	tPtrFunc := types.NewPointer(tFunc)

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
		emitStore(fn, pname, NewConst(exact.MakeString(testfunc.Name()), tString))

		// Emit: pfunc = &pitem.F
		fa = &FieldAddr{X: pitem, Field: 1} // .F
		fa.setType(tPtrFunc)
		pfunc := fn.emit(fa)

		// Emit: *pfunc = testfunc
		emitStore(fn, pfunc, testfunc)
	}

	// Emit: slice array[:]
	sl := &Slice{X: array}
	sl.setType(slice)
	return fn.emit(sl)
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
	rune, _ := utf8.DecodeRuneInString(name[len(prefix):])
	return !unicode.IsLower(rune)
}
