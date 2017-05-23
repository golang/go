// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"go/types"
	"strings"
	"unicode"
	"unicode/utf8"
)

func init() {
	register("tests",
		"check for common mistaken usages of tests/documentation examples",
		checkTestFunctions,
		funcDecl)
}

func isExampleSuffix(s string) bool {
	r, size := utf8.DecodeRuneInString(s)
	return size > 0 && unicode.IsLower(r)
}

func isTestSuffix(name string) bool {
	if len(name) == 0 {
		// "Test" is ok.
		return true
	}
	r, _ := utf8.DecodeRuneInString(name)
	return !unicode.IsLower(r)
}

func isTestParam(typ ast.Expr, wantType string) bool {
	ptr, ok := typ.(*ast.StarExpr)
	if !ok {
		// Not a pointer.
		return false
	}
	// No easy way of making sure it's a *testing.T or *testing.B:
	// ensure the name of the type matches.
	if name, ok := ptr.X.(*ast.Ident); ok {
		return name.Name == wantType
	}
	if sel, ok := ptr.X.(*ast.SelectorExpr); ok {
		return sel.Sel.Name == wantType
	}
	return false
}

func lookup(name string, scopes []*types.Scope) types.Object {
	for _, scope := range scopes {
		if o := scope.Lookup(name); o != nil {
			return o
		}
	}
	return nil
}

func extendedScope(f *File) []*types.Scope {
	scopes := []*types.Scope{f.pkg.typesPkg.Scope()}
	if f.basePkg != nil {
		scopes = append(scopes, f.basePkg.typesPkg.Scope())
	} else {
		// If basePkg is not specified (e.g. when checking a single file) try to
		// find it among imports.
		pkgName := f.pkg.typesPkg.Name()
		if strings.HasSuffix(pkgName, "_test") {
			basePkgName := strings.TrimSuffix(pkgName, "_test")
			for _, p := range f.pkg.typesPkg.Imports() {
				if p.Name() == basePkgName {
					scopes = append(scopes, p.Scope())
					break
				}
			}
		}
	}
	return scopes
}

func checkExample(fn *ast.FuncDecl, f *File, report reporter) {
	fnName := fn.Name.Name
	if params := fn.Type.Params; len(params.List) != 0 {
		report("%s should be niladic", fnName)
	}
	if results := fn.Type.Results; results != nil && len(results.List) != 0 {
		report("%s should return nothing", fnName)
	}

	if filesRun && !includesNonTest {
		// The coherence checks between a test and the package it tests
		// will report false positives if no non-test files have
		// been provided.
		return
	}

	if fnName == "Example" {
		// Nothing more to do.
		return
	}

	var (
		exName = strings.TrimPrefix(fnName, "Example")
		elems  = strings.SplitN(exName, "_", 3)
		ident  = elems[0]
		obj    = lookup(ident, extendedScope(f))
	)
	if ident != "" && obj == nil {
		// Check ExampleFoo and ExampleBadFoo.
		report("%s refers to unknown identifier: %s", fnName, ident)
		// Abort since obj is absent and no subsequent checks can be performed.
		return
	}
	if len(elems) < 2 {
		// Nothing more to do.
		return
	}

	if ident == "" {
		// Check Example_suffix and Example_BadSuffix.
		if residual := strings.TrimPrefix(exName, "_"); !isExampleSuffix(residual) {
			report("%s has malformed example suffix: %s", fnName, residual)
		}
		return
	}

	mmbr := elems[1]
	if !isExampleSuffix(mmbr) {
		// Check ExampleFoo_Method and ExampleFoo_BadMethod.
		if obj, _, _ := types.LookupFieldOrMethod(obj.Type(), true, obj.Pkg(), mmbr); obj == nil {
			report("%s refers to unknown field or method: %s.%s", fnName, ident, mmbr)
		}
	}
	if len(elems) == 3 && !isExampleSuffix(elems[2]) {
		// Check ExampleFoo_Method_suffix and ExampleFoo_Method_Badsuffix.
		report("%s has malformed example suffix: %s", fnName, elems[2])
	}
}

func checkTest(fn *ast.FuncDecl, prefix string, report reporter) {
	// Want functions with 0 results and 1 parameter.
	if fn.Type.Results != nil && len(fn.Type.Results.List) > 0 ||
		fn.Type.Params == nil ||
		len(fn.Type.Params.List) != 1 ||
		len(fn.Type.Params.List[0].Names) > 1 {
		return
	}

	// The param must look like a *testing.T or *testing.B.
	if !isTestParam(fn.Type.Params.List[0].Type, prefix[:1]) {
		return
	}

	if !isTestSuffix(fn.Name.Name[len(prefix):]) {
		report("%s has malformed name: first letter after '%s' must not be lowercase", fn.Name.Name, prefix)
	}
}

type reporter func(format string, args ...interface{})

// checkTestFunctions walks Test, Benchmark and Example functions checking
// malformed names, wrong signatures and examples documenting inexistent
// identifiers.
func checkTestFunctions(f *File, node ast.Node) {
	if !strings.HasSuffix(f.name, "_test.go") {
		return
	}

	fn, ok := node.(*ast.FuncDecl)
	if !ok || fn.Recv != nil {
		// Ignore non-functions or functions with receivers.
		return
	}

	report := func(format string, args ...interface{}) { f.Badf(node.Pos(), format, args...) }

	switch {
	case strings.HasPrefix(fn.Name.Name, "Example"):
		checkExample(fn, f, report)
	case strings.HasPrefix(fn.Name.Name, "Test"):
		checkTest(fn, "Test", report)
	case strings.HasPrefix(fn.Name.Name, "Benchmark"):
		checkTest(fn, "Benchmark", report)
	}
}
