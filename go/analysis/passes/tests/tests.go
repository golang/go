// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tests

import (
	"go/ast"
	"go/types"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/tools/go/analysis"
)

var Analyzer = &analysis.Analyzer{
	Name: "tests",
	Doc: `check for common mistaken usages of tests and examples

The tests checker walks Test, Benchmark and Example functions checking
malformed names, wrong signatures and examples documenting non-existent
identifiers.`,
	Run: run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	for _, f := range pass.Files {
		if !strings.HasSuffix(pass.Fset.File(f.Pos()).Name(), "_test.go") {
			continue
		}
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Recv != nil {
				// Ignore non-functions or functions with receivers.
				continue
			}

			switch {
			case strings.HasPrefix(fn.Name.Name, "Example"):
				checkExample(pass, fn)
			case strings.HasPrefix(fn.Name.Name, "Test"):
				checkTest(pass, fn, "Test")
			case strings.HasPrefix(fn.Name.Name, "Benchmark"):
				checkTest(pass, fn, "Benchmark")
			}
		}
	}
	return nil, nil
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

func lookup(pkg *types.Package, name string) types.Object {
	if o := pkg.Scope().Lookup(name); o != nil {
		return o
	}

	// If this package is ".../foo_test" and it imports a package
	// ".../foo", try looking in the latter package.
	// This heuristic should work even on build systems that do not
	// record any special link between the packages.
	if basePath := strings.TrimSuffix(pkg.Path(), "_test"); basePath != pkg.Path() {
		for _, imp := range pkg.Imports() {
			if imp.Path() == basePath {
				return imp.Scope().Lookup(name)
			}
		}
	}
	return nil
}

func checkExample(pass *analysis.Pass, fn *ast.FuncDecl) {
	fnName := fn.Name.Name
	if params := fn.Type.Params; len(params.List) != 0 {
		pass.Reportf(fn.Pos(), "%s should be niladic", fnName)
	}
	if results := fn.Type.Results; results != nil && len(results.List) != 0 {
		pass.Reportf(fn.Pos(), "%s should return nothing", fnName)
	}

	if fnName == "Example" {
		// Nothing more to do.
		return
	}

	var (
		exName = strings.TrimPrefix(fnName, "Example")
		elems  = strings.SplitN(exName, "_", 3)
		ident  = elems[0]
		obj    = lookup(pass.Pkg, ident)
	)
	if ident != "" && obj == nil {
		// Check ExampleFoo and ExampleBadFoo.
		pass.Reportf(fn.Pos(), "%s refers to unknown identifier: %s", fnName, ident)
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
			pass.Reportf(fn.Pos(), "%s has malformed example suffix: %s", fnName, residual)
		}
		return
	}

	mmbr := elems[1]
	if !isExampleSuffix(mmbr) {
		// Check ExampleFoo_Method and ExampleFoo_BadMethod.
		if obj, _, _ := types.LookupFieldOrMethod(obj.Type(), true, obj.Pkg(), mmbr); obj == nil {
			pass.Reportf(fn.Pos(), "%s refers to unknown field or method: %s.%s", fnName, ident, mmbr)
		}
	}
	if len(elems) == 3 && !isExampleSuffix(elems[2]) {
		// Check ExampleFoo_Method_suffix and ExampleFoo_Method_Badsuffix.
		pass.Reportf(fn.Pos(), "%s has malformed example suffix: %s", fnName, elems[2])
	}
}

func checkTest(pass *analysis.Pass, fn *ast.FuncDecl, prefix string) {
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
		pass.Reportf(fn.Pos(), "%s has malformed name: first letter after '%s' must not be lowercase", fn.Name.Name, prefix)
	}
}
