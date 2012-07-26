// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests verifying the types associated with an AST after
// type checking.

package types

import (
	"go/ast"
	"go/parser"
	"testing"
)

func checkSource(t *testing.T, src string) *ast.Package {
	const filename = "<src>"
	file, err := parser.ParseFile(fset, filename, src, parser.DeclarationErrors)
	if err != nil {
		t.Fatal(err)
	}
	files := map[string]*ast.File{filename: file}
	pkg, err := ast.NewPackage(fset, files, GcImport, Universe)
	if err != nil {
		t.Fatal(err)
	}
	_, err = Check(fset, pkg)
	if err != nil {
		t.Fatal(err)
	}
	return pkg
}

func TestVariadicFunctions(t *testing.T) {
	pkg := checkSource(t, `
package p
func f1(arg ...int)
func f2(arg1 string, arg2 ...int)
func f3()
func f4(arg int)
	`)
	f1 := pkg.Scope.Lookup("f1")
	f2 := pkg.Scope.Lookup("f2")
	for _, f := range [...](*ast.Object){f1, f2} {
		ftype := f.Type.(*Func)
		if !ftype.IsVariadic {
			t.Errorf("expected %s to be variadic", f.Name)
		}
		param := ftype.Params[len(ftype.Params)-1]
		if param.Type != Int {
			t.Errorf("expected last parameter of %s to have type int, found %T", f.Name, param.Type)
		}
	}

	f3 := pkg.Scope.Lookup("f3")
	f4 := pkg.Scope.Lookup("f4")
	for _, f := range [...](*ast.Object){f3, f4} {
		ftype := f.Type.(*Func)
		if ftype.IsVariadic {
			t.Fatalf("expected %s to not be variadic", f.Name)
		}
	}
	// TODO(axw) replace this function's innards with table driven tests.
	// We should have a helper function that prints a type signature. Then
	// we can have a table of function declarations and expected type
	// signatures which can be easily expanded.
}
