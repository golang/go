// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Alias-related code. Keep for now.
// +build ignore

package gcimporter_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"testing"

	gcimporter "golang.org/x/tools/go/gcimporter15"
)

func disabledTestInvalidAlias(t *testing.T) {
	// parse and typecheck
	const src = "package p; func InvalidAlias => foo.f"
	fset1 := token.NewFileSet()
	f, err := parser.ParseFile(fset1, "p.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	var conf types.Config
	pkg, err := conf.Check("p", fset1, []*ast.File{f}, nil)
	if err == nil {
		t.Fatal("invalid source type-checked without error")
	}
	if pkg == nil {
		t.Fatal("nil package returned")
	}

	// export
	exportdata := gcimporter.BExportData(fset1, pkg)

	// import
	imports := make(map[string]*types.Package)
	fset2 := token.NewFileSet()
	_, pkg2, err := gcimporter.BImportData(fset2, imports, exportdata, pkg.Path())
	if err != nil {
		t.Fatalf("BImportData(%s): %v", pkg.Path(), err)
	}

	// pkg2 must contain InvalidAlias as an invalid Alias
	obj := pkg2.Scope().Lookup("InvalidAlias")
	if obj == nil {
		t.Fatal("InvalidAlias not found")
	}
	alias, ok := obj.(*types.Alias)
	if !ok {
		t.Fatalf("got %v; want alias", alias)
	}
	if alias.Type() != types.Typ[types.Invalid] || alias.Orig() != nil {
		t.Fatalf("got %v (orig = %v); want invalid alias", alias, alias.Orig())
	}
}
