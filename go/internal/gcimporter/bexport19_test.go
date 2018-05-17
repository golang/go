// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.9

package gcimporter_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"testing"

	"golang.org/x/tools/go/internal/gcimporter"
)

const src = `
package p

type (
	T0 = int32
	T1 = struct{}
	T2 = struct{ T1 }
	Invalid = foo // foo is undeclared
)
`

func checkPkg(t *testing.T, pkg *types.Package, label string) {
	T1 := types.NewStruct(nil, nil)
	T2 := types.NewStruct([]*types.Var{types.NewField(0, pkg, "T1", T1, true)}, nil)

	for _, test := range []struct {
		name string
		typ  types.Type
	}{
		{"T0", types.Typ[types.Int32]},
		{"T1", T1},
		{"T2", T2},
		{"Invalid", types.Typ[types.Invalid]},
	} {
		obj := pkg.Scope().Lookup(test.name)
		if obj == nil {
			t.Errorf("%s: %s not found", label, test.name)
			continue
		}
		tname, _ := obj.(*types.TypeName)
		if tname == nil {
			t.Errorf("%s: %v not a type name", label, obj)
			continue
		}
		if !tname.IsAlias() {
			t.Errorf("%s: %v: not marked as alias", label, tname)
			continue
		}
		if got := tname.Type(); !types.Identical(got, test.typ) {
			t.Errorf("%s: %v: got %v; want %v", label, tname, got, test.typ)
		}
	}
}

func TestTypeAliases(t *testing.T) {
	// parse and typecheck
	fset1 := token.NewFileSet()
	f, err := parser.ParseFile(fset1, "p.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	var conf types.Config
	pkg1, err := conf.Check("p", fset1, []*ast.File{f}, nil)
	if err == nil {
		// foo in undeclared in src; we should see an error
		t.Fatal("invalid source type-checked without error")
	}
	if pkg1 == nil {
		// despite incorrect src we should see a (partially) type-checked package
		t.Fatal("nil package returned")
	}
	checkPkg(t, pkg1, "export")

	// export
	exportdata, err := gcimporter.BExportData(fset1, pkg1)
	if err != nil {
		t.Fatal(err)
	}

	// import
	imports := make(map[string]*types.Package)
	fset2 := token.NewFileSet()
	_, pkg2, err := gcimporter.BImportData(fset2, imports, exportdata, pkg1.Path())
	if err != nil {
		t.Fatalf("BImportData(%s): %v", pkg1.Path(), err)
	}
	checkPkg(t, pkg2, "import")
}
