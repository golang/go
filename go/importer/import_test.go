// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importer

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"path/filepath"
	"testing"

	_ "code.google.com/p/go.tools/go/gcimporter"
	"code.google.com/p/go.tools/go/types"
)

var tests = []string{
	`package p`,

	// consts
	`package p; const X = true`,
	`package p; const X, y, Z = true, false, 0 != 0`,
	`package p; const ( A float32 = 1<<iota; B; C; D)`,
	`package p; const X = "foo"`,
	`package p; const X string = "foo"`,
	`package p; const X = 0`,
	`package p; const X = -42`,
	`package p; const X = 3.14159265`,
	`package p; const X = -1e-10`,
	`package p; const X = 1.2 + 2.3i`,
	`package p; const X = -1i`,
	`package p; import "math"; const Pi = math.Pi`,
	`package p; import m "math"; const Pi = m.Pi`,

	// types
	`package p; type T int`,
	`package p; type T [10]int`,
	`package p; type T []int`,
	`package p; type T struct{}`,
	`package p; type T struct{x int}`,
	`package p; type T *int`,
	`package p; type T func()`,
	`package p; type T *T`,
	`package p; type T interface{}`,
	`package p; type T interface{ foo() }`,
	`package p; type T interface{ m() T }`,
	// TODO(gri) disabled for now - import/export works but
	// types.Type.String() used in the test cannot handle cases
	// like this yet
	// `package p; type T interface{ m() interface{T} }`,
	`package p; type T map[string]bool`,
	`package p; type T chan int`,
	`package p; type T <-chan complex64`,
	`package p; type T chan<- map[int]string`,

	// vars
	`package p; var X int`,
	`package p; var X, Y, Z struct{f int "tag"}`,

	// funcs
	`package p; func F()`,
	`package p; func F(x int, y struct{}) bool`,
	`package p; type T int; func (*T) F(x int, y struct{}) T`,

	// selected special cases
	`package p; type T int`,
	`package p; type T uint8`,
	`package p; type T byte`,
	`package p; type T error`,
	`package p; import "net/http"; type T http.Client`,
	`package p; import "net/http"; type ( T1 http.Client; T2 struct { http.Client } )`,
	`package p; import "unsafe"; type ( T1 unsafe.Pointer; T2 unsafe.Pointer )`,
	`package p; import "unsafe"; type T struct { p unsafe.Pointer }`,
}

func TestImportSrc(t *testing.T) {
	for _, src := range tests {
		pkg, err := pkgForSource(src)
		if err != nil {
			t.Errorf("typecheck failed: %s", err)
			continue
		}
		testExportImport(t, pkg)
	}
}

// TODO(gri) expand to std library
var libs = []string{
	"../exact",
	"../gcimporter",
	"../importer",
	"../types",
	"../types/typemap",
}

func TestImportLib(t *testing.T) {
	for _, lib := range libs {
		pkg, err := pkgFor(lib)
		if err != nil {
			t.Errorf("typecheck failed: %s", err)
			continue
		}
		testExportImport(t, pkg)
	}
}

func testExportImport(t *testing.T, pkg0 *types.Package) {
	data := ExportData(pkg0)

	imports := make(map[string]*types.Package)
	pkg1, err := ImportData(imports, data)
	if err != nil {
		t.Errorf("package %s: import failed: %s", pkg0.Name(), err)
		return
	}

	s0 := pkgString(pkg0)
	s1 := pkgString(pkg1)
	if s1 != s0 {
		t.Errorf("package %s: \ngot:\n%s\nwant:\n%s\n", pkg0.Name(), s1, s0)
	}
}

func pkgForSource(src string) (*types.Package, error) {
	// parse file
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		return nil, err
	}

	// typecheck file
	return types.Check("import-test", fset, []*ast.File{f})
}

func pkgFor(path string) (*types.Package, error) {
	// collect filenames
	ctxt := build.Default
	pkginfo, err := ctxt.ImportDir(path, 0)
	if _, nogo := err.(*build.NoGoError); err != nil && !nogo {
		return nil, err
	}
	filenames := append(pkginfo.GoFiles, pkginfo.CgoFiles...)

	// parse files
	fset := token.NewFileSet()
	files := make([]*ast.File, len(filenames))
	for i, filename := range filenames {
		var err error
		files[i], err = parser.ParseFile(fset, filepath.Join(path, filename), nil, 0)
		if err != nil {
			return nil, err
		}
	}

	// typecheck files
	return types.Check(path, fset, files)
}

func pkgString(pkg *types.Package) string {
	var buf bytes.Buffer

	fmt.Fprintf(&buf, "package %s\n", pkg.Name())

	scope := pkg.Scope()
	for _, name := range scope.Names() {
		if exported(name) {
			obj := scope.Lookup(name)
			buf.WriteString(obj.String())

			switch obj := obj.(type) {
			case *types.Const:
				fmt.Fprintf(&buf, " = %s", obj.Val())

			case *types.TypeName:
				// Basic types (e.g., unsafe.Pointer) have *types.Basic
				// type rather than *types.Named; so we need to check.
				if typ, _ := obj.Type().(*types.Named); typ != nil {
					if n := typ.NumMethods(); n > 0 {
						buf.WriteString("\nmethods (\n")
						for i := 0; i < n; i++ {
							fmt.Fprintf(&buf, "\t%s\n", typ.Method(i))
						}
						buf.WriteString(")")
					}
				}
			}
			buf.WriteByte('\n')
		}
	}

	return buf.String()
}
