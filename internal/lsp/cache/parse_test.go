// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"go/types"
	"reflect"
	"sort"
	"testing"

	"golang.org/x/tools/go/packages"
)

func TestArrayLength(t *testing.T) {
	tests := []struct {
		expr   string
		length int
	}{
		{`[...]int{0,1,2,3,4,5,6,7,8,9}`, 10},
		{`[...]int{9:0}`, 10},
		{`[...]int{19-10:0}`, 10},
		{`[...]int{19-10:0, 17-10:0, 18-10:0}`, 10},
	}

	for _, tt := range tests {
		expr, err := parser.ParseExpr(tt.expr)
		if err != nil {
			t.Fatal(err)
		}
		l, ok := arrayLength(expr.(*ast.CompositeLit))
		if !ok {
			t.Errorf("arrayLength did not recognize expression %#v", expr)
		}
		if l != tt.length {
			t.Errorf("arrayLength(%#v) = %v, want %v", expr, l, tt.length)
		}
	}
}

func TestTrim(t *testing.T) {
	tests := []struct {
		name string
		file string
		kept []string
	}{
		{
			name: "delete_unused",
			file: `
type x struct{}
func y()
var z int
`,
			kept: []string{},
		},
		{
			// From the common type in testing.
			name: "unexported_embedded",
			file: `
type x struct {}
type Exported struct { x }
`,
			kept: []string{"Exported", "x"},
		},
		{
			// From the d type in unicode.
			name: "exported_field_unexported_type",
			file: `
type x struct {}
type Exported struct {
	X x
}
`,
			kept: []string{"Exported", "x"},
		},
		{
			// From errNotExist in io/fs.
			name: "exported_var_function_call",
			file: `
func x() int { return 0 }
var Exported = x()
`,
			kept: []string{"Exported", "x"},
		},
		{
			// From DefaultServeMux in net/http.
			name: "exported_pointer_to_unexported_var",
			file: `
var Exported = &x
var x int
`,
			kept: []string{"Exported", "x"},
		},
		{
			// From DefaultWriter in goldmark/renderer/html.
			name: "exported_pointer_to_composite_lit",
			file: `
var Exported = &x{}
type x struct{}
`,
			kept: []string{"Exported", "x"},
		},
		{
			// From SelectDir in reflect.
			name: "leave_constants",
			file: `
type Enum int
const (
	_             Enum = iota
	EnumOne
)
`,
			kept: []string{"Enum", "EnumOne"},
		},
		{
			name: "constant_conversion",
			file: `
type x int
const (
	foo x = 0
)
`,
			kept: []string{"x", "foo"},
		},
		{
			name: "unexported_return",
			file: `
type x int
func Exported() x {}
type y int
type Interface interface {
	Exported() y
}
`,
			kept: []string{"Exported", "Interface", "x", "y"},
		},
		{
			name: "drop_composite_literals",
			file: `
type x int
type Exported struct {
	foo x
}
var Var = Exported{foo:1}
`,
			kept: []string{"Exported", "Var"},
		},
		{
			name: "drop_function_literals",
			file: `
type x int
var Exported = func() { return x(0) }
`,
			kept: []string{"Exported"},
		},
		{
			name: "missing_receiver_panic",
			file: `
			func() foo() {}
`,
			kept: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fset := token.NewFileSet()
			file, err := parser.ParseFile(fset, "main.go", "package main\n\n"+tt.file, parser.AllErrors)
			if err != nil {
				t.Fatal(err)
			}
			filter := &unexportedFilter{uses: map[string]bool{}}
			filter.Filter([]*ast.File{file})
			pkg := types.NewPackage("main", "main")
			checker := types.NewChecker(&types.Config{
				DisableUnusedImportCheck: true,
			}, fset, pkg, nil)
			if err := checker.Files([]*ast.File{file}); err != nil {
				t.Error(err)
			}
			names := pkg.Scope().Names()
			sort.Strings(names)
			sort.Strings(tt.kept)
			if !reflect.DeepEqual(names, tt.kept) {
				t.Errorf("package contains names %v, wanted %v", names, tt.kept)
			}
		})
	}
}

func TestPkg(t *testing.T) {
	t.Skip("for manual debugging")
	fset := token.NewFileSet()
	pkgs, err := packages.Load(&packages.Config{
		Mode: packages.NeedSyntax | packages.NeedFiles,
		Fset: fset,
	}, "io")
	if err != nil {
		t.Fatal(err)
	}
	if len(pkgs[0].Errors) != 0 {
		t.Fatal(pkgs[0].Errors)
	}
	filter := &unexportedFilter{uses: map[string]bool{}}
	filter.Filter(pkgs[0].Syntax)
	for _, file := range pkgs[0].Syntax {
		buf := &bytes.Buffer{}
		format.Node(buf, fset, file)
		t.Log(buf.String())
	}
}
