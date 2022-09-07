// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"internal/testenv"
	"testing"

	. "go/types"
)

const filename = "<src>"

func makePkg(src string) (*Package, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filename, src, parser.DeclarationErrors)
	if err != nil {
		return nil, err
	}
	// use the package name as package path
	conf := Config{Importer: importer.Default()}
	return conf.Check(file.Name.Name, fset, []*ast.File{file}, nil)
}

type testEntry struct {
	src, str string
}

// dup returns a testEntry where both src and str are the same.
func dup(s string) testEntry {
	return testEntry{s, s}
}

// types that don't depend on any other type declarations
var independentTestTypes = []testEntry{
	// basic types
	dup("int"),
	dup("float32"),
	dup("string"),

	// arrays
	dup("[10]int"),

	// slices
	dup("[]int"),
	dup("[][]int"),

	// structs
	dup("struct{}"),
	dup("struct{x int}"),
	{`struct {
		x, y int
		z float32 "foo"
	}`, `struct{x int; y int; z float32 "foo"}`},
	{`struct {
		string
		elems []complex128
	}`, `struct{string; elems []complex128}`},

	// pointers
	dup("*int"),
	dup("***struct{}"),
	dup("*struct{a int; b float32}"),

	// functions
	dup("func()"),
	dup("func(x int)"),
	{"func(x, y int)", "func(x int, y int)"},
	{"func(x, y int, z string)", "func(x int, y int, z string)"},
	dup("func(int)"),
	{"func(int, string, byte)", "func(int, string, byte)"},

	dup("func() int"),
	{"func() (string)", "func() string"},
	dup("func() (u int)"),
	{"func() (u, v int, w string)", "func() (u int, v int, w string)"},

	dup("func(int) string"),
	dup("func(x int) string"),
	dup("func(x int) (u string)"),
	{"func(x, y int) (u string)", "func(x int, y int) (u string)"},

	dup("func(...int) string"),
	dup("func(x ...int) string"),
	dup("func(x ...int) (u string)"),
	{"func(x int, y ...int) (u string)", "func(x int, y ...int) (u string)"},

	// interfaces
	dup("interface{}"),
	dup("interface{m()}"),
	dup(`interface{String() string; m(int) float32}`),
	dup("interface{int | float32 | complex128}"),
	dup("interface{int | ~float32 | ~complex128}"),
	dup("any"),
	dup("interface{comparable}"),
	// TODO(gri) adjust test for EvalCompositeTest
	// {"comparable", "interface{comparable}"},
	// {"error", "interface{Error() string}"},

	// maps
	dup("map[string]int"),
	{"map[struct{x, y int}][]byte", "map[struct{x int; y int}][]byte"},

	// channels
	dup("chan<- chan int"),
	dup("chan<- <-chan int"),
	dup("<-chan <-chan int"),
	dup("chan (<-chan int)"),
	dup("chan<- func()"),
	dup("<-chan []func() int"),
}

// types that depend on other type declarations (src in TestTypes)
var dependentTestTypes = []testEntry{
	// interfaces
	dup(`interface{io.Reader; io.Writer}`),
	dup(`interface{m() int; io.Writer}`),
	{`interface{m() interface{T}}`, `interface{m() interface{p.T}}`},
}

func TestTypeString(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	var tests []testEntry
	tests = append(tests, independentTestTypes...)
	tests = append(tests, dependentTestTypes...)

	for _, test := range tests {
		src := `package p; import "io"; type _ io.Writer; type T ` + test.src
		pkg, err := makePkg(src)
		if err != nil {
			t.Errorf("%s: %s", src, err)
			continue
		}
		obj := pkg.Scope().Lookup("T")
		if obj == nil {
			t.Errorf("%s: T not found", test.src)
			continue
		}
		typ := obj.Type().Underlying()
		if got := typ.String(); got != test.str {
			t.Errorf("%s: got %s, want %s", test.src, got, test.str)
		}
	}
}

func TestQualifiedTypeString(t *testing.T) {
	p, _ := pkgFor("p.go", "package p; type T int", nil)
	q, _ := pkgFor("q.go", "package q", nil)

	pT := p.Scope().Lookup("T").Type()
	for _, test := range []struct {
		typ  Type
		this *Package
		want string
	}{
		{nil, nil, "<nil>"},
		{pT, nil, "p.T"},
		{pT, p, "T"},
		{pT, q, "p.T"},
		{NewPointer(pT), p, "*T"},
		{NewPointer(pT), q, "*p.T"},
	} {
		qualifier := func(pkg *Package) string {
			if pkg != test.this {
				return pkg.Name()
			}
			return ""
		}
		if got := TypeString(test.typ, qualifier); got != test.want {
			t.Errorf("TypeString(%s, %s) = %s, want %s",
				test.this, test.typ, got, test.want)
		}
	}
}
