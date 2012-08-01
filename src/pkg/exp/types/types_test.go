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

func makePkg(t *testing.T, src string) (*ast.Package, error) {
	const filename = "<src>"
	file, err := parser.ParseFile(fset, filename, src, parser.DeclarationErrors)
	if err != nil {
		return nil, err
	}
	files := map[string]*ast.File{filename: file}
	pkg, err := ast.NewPackage(fset, files, GcImport, Universe)
	if err != nil {
		return nil, err
	}
	if _, err := Check(fset, pkg); err != nil {
		return nil, err
	}
	return pkg, nil
}

type testEntry struct {
	src, str string
}

// dup returns a testEntry where both src and str are the same.
func dup(s string) testEntry {
	return testEntry{s, s}
}

var testTypes = []testEntry{
	// basic types
	dup("int"),
	dup("float32"),
	dup("string"),

	// arrays
	{"[10]int", "[0]int"}, // TODO(gri) fix array length, add more array tests

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
		elems []T
	}`, `struct{string; elems []T}`},

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
	dup("func(int, string, byte)"),

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
	{"func(x, y ...int) (u string)", "func(x int, y ...int) (u string)"},

	// interfaces
	dup("interface{}"),
	dup("interface{m()}"),
	{`interface{
		m(int) float32
		String() string
	}`, `interface{String() string; m(int) float32}`}, // methods are sorted
	// TODO(gri) add test for interface w/ anonymous field

	// maps
	dup("map[string]int"),
	{"map[struct{x, y int}][]byte", "map[struct{x int; y int}][]byte"},

	// channels
	dup("chan int"),
	dup("chan<- func()"),
	dup("<-chan []func() int"),
}

func TestTypes(t *testing.T) {
	for _, test := range testTypes {
		src := "package p; type T " + test.src
		pkg, err := makePkg(t, src)
		if err != nil {
			t.Errorf("%s: %s", src, err)
			continue
		}
		typ := Underlying(pkg.Scope.Lookup("T").Type.(Type))
		str := TypeString(typ)
		if str != test.str {
			t.Errorf("%s: got %s, want %s", test.src, str, test.str)
		}
	}
}
