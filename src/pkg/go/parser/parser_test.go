// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parser

import (
	"go/ast"
	"go/token"
	"os"
	"testing"
)

var fset = token.NewFileSet()

var illegalInputs = []interface{}{
	nil,
	3.14,
	[]byte(nil),
	"foo!",
	`package p; func f() { if /* should have condition */ {} };`,
	`package p; func f() { if ; /* should have condition */ {} };`,
	`package p; func f() { if f(); /* should have condition */ {} };`,
	`package p; const c; /* should have constant value */`,
	`package p; func f() { if _ = range x; true {} };`,
	`package p; func f() { switch _ = range x; true {} };`,
	`package p; func f() { for _ = range x ; ; {} };`,
	`package p; func f() { for ; ; _ = range x {} };`,
	`package p; func f() { for ; _ = range x ; {} };`,
	`package p; func f() { switch t = t.(type) {} };`,
	`package p; func f() { switch t, t = t.(type) {} };`,
	`package p; func f() { switch t = t.(type), t {} };`,
	`package p; var a = [1]int; /* illegal expression */`,
	`package p; var a = [...]int; /* illegal expression */`,
	`package p; var a = struct{} /* illegal expression */`,
	`package p; var a = func(); /* illegal expression */`,
	`package p; var a = interface{} /* illegal expression */`,
	`package p; var a = []int /* illegal expression */`,
	`package p; var a = map[int]int /* illegal expression */`,
	`package p; var a = chan int; /* illegal expression */`,
	`package p; var a = []int{[]int}; /* illegal expression */`,
	`package p; var a = ([]int); /* illegal expression */`,
	`package p; var a = a[[]int:[]int]; /* illegal expression */`,
	`package p; var a = <- chan int; /* illegal expression */`,
	`package p; func f() { select { case _ <- chan int: } };`,
}

func TestParseIllegalInputs(t *testing.T) {
	for _, src := range illegalInputs {
		_, err := ParseFile(fset, "", src, 0)
		if err == nil {
			t.Errorf("ParseFile(%v) should have failed", src)
		}
	}
}

var validPrograms = []interface{}{
	"package p\n",
	`package p;`,
	`package p; import "fmt"; func f() { fmt.Println("Hello, World!") };`,
	`package p; func f() { if f(T{}) {} };`,
	`package p; func f() { _ = (<-chan int)(x) };`,
	`package p; func f() { _ = (<-chan <-chan int)(x) };`,
	`package p; func f(func() func() func());`,
	`package p; func f(...T);`,
	`package p; func f(float, ...int);`,
	`package p; func f(x int, a ...int) { f(0, a...); f(1, a...,) };`,
	`package p; func f(int,) {};`,
	`package p; func f(...int,) {};`,
	`package p; func f(x ...int,) {};`,
	`package p; type T []int; var a []bool; func f() { if a[T{42}[0]] {} };`,
	`package p; type T []int; func g(int) bool { return true }; func f() { if g(T{42}[0]) {} };`,
	`package p; type T []int; func f() { for _ = range []int{T{42}[0]} {} };`,
	`package p; var a = T{{1, 2}, {3, 4}}`,
	`package p; func f() { select { case <- c: case c <- d: case c <- <- d: case <-c <- d: } };`,
	`package p; func f() { select { case x := (<-c): } };`,
	`package p; func f() { if ; true {} };`,
	`package p; func f() { switch ; {} };`,
	`package p; func f() { for _ = range "foo" + "bar" {} };`,
}

func TestParseValidPrograms(t *testing.T) {
	for _, src := range validPrograms {
		_, err := ParseFile(fset, "", src, SpuriousErrors)
		if err != nil {
			t.Errorf("ParseFile(%q): %v", src, err)
		}
	}
}

var validFiles = []string{
	"parser.go",
	"parser_test.go",
}

func TestParse3(t *testing.T) {
	for _, filename := range validFiles {
		_, err := ParseFile(fset, filename, nil, DeclarationErrors)
		if err != nil {
			t.Errorf("ParseFile(%s): %v", filename, err)
		}
	}
}

func nameFilter(filename string) bool {
	switch filename {
	case "parser.go":
	case "interface.go":
	case "parser_test.go":
	default:
		return false
	}
	return true
}

func dirFilter(f *os.FileInfo) bool { return nameFilter(f.Name) }

func TestParse4(t *testing.T) {
	path := "."
	pkgs, err := ParseDir(fset, path, dirFilter, 0)
	if err != nil {
		t.Fatalf("ParseDir(%s): %v", path, err)
	}
	if len(pkgs) != 1 {
		t.Errorf("incorrect number of packages: %d", len(pkgs))
	}
	pkg := pkgs["parser"]
	if pkg == nil {
		t.Errorf(`package "parser" not found`)
		return
	}
	for filename := range pkg.Files {
		if !nameFilter(filename) {
			t.Errorf("unexpected package file: %s", filename)
		}
	}
}

func TestColonEqualsScope(t *testing.T) {
	f, err := ParseFile(fset, "", `package p; func f() { x, y, z := x, y, z }`, 0)
	if err != nil {
		t.Errorf("parse: %s", err)
	}

	// RHS refers to undefined globals; LHS does not.
	as := f.Decls[0].(*ast.FuncDecl).Body.List[0].(*ast.AssignStmt)
	for _, v := range as.Rhs {
		id := v.(*ast.Ident)
		if id.Obj != nil {
			t.Errorf("rhs %s has Obj, should not", id.Name)
		}
	}
	for _, v := range as.Lhs {
		id := v.(*ast.Ident)
		if id.Obj == nil {
			t.Errorf("lhs %s does not have Obj, should", id.Name)
		}
	}
}

func TestVarScope(t *testing.T) {
	f, err := ParseFile(fset, "", `package p; func f() { var x, y, z = x, y, z }`, 0)
	if err != nil {
		t.Errorf("parse: %s", err)
	}

	// RHS refers to undefined globals; LHS does not.
	as := f.Decls[0].(*ast.FuncDecl).Body.List[0].(*ast.DeclStmt).Decl.(*ast.GenDecl).Specs[0].(*ast.ValueSpec)
	for _, v := range as.Values {
		id := v.(*ast.Ident)
		if id.Obj != nil {
			t.Errorf("rhs %s has Obj, should not", id.Name)
		}
	}
	for _, id := range as.Names {
		if id.Obj == nil {
			t.Errorf("lhs %s does not have Obj, should", id.Name)
		}
	}
}
