// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parser

import (
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
	"package main\n",
	`package main;`,
	`package main; import "fmt"; func main() { fmt.Println("Hello, World!") };`,
	`package main; func main() { if f(T{}) {} };`,
	`package main; func main() { _ = (<-chan int)(x) };`,
	`package main; func main() { _ = (<-chan <-chan int)(x) };`,
	`package main; func f(func() func() func());`,
	`package main; func f(...T);`,
	`package main; func f(float, ...int);`,
	`package main; func f(x int, a ...int) { f(0, a...); f(1, a...,) };`,
	`package main; type T []int; var a []bool; func f() { if a[T{42}[0]] {} };`,
	`package main; type T []int; func g(int) bool { return true }; func f() { if g(T{42}[0]) {} };`,
	`package main; type T []int; func f() { for _ = range []int{T{42}[0]} {} };`,
	`package main; var a = T{{1, 2}, {3, 4}}`,
}


func TestParseValidPrograms(t *testing.T) {
	for _, src := range validPrograms {
		_, err := ParseFile(fset, "", src, 0)
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
		_, err := ParseFile(fset, filename, nil, 0)
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
