// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parser

import (
	"os";
	"testing";
)


var illegalInputs = []interface{}{
	nil,
	3.14,
	[]byte(nil),
	"foo!",
}


func TestParseIllegalInputs(t *testing.T) {
	for _, src := range illegalInputs {
		_, err := ParseFile("", src, 0);
		if err == nil {
			t.Errorf("ParseFile(%v) should have failed", src)
		}
	}
}


var validPrograms = []interface{}{
	`package main`,
	`package main import "fmt" func main() { fmt.Println("Hello, World!") }`,
	`package main func main() { if f(T{}) {} }`,
}


func TestParseValidPrograms(t *testing.T) {
	for _, src := range validPrograms {
		_, err := ParseFile("", src, 0);
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
		_, err := ParseFile(filename, nil, 0);
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
	return true;
}


func dirFilter(d *os.Dir) bool	{ return nameFilter(d.Name) }


func TestParse4(t *testing.T) {
	path := ".";
	pkg, err := ParsePackage(path, dirFilter, 0);
	if err != nil {
		t.Fatalf("ParsePackage(%s): %v", path, err)
	}
	if pkg.Name != "parser" {
		t.Errorf("incorrect package name: %s", pkg.Name)
	}
	for filename, _ := range pkg.Files {
		if !nameFilter(filename) {
			t.Errorf("unexpected package file: %s", filename)
		}
	}
}
