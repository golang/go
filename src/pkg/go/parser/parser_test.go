// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parser

import (
	"go/ast";
	"go/parser";
	"os";
	"testing";
)


var illegalInputs = []interface{} {
	nil,
	3.14,
	[]byte(nil),
	"foo!",
}


func TestParseIllegalInputs(t *testing.T) {
	for _, src := range illegalInputs {
		prog, err := ParseFile("", src, 0);
		if err == nil {
			t.Errorf("ParseFile(%v) should have failed", src);
		}
	}
}


var validPrograms = []interface{} {
	`package main`,
	`package main import "fmt" func main() { fmt.Println("Hello, World!") }`,
}


func TestParseValidPrograms(t *testing.T) {
	for _, src := range validPrograms {
		prog, err := ParseFile("", src, 0);
		if err != nil {
			t.Errorf("ParseFile(%q): %v", src, err);
		}
	}
}


var validFiles = []string {
	"parser.go",
	"parser_test.go",
}


func TestParse3(t *testing.T) {
	for _, filename := range validFiles {
		prog, err := ParseFile(filename, nil, 0);
		if err != nil {
			t.Errorf("ParseFile(%s): %v", filename, err);
		}
	}
}


func filter(filename string) bool {
	switch filename {
	case "parser.go":
	case "interface.go":
	case "parser_test.go":
	default:
		return false;
	}
	return true;
}


func TestParse4(t *testing.T) {
	path := ".";
	pkg, err := ParsePackage(path, filter, 0);
	if err != nil {
		t.Errorf("ParsePackage(%s): %v", path, err);
	}
	if pkg.Name != "parser" {
		t.Errorf("incorrect package name: %s", pkg.Name);
	}
	for filename, _ := range pkg.Files {
		if !filter(filename) {
			t.Errorf("unexpected package file: %s", filename);
		}
	}
}
